from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCD(nn.Module):
    """PCD trains a multinomial RBM with persistent Gibbs chains and property-guided discrete sampling.

    Originally persistent contrastive divergence for binary RBMs (Tieleman, ICML 2008). This RNA
    adaptation uses multinomial visibles (one categorical unit per nucleotide site), paper-style
    PCD training (persistent fantasies, one Gibbs step per update, chains = minibatch), and
    discrete property-guided Gibbs for seed-started optimization.

    Paper: Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient (ICML 2008)
    Github: N/A
    """

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        hidden_dim: int = 256,
        n_gibbs: int = 1,
        buffer_size: int = 128,
        num_properties: int = 1,
        prop_hidden: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.n_gibbs = int(n_gibbs)
        self.buffer_size = int(buffer_size)
        self.num_properties = int(num_properties)
        self.pad_token_id = int(pad_token_id)

        # Multinomial RBM: one categorical visible per site.
        # E(v,h) = -sum_i b[i,v_i] - sum_j c_j h_j - sum_{i,j} W[i,v_i,j] h_j
        self.W = nn.Parameter(torch.randn(self.seq_len, self.vocab_size, self.hidden_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.seq_len, self.vocab_size))
        self.c = nn.Parameter(torch.zeros(self.hidden_dim))

        # Persistent fantasy particles as token ids [buffer, L] (paper: chains ~= minibatch).
        # Training-only state; not required for guided optimize from seeds.
        init = torch.randint(1, max(self.vocab_size, 2), (self.buffer_size, self.seq_len))
        self.register_buffer("fantasy_tokens", init, persistent=False)

        self.prop_input = nn.Linear(self.vocab_size, prop_hidden)
        self.property_head = nn.Sequential(
            nn.Linear(prop_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_properties),
        )
        self.to(self.device)

    # ------------------------------------------------------------------ utils
    def _one_hot(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.one_hot(tokens.long(), num_classes=self.vocab_size).float()

    def _default_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens != self.pad_token_id

    def free_energy(self, tokens: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """RBM free energy F(v) for multinomial token sequences ``[B, L]``."""
        oh = self._one_hot(tokens)
        if pad_mask is None:
            pad_mask = self._default_pad_mask(tokens)
        oh = oh * pad_mask.unsqueeze(-1).float()
        fe = -(oh * self.b.unsqueeze(0)).sum(dim=(1, 2))
        pre_h = self.c.unsqueeze(0) + torch.einsum("blv,lvh->bh", oh, self.W)
        return fe - F.softplus(pre_h).sum(dim=-1)

    # --------------------------------------------------------------- Gibbs
    @torch.no_grad()
    def _sample_h_given_v(self, tokens: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        oh = self._one_hot(tokens)
        if pad_mask is None:
            pad_mask = self._default_pad_mask(tokens)
        oh = oh * pad_mask.unsqueeze(-1).float()
        p_h = torch.sigmoid(self.c.unsqueeze(0) + torch.einsum("blv,lvh->bh", oh, self.W))
        return torch.bernoulli(p_h)

    def _visible_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Categorical logits ``[B, L, V]`` for visibles given hidden sample ``[B, H]``."""
        return self.b.unsqueeze(0) + torch.einsum("bh,lvh->blv", h, self.W)

    @torch.no_grad()
    def _sample_v_given_h(
        self,
        h: torch.Tensor,
        *,
        pad_mask: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multinomial Gibbs update of visibles (pad sites stay pad; pad id blocked elsewhere)."""
        if logits is None:
            logits = self._visible_logits(h)
        logits = logits.clone()
        # Never emit pad on content sites.
        logits[..., self.pad_token_id] = -1e9
        probs = F.softmax(logits, dim=-1)
        flat = probs.reshape(-1, self.vocab_size)
        sampled = torch.multinomial(flat, num_samples=1).reshape(h.shape[0], self.seq_len)
        if pad_mask is None:
            return sampled
        pad_mask = pad_mask.bool()
        return torch.where(pad_mask, sampled, torch.full_like(sampled, self.pad_token_id))

    @torch.no_grad()
    def _gibbs_k(
        self,
        tokens: torch.Tensor,
        k: int,
        *,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = tokens
        for _ in range(max(int(k), 1)):
            h = self._sample_h_given_v(out, pad_mask=pad_mask)
            out = self._sample_v_given_h(h, pad_mask=pad_mask)
        return out

    def _ensure_fantasy(self, batch_size: int, template: torch.Tensor) -> None:
        """Ensure fantasy buffer covers ``batch_size`` chains (never shrink mid-epoch)."""
        need_l = int(template.shape[1])
        have = self.fantasy_tokens
        if have is not None and have.shape[0] >= batch_size and have.shape[1] == need_l:
            return
        n = max(int(self.buffer_size), int(batch_size), 1 if have is None else int(have.shape[0]))
        init = torch.randint(
            1,
            max(self.vocab_size, 2),
            (n, need_l),
            device=template.device,
            dtype=torch.long,
        )
        pad_mask = template[0] != self.pad_token_id
        init = torch.where(pad_mask.unsqueeze(0), init, torch.full_like(init, self.pad_token_id))
        if have is not None and have.shape[1] == need_l and have.shape[0] > 0:
            n_copy = min(have.shape[0], n)
            init[:n_copy] = have[:n_copy]
        self.register_buffer("fantasy_tokens", init, persistent=False)
        self.buffer_size = n

    @torch.no_grad()
    def _pcd_negatives(self, tokens_pos: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Advance persistent chains by ``n_gibbs`` Gibbs steps; return negatives."""
        batch_size = tokens_pos.shape[0]
        self._ensure_fantasy(batch_size, tokens_pos)
        chains = self.fantasy_tokens[:batch_size]
        v_k = self._gibbs_k(chains, self.n_gibbs, pad_mask=pad_mask)
        self.fantasy_tokens[:batch_size].copy_(v_k)
        return v_k

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Drop training-only fantasy chains (size varies with last minibatch).
        state_dict = {k: v for k, v in state_dict.items() if k != "fantasy_tokens"}
        return super().load_state_dict(state_dict, strict=False, assign=assign)

    # ------------------------------------------------------------- property
    def predict_property(
        self,
        tokens_or_oh: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property head on one-hot ``[B, L, V]`` or token ids ``[B, L]``."""
        if tokens_or_oh.dim() == 2:
            oh = self._one_hot(tokens_or_oh)
            if pad_mask is None:
                pad_mask = self._default_pad_mask(tokens_or_oh)
        else:
            oh = tokens_or_oh.float()
        h = self.prop_input(oh)
        if pad_mask is None:
            pooled = h.mean(dim=1)
        else:
            w = pad_mask.float().unsqueeze(-1)
            pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return self.property_head(pooled)

    # ---------------------------------------------------------------- training
    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PCD loss ``F(data) - F(fantasy)`` + property MSE (Tieleman 2008 + RNA property head)."""
        if x_0.dim() != 2:
            raise ValueError(f"Expected token ids [B, L], got shape {tuple(x_0.shape)}")

        tokens = x_0.long()
        pad_mask = mask if mask is not None else self._default_pad_mask(tokens)
        v_neg = self._pcd_negatives(tokens, pad_mask)

        fe_pos = self.free_energy(tokens, pad_mask=pad_mask)
        fe_neg = self.free_energy(v_neg, pad_mask=pad_mask)
        gen_loss = fe_pos.mean() - fe_neg.mean()

        prop_pred = self.predict_property(tokens, pad_mask=pad_mask)
        if targets is None:
            prop_loss = torch.zeros((), device=tokens.device)
        else:
            targets = targets.float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            prop_loss = F.mse_loss(prop_pred, targets)

        total = recon_weight * gen_loss + prop_loss
        return total, gen_loss, prop_loss, prop_pred

    # ------------------------------------------------------ discrete sample
    def _guided_visible_logits(
        self,
        tokens: torch.Tensor,
        h: torch.Tensor,
        *,
        pad_mask: Optional[torch.Tensor],
        guidance: bool,
        direction: float,
        guide_scale: float,
    ) -> torch.Tensor:
        """RBM visible logits plus a discrete property bias from ``∇_onehot y``."""
        logits = self._visible_logits(h)
        if not (guidance and self.num_properties > 0 and guide_scale != 0.0):
            return logits
        oh = self._one_hot(tokens).detach().requires_grad_(True)
        pred = self.predict_property(oh, pad_mask=pad_mask)
        grad_oh = torch.autograd.grad(pred.sum(), oh, create_graph=False)[0].detach()
        return logits + float(direction) * float(guide_scale) * grad_oh

    def sample(
        self,
        batch_size: int,
        *,
        seed_tokens: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance: bool = True,
        direction: float = 1.0,
        guide_scale: float = 1.0,
        pad_mask: Optional[torch.Tensor] = None,
        return_traj: bool = False,
    ):
        """Persistent-style multinomial Gibbs with optional property-biased visibles."""
        if seed_tokens is None:
            self._ensure_fantasy(batch_size, self.fantasy_tokens)
            tokens = self.fantasy_tokens[:batch_size].clone()
            if pad_mask is None:
                pad_mask = torch.ones(batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        else:
            tokens = seed_tokens.long().to(self.device)
            batch_size = tokens.shape[0]
            if pad_mask is None:
                pad_mask = self._default_pad_mask(tokens)
            else:
                pad_mask = pad_mask.to(self.device).bool()

        traj = []
        for _ in range(max(int(num_steps), 1)):
            with torch.no_grad():
                h = self._sample_h_given_v(tokens, pad_mask=pad_mask)
            logits = self._guided_visible_logits(
                tokens,
                h,
                pad_mask=pad_mask,
                guidance=guidance,
                direction=float(direction),
                guide_scale=float(guide_scale),
            )
            with torch.no_grad():
                tokens = self._sample_v_given_h(h, pad_mask=pad_mask, logits=logits.detach())
            if return_traj:
                traj.append(tokens.detach().clone())

        if return_traj:
            return tokens, torch.stack(traj, dim=0) if traj else tokens.unsqueeze(0)
        return tokens

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        num_steps: int = 50,
        guide_scale: float = 1.0,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property-guided multinomial Gibbs starting from seed sequences ``[B, L]``."""
        direction = 1.0 if target_direction == "increase" else -1.0
        return self.sample(
            batch_size=sequences.shape[0],
            seed_tokens=sequences,
            num_steps=num_steps,
            guidance=True,
            direction=direction,
            guide_scale=guide_scale,
            pad_mask=pad_mask,
            return_traj=False,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 8, 32, 7
    model = PCD(vocab_size=vocab_size, seq_len=seq_len, device="cpu", buffer_size=8, n_gibbs=1)
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    mask = sequences != 0

    total, gen, prop, pred = model.compute_loss(sequences, targets=targets, mask=mask)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)
    assert torch.isfinite(total), total
    total.backward()

    generated = model.sample(batch_size=4, num_steps=5, guidance=False)
    assert generated.shape == (4, seq_len)
    assert (generated != 0).all()

    optimized = model.optimize(sequences[:4], target_direction="increase", num_steps=5)
    assert optimized.shape == (4, seq_len)
    print("PCD unit tests passed.")
