from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCD(nn.Module):
    """PCD trains an energy-based model with persistent Markov chains and samples with property-guided dynamics.

    Originally persistent contrastive divergence for energy-based / RBM training. This RNA adaptation
    applies PCD to nucleotide sequence modeling with property-guided sampling.

    Paper: Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient (ICML 2008)
    Github: N/A
    """

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        hidden_dim: int = 256,
        n_gibbs: int = 5,
        buffer_size: int = 256,
        langevin_step: float = 0.05,
        langevin_noise: float = 0.01,
        num_properties: int = 1,
        prop_hidden: int = 128,
        dropout: float = 0.1,
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
        self.visible_dim = self.seq_len * self.vocab_size
        self.hidden_dim = int(hidden_dim)
        self.n_gibbs = int(n_gibbs)
        self.buffer_size = int(buffer_size)
        self.langevin_step = float(langevin_step)
        self.langevin_noise = float(langevin_noise)
        self.num_properties = int(num_properties)

        # Classical RBM parameters: E(v,h) = -v^T W h - b^T v - c^T h
        self.W = nn.Parameter(torch.randn(self.visible_dim, self.hidden_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.visible_dim))
        self.c = nn.Parameter(torch.zeros(self.hidden_dim))

        # Persistent fantasy particles in {0,1}^{D} (registered buffer, not a param).
        self.register_buffer(
            "fantasy_v",
            torch.bernoulli(torch.full((self.buffer_size, self.visible_dim), 0.5)),
            persistent=True,
        )
        self.register_buffer("_fantasy_ptr", torch.zeros((), dtype=torch.long), persistent=True)

        # Property head on continuous sequence view (joint training).
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
    def _tokens_to_binary(self, token_ids: torch.Tensor) -> torch.Tensor:
        """One-hot flatten to binary visibles ``[B, L*V]`` in {0,1}."""
        oh = F.one_hot(token_ids.long(), num_classes=self.vocab_size).float()
        return oh.reshape(token_ids.shape[0], self.visible_dim)

    def _tokens_to_continuous(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Continuous one-hots in [-1,1], shape ``[B, L, V]``."""
        return F.one_hot(token_ids.long(), num_classes=self.vocab_size).float() * 2.0 - 1.0

    def _binary_to_seq(self, v: torch.Tensor) -> torch.Tensor:
        """Map flat visibles ``[B, L*V]`` -> sequence continuous ``[B, L, V]`` in [-1,1]."""
        x = v.reshape(-1, self.seq_len, self.vocab_size)
        return x * 2.0 - 1.0

    def _seq_to_flat01(self, x: torch.Tensor) -> torch.Tensor:
        """Map continuous ``[B,L,V]`` in [-1,1] -> soft visibles in [0,1]."""
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0).reshape(x.shape[0], self.visible_dim)

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """RBM free energy ``F(v)`` for soft/binary visibles ``[B, D]``.

        ``F(v) = -b^T v - sum_j softplus(c_j + W_j^T v)``
        (up to an additive constant; lower = more likely under the model).
        """
        pre_h = self.c.unsqueeze(0) + v @ self.W
        return -v @ self.b - F.softplus(pre_h).sum(dim=-1)

    def energy(self, continuous_x: torch.Tensor) -> torch.Tensor:
        """Free energy of continuous sequence ``[B, L, V]`` in [-1,1]."""
        return self.free_energy(self._seq_to_flat01(continuous_x))

    # --------------------------------------------------------------- PCD Gibbs
    @torch.no_grad()
    def _sample_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        p_h = torch.sigmoid(self.c.unsqueeze(0) + v @ self.W)
        return torch.bernoulli(p_h)

    @torch.no_grad()
    def _sample_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        p_v = torch.sigmoid(self.b.unsqueeze(0) + h @ self.W.t())
        return torch.bernoulli(p_v)

    @torch.no_grad()
    def _gibbs_k(self, v: torch.Tensor, k: int) -> torch.Tensor:
        out = v
        for _ in range(max(int(k), 1)):
            h = self._sample_h_given_v(out)
            out = self._sample_v_given_h(h)
        return out

    @torch.no_grad()
    def _pcd_negatives(self, batch_size: int) -> torch.Tensor:
        """Advance persistent chains and return ``batch_size`` negatives."""
        n = self.fantasy_v.shape[0]
        # Round-robin slice from the fantasy buffer.
        ptr = int(self._fantasy_ptr.item()) % n
        idx = [(ptr + i) % n for i in range(batch_size)]
        idx_t = torch.as_tensor(idx, device=self.fantasy_v.device, dtype=torch.long)
        v0 = self.fantasy_v[idx_t]
        v_k = self._gibbs_k(v0, self.n_gibbs)
        self.fantasy_v[idx_t] = v_k
        self._fantasy_ptr.fill_((ptr + batch_size) % n)
        return v_k

    # ------------------------------------------------------------- property
    def predict_property(
        self,
        continuous_x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.prop_input(continuous_x)
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
        """PCD contrastive loss ``E_data - E_model`` + property MSE.

        Args:
            x_0: token ids ``[B, L]``
            targets: property labels ``[B]`` or ``[B, P]``
            mask: bool pad mask True=valid (used by property head)
            recon_weight: weight on generative loss

        Returns:
            total_loss, gen_loss (PCD), property_loss, property_pred ``[B, P]``
        """
        if x_0.dim() != 2:
            raise ValueError(f"Expected token ids [B, L], got shape {tuple(x_0.shape)}")

        v_pos = self._tokens_to_binary(x_0)
        v_neg = self._pcd_negatives(v_pos.shape[0]).to(dtype=v_pos.dtype)

        # Contrastive divergence objective on free energies.
        fe_pos = self.free_energy(v_pos)
        fe_neg = self.free_energy(v_neg)
        gen_loss = fe_pos.mean() - fe_neg.mean()

        continuous = self._tokens_to_continuous(x_0)
        prop_pred = self.predict_property(continuous, pad_mask=mask)
        if targets is None:
            prop_loss = torch.zeros((), device=x_0.device)
        else:
            targets = targets.float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            prop_loss = F.mse_loss(prop_pred, targets)

        total = recon_weight * gen_loss + prop_loss
        return total, gen_loss, prop_loss, prop_pred

    # ------------------------------------------------------ Langevin sample
    def _energy_grad(
        self,
        x: torch.Tensor,
        *,
        pad_mask: Optional[torch.Tensor],
        guidance: bool,
        direction: float,
        guide_scale: float,
    ) -> torch.Tensor:
        """``∇_x ( F(x) - direction * guide_scale * property(x) )``."""
        x_req = x.detach().requires_grad_(True)
        fe = self.energy(x_req)
        objective = fe.sum()
        if guidance and self.num_properties > 0 and guide_scale != 0.0:
            pred = self.predict_property(x_req, pad_mask=pad_mask)
            # Ascend property when direction > 0 => subtract from energy objective.
            objective = objective - float(direction) * float(guide_scale) * pred.sum()
        return torch.autograd.grad(objective, x_req, create_graph=False)[0]

    @torch.no_grad()
    def _decode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=-1)

    def sample(
        self,
        batch_size: int,
        *,
        seed_tokens: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        step_size: Optional[float] = None,
        noise_std: Optional[float] = None,
        guidance: bool = True,
        direction: float = 1.0,
        guide_scale: float = 1.0,
        pad_mask: Optional[torch.Tensor] = None,
        return_traj: bool = False,
    ):
        """Langevin dynamics on free energy (+ optional property guidance).

        ``dx = -∇F dt + σ dW``, with optional ``- direction * guide * ∇property``.
        """
        dt = float(self.langevin_step if step_size is None else step_size)
        sigma = float(self.langevin_noise if noise_std is None else noise_std)

        if seed_tokens is None:
            # Start near the PCD fantasy manifold (random buffer rows) for de novo.
            with torch.no_grad():
                n = self.fantasy_v.shape[0]
                idx = torch.randint(0, n, (batch_size,), device=self.device)
                v0 = self._gibbs_k(self.fantasy_v[idx], k=max(self.n_gibbs, 1))
            x = self._binary_to_seq(v0.float())
            if pad_mask is None:
                pad_mask = torch.ones(
                    batch_size, self.seq_len, dtype=torch.bool, device=self.device
                )
        else:
            seed_tokens = seed_tokens.to(self.device)
            batch_size = seed_tokens.shape[0]
            x = self._tokens_to_continuous(seed_tokens)
            if pad_mask is None:
                pad_mask = seed_tokens != 0

        traj = []
        for _ in range(max(int(num_steps), 1)):
            with torch.enable_grad():
                grad = self._energy_grad(
                    x,
                    pad_mask=pad_mask,
                    guidance=guidance,
                    direction=float(direction),
                    guide_scale=float(guide_scale),
                )
            noise = torch.randn_like(x) * sigma
            x = (x - dt * grad.detach() + noise).clamp(-1.0, 1.0)
            if return_traj:
                traj.append(self._decode_tokens(x).detach())

        tokens = self._decode_tokens(x)
        if return_traj:
            return tokens, torch.stack(traj, dim=0) if traj else tokens.unsqueeze(0)
        return tokens

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        target_direction: str = "increase",
        num_steps: int = 50,
        step_size: Optional[float] = None,
        noise_std: Optional[float] = None,
        guide_scale: float = 1.0,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property-guided Langevin starting from seed sequences ``[B, L]``."""
        direction = 1.0 if target_direction == "increase" else -1.0
        return self.sample(
            batch_size=sequences.shape[0],
            seed_tokens=sequences,
            num_steps=num_steps,
            step_size=step_size,
            noise_std=noise_std,
            guidance=True,
            direction=direction,
            guide_scale=guide_scale,
            pad_mask=pad_mask,
            return_traj=False,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 8, 32, 7
    model = PCD(vocab_size=vocab_size, seq_len=seq_len, device="cpu", buffer_size=32)
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    mask = sequences != 0

    total, gen, prop, pred = model.compute_loss(sequences, targets=targets, mask=mask)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)
    assert torch.isfinite(total), total
    total.backward()

    generated = model.sample(batch_size=4, num_steps=5, guidance=False)
    assert generated.shape == (4, seq_len)

    optimized = model.optimize(sequences[:4], target_direction="increase", num_steps=5)
    assert optimized.shape == (4, seq_len)
    print("PCD unit tests passed.")
