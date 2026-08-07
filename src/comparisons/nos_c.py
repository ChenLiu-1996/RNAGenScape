from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class NOS_C(nn.Module):
    """NOS-C guides continuous embedding-space diffusion with property gradients.

    Originally continuous guided diffusion for protein design (Gaussian corruption /
    transitions in token embedding space, with a categorical denoiser). This RNA
    adaptation applies the approach to nucleotide sequences.

    Paper: Protein Design with Guided Discrete Diffusion (NeurIPS 2023)
    Github: https://github.com/ngruver/NOS
    """

    def __init__(
        self,
        seq_len: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_timesteps: int = 100,
        num_properties: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        # Kept for API compatibility with older call sites; unused (no noise MSE).
        noise_scale: float = 1.0,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.max_timesteps = int(max_timesteps)
        self.noise_scale = float(noise_scale)  # unused; continuous schedule is unit Gaussian
        self.num_properties = num_properties

        # Continuous path has no [MASK] token (that is NOS-D).
        self.rna_vocab = {"PAD": 0, "A": 1, "G": 2, "C": 3, "T": 4, "U": 5, "N": 6}
        self.idx_to_base = {0: "PAD", 1: "A", 2: "G", 3: "C", 4: "T", 5: "U", 6: "N"}
        self.vocab_size = len(self.rna_vocab)
        self.pad_id = 0

        self.token_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, seq_len + 100)
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.transformer_layers = nn.ModuleList(
            [NOSTransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        # Predict clean tokens from continuous noisy embeddings (paper §3 continuous noise).
        self.token_pred_head = nn.Linear(hidden_dim, self.vocab_size)
        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_properties),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Linear cumulative schedule ``ᾱ_t = 1 - t / T`` in ``(eps, 1]``."""
        T = float(max(self.max_timesteps, 1))
        ab = 1.0 - (t.float() / T)
        return ab.clamp(min=1e-4, max=1.0)

    def sequence_to_ids(self, sequences):
        if isinstance(sequences, str):
            sequences = [sequences]
        batch_ids = []
        for seq in sequences:
            ids = [self.rna_vocab.get(base.upper(), 0) for base in seq]
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long, device=self.device)

    def ids_to_sequence(self, token_ids):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        sequences = []
        for ids in token_ids:
            seq = "".join(
                [self.idx_to_base.get(idx.item(), "N") for idx in ids if idx.item() != 0]
            )
            sequences.append(seq)
        return sequences if len(sequences) > 1 else sequences[0]

    def encode_backbone(self, x: torch.Tensor, timesteps: torch.Tensor, mask=None):
        """Run continuous embeddings ``x`` [B,L,H] through the time-conditioned backbone."""
        if mask is not None:
            mask = mask.to(x.device)
        h = self.pos_encoding(x)
        time_emb = self.time_embedding(timesteps)
        for layer in self.transformer_layers:
            h = layer(h, time_emb, mask)
        return self.output_norm(h)

    def forward(self, x, timesteps, mask=None, return_properties=True):
        """
        Args:
            x: continuous embeddings [B, L, H] (or token ids [B, L], embedded first)
            timesteps: [B]
            mask: [B, L] True = valid (non-pad) positions
        Returns:
            token_logits [B, L, V], optional property_pred [B, P]
        """
        if x.dim() == 2:
            x = self.token_embedding(x)
        h = self.encode_backbone(x, timesteps, mask)
        token_logits = self.token_pred_head(h)
        if not return_properties:
            return token_logits
        if mask is not None:
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(
                min=1
            )
        else:
            pooled = h.mean(dim=1)
        property_pred = self.property_head(pooled)
        return token_logits, property_pred

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor):
        """Gaussian forward corruption on embeddings: ``x_t = √ᾱ x_0 + √(1-ᾱ) ε``."""
        if x_0.dim() == 2:
            x_0 = self.token_embedding(x_0)
        ab = self.alpha_bar(t).view(-1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1.0 - ab) * noise
        return x_t, noise

    def compute_loss(self, x_0, targets=None, mask=None, recon_weight=1.0):
        """
        Train token CE from continuous noisy embeddings (+ property MSE).

        Args:
            x_0: clean token ids [B, L]
            targets: property targets [B, P]
            mask: attention / non-pad mask [B, L]
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.max_timesteps, (batch_size,), device=x_0.device)
        x_t, _noise = self.add_noise(x_0, t)
        token_logits, property_pred = self.forward(x_t, t, mask)

        # CE over non-pad positions (continuous denoising of the sequence).
        ce = F.cross_entropy(
            token_logits.reshape(-1, self.vocab_size),
            x_0.reshape(-1),
            reduction="none",
        ).view(batch_size, -1)
        if mask is not None:
            m = mask.float()
            recon_loss = (ce * m).sum() / m.sum().clamp(min=1.0)
        else:
            recon_loss = ce.mean()

        prop_loss = torch.zeros((), device=x_0.device)
        if targets is not None:
            prop_loss = F.mse_loss(property_pred, targets)

        total_loss = recon_weight * recon_loss + prop_loss
        return total_loss, recon_loss, prop_loss, property_pred

    def _decode_tokens(self, token_logits: torch.Tensor, *, temperature: float = 1.0):
        """Sample / argmax tokens from logits; never emit PAD during generation."""
        logits = token_logits.clone()
        logits[..., self.pad_id] = float("-inf")
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs.view(-1, self.vocab_size), 1).view(
            token_logits.shape[0], token_logits.shape[1]
        )

    def _continuous_transition(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        mask=None,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """NOS-C reverse step: decode tokens → re-embed → Gaussian re-noise to ``t_prev``.

        This is the continuous transition (embedding-space), not a discrete token jump.
        """
        token_logits, _ = self.forward(x_t, t, mask)
        tokens = self._decode_tokens(token_logits, temperature=temperature)
        # Respect pad positions in the attention mask if provided.
        if mask is not None:
            tokens = torch.where(mask, tokens, torch.zeros_like(tokens))
        x0_hat = self.token_embedding(tokens)
        ab_prev = self.alpha_bar(t_prev).view(-1, 1, 1)
        # When t_prev == 0, return the clean embedding estimate (no extra noise).
        noise = torch.randn_like(x0_hat)
        x_prev = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * noise
        at_zero = (t_prev <= 0).view(-1, 1, 1)
        return torch.where(at_zero, x0_hat, x_prev)

    def _langevin_guide(self, x_t: torch.Tensor, t: torch.Tensor, guidance_kwargs, mask):
        """Property gradient steps in continuous embedding space (NOS guidance)."""
        step_size = float(guidance_kwargs.get("step_size", 0.1))
        stability_coef = float(guidance_kwargs.get("stability_coef", 0.01))
        n_langevin = int(guidance_kwargs.get("n_langevin", 1))
        target_values = guidance_kwargs["target_values"]

        x = x_t
        for _ in range(max(n_langevin, 1)):
            x = x.detach().requires_grad_(True)
            _logits, property_pred = self.forward(x, t, mask)
            if isinstance(target_values, (int, float)):
                tgt = torch.full_like(property_pred, float(target_values))
            elif isinstance(target_values, (list, tuple)):
                tgt = torch.tensor(target_values, device=self.device, dtype=property_pred.dtype)
                tgt = tgt.view(1, -1).expand_as(property_pred)
            else:
                tgt = target_values.to(device=self.device, dtype=property_pred.dtype)
                if tgt.dim() == 1:
                    tgt = tgt.unsqueeze(-1)
                tgt = tgt.expand_as(property_pred)

            guidance_loss = F.mse_loss(property_pred, tgt)
            grad = torch.autograd.grad(guidance_loss, x, retain_graph=False)[0]
            with torch.no_grad():
                x = x - step_size * grad
                if stability_coef > 0:
                    x = x + stability_coef * torch.randn_like(x)
        return x.detach()

    def sample(
        self,
        batch_size,
        num_steps=50,
        hijack_step=None,
        guidance_kwargs=None,
        mask=None,
        return_traj=False,
        temperature: float = 1.0,
    ):
        """Sample via continuous reverse diffusion in embedding space."""
        if guidance_kwargs is None:
            guidance_kwargs = {}
        if mask is None:
            mask = torch.ones(batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(self.device)
            if mask.dtype != torch.bool:
                mask = mask.bool()

        # Prior: isotropic Gaussian in embedding space (continuous path).
        x = torch.randn(batch_size, self.seq_len, self.hidden_dim, device=self.device)
        trajs = []
        num_steps = int(num_steps)
        step = max(self.max_timesteps // max(num_steps, 1), 1)

        for i in reversed(range(num_steps)):
            t_val = i * step
            t_prev_val = max((i - 1) * step, 0) if i > 0 else 0
            t = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
            t_prev = torch.full((batch_size,), t_prev_val, device=self.device, dtype=torch.long)

            if "target_values" in guidance_kwargs and self.num_properties > 0:
                x = self._langevin_guide(x, t, guidance_kwargs, mask)

            with torch.no_grad():
                x = self._continuous_transition(
                    x, t, t_prev, mask, temperature=temperature
                )

            if return_traj:
                with torch.no_grad():
                    logits, _ = self.forward(x, t_prev, mask)
                    trajs.append(self._decode_tokens(logits, temperature=0.0)[None, :])

            if hijack_step is not None and i == (num_steps - hijack_step):
                break

        with torch.no_grad():
            t0 = torch.zeros(batch_size, device=self.device, dtype=torch.long)
            logits, _ = self.forward(x, t0, mask)
            tokens = self._decode_tokens(logits, temperature=0.0)
            if mask is not None:
                tokens = torch.where(mask, tokens, torch.zeros_like(tokens))
        return tokens, trajs

    def optimize(
        self,
        sequences,
        *,
        target_direction="increase",
        num_steps=50,
        step_size=0.1,
        stability_coef=0.01,
        target_abs=1.0,
        mask=None,
        n_langevin: int = 1,
        temperature: float = 1.0,
    ):
        """Property-guided NOS-C sampling (continuous embedding reverse chain).

        ``sequences`` only sets batch size / optional mask shape (de-novo from noise).
        """
        sequences = sequences.to(self.device)
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        batch_size, seq_len = sequences.shape
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(self.device)
            if mask.dtype != torch.bool:
                mask = mask.bool()

        target = float(target_abs) if target_direction == "increase" else -float(target_abs)
        guidance_kwargs = {
            "step_size": float(step_size),
            "stability_coef": float(stability_coef),
            "target_values": [target],
            "n_langevin": int(n_langevin),
        }
        tokens, _traj = self.sample(
            batch_size,
            num_steps=num_steps,
            guidance_kwargs=guidance_kwargs,
            mask=mask,
            return_traj=False,
            temperature=temperature,
        )
        return tokens

    def generate_sequences(self, batch_size, num_steps=50, guidance_kwargs=None, mask=None):
        tokens, _ = self.sample(batch_size, num_steps, guidance_kwargs, mask)
        return self.ids_to_sequence(tokens)


class NOSTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb, mask=None):
        time_out = self.time_proj(time_emb)
        scale, shift = time_out.chunk(2, dim=-1)
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        key_padding_mask = (~mask.bool()) if mask is not None else None
        attn_out, _ = self.self_attention(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps):
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.time_mlp(emb)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len = 2, 16
    model = NOS_C(seq_len=seq_len, num_properties=1, device="cpu", max_timesteps=20)
    sequences = torch.randint(1, model.vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    total, recon, prop, pred = model.compute_loss(sequences, targets, mask)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)

    tokens, _traj = model.sample(batch_size=batch_size, num_steps=10, return_traj=False)
    assert tokens.shape == (batch_size, seq_len)
    assert not torch.isnan(tokens.float()).any()
    # Should not collapse to all PAD.
    assert (tokens != 0).any(), f"all-pad collapse: {tokens}"

    guided, _ = model.sample(
        batch_size=batch_size,
        num_steps=10,
        guidance_kwargs={"step_size": 0.1, "stability_coef": 0.01, "target_values": [0.5]},
    )
    assert guided.shape == (batch_size, seq_len)
    assert (guided != 0).any()

    # Norms should stay finite across a longer untrained sample.
    x = torch.randn(2, seq_len, model.hidden_dim)
    for i in reversed(range(20)):
        t = torch.full((2,), i, dtype=torch.long)
        t_prev = torch.full((2,), max(i - 1, 0), dtype=torch.long)
        with torch.no_grad():
            x = model._continuous_transition(x, t, t_prev, mask)
        assert torch.isfinite(x).all(), f"non-finite at step t={i}"
    print("NOS_C unit tests passed.")
