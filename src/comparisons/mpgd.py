from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPGD(nn.Module):
    """Manifold Preserving Guided Diffusion for property-directed sequence design.

    Paper: Manifold Preserving Guided Diffusion (ICLR 2024)
    Github: https://github.com/KellyYutongHe/mpgd_pytorch
    """

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 150,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        pot_hidden: int = 512,
        pot_layers: int = 4,
        recon_p: float = 0.5,
        bank_size: int = 8192,
        scale: float = 0.1,
        t0_frac: float = 0.8,
        y_delta: float = 1.0,
        eta: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        del hidden_dim, num_layers, num_heads, dropout  # API compat with other baselines
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)
        self.num_timesteps = int(num_timesteps)
        self.recon_p = float(recon_p)
        self.bank_size = int(bank_size)
        self.scale = float(scale)
        self.t0_frac = float(t0_frac)
        self.y_delta = float(y_delta)
        self.eta = float(eta)

        self.ae = _ConvAE(vocab_size=self.vocab_size, length=self.seq_len, latent_dim=self.latent_dim)
        self.eps_net = _EpsNet(self.latent_dim, hidden=pot_hidden, n_layers=pot_layers)
        self.predictor = _PropertyNet(self.vocab_size, self.seq_len)

        alphas_bar = _cosine_alphas_bar(self.num_timesteps)
        self.register_buffer("alphas_bar", alphas_bar)

        self.register_buffer("z_mu", torch.zeros(self.latent_dim))
        self.register_buffer("z_sd", torch.ones(self.latent_dim))
        self.register_buffer("_stats_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_bank_z", torch.zeros(self.bank_size, self.latent_dim))
        self.register_buffer("_bank_fill", torch.zeros((), dtype=torch.long))
        self.register_buffer("_bank_ptr", torch.zeros((), dtype=torch.long))

        self.to(self.device)

    # ------------------------------------------------------------------ stats / bank
    def _update_latent_stats(self, z: torch.Tensor) -> None:
        with torch.no_grad():
            b = z.shape[0]
            if b < 1:
                return
            batch_mu = z.mean(dim=0)
            batch_var = z.var(dim=0, unbiased=False).clamp_min(0.0)
            n0 = int(self._stats_count.item())
            if n0 == 0:
                self.z_mu.copy_(batch_mu)
                self.z_sd.copy_(batch_var.sqrt().clamp_min(1e-6))
                self._stats_count.fill_(b)
                return
            n1 = n0 + b
            delta = batch_mu - self.z_mu
            new_mu = self.z_mu + delta * (b / float(n1))
            m2_0 = self.z_sd.square() * float(n0)
            m2_1 = batch_var * float(b)
            m2 = m2_0 + m2_1 + delta.square() * (float(n0) * float(b) / float(n1))
            self.z_mu.copy_(new_mu)
            self.z_sd.copy_((m2 / float(n1)).sqrt().clamp_min(1e-6))
            self._stats_count.fill_(n1)

    def _standardize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.z_mu) / self.z_sd.clamp_min(1e-6)

    def _unstandardize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.z_sd.clamp_min(1e-6) + self.z_mu

    def _push_bank(self, z_norm: torch.Tensor) -> None:
        with torch.no_grad():
            b = z_norm.shape[0]
            if b < 1:
                return
            ptr = int(self._bank_ptr.item())
            for i in range(b):
                self._bank_z[ptr] = z_norm[i]
                ptr = (ptr + 1) % self.bank_size
            self._bank_ptr.fill_(ptr)
            fill = int(self._bank_fill.item())
            self._bank_fill.fill_(min(fill + b, self.bank_size))

    def _sample_bank(self, batch_size: int) -> Optional[torch.Tensor]:
        fill = int(self._bank_fill.item())
        if fill < 1:
            return None
        idx = torch.randint(0, fill, (batch_size,), device=self._bank_z.device)
        return self._bank_z[idx]

    # ------------------------------------------------------------------ public API
    def compute_loss(
        self,
        x_0: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recon_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """AE CE + latent DDPM eps-MSE + property MSE.

        Args:
            x_0: token ids [B, L]
            targets: property labels [B] or [B, P] (standardized by the dataloader)
            mask: unused (latent MPGD); kept for API parity
            recon_weight: weight on generative loss (AE + DDPM)

        Returns:
            total_loss, gen_loss, property_loss, property_pred [B, 1]
        """
        del mask
        if x_0.dim() != 2:
            raise ValueError(f"Expected token ids [B, L], got shape {tuple(x_0.shape)}")

        b = x_0.shape[0]
        logits, z = self.ae(x_0)
        ae_loss = F.cross_entropy(logits, x_0)

        # Property net on hard one-hot or AE soft recon.
        if targets is None:
            prop_pred = torch.zeros(b, 1, device=x_0.device)
            prop_loss = torch.zeros((), device=x_0.device)
        else:
            targets = targets.float()
            y_vec = targets.view(b, -1)[:, 0]
            if self.training and torch.rand(()) < self.recon_p:
                x_soft = torch.softmax(logits, dim=1)  # [B, V, L]
            else:
                x_soft = F.one_hot(x_0.long(), self.vocab_size).float().permute(0, 2, 1)
            prop_pred = self.predictor(x_soft).view(b, 1)
            prop_loss = F.mse_loss(prop_pred.view(-1), y_vec)

        self._update_latent_stats(z.detach())
        z_norm = self._standardize(z.detach())
        self._push_bank(z_norm)

        z0 = self._sample_bank(b)
        if z0 is None:
            z0 = z_norm
        t = torch.randint(0, self.num_timesteps, (b,), device=x_0.device)
        ab = self.alphas_bar[t].view(-1, 1)
        noise = torch.randn_like(z0)
        zt = ab.sqrt() * z0 + (1.0 - ab).sqrt() * noise
        eps_pred = self.eps_net(zt, t.float())
        ddpm_loss = F.mse_loss(eps_pred, noise)

        gen_loss = ae_loss + ddpm_loss
        total = recon_weight * gen_loss + prop_loss
        return total, gen_loss, prop_loss, prop_pred

    def sample(
        self,
        batch_size: int,
        *,
        seed_tokens: Optional[torch.Tensor] = None,
        seed_labels: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance: bool = True,
        direction: float = 1.0,
        scale: Optional[float] = None,
        t0_frac: Optional[float] = None,
        y_delta: Optional[float] = None,
        eta: Optional[float] = None,
        pad_mask: Optional[torch.Tensor] = None,
        return_traj: bool = False,
    ):
        """SDEdit-seeded MPGD sampling with optional property guidance."""
        del pad_mask
        if seed_tokens is None:
            raise ValueError("MPGD mpgd_z sampling requires seed_tokens (SDEdit init).")
        if guidance and seed_labels is None:
            raise ValueError("Guided MPGD requires seed_labels for y* = y_seed + direction*y_delta.")

        seed_tokens = seed_tokens.to(self.device)
        batch_size = seed_tokens.shape[0]
        scale = float(self.scale if scale is None else scale)
        t0_frac = float(self.t0_frac if t0_frac is None else t0_frac)
        y_delta = float(self.y_delta if y_delta is None else y_delta)
        eta = float(self.eta if eta is None else eta)
        ddim_steps = max(int(num_steps), 1)

        self.ae.eval()
        self.eps_net.eval()
        self.predictor.eval()

        z = self._standardize(self.ae.encode(seed_tokens))
        if seed_labels is not None:
            y_seed = seed_labels.float().view(-1).to(self.device)
            y_target = y_seed + float(direction) * y_delta
        else:
            y_target = None

        t0 = min(self.num_timesteps - 1, max(0, int(round(t0_frac * self.num_timesteps)) - 1))
        ts = sorted(
            {int(round(v)) for v in torch.linspace(t0, 0, ddim_steps).tolist()},
            reverse=True,
        )
        ab_t0 = self.alphas_bar[t0]
        z = ab_t0.sqrt() * z + (1.0 - ab_t0).sqrt() * torch.randn_like(z)

        traj = []
        one = torch.ones((), device=self.device, dtype=self.alphas_bar.dtype)
        for i, t in enumerate(ts):
            ab = self.alphas_bar[t]
            ab_prev = self.alphas_bar[ts[i + 1]] if i + 1 < len(ts) else one
            tt = torch.full((batch_size,), float(t), device=self.device)
            with torch.no_grad():
                eps = self.eps_net(z, tt)
            z0t = (z - (1.0 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-8)

            if guidance and scale != 0.0 and y_target is not None:
                # Shortcut: grad only w.r.t. clean estimate (no backprop through eps).
                c_t = scale / ab.sqrt().clamp_min(1e-8)
                zg = z0t.detach().requires_grad_(True)
                x_soft = torch.softmax(self.ae.decode(self._unstandardize(zg)), dim=1)
                loss = 0.5 * ((self.predictor(x_soft) - y_target) ** 2).sum()
                g = torch.autograd.grad(loss, zg, create_graph=False)[0]
                z0t = zg.detach() - c_t * g

            sigma = eta * ((1.0 - ab_prev) / (1.0 - ab).clamp_min(1e-8)).sqrt() * (
                1.0 - ab / ab_prev.clamp_min(1e-8)
            ).clamp_min(0.0).sqrt()
            if eta > 0.0 and i + 1 < len(ts):
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            z = (
                ab_prev.sqrt() * z0t
                + (1.0 - ab_prev - sigma**2).clamp_min(0.0).sqrt() * eps
                + sigma * noise
            )
            if return_traj:
                ids = self.ae.decode(self._unstandardize(z)).argmax(dim=1)
                traj.append(ids.detach())

        tokens = self.ae.decode(self._unstandardize(z.detach())).argmax(dim=1)
        if return_traj:
            traj.append(tokens.detach())
            return tokens, torch.stack(traj, dim=0) if traj else tokens.unsqueeze(0)
        return tokens

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        seed_labels: torch.Tensor,
        target_direction: str = "increase",
        num_steps: int = 50,
        scale: Optional[float] = None,
        t0_frac: Optional[float] = None,
        y_delta: Optional[float] = None,
        eta: Optional[float] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SDEdit MPGD-LDM from seed sequences [B, L] and normalized seed labels [B]."""
        direction = 1.0 if target_direction == "increase" else -1.0
        return self.sample(
            batch_size=sequences.shape[0],
            seed_tokens=sequences,
            seed_labels=seed_labels,
            num_steps=num_steps,
            guidance=True,
            direction=direction,
            scale=scale,
            t0_frac=t0_frac,
            y_delta=y_delta,
            eta=eta,
            pad_mask=pad_mask,
            return_traj=False,
        )


# --------------------------------------------------------------------------- submodules
def _cosine_alphas_bar(T: int) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule; returns abar[t] for t=0..T-1."""
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    abar = f / f[0]
    betas = (1.0 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()


class _ConvAE(nn.Module):
    """1D-conv autoencoder over token sequences."""

    def __init__(self, vocab_size: int = 7, length: int = 150, latent_dim: int = 128):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.length = int(length)
        self.latent_dim = int(latent_dim)
        self.enc_conv = nn.Sequential(
            nn.Conv1d(vocab_size, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
        )
        enc_len = (length + 1) // 2
        enc_len = (enc_len + 1) // 2
        self.enc_len = enc_len
        self.enc_proj = nn.Linear(64 * enc_len, latent_dim)
        self.dec_proj = nn.Linear(latent_dim, 64 * enc_len)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, vocab_size, 3, stride=2, padding=1, output_padding=0),
        )

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(ids.long(), num_classes=self.vocab_size).float().transpose(1, 2)
        return self.enc_proj(self.enc_conv(onehot).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_proj(z).view(z.size(0), 64, self.enc_len)
        logits = self.dec_conv(h)
        if logits.size(-1) < self.length:
            logits = F.pad(logits, (0, self.length - logits.size(-1)))
        elif logits.size(-1) > self.length:
            logits = logits[..., : self.length]
        return logits

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(ids)
        return self.decode(z), z


class _EpsNet(nn.Module):
    """Residual MLP noise predictor over standardized latents."""

    def __init__(self, dim: int, hidden: int = 512, n_layers: int = 4, t_dim: int = 128):
        super().__init__()
        self.t_dim = int(t_dim)
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        )
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.inp(z) + self.t_mlp(_timestep_embedding(t, self.t_dim))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


class _PropertyNet(nn.Module):
    """Property head: soft one-hot [B, V, L] -> standardized scalar."""

    def __init__(self, vocab_size: int, length: int, width: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(vocab_size, width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2),
            nn.GELU(),
        )
        L = int(length)
        for _ in range(3):
            L = (L + 1) // 2
        self.head = nn.Sequential(nn.Linear(width * L, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x_soft: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x_soft).flatten(1)).squeeze(-1)


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 4, 32, 7
    model = MPGD(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_timesteps=50,
        bank_size=64,
        device="cpu",
    )
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    for _ in range(4):
        total, gen, prop, pred = model.compute_loss(sequences, targets=targets)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)
    assert torch.isfinite(total)

    model.eval()
    out = model.optimize(
        sequences,
        seed_labels=targets.view(-1),
        target_direction="increase",
        num_steps=5,
    )
    assert out.shape == sequences.shape
    print("MPGD unit tests passed.")
