"""Organized Autoencoder (OAE) for RNAGenScape.

Continuous latent AE + property head for manifold Langevin guidance.

Architecture notes (domain-agnostic):
* Deeper residual Conv1d encoder with strided downsampling (SD-VAE-style stem)
  instead of a shallow stack + aggressive global pool.
* Attention pooling to a fixed token grid before the bottleneck.
* Mild beta-VAE (``kl_w``) so the latent is smooth enough for Langevin; set
  ``kl_w=0`` for a deterministic AE.
* LayerNorm on ``z`` to stabilize latent scale for the projector / Langevin.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

# Allow `python src/modules/oae.py`.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.metrics import VOCAB_SIZE

DEFAULT_LATENT_DIM = 128


class ChannelSELayer(nn.Module):
    """Squeeze-and-Excitation layer."""

    def __init__(self, num_channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        if num_channels <= 4:
            self.identity = nn.Identity()
            self.fc1 = None
            self.fc2 = None
            return

        reduced = max(1, num_channels // reduction_ratio)
        self.fc1 = nn.Linear(num_channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.identity = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity is not None:
            return self.identity(x)
        batch_size, num_channels, _ = x.size()
        squeeze = x.view(batch_size, num_channels, -1).mean(dim=2)
        excite = self.sigmoid(self.fc2(self.relu(self.fc1(squeeze))))
        return x * excite.view(batch_size, num_channels, 1)


class ResBlock1D(nn.Module):
    """Pre-norm residual Conv1d block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.se = ChannelSELayer(out_channels)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(self.activation(self.norm1(x)))
        out = self.conv2(self.dropout(self.activation(self.norm2(out))))
        return self.se(out + residual)


class AttentionPool1D(nn.Module):
    """Learned attention pooling to a fixed number of summary tokens."""

    def __init__(self, channels: int, num_tokens: int = 8) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.query = nn.Parameter(torch.randn(1, num_tokens, channels) * 0.02)
        self.norm = nn.LayerNorm(channels)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> tokens [B, T, C]
        h = x.transpose(1, 2)
        h = self.norm(h)
        q = self.query.expand(h.shape[0], -1, -1)
        attn = torch.softmax(torch.matmul(q, h.transpose(1, 2)) * self.scale, dim=-1)
        return torch.matmul(attn, h)


class UpsampleBlock1D(nn.Module):
    """Upsample block with learnable residual convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = ResBlock1D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class ProgressiveDecoder1D(nn.Module):
    """Progressive decoder that builds spatial structure gradually."""

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        target_length: int = 120,
        output_channels: int = VOCAB_SIZE,
        base_channels: int = 128,
        initial_length: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.target_length = target_length
        self.output_channels = output_channels
        self.initial_length = initial_length
        self.base_channels = base_channels
        self.num_upsample_steps = math.ceil(math.log2(max(target_length, 1) / initial_length))

        self.initial_linear = nn.Sequential(
            nn.Linear(latent_dim, base_channels * initial_length),
            nn.GELU(),
        )

        self.upsample_blocks = nn.ModuleList()
        in_channels = base_channels
        for _ in range(self.num_upsample_steps):
            out_channels = max(32, in_channels // 2)
            self.upsample_blocks.append(UpsampleBlock1D(in_channels, out_channels))
            in_channels = out_channels

        self.final_layers = nn.Sequential(
            ResBlock1D(in_channels, in_channels),
            ResBlock1D(in_channels, in_channels),
            nn.Conv1d(in_channels, output_channels, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        x = self.initial_linear(z).view(batch_size, self.base_channels, self.initial_length)
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)
        if x.shape[-1] != self.target_length:
            x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)
        return self.final_layers(x)


class OAE(nn.Module):
    """Organized Autoencoder: encode / decode / regress with compressed latent."""

    def __init__(
        self,
        device,
        seq_len: int,
        vocab_size: int = VOCAB_SIZE,
        latent_dim: int = DEFAULT_LATENT_DIM,
        dropout: float = 0.2,
        kl_w: float = 1e-3,
        pool_tokens: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.device = device
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.latent_dim = int(latent_dim)
        self.kl_w = float(kl_w)
        self.loss_fn = nn.MSELoss()

        # Residual encoder with strided downsampling (domain-agnostic).
        self.stem = nn.Sequential(
            nn.Conv1d(vocab_size, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.GELU(),
        )
        self.enc_blocks = nn.ModuleList(
            [
                ResBlock1D(32, 32, dropout=dropout),
                ResBlock1D(32, 64, stride=2, dropout=dropout),
                ResBlock1D(64, 64, dropout=dropout),
                ResBlock1D(64, 128, stride=2, dropout=dropout),
                ResBlock1D(128, 128, dropout=dropout),
                ResBlock1D(128, 128, dropout=dropout),
            ]
        )
        self.pool = AttentionPool1D(128, num_tokens=pool_tokens)
        flat_dim = 128 * pool_tokens
        self.to_mu = nn.Linear(flat_dim, latent_dim)
        self.to_logvar = nn.Linear(flat_dim, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)

        self.decoder = ProgressiveDecoder1D(
            latent_dim=latent_dim,
            target_length=seq_len,
            output_channels=vocab_size,
            base_channels=128,
            initial_length=pool_tokens,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, max(64, latent_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, latent_dim), 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Start near a standard Normal prior (helps early KL).
        nn.init.zeros_(self.to_mu.weight)
        nn.init.zeros_(self.to_mu.bias)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -2.0)

    def _as_channels_first(self, x: torch.Tensor) -> torch.Tensor:
        """Accept ``[B, L, V]`` or ``[B, V, L]`` -> ``[B, V, L]``."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D one-hot, got shape {tuple(x.shape)}")
        if x.shape[1] == self.vocab_size:
            return x.float()
        if x.shape[2] == self.vocab_size:
            return x.float().transpose(1, 2).contiguous()
        raise ValueError(
            f"Cannot infer layout for shape {tuple(x.shape)} with vocab_size={self.vocab_size}"
        )

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.enc_blocks:
            h = block(h)
        pooled = self.pool(h)  # [B, T, C]
        return pooled.reshape(pooled.shape[0], -1)

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)`` before LayerNorm / sampling."""
        feat = self._backbone(self._as_channels_first(x).to(self.device))
        return self.to_mu(feat), self.to_logvar(feat)

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Mean KL(q(z|x) || N(0, I))."""
        return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()

    def encode(self, x: torch.Tensor, sample: Optional[bool] = None) -> torch.Tensor:
        """Map one-hot sequences to latent ``z`` ``[B, D]``.

        By default samples during training when ``kl_w > 0``, otherwise uses ``mu``.
        Inference / generation starts should call with ``sample=False`` (default in eval).
        """
        mu, logvar = self.encode_stats(x)
        if sample is None:
            sample = bool(self.training and self.kl_w > 0.0)
        if sample:
            std = torch.exp(0.5 * logvar.clamp(-20.0, 20.0))
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return self.latent_norm(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return rearrange(self.decoder(z), "b c l -> b l c")

    def regress(self, z: torch.Tensor) -> torch.Tensor:
        return self.regression_head(z).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regress(self.encode(x, sample=False))

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.decode(z), dim=-1)

    def regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 2:
            pred = pred.squeeze(-1)
        return self.loss_fn(pred, target)

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            rearrange(pred, "b l c -> (b l) c"),
            rearrange(target, "b l -> (b l)"),
            reduction="mean",
        )


def infer_latent_dim_from_state(state: dict) -> int:
    """Infer ``latent_dim`` from an OAE ``state_dict`` (new or legacy)."""
    if "to_mu.weight" in state:
        return int(state["to_mu.weight"].shape[0])
    if "latent_norm.weight" in state:
        return int(state["latent_norm.weight"].shape[0])
    # Legacy shallow encoder: last Linear in Sequential.
    if "encoder.5.weight" in state:
        return int(state["encoder.5.weight"].shape[0])
    if "regression_head.0.weight" in state:
        return int(state["regression_head.0.weight"].shape[1])
    raise KeyError("Could not infer latent_dim from OAE state_dict keys.")


def load_oae(
    ckpt_path: str,
    *,
    device: str,
    seq_len: int,
    vocab_size: int = VOCAB_SIZE,
    kl_w: float = 1e-3,
    strict: bool = True,
) -> OAE:
    """Build OAE and load ``model.pt``, inferring ``latent_dim`` from weights."""
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    latent_dim = infer_latent_dim_from_state(state)
    model = OAE(
        device=device,
        seq_len=seq_len,
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        kl_w=kl_w,
    )
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint incompatible with current OAE architecture "
            f"(inferred latent_dim={latent_dim}). Retrain with src/train_oae.py. "
            f"Original error: {exc}"
        ) from exc
    model.eval()
    return model


if __name__ == "__main__":
    device = "cpu"
    for seq_len in (50, 107, 124):
        for latent_dim in (64, 128):
            model = OAE(
                device=device,
                seq_len=seq_len,
                vocab_size=VOCAB_SIZE,
                latent_dim=latent_dim,
                kl_w=1e-3,
            )
            x = torch.randn(2, seq_len, VOCAB_SIZE)
            model.train()
            z_s = model.encode(x, sample=True)
            model.eval()
            z = model.encode(x, sample=False)
            mu, logvar = model.encode_stats(x)
            logits = model.decode(z)
            y = model.regress(z)
            ids = model.generate(z)
            kl = model.kl_divergence(mu, logvar)
            assert z.shape == (2, latent_dim), z.shape
            assert z_s.shape == (2, latent_dim), z_s.shape
            assert logits.shape == (2, seq_len, VOCAB_SIZE), logits.shape
            assert y.shape == (2,), y.shape
            assert ids.shape == (2, seq_len), ids.shape
            assert torch.isfinite(kl)
            print(
                f"seq_len={seq_len} D={latent_dim}: z={tuple(z.shape)} "
                f"decode={tuple(logits.shape)} kl={float(kl):.4f} "
                f"params={sum(p.numel() for p in model.parameters()):,}"
            )
    print("OAE unit test finished")
