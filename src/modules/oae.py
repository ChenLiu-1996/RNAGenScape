"""Organized Autoencoder (OAE) for RNAGenScape.

Faithful port of the active ``OrganizedAE`` from mRNA-translation:
Conv1d encoder -> latent (320) -> progressive decoder + regression head.
Used by ``train_oae.py``; guidance head is separate from eval oracles.
"""

from __future__ import annotations

import math
import os
import sys

# Allow `python src/modules/oae.py`.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.metrics import VOCAB_SIZE


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


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.se = ChannelSELayer(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.norm(self.conv(x))))


class ResBlock1D(nn.Module):
    """Residual block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_conv(x)
        out = self.conv1(self.activation(self.norm1(x)))
        out = self.conv2(self.dropout(self.activation(self.norm2(out))))
        return out + residual


class UpsampleBlock1D(nn.Module):
    """Upsample block with learnable convolution."""

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
        latent_dim: int = 320,
        target_length: int = 120,
        output_channels: int = VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.target_length = target_length
        self.output_channels = output_channels
        self.initial_length = 8
        self.num_upsample_steps = math.ceil(math.log2(target_length / self.initial_length))

        self.initial_linear = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.initial_length),
            nn.GELU(),
        )

        self.upsample_blocks = nn.ModuleList()
        in_channels = 128
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
        x = self.initial_linear(z).view(batch_size, 128, self.initial_length)
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)
        if x.shape[-1] != self.target_length:
            x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)
        return self.final_layers(x)


class OAE(nn.Module):
    """Organized Autoencoder: encode / decode / regress (latent dim 320)."""

    def __init__(
        self,
        device,
        seq_len: int,
        vocab_size: int = VOCAB_SIZE,
        latent_dim: int = 320,
        dropout: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.device = device
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.loss_fn = nn.MSELoss()

        self.encoder = nn.Sequential(
            ConvBlock1D(vocab_size, 16),
            ConvBlock1D(16, 32),
            ConvBlock1D(32, 64),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, latent_dim),
        )
        self.decoder = ProgressiveDecoder1D(
            latent_dim=latent_dim,
            target_length=seq_len,
            output_channels=vocab_size,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
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
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _as_channels_first(self, x: torch.Tensor) -> torch.Tensor:
        """Accept ``[B, L, V]`` or ``[B, V, L]`` -> ``[B, V, L]`` (channels-first)."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D one-hot, got shape {tuple(x.shape)}")
        if x.shape[1] == self.vocab_size:
            return x.float()
        if x.shape[2] == self.vocab_size:
            return x.float().transpose(1, 2).contiguous()
        raise ValueError(
            f"Cannot infer layout for shape {tuple(x.shape)} with vocab_size={self.vocab_size}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._as_channels_first(x).to(self.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return rearrange(self.decoder(z), "b c l -> b l c")

    def regress(self, z: torch.Tensor) -> torch.Tensor:
        return self.regression_head(z).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regress(self.encode(x))

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


if __name__ == "__main__":
    device = "cpu"
    for seq_len in (50, 107, 124):
        model = OAE(device=device, seq_len=seq_len, vocab_size=VOCAB_SIZE, latent_dim=320)
        x = torch.randn(2, seq_len, VOCAB_SIZE)
        z = model.encode(x)
        logits = model.decode(z)
        y = model.regress(z)
        ids = model.generate(z)
        assert z.shape == (2, 320), z.shape
        assert logits.shape == (2, seq_len, VOCAB_SIZE), logits.shape
        assert y.shape == (2,), y.shape
        assert ids.shape == (2, seq_len), ids.shape
        print(
            f"seq_len={seq_len}: z={tuple(z.shape)} decode={tuple(logits.shape)} "
            f"y={tuple(y.shape)} generate={tuple(ids.shape)} "
            f"params={sum(p.numel() for p in model.parameters()):,}"
        )
    print("OAE unit test finished")
