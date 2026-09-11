"""Shared 1D conv / progressive decoder blocks for de novo baselines.

Adapted from mRNA-translation ``models/organized_ae.py`` (no external project import).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSELayer(nn.Module):
    """Squeeze-and-Excitation over channels."""

    def __init__(self, num_channels: int, reduction_ratio: int = 4):
        super().__init__()
        if num_channels <= 4:
            self.identity = nn.Identity()
            self.fc1 = None
            self.fc2 = None
            return
        reduced = max(1, num_channels // reduction_ratio)
        self.identity = None
        self.fc1 = nn.Linear(num_channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity is not None:
            return self.identity(x)
        b, c, _ = x.shape
        squeeze = x.view(b, c, -1).mean(dim=2)
        excite = self.sigmoid(self.fc2(self.relu(self.fc1(squeeze))))
        return x * excite.view(b, c, 1)


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.se = ChannelSELayer(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.norm(self.conv(x))))


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
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
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = ResBlock1D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class ProgressiveDecoder1D(nn.Module):
    """Progressive 1D decoder: latent vector -> [B, C, L]."""

    def __init__(self, latent_dim: int = 32, target_length: int = 120, output_channels: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_length = target_length
        self.output_channels = output_channels
        self.initial_length = 8
        self.num_upsample_steps = math.ceil(math.log2(max(target_length / self.initial_length, 1.0)))

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
        b = z.shape[0]
        x = self.initial_linear(z).view(b, 128, self.initial_length)
        for block in self.upsample_blocks:
            x = block(x)
        if x.shape[-1] != self.target_length:
            x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)
        return self.final_layers(x)


def tokens_to_one_hot(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Token ids [B, L] -> float one-hot [B, L, V]."""
    return F.one_hot(tokens.long(), num_classes=vocab_size).float()


def mlp_regressor(in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
    )
