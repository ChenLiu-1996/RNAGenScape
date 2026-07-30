"""Conv1d property oracle (trainable).

``Conv1d`` is a length-agnostic residual 1D CNN regressor: global average pooling
makes it work for any sequence length. Used via ``train_oracle.py`` only
(alongside ``UTRLM``); never confused with frozen UTRLM_TE / UTRLM_MRL.
"""

from __future__ import annotations

import os
import sys

# Allow `python src/modules/conv1d.py`.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import TOKENS, VOCAB_SIZE, to_token_ids


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-init module parameters (Fixup-style residual start)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ChannelSELayer(nn.Module):
    """Squeeze-and-Excitation block (Hu et al.)."""

    def __init__(self, num_channels: int, reduction_ratio: int = 2) -> None:
        super().__init__()
        reduced = max(1, num_channels // reduction_ratio)
        self.fc1 = nn.Linear(num_channels, reduced)
        self.fc2 = nn.Linear(reduced, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        squeeze = x.mean(dim=-1)
        excite = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        return x * excite.unsqueeze(-1)


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = zero_module(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
        self.norm = nn.GroupNorm(num_groups=min(16, out_channels), num_channels=out_channels)
        self.se = ChannelSELayer(out_channels)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.norm(self.conv(x))
        out = self.se(out)
        return F.gelu(out + residual)


class Conv1d(nn.Module):
    """Trainable Conv1d oracle (arbitrary sequence length).

    Input at train time: one-hot ``[B, L, V]`` or ``[B, V, L]``.
    ``encode`` also accepts nucleotide strings / token ids for scoring.
    """

    def __init__(
        self,
        device,
        seq_len: int,
        vocab_size: int = VOCAB_SIZE,
        latent_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        # kwargs absorbs unused train_oracle args (e.g. freeze_backbone).
        del kwargs
        self.device = device
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.loss_fn = nn.MSELoss()

        self.input_conv = nn.Conv1d(vocab_size, latent_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(latent_dim, latent_dim) for _ in range(num_blocks)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GroupNorm(num_groups=min(16, latent_dim), num_channels=latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )
        self._initialize_weights()
        # Re-zero residual branch convs after kaiming init (Fixup-style).
        for block in self.blocks:
            zero_module(block.conv)
        self.to(device)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _as_channels_first(self, x) -> torch.Tensor:
        """Convert strings / token ids / one-hot to ``[B, V, L]`` (channels-first)."""
        if isinstance(x, (list, tuple)) and (len(x) == 0 or isinstance(x[0], str)):
            ids = to_token_ids(list(x))
            eye = torch.eye(self.vocab_size, dtype=torch.float32)
            x = eye[ids]  # [B, L, V]
        elif not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        if x.dim() == 2:
            # Token ids [B, L]
            eye = torch.eye(self.vocab_size, dtype=torch.float32, device=x.device)
            x = eye[x.long()]

        if x.dim() != 3:
            raise ValueError(f"Expected 3D one-hot or convertible input, got shape {tuple(x.shape)}")

        # Prefer channel-second layout detection via vocab size (not seq_len).
        if x.shape[1] == self.vocab_size:
            channels_first = x.float()
        elif x.shape[2] == self.vocab_size:
            channels_first = x.float().transpose(1, 2).contiguous()
        else:
            raise ValueError(
                f"Cannot infer layout for shape {tuple(x.shape)} with vocab_size={self.vocab_size}"
            )
        return channels_first.to(self.device)

    def encode(self, x) -> torch.Tensor:
        """Return pooled embedding ``[B, latent_dim]`` (length-agnostic)."""
        h = self.input_conv(self._as_channels_first(x))
        for block in self.blocks:
            h = block(h)
        return self.global_avg_pool(h).squeeze(-1)

    def regress(self, z: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.regression_head(z), dim=-1)

    def forward(self, x) -> torch.Tensor:
        return self.regress(self.encode(x))

    def regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 2:
            pred = pred.squeeze(-1)
        return self.loss_fn(pred, target)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Dummy decode for API compatibility."""
        batch_size = z.shape[0]
        return torch.randn(batch_size, self.seq_len, self.vocab_size, device=z.device)


if __name__ == "__main__":
    device = "cpu"
    for seq_len in (50, 107, 124, 200):
        model = Conv1d(device=device, seq_len=seq_len, vocab_size=VOCAB_SIZE, latent_dim=64)
        one_hot = torch.randn(4, seq_len, VOCAB_SIZE)
        strings = ["".join(TOKENS[1 + (i + j) % 4] for j in range(seq_len)) for i in range(4)]
        y1 = model(one_hot)
        y2 = model(strings)
        z = model.encode(one_hot)
        print(
            f"seq_len={seq_len}: one_hot->{tuple(y1.shape)}, "
            f"strings->{tuple(y2.shape)}, z->{tuple(z.shape)}"
        )
    print("Conv1d unit test finished")
