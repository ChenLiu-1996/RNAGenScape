"""Organized Autoencoder (OAE) for RNAGenScape.

Deterministic continuous latent AE + property head for manifold Langevin.

Design (single vector ``z in R^D``)::

    [B,L,V] -> [B,V,L]
      -> Conv tower (stem + num_down stride-2 stages)
      -> [B,C,L']
      -> AdaptiveAvgPool1d(K)  [B,C,K]   # fixed positional bins
      -> flatten + Linear(C*K -> D)       # capacity bottleneck
      <- Linear(D -> C*K) + reshape [B,C,K]
      <- x2 upsample until length >= L, then interpolate to exact L
      -> token logits [B,L,V]

Why AdaptiveAvgPool (not GAP, not AttentionPool):
* GAP collapses L' to one mean and drops position. Property heads can then
  fit composition alone, so correlation metrics rise while token_acc stalls.
* AttentionPool can re-learn a soft GAP (queries attend globally) and still
  under-serve base-level recon.
* AdaptiveAvgPool1d(K) forces K equal-length positional bins. Position reaches
  z by construction; D remains the only Langevin capacity bottleneck.

Defaults: base_channels=64, num_down=2, pool_tokens=16
  (C=256, path approximately 64@L -> 128@L/2 -> 256@L/4 -> AdaptPool(16)).

Blocks use GroupNorm + SiLU + Squeeze-Excite (pre-norm residual).
No VAE / KL. No LayerNorm on z.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List

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
DEFAULT_BASE_CHANNELS = 64
DEFAULT_NUM_DOWN = 2
DEFAULT_POOL_TOKENS = 16


def gn_groups(num_channels: int, max_groups: int = 32) -> int:
    """Largest group count <= max_groups that divides num_channels."""
    g = min(int(max_groups), int(num_channels))
    while g > 1 and num_channels % g != 0:
        g -= 1
    return max(1, g)


def conv1d_out_len(length: int, *, kernel: int = 3, stride: int = 1, padding: int = 1) -> int:
    """Output length of a Conv1d with the given hyper-params (PyTorch formula)."""
    return (int(length) + 2 * padding - kernel) // stride + 1


def downsample_length(length: int, num_down: int, *, kernel: int = 3, padding: int = 1) -> int:
    """Spatial length after num_down stride-2 convolutions (kernel=3, pad=1)."""
    out = int(length)
    for _ in range(int(num_down)):
        out = conv1d_out_len(out, kernel=kernel, stride=2, padding=padding)
    return out


def num_upsamples_to_cover(start_len: int, target_len: int) -> int:
    """How many x2 ups are needed so start_len * 2^n >= target_len."""
    if start_len >= target_len:
        return 0
    return int(math.ceil(math.log2(target_len / float(start_len))))


class ChannelSELayer(nn.Module):
    """Squeeze-and-Excitation over channels (no spatial compression)."""

    def __init__(self, num_channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        if num_channels <= 4:
            self.fc1 = None
            self.fc2 = None
            return
        reduced = max(1, num_channels // reduction_ratio)
        self.fc1 = nn.Linear(num_channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, num_channels, bias=True)
        self.act = nn.SiLU()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc1 is None:
            return x
        squeeze = x.mean(dim=2)
        excite = self.gate(self.fc2(self.act(self.fc1(squeeze))))
        return x * excite.unsqueeze(-1)


class ResBlock1D(nn.Module):
    """Pre-norm residual Conv1d block (optional stride for downsampling)."""

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
        self.norm1 = nn.GroupNorm(gn_groups(in_channels), in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.norm2 = nn.GroupNorm(gn_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.activation = nn.SiLU()
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


class UpsampleBlock1D(nn.Module):
    """x2 linear upsample + residual conv (inverse of a stride-2 block)."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = ResBlock1D(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        return self.conv(x)


class OAE(nn.Module):
    """Organized Autoencoder: encode / decode / regress with a single continuous z."""

    def __init__(
        self,
        device,
        seq_len: int,
        vocab_size: int = VOCAB_SIZE,
        latent_dim: int = DEFAULT_LATENT_DIM,
        dropout: float = 0.1,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        num_down: int = DEFAULT_NUM_DOWN,
        pool_tokens: int = DEFAULT_POOL_TOKENS,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        if num_down < 1:
            raise ValueError(f"num_down must be >= 1, got {num_down}")
        if pool_tokens < 1:
            raise ValueError(f"pool_tokens must be >= 1, got {pool_tokens}")
        self.device = device
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.num_down = int(num_down)
        self.pool_tokens = int(pool_tokens)
        self.conv_len = downsample_length(self.seq_len, self.num_down)
        self.enc_channels = self.base_channels * (2 ** self.num_down)
        self.num_up = num_upsamples_to_cover(self.pool_tokens, self.seq_len)
        self.loss_fn = nn.MSELoss()

        # ----- Encoder conv tower -----
        # Default: 64@L -> 128@L/2 -> 256@L/4
        self.stem = nn.Sequential(
            nn.Conv1d(vocab_size, self.base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups(self.base_channels), self.base_channels),
            nn.SiLU(),
        )
        enc_blocks: List[nn.Module] = []
        ch = self.base_channels
        for _ in range(self.num_down):
            enc_blocks.append(ResBlock1D(ch, ch, dropout=dropout, stride=1))
            next_ch = ch * 2
            enc_blocks.append(ResBlock1D(ch, next_ch, dropout=dropout, stride=2))
            ch = next_ch
        enc_blocks.append(ResBlock1D(ch, ch, dropout=dropout, stride=1))
        self.enc_blocks = nn.ModuleList(enc_blocks)

        # Fixed positional bins, then capacity bottleneck D.
        self.pool = nn.AdaptiveAvgPool1d(self.pool_tokens)
        self.to_latent = nn.Linear(self.enc_channels * self.pool_tokens, self.latent_dim)

        # ----- Decoder: inverse of pool -> latent -----
        self.from_latent = nn.Sequential(
            nn.Linear(self.latent_dim, self.enc_channels * self.pool_tokens),
            nn.SiLU(),
        )
        dec_blocks: List[nn.Module] = []
        ch = self.enc_channels
        for _ in range(self.num_up):
            next_ch = max(self.base_channels, ch // 2)
            dec_blocks.append(UpsampleBlock1D(ch, next_ch, dropout=dropout))
            ch = next_ch
        self.dec_up = nn.ModuleList(dec_blocks)
        self.dec_tail = nn.Sequential(
            ResBlock1D(ch, ch, dropout=dropout),
            ResBlock1D(ch, ch, dropout=dropout),
            nn.Conv1d(ch, vocab_size, kernel_size=3, padding=1),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(self.latent_dim, max(64, self.latent_dim)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, self.latent_dim), 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._initialize_weights()
        self.to(device)

    @property
    def bottleneck_len(self) -> int:
        """Decoder start length (= pool_tokens); alias for logs/meta."""
        return self.pool_tokens

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
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

    def _as_channels_first(self, x: torch.Tensor) -> torch.Tensor:
        """Accept [B, L, V] or [B, V, L] -> [B, V, L]."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D one-hot, got shape {tuple(x.shape)}")
        if x.shape[1] == self.vocab_size:
            return x.float()
        if x.shape[2] == self.vocab_size:
            return x.float().transpose(1, 2).contiguous()
        raise ValueError(
            f"Cannot infer layout for shape {tuple(x.shape)} with vocab_size={self.vocab_size}"
        )

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder feature map [B, C, L'] (before pool)."""
        h = self.stem(self._as_channels_first(x).to(self.device))
        for block in self.enc_blocks:
            h = block(h)
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map one-hot sequences to latent z [B, D] (deterministic)."""
        h = self.pool(self.encode_features(x))  # [B, C, K]
        flat = rearrange(h, "b c k -> b (c k)")
        return self.to_latent(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode z [B, D] to token logits [B, L, V]."""
        batch = z.shape[0]
        x = self.from_latent(z).view(batch, self.enc_channels, self.pool_tokens)
        for up in self.dec_up:
            x = up(x)
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)
        return rearrange(self.dec_tail(x), "b c l -> b l c")

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


def infer_latent_dim_from_state(state: dict) -> int:
    """Infer latent_dim from a current OAE state_dict."""
    if "to_latent.weight" in state:
        return int(state["to_latent.weight"].shape[0])
    if "regression_head.0.weight" in state:
        return int(state["regression_head.0.weight"].shape[1])
    raise KeyError("Could not infer latent_dim from OAE state_dict keys.")


def load_oae(
    ckpt_path: str,
    *,
    device: str,
    seq_len: int,
    vocab_size: int = VOCAB_SIZE,
    strict: bool = True,
) -> OAE:
    """Build OAE and load model.pt, inferring latent_dim from weights."""
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    latent_dim = infer_latent_dim_from_state(state)
    model = OAE(
        device=device,
        seq_len=seq_len,
        vocab_size=vocab_size,
        latent_dim=latent_dim,
    )
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint does not match current OAE architecture "
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
            )
            x = torch.randn(2, seq_len, VOCAB_SIZE)
            if latent_dim == 128:
                h = model._as_channels_first(x)
                print(
                    f"\n### seq_len={seq_len} base={model.base_channels} "
                    f"downs={model.num_down} K={model.pool_tokens} ups={model.num_up}"
                )
                print(f"input                  {tuple(x.shape)}")
                print(f"channels-first         {tuple(h.shape)}")
                h = model.stem(h)
                print(f"stem                   {tuple(h.shape)}")
                for i, block in enumerate(model.enc_blocks):
                    h = block(h)
                    print(f"enc_blocks[{i}]          {tuple(h.shape)}")
                pooled = model.pool(h)
                print(f"AdaptiveAvgPool        {tuple(pooled.shape)}")
                flat = rearrange(pooled, "b c k -> b (c k)")
                print(f"flatten                {tuple(flat.shape)}")
                z = model.to_latent(flat)
                print(f"to_latent              {tuple(z.shape)}")
            z = model.encode(x)
            logits = model.decode(z)
            y = model.regress(z)
            ids = model.generate(z)
            assert z.shape == (2, latent_dim), z.shape
            assert logits.shape == (2, seq_len, VOCAB_SIZE), logits.shape
            assert y.shape == (2,), y.shape
            assert ids.shape == (2, seq_len), ids.shape
            assert model.enc_channels == 256
            assert model.pool_tokens == DEFAULT_POOL_TOKENS
            print(
                f"OK seq_len={seq_len} D={latent_dim} conv_L'={model.conv_len} "
                f"K={model.pool_tokens} C={model.enc_channels} "
                f"params={sum(p.numel() for p in model.parameters()):,}"
            )
    print("OAE unit test finished")
