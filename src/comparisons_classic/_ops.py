"""Shared helpers for OAE-latent classic optimizers."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def direction_sign(target_direction: str | float) -> float:
    if isinstance(target_direction, str):
        return 1.0 if target_direction == "increase" else -1.0
    return 1.0 if float(target_direction) > 0 else -1.0


def tokens_to_one_hot(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return F.one_hot(tokens.long(), num_classes=int(vocab_size)).float()


def encode_tokens(oae, tokens: torch.Tensor) -> torch.Tensor:
    """Token ids [B, L] -> latent [B, D] via OAE one-hot encode."""
    oh = tokens_to_one_hot(tokens, oae.vocab_size).to(tokens.device)
    return oae.encode(oh)


def decode_latents(oae, z: torch.Tensor) -> torch.Tensor:
    """Latent [B, D] -> token ids [B, L]."""
    return oae.generate(z)


def property_fn(oae) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda z: oae.regress(z)
