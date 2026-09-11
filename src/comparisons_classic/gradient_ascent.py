"""OAE + gradient ascent in latent space (mRNA-translation ``gradient_ascent``)."""

from __future__ import annotations

from typing import Optional

import torch

from comparisons_classic._ops import (
    decode_latents,
    direction_sign,
    encode_tokens,
    property_fn,
)


class GradientAscent:
    """Latent gradient ascent on ``oae.regress`` (Kingma-style AE property head).

    Adapted from: mRNA-translation ``optimization_baselines.optimization.gradient_ascent``
    """

    def __init__(
        self,
        num_steps: int = 100,
        step_size: float = 5e-3,
        cycle: bool = False,
        noise: bool = False,
        sigma: float = 1e-3,
    ):
        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.cycle = bool(cycle)
        self.noise = bool(noise)
        self.sigma = float(sigma)

    def optimize(
        self,
        sequences: torch.Tensor,
        *,
        oae,
        target_direction: str = "increase",
        pad_mask: Optional[torch.Tensor] = None,
        train_latents: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Property-guided latent gradient ascent from seed token ids ``[B, L]``."""
        del pad_mask, train_latents, _kwargs
        sign = direction_sign(target_direction)
        oracle = property_fn(oae)
        z = encode_tokens(oae, sequences).detach().requires_grad_(True)

        for _ in range(max(self.num_steps, 1)):
            fitness = oracle(z) * sign
            grad = torch.autograd.grad(fitness.sum(), z, create_graph=False)[0]
            if self.noise:
                grad = grad + self.sigma * torch.randn_like(grad)
            z = (z + self.step_size * grad).detach()
            if self.cycle:
                with torch.no_grad():
                    tokens = decode_latents(oae, z)
                    z = encode_tokens(oae, tokens).detach()
            z = z.requires_grad_(True)

        with torch.no_grad():
            return decode_latents(oae, z.detach())
