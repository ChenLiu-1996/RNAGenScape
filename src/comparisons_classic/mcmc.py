"""OAE + latent Metropolis-Hastings MCMC (mRNA-translation ``metropolisMCMC_embedding``)."""

from __future__ import annotations

from typing import Optional

import torch

from comparisons_classic._ops import (
    decode_latents,
    direction_sign,
    encode_tokens,
    property_fn,
)


class MCMC:
    """Gaussian-proposal Metropolis-Hastings on OAE latents.

    Adapted from: mRNA-translation ``optimization_baselines.optimization.metropolisMCMC_embedding``
    """

    def __init__(
        self,
        num_steps: int = 100,
        temperature: float = 0.5,
        delta: float = 0.1,
    ):
        self.num_steps = int(num_steps)
        self.temperature = float(temperature)
        self.delta = float(delta)

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
        """Property-guided latent MCMC from seed token ids ``[B, L]``."""
        del pad_mask, train_latents, _kwargs
        sign = direction_sign(target_direction)
        oracle = property_fn(oae)
        curr = encode_tokens(oae, sequences).detach()
        curr_fit = (oracle(curr) * sign).detach()
        t = max(self.temperature, 1e-12)

        with torch.no_grad():
            for _ in range(max(self.num_steps, 1)):
                prop = curr + self.delta * torch.randn_like(curr)
                prop_fit = oracle(prop) * sign
                delta_fit = prop_fit - curr_fit
                accept = (delta_fit >= 0) | (
                    torch.rand_like(delta_fit) < (delta_fit / t).exp()
                )
                curr = torch.where(accept.unsqueeze(-1), prop, curr)
                curr_fit = torch.where(accept, prop_fit, curr_fit)
            return decode_latents(oae, curr)
