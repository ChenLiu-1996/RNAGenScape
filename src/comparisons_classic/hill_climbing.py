"""OAE + nearest-neighbor hill climbing (mRNA-translation ``nn_hill_climbing_embedding``)."""

from __future__ import annotations

from typing import Optional

import torch

from comparisons_classic._ops import (
    decode_latents,
    direction_sign,
    encode_tokens,
    property_fn,
)


class HillClimbing:
    """Move toward the best kNN train-latent neighbor (deterministic).

    Adapted from: mRNA-translation ``nn_hill_climbing_embedding`` (stochastic=False).
    Neighbor pool is the OAE train-set latents (manifold), not the start batch.
    """

    def __init__(
        self,
        num_steps: int = 100,
        step_size: float = 5e-3,
        k_neighbors: int = 10,
        stochastic: bool = False,
    ):
        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.k_neighbors = int(k_neighbors)
        self.stochastic = bool(stochastic)

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
        """Property-guided hill climb from seed token ids ``[B, L]``."""
        del pad_mask, _kwargs
        if train_latents is None:
            raise ValueError(f"{type(self).__name__}.optimize requires train_latents [N, D].")
        sign = direction_sign(target_direction)
        oracle = property_fn(oae)
        pool = train_latents.to(sequences.device).float()
        curr = encode_tokens(oae, sequences).detach()
        k = min(max(self.k_neighbors, 1), max(pool.shape[0] - 1, 1))

        with torch.no_grad():
            for _ in range(max(self.num_steps, 1)):
                # Pairwise distances [B, N]
                dists = torch.cdist(curr, pool)
                # Exclude exact self-matches when starts are in the pool (set large).
                topk = torch.topk(dists, k=k, largest=False).indices  # [B, K]
                neighbors = pool[topk]  # [B, K, D]
                # Score neighbors under directed property
                flat = neighbors.reshape(-1, neighbors.shape[-1])
                scores = (oracle(flat) * sign).view(neighbors.shape[0], neighbors.shape[1])

                if self.stochastic:
                    curr_fit = oracle(curr) * sign
                    # Prefer improving neighbors; else random among k.
                    improve = scores > curr_fit.unsqueeze(-1)
                    choice = []
                    for b in range(curr.shape[0]):
                        inds = torch.where(improve[b])[0]
                        if inds.numel() == 0:
                            inds = torch.arange(k, device=curr.device)
                        pick = inds[torch.randint(inds.numel(), (1,), device=curr.device)]
                        choice.append(neighbors[b, pick].squeeze(0))
                    target = torch.stack(choice, dim=0)
                else:
                    best = scores.argmax(dim=1)  # [B]
                    target = neighbors[torch.arange(curr.shape[0], device=curr.device), best]

                direction = target - curr
                curr = curr + self.step_size * direction

            return decode_latents(oae, curr)


class StochasticHillClimbing(HillClimbing):
    """Same as HillClimbing with random improving neighbor selection."""

    def __init__(
        self,
        num_steps: int = 100,
        step_size: float = 5e-3,
        k_neighbors: int = 10,
    ):
        super().__init__(
            num_steps=num_steps,
            step_size=step_size,
            k_neighbors=k_neighbors,
            stochastic=True,
        )
