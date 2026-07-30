"""Rule-based kNN manifold projector: retract onto nearest train latent neighbors."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# Allow `python src/modules/manifold_projector_knn.py`.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class ManifoldProjectorKNN(nn.Module):
    """Project latents onto the empirical manifold via k-NN retraction.

    Duck-typed like ``ManifoldProjectorDAE``: ``forward(x) -> x'`` and ``.device``.
    Not trainable: build at inference from cached train latents
    (``latent_trainset.pt``) and a chosen ``k``.
    For ``k=1`` each point snaps to its nearest training neighbor; for ``k>1``
    it is replaced by the mean of its ``k`` nearest neighbors.
    """

    def __init__(
        self,
        manifold_points: torch.Tensor,
        k: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        points = manifold_points.detach().float()
        if points.dim() != 2:
            raise ValueError(f"manifold_points must be [N, D], got {tuple(points.shape)}")
        self.k = min(k, points.shape[0])
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.register_buffer("manifold", points.to(self.device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        dists = torch.cdist(x, self.manifold)
        knn_idx = torch.topk(dists, k=self.k, largest=False).indices
        neighbors = self.manifold[knn_idx]
        if self.k == 1:
            return neighbors.squeeze(1)
        return neighbors.mean(dim=1)

    @classmethod
    def from_latent_trainset(
        cls,
        path: str,
        k: int = 1,
        device: str | torch.device = "cpu",
    ) -> "ManifoldProjectorKNN":
        """Load ``latent_trainset.pt`` and build a projector with the given ``k``."""
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            latents = payload
        else:
            latents = payload["latents"]
        return cls(manifold_points=latents, k=k, device=device)


if __name__ == "__main__":
    manifold = torch.randn(50, 320)
    for k in (1, 3, 5, 10):
        projector = ManifoldProjectorKNN(manifold, k=k, device="cpu")
        out = projector(torch.randn(4, 320))
        assert out.shape == (4, 320), out.shape
        print(f"k={k}: out={tuple(out.shape)}")
    print("ManifoldProjectorKNN unit test finished")
