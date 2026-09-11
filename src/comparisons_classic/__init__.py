"""Classic optimizers on the OAE latent manifold (from mRNA-translation).

Algorithms (not trainable models). Generation loads a trained OAE and runs
``optimize`` from test starts. Results folders use the optimizer name
(``GradientAscent``, ``MCMC``, ``HillClimbing``, ``StochasticHillClimbing``).
"""

from comparisons_classic.gradient_ascent import GradientAscent
from comparisons_classic.hill_climbing import HillClimbing, StochasticHillClimbing
from comparisons_classic.mcmc import MCMC

CLASSIC_MODELS = ("GradientAscent", "MCMC", "HillClimbing", "StochasticHillClimbing")

__all__ = [
    "CLASSIC_MODELS",
    "GradientAscent",
    "MCMC",
    "HillClimbing",
    "StochasticHillClimbing",
    "build_classic_optimizer",
]


def build_classic_optimizer(name: str, **kwargs):
    """Construct a classic optimizer by results-folder name."""
    if name == "GradientAscent":
        return GradientAscent(**kwargs)
    if name == "MCMC":
        return MCMC(**kwargs)
    if name == "HillClimbing":
        return HillClimbing(**kwargs)
    if name == "StochasticHillClimbing":
        return StochasticHillClimbing(**kwargs)
    raise ValueError(f"Unknown classic optimizer '{name}'. Choose from {CLASSIC_MODELS}.")
