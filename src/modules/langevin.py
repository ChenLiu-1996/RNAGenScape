"""Manifold Langevin dynamics for latent-space generation.

Fitness-guided (or unconstrained) Langevin updates in latent space, with an
optional manifold projector (DAE / kNN) applied after each step.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm

PotentialFn = Callable[[torch.Tensor], torch.Tensor]
ForceFn = Callable[[torch.Tensor], torch.Tensor]
StopFn = Callable[[torch.Tensor], bool]


def langevin_dynamics(
    potential_fn: Optional[PotentialFn],
    x_init: torch.Tensor,
    num_steps: int,
    step_size: float,
    *,
    return_history: bool = False,
    stop_fn: Optional[StopFn] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Plain Langevin dynamics (no manifold projector).

    ``potential_fn(x)`` is treated as an energy / negative-log-density; the
    update uses score = -grad potential.
    """
    if seed is not None:
        torch.manual_seed(seed)
    x = x_init.clone().detach().requires_grad_(True)
    history = [] if return_history else None

    for step in tqdm(range(num_steps), desc="langevin"):
        if potential_fn is not None:
            energy = potential_fn(x)
            grad_energy = torch.autograd.grad(
                energy, x, grad_outputs=torch.ones_like(energy), create_graph=True
            )[0]
            score = -grad_energy
            x = x + step_size * score + ((2 * step_size) ** 0.5) * torch.randn_like(x)
        else:
            x = x + ((2 * step_size) ** 0.5) * torch.randn_like(x)
        x = x.detach().requires_grad_(True)
        if history is not None:
            history.append(x)
        if stop_fn is not None and stop_fn(x):
            print(f"Early stopping at step {step}")
            break

    if history is not None:
        return torch.stack(history, dim=0).detach()
    return x.detach()


def manifold_langevin_dynamics(
    potential_fn: Optional[PotentialFn],
    projector,
    x_init: torch.Tensor,
    num_steps: int,
    step_size: float,
    *,
    temperature: float = 1.0,
    step_size_rescale: Optional[float] = None,
    force_fn: Optional[ForceFn] = None,
    use_projector: bool = True,
    projector_iters: int = 1,
    return_history: bool = False,
    stop_fn: Optional[StopFn] = None,
    debug: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Langevin update followed by optional manifold projection.

    ``potential_fn(x)`` is an energy / drift potential; score = -grad / temperature.
    ``projector`` must be callable ``forward(x) -> x'`` (DAE or kNN).
    """
    x = x_init.clone().detach().requires_grad_(True)
    history = [x.clone()] if return_history else None

    for step in tqdm(range(num_steps), desc="manifold_langevin"):
        score = None
        if potential_fn is not None:
            energy = potential_fn(x)
            grad_energy = torch.autograd.grad(
                energy, x, grad_outputs=torch.ones_like(energy), create_graph=True
            )[0]
            score = (-grad_energy) / temperature

        if force_fn is not None:
            force = force_fn(x)
            force_magnitude = torch.clip(torch.norm(force, dim=1), min=1e-6)
            force = force / force_magnitude.unsqueeze(1) / temperature
            score = force if score is None else score + force

        noise = torch.randn_like(x)
        sqrt_step_size = (2 * step_size) ** 0.5
        if score is not None:
            delta_x = step_size * score + sqrt_step_size * noise
        else:
            delta_x = sqrt_step_size * noise
        if step_size_rescale is not None:
            delta_x = step_size_rescale * torch.nn.functional.normalize(delta_x, p=2, dim=1)
        x = x + delta_x

        if debug:
            print(
                f"step={step} ||x|| mean={torch.norm(x, dim=1).mean().item():.4f} "
                f"||delta|| mean={delta_x.norm(dim=1).mean().item():.4f}"
            )

        if use_projector and projector is not None:
            for _ in range(projector_iters):
                x = projector(x)

        x = x.detach().requires_grad_(True)
        if history is not None:
            history.append(x)
        if stop_fn is not None and stop_fn(x):
            print(f"Early stopping at step {step}")
            break

    if history is not None:
        return x.detach(), torch.stack(history, dim=0).detach()
    return x.detach()


def annealed_manifold_langevin_dynamics(
    potential_fn: Optional[PotentialFn],
    projector,
    x_init: torch.Tensor,
    noise_scales: Sequence[float],
    *,
    base_step_size: float = 0.01,
    n_steps_per_scale: int = 10,
    force_fn: Optional[ForceFn] = None,
    use_projector: bool = True,
    return_history: bool = False,
    stop_fn: Optional[StopFn] = None,
    seed: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Annealed Langevin over a noise schedule, with optional manifold projection."""
    if seed is not None:
        torch.manual_seed(seed)

    x = x_init.clone().detach().requires_grad_(True)
    history = [x.clone()] if return_history else None

    for sigma in noise_scales:
        step_size = (float(sigma) ** 2) * base_step_size
        sqrt_step_size = (2 * step_size) ** 0.5
        for step in range(n_steps_per_scale):
            score = None
            if potential_fn is not None:
                energy = potential_fn(x)
                grad_energy = torch.autograd.grad(
                    energy, x, grad_outputs=torch.ones_like(energy), create_graph=True
                )[0]
                score = (-grad_energy) / (float(sigma) ** 2)

            if force_fn is not None:
                force = force_fn(x)
                force_mag = torch.norm(force, dim=1).clamp(min=1e-6)
                desired = (x_init.shape[1] ** 0.5) * 0.75
                force = force / force_mag.unsqueeze(1) * desired
                score = force if score is None else score + force

            noise = torch.randn_like(x)
            if score is not None:
                x = x + step_size * score + sqrt_step_size * noise
            else:
                x = x + sqrt_step_size * noise

            if use_projector and projector is not None:
                x = projector(x)

            x = x.detach().requires_grad_(True)
            if history is not None:
                history.append(x)
            if stop_fn is not None and stop_fn(x):
                print(f"Early stopping at noise={sigma} step={step}")
                break

    if history is not None:
        return x.detach(), torch.stack(history, dim=0).detach()
    return x.detach()


def run_manifold_langevin(
    x_init: torch.Tensor,
    projector,
    *,
    fitness_fn: Optional[PotentialFn] = None,
    direction: float = 1.0,
    num_steps: int = 100,
    step_size: float = 0.01,
    temperature: float = 1.0,
    step_size_rescale: Optional[float] = None,
    use_projector: bool = True,
    projector_iters: int = 1,
    annealed: bool = False,
    batch_size: int = 256,
    return_history: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Batched fitness-guided manifold Langevin over ``x_init`` ``[N, D]``.

    Returns ``(z_final [N, D], trajectories [T+1, N, D] or None)``.
    """
    potential_fn = None
    if fitness_fn is not None:
        potential_fn = lambda z: -fitness_fn(z) * direction  # noqa: E731

    generated = []
    trajectories = []
    for i in range(0, x_init.shape[0], batch_size):
        batch = x_init[i : i + batch_size]
        if annealed:
            scales = torch.logspace(start=0, end=-2, steps=20)
            out = annealed_manifold_langevin_dynamics(
                potential_fn,
                projector,
                batch,
                scales,
                base_step_size=step_size,
                use_projector=use_projector,
                return_history=return_history,
            )
        else:
            out = manifold_langevin_dynamics(
                potential_fn,
                projector,
                batch,
                num_steps,
                step_size,
                temperature=temperature,
                step_size_rescale=step_size_rescale,
                use_projector=use_projector,
                projector_iters=projector_iters,
                return_history=return_history,
            )
        if return_history:
            z_final, traj = out
            generated.append(z_final)
            trajectories.append(traj)
        else:
            generated.append(out)

    z_final = torch.cat(generated, dim=0)
    if not return_history:
        return z_final, None
    # traj batches are [T, B, D] -> concat on batch dim -> [T, N, D]
    traj_all = torch.cat(trajectories, dim=1)
    return z_final, traj_all
