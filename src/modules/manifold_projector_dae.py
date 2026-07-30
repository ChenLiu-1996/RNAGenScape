"""Learnable DAE manifold projector (latent denoising autoencoder).

MLP trained with a noise curriculum and optional two-step denoising.
Duck-typed with ``ManifoldProjectorKNN`` (``forward`` + ``.device``).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow `python src/modules/manifold_projector_dae.py`.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils.training_utils import EarlyStopping


class ManifoldProjectorDAE(nn.Module):
    """MLP denoiser: noisy latent -> clean latent (learnable manifold projector)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 16, 32),
        activation: Union[str, nn.Module] = "relu",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.device = device

        if isinstance(activation, str):
            act = {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "gelu": nn.GELU(),
            }[activation.lower()]
        else:
            act = activation

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), act])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.mlp = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.to(self.device))


def train_dae_epoch_loop(
    model: ManifoldProjectorDAE,
    train_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    val_loader: Optional[DataLoader] = None,
    noise: float = 0.0,
    two_step_denoising: bool = True,
    early_stopper: Optional[EarlyStopping] = None,
) -> Dict[str, List[float]]:
    """Train one noise-level stage (Adam + MSE)."""
    model = model.to(device)
    model.device = device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    pbar = tqdm(range(num_epochs), desc=f"dae noise={noise}")
    for _ in pbar:
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            clean = batch.to(device)

            if noise > 0:
                noisy = clean + noise * torch.randn_like(clean)
                if two_step_denoising:
                    optimizer.zero_grad()
                    loss_std = criterion(model(noisy), clean)
                    doubly = noisy + noise * torch.randn_like(clean)
                    loss_two = criterion(model(doubly), noisy)
                    loss = loss_std + loss_two
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss = criterion(model(noisy), clean)
                    loss.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                loss = criterion(model(clean), clean)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_train)

        if val_loader is None:
            pbar.set_postfix(train_loss=f"{avg_train:.4f}")
            continue

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                clean = batch.to(device)
                if noise > 0:
                    noisy = clean + noise * torch.randn_like(clean)
                    if two_step_denoising:
                        loss_std = criterion(model(noisy), clean)
                        doubly = noisy + noise * torch.randn_like(clean)
                        loss_two = criterion(model(doubly), noisy)
                        batch_loss = loss_std + loss_two
                    else:
                        batch_loss = criterion(model(noisy), clean)
                else:
                    batch_loss = criterion(model(clean), clean)
                val_loss += batch_loss.item()

        avg_val = val_loss / max(len(val_loader), 1)
        history["val_loss"].append(avg_val)
        pbar.set_postfix(train_loss=f"{avg_train:.4f}", val_loss=f"{avg_val:.4f}")
        if early_stopper is not None and early_stopper.step(avg_val):
            print("Early stopping criterion met. Ending this noise stage.")
            break

    return history


def train_manifold_projector_dae(
    X: torch.Tensor,
    *,
    noise_levels: Sequence[float] = (0.5, 0.2, 0.1),
    hidden_dims: Sequence[int] = (32, 16, 32),
    num_epochs: int = 1000,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    early_stop_patience: int = 50,
    device: str = "cpu",
    random_state: int = 1,
    two_step_denoising: bool = True,
) -> tuple[ManifoldProjectorDAE, List[Dict[str, List[float]]], Dict[str, float]]:
    """Noise-curriculum DAE training on latent points ``X`` ``[N, D]``.

    Split is 20% train / 40% val / 40% test (legacy RNAGenScape / paper path).
    """
    noise_levels = list(noise_levels)
    learning_rates = [learning_rate] * len(noise_levels)

    data = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    train_np, temp_np = train_test_split(data, test_size=0.8, random_state=random_state)
    val_np, test_np = train_test_split(temp_np, test_size=0.5, random_state=random_state)

    train_data = torch.tensor(train_np, dtype=torch.float32)
    val_data = torch.tensor(val_np, dtype=torch.float32)
    test_data = torch.tensor(test_np, dtype=torch.float32)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = ManifoldProjectorDAE(
        input_dim=data.shape[1], hidden_dims=hidden_dims, device=device
    )
    histories: List[Dict[str, List[float]]] = []

    for noise_level, lr in zip(noise_levels, learning_rates):
        stopper = (
            EarlyStopping(mode="min", patience=early_stop_patience)
            if early_stop_patience is not None and early_stop_patience > 0
            else None
        )
        history = train_dae_epoch_loop(
            model,
            train_loader,
            num_epochs=num_epochs,
            learning_rate=lr,
            device=device,
            val_loader=val_loader,
            noise=float(noise_level),
            two_step_denoising=two_step_denoising,
            early_stopper=stopper,
        )
        histories.append(history)

    model.eval()
    metrics = _identity_mse(model, train_loader, val_loader, test_loader, device)
    print(
        f"DAE identity MSE  train={metrics['train_mse']:.6f} "
        f"val={metrics['val_mse']:.6f} test={metrics['test_mse']:.6f}"
    )
    return model, histories, metrics


@torch.no_grad()
def _identity_mse(
    model: ManifoldProjectorDAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    def _mse(loader: DataLoader) -> float:
        total = 0.0
        n = 0
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            out = model(batch)
            total += torch.sum(torch.square(out - batch)).item() / batch.size(-1)
            n += batch.size(0)
        return total / max(n, 1)

    return {
        "train_mse": _mse(train_loader),
        "val_mse": _mse(val_loader),
        "test_mse": _mse(test_loader),
    }


def load_manifold_projector_dae(
    path: str,
    input_dim: int,
    hidden_dims: Sequence[int] = (32, 16, 32),
    device: str = "cpu",
) -> ManifoldProjectorDAE:
    model = ManifoldProjectorDAE(
        input_dim=input_dim, hidden_dims=hidden_dims, device=device
    )
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


if __name__ == "__main__":
    device = "cpu"
    X = torch.randn(200, 320)
    model, _, metrics = train_manifold_projector_dae(
        X,
        noise_levels=(0.5, 0.2),
        num_epochs=2,
        batch_size=64,
        early_stop_patience=2,
        device=device,
        random_state=1,
    )
    y = model(X[:4])
    assert y.shape == (4, 320), y.shape
    print("ManifoldProjectorDAE unit test finished", metrics, y.shape)
