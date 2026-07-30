"""Results paths, generation artifacts, and evaluation reporting."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


GENERATION_FILENAME = "generation.npz"

_TRAIN_STATS_KEYS = (
    "label_mean",
    "label_std",
    "label_min",
    "label_max",
    "latent_mean",
    "latent_std",
    "latent_min",
    "latent_max",
)

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def repo_root() -> str:
    """RNAGenScape repository root (parent of src/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def results_root(root: Optional[str] = None) -> str:
    return os.path.join(root or repo_root(), "results")


def experiment_dir(
    dataset: str,
    model: str,
    experiment: str,
    root: Optional[str] = None,
) -> str:
    return os.path.join(results_root(root), dataset, model, experiment)


def oae_checkpoint_path(dataset: str, root: Optional[str] = None) -> str:
    """Path to trained OAE weights: ``results/<dataset>/OAE/model.pt``."""
    return os.path.join(results_root(root), dataset, "OAE", "model.pt")


def _sugar_tag(sugar_w: float) -> str:
    """Encode sugar weight for path segments (``1.0`` -> ``1p0``)."""
    return str(float(sugar_w)).replace(".", "p")


def normalize_latent_norm_name(latent_normalization: str) -> str:
    """Canonical lowercase name for DAE checkpoint filenames."""
    name = str(latent_normalization).strip().lower()
    if name in ("", "none", "null"):
        return "none"
    return name


def manifold_projector_dae_dir(
    dataset: str, sugar_w: float = 0.0, root: Optional[str] = None
) -> str:
    """``results/<dataset>/OAE/manifold_projector_dae_sugar{w}/``."""
    return os.path.join(
        results_root(root),
        dataset,
        "OAE",
        f"manifold_projector_dae_sugar{_sugar_tag(sugar_w)}",
    )


def manifold_projector_dae_checkpoint_path(
    dataset: str,
    sugar_w: float = 0.0,
    latent_normalization: str = "none",
    root: Optional[str] = None,
) -> str:
    """``.../manifold_projector_dae_sugar{w}/model_latentnorm_{norm}.pt``."""
    norm = normalize_latent_norm_name(latent_normalization)
    return os.path.join(
        manifold_projector_dae_dir(dataset, sugar_w=sugar_w, root=root),
        f"model_latentnorm_{norm}.pt",
    )


def manifold_projector_knn_dir(
    dataset: str, sugar_w: float = 0.0, root: Optional[str] = None
) -> str:
    """``results/<dataset>/OAE/manifold_projector_knn_sugar{w}/``."""
    return os.path.join(
        results_root(root),
        dataset,
        "OAE",
        f"manifold_projector_knn_sugar{_sugar_tag(sugar_w)}",
    )


def latent_trainset_path(
    dataset: str,
    sugar_w: float = 0.0,
    root: Optional[str] = None,
) -> str:
    """``.../manifold_projector_knn_sugar{w}/latent_trainset.pt`` (k-agnostic)."""
    return os.path.join(
        manifold_projector_knn_dir(dataset, sugar_w=sugar_w, root=root),
        "latent_trainset.pt",
    )


def evaluation_dir(exp_dir: str) -> str:
    path = os.path.join(exp_dir, "evaluation")
    os.makedirs(path, exist_ok=True)
    return path


def discover_seed_runs(exp_dir: str) -> List[Tuple[int, str]]:
    """Return sorted (seed, run_dir) for directories with generation.npz under exp_dir/seed_*.

    Also accepts legacy flat dirs named eval_seed{N}_... containing generation.npz.
    """
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    runs: List[Tuple[int, str]] = []

    for name in sorted(os.listdir(exp_dir)):
        path = os.path.join(exp_dir, name)
        if not os.path.isdir(path):
            continue
        m = _SEED_DIR_RE.match(name)
        if m and os.path.isfile(os.path.join(path, GENERATION_FILENAME)):
            runs.append((int(m.group(1)), path))

    if runs:
        return sorted(runs, key=lambda x: x[0])

    # Legacy: eval_seed{N}_tag/.../generation.npz
    legacy_re = re.compile(r"eval_seed(\d+)_")
    for name in sorted(os.listdir(exp_dir)):
        path = os.path.join(exp_dir, name)
        if not os.path.isdir(path):
            continue
        m = legacy_re.match(name)
        if m and os.path.isfile(os.path.join(path, GENERATION_FILENAME)):
            runs.append((int(m.group(1)), path))

    if not runs:
        raise FileNotFoundError(
            f"No seed runs with {GENERATION_FILENAME} under {exp_dir}. "
            "Expected seed_*/generation.npz (or legacy eval_seed*_*/generation.npz)."
        )
    return sorted(runs, key=lambda x: x[0])


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_generation_artifact(
    path: str,
    *,
    new_sequences,
    sampled_X,
    sampled_Y,
    sampled_indices,
    sampling_pool_X,
    direction: float,
    train_stats: Dict[str, Any],
    model_type: str,
    data: str,
    seed: int,
    subsample_seed: int = 42,
    trajectories=None,
    trajectories_are_sequences: bool = False,
    sugar_w: float = 0.0,
    target_value: Optional[float] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Write generation.npz (or path/dir + filename). Returns the path written."""
    if os.path.isdir(path):
        path = os.path.join(path, GENERATION_FILENAME)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload: Dict[str, Any] = {
        "new_sequences": _to_numpy(new_sequences),
        "sampled_X": _to_numpy(sampled_X),
        "sampled_Y": _to_numpy(sampled_Y),
        "sampled_indices": _to_numpy(sampled_indices),
        "sampling_pool_X": _to_numpy(sampling_pool_X),
        "direction": np.asarray(direction, dtype=np.float64),
        "trajectories_are_sequences": np.asarray(
            1 if trajectories_are_sequences else 0, dtype=np.int64
        ),
        "meta__data": np.asarray(data),
        "meta__model_type": np.asarray(model_type),
        "meta__seed": np.asarray(seed, dtype=np.int64),
        "meta__subsample_seed": np.asarray(subsample_seed, dtype=np.int64),
        "meta__sugar_w": np.asarray(sugar_w, dtype=np.float64),
    }
    if target_value is not None:
        payload["meta__target_value"] = np.asarray(target_value, dtype=np.float64)

    if trajectories is not None and (
        not hasattr(trajectories, "__len__") or len(trajectories) > 0
    ):
        traj_np = _to_numpy(trajectories)
        if traj_np is not None and traj_np.size > 0:
            payload["trajectories"] = traj_np

    for key in _TRAIN_STATS_KEYS:
        if key in train_stats and train_stats[key] is not None:
            val = train_stats[key]
            if isinstance(val, torch.Tensor):
                val = (
                    val.detach().cpu().item()
                    if val.numel() == 1
                    else val.detach().cpu().numpy()
                )
            payload[f"train_stats__{key}"] = np.asarray(val)

    if extra_meta:
        payload["meta__extra_json"] = np.asarray(json.dumps(extra_meta))

    np.savez_compressed(path, **payload)
    return path


def load_generation_artifact(path: str) -> Dict[str, Any]:
    """Load generation.npz into a dict (sequence fields as torch tensors)."""
    if os.path.isdir(path):
        path = os.path.join(path, GENERATION_FILENAME)
    data = np.load(path, allow_pickle=True)

    out: Dict[str, Any] = {
        "new_sequences": torch.from_numpy(np.asarray(data["new_sequences"])),
        "sampled_X": torch.from_numpy(np.asarray(data["sampled_X"])),
        "sampled_Y": torch.from_numpy(
            np.asarray(data["sampled_Y"]).astype(np.float32)
        ),
        "sampled_indices": np.asarray(data["sampled_indices"]),
        "sampling_pool_X": torch.from_numpy(np.asarray(data["sampling_pool_X"])),
        "direction": float(np.asarray(data["direction"]).item()),
        "trajectories_are_sequences": bool(
            np.asarray(data["trajectories_are_sequences"]).item()
        ),
        "data": str(np.asarray(data["meta__data"]).item()),
        "model_type": str(np.asarray(data["meta__model_type"]).item()),
        "seed": int(np.asarray(data["meta__seed"]).item()),
        "subsample_seed": int(np.asarray(data["meta__subsample_seed"]).item()),
        "sugar_w": (
            float(np.asarray(data["meta__sugar_w"]).item())
            if "meta__sugar_w" in data.files
            else 0.0
        ),
        "path": path,
    }
    if "meta__target_value" in data.files:
        out["target_value"] = float(np.asarray(data["meta__target_value"]).item())

    if "trajectories" in data.files:
        traj = np.asarray(data["trajectories"])
        out["trajectories"] = torch.from_numpy(traj) if traj.size > 0 else None
    else:
        out["trajectories"] = None

    train_stats: Dict[str, Any] = {}
    for key in _TRAIN_STATS_KEYS:
        fname = f"train_stats__{key}"
        if fname in data.files:
            arr = np.asarray(data[fname])
            train_stats[key] = arr.item() if arr.ndim == 0 else arr
    out["train_stats"] = train_stats

    if "meta__extra_json" in data.files:
        out["extra_meta"] = json.loads(str(np.asarray(data["meta__extra_json"]).item()))
    else:
        out["extra_meta"] = {}

    return out


def write_per_seed_csv(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    """Write one row per seed to CSV. Returns path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(list(rows))
    if "seed" in df.columns:
        df = df.sort_values("seed")
    df.to_csv(path, index=False)
    return path


def summarize_per_seed_csv(
    per_seed_csv: str,
    summary_csv: str,
    skip_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Aggregate numeric columns as mean +/- std across seeds; write summary CSV and return it."""
    df = pd.read_csv(per_seed_csv)
    skip = set(skip_cols or ()) | {"seed"}
    rows = []
    for col in df.columns:
        if col in skip:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].astype(float).to_numpy()
        rows.append(
            {
                "metric": col,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "n_seeds": int(len(values)),
            }
        )
    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    return summary


def format_summary_table(summary: pd.DataFrame) -> str:
    """ASCII table: metric | mean \u00B1 std (std across random seeds)."""
    if summary.empty:
        return "(no numeric metrics)"
    pm = "\u00B1"
    lines = [f"metric                         mean {pm} std", "-" * 48]
    for _, row in summary.iterrows():
        name = str(row["metric"])
        mean = float(row["mean"])
        std = float(row["std"])
        lines.append(f"{name:<30} {mean:.6g} {pm} {std:.6g}")
    n = int(summary["n_seeds"].iloc[0]) if "n_seeds" in summary.columns else 0
    lines.append("-" * 48)
    lines.append(f"n_seeds                        {n}")
    return "\n".join(lines)
