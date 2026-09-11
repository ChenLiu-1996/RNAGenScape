"""Results paths, generation artifacts, and evaluation reporting."""

from __future__ import annotations

import json
import os
import re
from decimal import Decimal
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

# Printed optimization table: core paper metrics first (Table 1 then Table 2),
# then remaining metrics. ``None`` inserts a horizontal rule.
# Each entry is (metric_key, display_label, value_scale).
# value_scale=100 shows fraction metrics as percentages; "%" stays in the label.
OPTIMIZATION_SUMMARY_CORE: Tuple[Optional[Tuple[str, str, float]], ...] = (
    ("median_property_change", "Median property change", 1.0),
    ("pct_improved", "Percentage improved, i.e., success rate (%)", 1.0),
    None,
    ("generated_uorf_oof_aug_mean", "uORF OOF AUG % (generated)", 100.0),
    ("generated_kozak_mean", "Kozak Similarity % (generated)", 100.0),
    ("generated_mfe_mean", "Minimum Free Energy (generated)", 1.0),
    ("generated_mean_plddt", "Mean pLDDT (generated)", 1.0),
    None,
    ("root_uorf_oof_aug_mean", "uORF OOF AUG % (test data)", 100.0),
    ("root_kozak_mean", "Kozak Similarity % (test data)", 100.0),
    ("root_mfe_mean", "Minimum Free Energy (test data)", 1.0),
    ("root_mean_plddt", "Mean pLDDT (test data)", 1.0),
    None,
    ("elite_nn_hamming_gen_mean", "Elite NN edit distance (generated)", 1.0),
    ("elite_nn_hamming_root_mean", "Elite NN edit distance (test data)", 1.0),
    None,
)


def repo_root() -> str:
    """RNAGenScape repository root (parent of src/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def results_root(root: Optional[str] = None) -> str:
    return os.path.join(root or repo_root(), "results")


def float_tag(value: float) -> str:
    """Encode a float for path segments via compact scientific notation.

    Examples: ``5.0`` -> ``5e0``, ``0.5`` -> ``5e-1``, ``1e-4`` -> ``1e-4``.
    Coefficient is an integer (no ``.`` in the tag).
    """
    d = Decimal(f"{float(value):.12g}").normalize()
    sign, digits, exp = d.as_tuple()
    if not digits or digits == (0,):
        return "0e0"
    coeff = int("".join(str(dig) for dig in digits))
    prefix = "-" if sign else ""
    return f"{prefix}{coeff}e{int(exp)}"


def oae_config_tag(latent_dim: int, recon_w: float, reg_w: float = 1.0) -> str:
    """Ablation folder for an OAE variant: ``d128_recon5e0_reg1e0``."""
    return (
        f"d{int(latent_dim)}_recon{float_tag(recon_w)}_reg{float_tag(reg_w)}"
    )


def oae_config_dir(
    dataset: str,
    latent_dim: int,
    recon_w: float,
    reg_w: float = 1.0,
    root: Optional[str] = None,
) -> str:
    """``results/<dataset>/OAE/d{latent}_recon{w}_reg{reg}/``."""
    return os.path.join(
        results_root(root),
        dataset,
        "OAE",
        oae_config_tag(latent_dim, recon_w, reg_w),
    )


def experiment_dir(
    dataset: str,
    model: str,
    experiment: str,
    *,
    oae_latent_dim: Optional[int] = None,
    oae_recon_w: Optional[float] = None,
    oae_reg_w: float = 1.0,
    root: Optional[str] = None,
) -> str:
    """Experiment outputs dir.

    For OAE, generation/eval live under the ablation config:
    ``results/<dataset>/OAE/d{latent}_recon{w}_reg{reg}/<experiment>/``.
    """
    if model == "OAE":
        if oae_latent_dim is None or oae_recon_w is None:
            raise ValueError(
                "experiment_dir(..., model='OAE') requires oae_latent_dim and oae_recon_w"
            )
        return os.path.join(
            oae_config_dir(
                dataset,
                oae_latent_dim,
                oae_recon_w,
                reg_w=oae_reg_w,
                root=root,
            ),
            experiment,
        )
    return os.path.join(results_root(root), dataset, model, experiment)


def oae_seed_dir(
    dataset: str,
    seed: int,
    *,
    latent_dim: int,
    recon_w: float,
    reg_w: float = 1.0,
    root: Optional[str] = None,
) -> str:
    """``results/<dataset>/OAE/d{latent}_recon{w}_reg{reg}/seed_{seed}/``."""
    return os.path.join(
        oae_config_dir(dataset, latent_dim, recon_w, reg_w=reg_w, root=root),
        f"seed_{int(seed)}",
    )


def oae_checkpoint_path(
    dataset: str,
    seed: int,
    *,
    latent_dim: int,
    recon_w: float,
    reg_w: float = 1.0,
    root: Optional[str] = None,
) -> str:
    """``results/<dataset>/OAE/d{latent}_recon{w}_reg{reg}/seed_{seed}/model.pt``."""
    return os.path.join(
        oae_seed_dir(
            dataset,
            seed,
            latent_dim=latent_dim,
            recon_w=recon_w,
            reg_w=reg_w,
            root=root,
        ),
        "model.pt",
    )


def baseline_seed_dir(
    dataset: str,
    model: str,
    seed: int,
    root: Optional[str] = None,
) -> str:
    """``results/<dataset>/<model>/seed_{seed}/`` for comparison baselines."""
    return os.path.join(results_root(root), dataset, model, f"seed_{int(seed)}")


def baseline_checkpoint_path(
    dataset: str,
    model: str,
    seed: int,
    root: Optional[str] = None,
) -> str:
    """``results/<dataset>/<model>/seed_{seed}/model.pt``."""
    return os.path.join(baseline_seed_dir(dataset, model, seed, root=root), "model.pt")


def _sugar_tag(sugar_w: float) -> str:
    """Encode sugar weight for path segments (``1.0`` -> ``1e0``, ``0.5`` -> ``5e-1``)."""
    return float_tag(sugar_w)


def normalize_latent_norm_name(latent_normalization: str) -> str:
    """Canonical lowercase name for DAE checkpoint filenames."""
    name = str(latent_normalization).strip().lower()
    if name in ("", "none", "null"):
        return "none"
    return name


def _dae_projector_dirname(sugar_w: float, dae_tag: Optional[str] = None) -> str:
    """Directory leaf for a DAE projector.

    Default (multi-step / locked): ``manifold_projector_dae_sugar{w}``.
    Tagged (e.g. single-step): ``manifold_projector_dae_{tag}_sugar{w}``.
    """
    sugar = _sugar_tag(sugar_w)
    tag = str(dae_tag or "").strip()
    if tag:
        return f"manifold_projector_dae_{tag}_sugar{sugar}"
    return f"manifold_projector_dae_sugar{sugar}"


def manifold_projector_dae_dir(
    dataset: str,
    seed: int,
    *,
    latent_dim: int,
    recon_w: float,
    reg_w: float = 1.0,
    sugar_w: float = 0.0,
    dae_tag: Optional[str] = None,
    root: Optional[str] = None,
) -> str:
    """``.../seed_{seed}/manifold_projector_dae[_tag]_sugar{w}/``."""
    return os.path.join(
        oae_seed_dir(
            dataset,
            seed,
            latent_dim=latent_dim,
            recon_w=recon_w,
            reg_w=reg_w,
            root=root,
        ),
        _dae_projector_dirname(sugar_w, dae_tag=dae_tag),
    )


def manifold_projector_dae_checkpoint_path(
    dataset: str,
    seed: int,
    *,
    latent_dim: int,
    recon_w: float,
    reg_w: float = 1.0,
    sugar_w: float = 0.0,
    dae_tag: Optional[str] = None,
    latent_normalization: str = "none",
    root: Optional[str] = None,
) -> str:
    """``.../manifold_projector_dae[_tag]_sugar{w}/model_latentnorm_{norm}.pt``."""
    norm = normalize_latent_norm_name(latent_normalization)
    return os.path.join(
        manifold_projector_dae_dir(
            dataset,
            seed,
            latent_dim=latent_dim,
            recon_w=recon_w,
            reg_w=reg_w,
            sugar_w=sugar_w,
            dae_tag=dae_tag,
            root=root,
        ),
        f"model_latentnorm_{norm}.pt",
    )


def manifold_projector_knn_dir(
    dataset: str,
    seed: int,
    *,
    latent_dim: int,
    recon_w: float,
    reg_w: float = 1.0,
    sugar_w: float = 0.0,
    root: Optional[str] = None,
) -> str:
    """``.../OAE/d{latent}_recon{w}_reg{reg}/seed_{seed}/manifold_projector_knn_sugar{w}/``."""
    return os.path.join(
        oae_seed_dir(
            dataset,
            seed,
            latent_dim=latent_dim,
            recon_w=recon_w,
            reg_w=reg_w,
            root=root,
        ),
        f"manifold_projector_knn_sugar{_sugar_tag(sugar_w)}",
    )


def latent_trainset_path(
    dataset: str,
    seed: int,
    *,
    latent_dim: int,
    recon_w: float,
    reg_w: float = 1.0,
    sugar_w: float = 0.0,
    root: Optional[str] = None,
) -> str:
    """``.../seed_{seed}/manifold_projector_knn_sugar{w}/latent_trainset.pt`` (k-agnostic)."""
    return os.path.join(
        manifold_projector_knn_dir(
            dataset,
            seed,
            latent_dim=latent_dim,
            recon_w=recon_w,
            reg_w=reg_w,
            sugar_w=sugar_w,
            root=root,
        ),
        "latent_trainset.pt",
    )


def evaluation_dir(exp_dir: str) -> str:
    path = os.path.join(exp_dir, "evaluation")
    os.makedirs(path, exist_ok=True)
    return path


def discover_seed_runs(exp_dir: str) -> List[Tuple[int, str]]:
    """Return sorted (seed, run_dir) for directories with generation.npz under exp_dir/seed_*."""
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

    if not runs:
        raise FileNotFoundError(
            f"No seed runs with {GENERATION_FILENAME} under {exp_dir}. "
            "Expected seed_*/generation.npz."
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
    """ASCII table: metric | mean \u00B1 std (std across random seeds).

    Core RNAGenScape paper metrics (Table 1 then Table 2) are listed first with
    section breaks; remaining metrics follow. Fraction heuristics that should be
    read as percentages are scaled for display; the "%" lives in the label only.
    """
    if summary.empty:
        return "(no numeric metrics)"

    by_metric = {str(row["metric"]): row for _, row in summary.iterrows()}
    pm = "\u00B1"
    label_width = 48
    rule = "-" * (label_width + 22)
    lines = [f"{'metric':<{label_width}} mean {pm} std", rule]

    def _append_row(label: str, mean: float, std: float) -> None:
        lines.append(f"{label:<{label_width}} {mean:.6g} {pm} {std:.6g}")

    shown: set = set()
    for spec in OPTIMIZATION_SUMMARY_CORE:
        if spec is None:
            lines.append(rule)
            continue
        key, label, scale = spec
        if key not in by_metric:
            continue
        row = by_metric[key]
        _append_row(label, float(row["mean"]) * scale, float(row["std"]) * scale)
        shown.add(key)

    for _, row in summary.iterrows():
        key = str(row["metric"])
        if key in shown:
            continue
        _append_row(key, float(row["mean"]), float(row["std"]))

    n = int(summary["n_seeds"].iloc[0]) if "n_seeds" in summary.columns else 0
    lines.append(rule)
    lines.append(f"{'n_seeds':<{label_width}} {n}")
    return "\n".join(lines)
