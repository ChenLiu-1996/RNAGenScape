"""Metric primitives for RNAGenScape evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import ot
import scipy.spatial.distance as sc_dist
import torch

try:
    import RNA as ViennaRNA
except ImportError:  # pragma: no cover
    ViennaRNA = None


# ---------------------------------------------------------------------------
# Sequence vocabulary / decode
# ---------------------------------------------------------------------------

TOKENS = ["<pad>", "A", "G", "C", "T", "U", "N"]
VOCAB_SIZE = len(TOKENS)


def idx_to_token(idx: int) -> str:
    return TOKENS[int(idx)]


def token_to_idx(token: str) -> int:
    return TOKENS.index(token)


def decode_token_ids(tokens: np.ndarray) -> Union[str, List[str]]:
    """Decode integer token ids to nucleotide strings (pads stripped)."""
    arr = np.asarray(tokens)
    if arr.ndim == 1:
        chars = [idx_to_token(t) for t in arr if idx_to_token(t) != "<pad>"]
        return "".join(chars)
    if arr.ndim == 2:
        return [decode_token_ids(arr[i]) for i in range(arr.shape[0])]
    raise ValueError(f"tokens must be 1D or 2D, got shape {arr.shape}")


def to_token_ids(x) -> torch.Tensor:
    """Convert one-hot, token ids, or string batches to LongTensor [B, L]."""
    if isinstance(x, (list, tuple)):
        if len(x) == 0 or isinstance(x[0], str):
            rows = []
            for seq in x:
                ids = [token_to_idx(ch) for ch in seq]
                rows.append(ids)
            # Pad to max length within batch.
            max_len = max(len(r) for r in rows) if rows else 0
            pad_id = token_to_idx("<pad>")
            rows = [r + [pad_id] * (max_len - len(r)) for r in rows]
            return torch.tensor(rows, dtype=torch.long)
        x = torch.stack([torch.as_tensor(item) for item in x])

    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("U", "O", "S"):
            return to_token_ids(list(x))
        x = torch.from_numpy(x)

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Unsupported sequence batch type: {type(x)}")

    if x.dim() == 3:
        return torch.argmax(x, dim=-1).long()
    if x.dim() == 2:
        return x.long()
    raise ValueError(f"Expected 2D token ids or 3D one-hot, got shape {tuple(x.shape)}")


def _as_nucleotide_strings(sequences) -> List[str]:
    """Normalize mixed sequence inputs to a list of nucleotide strings."""
    if isinstance(sequences, str):
        return [sequences]
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.detach().cpu().numpy()
    if isinstance(sequences, np.ndarray):
        if sequences.dtype.kind in ("U", "O", "S"):
            return [str(s) for s in sequences.tolist()]
        if np.issubdtype(sequences.dtype, np.integer):
            decoded = decode_token_ids(sequences)
            return decoded if isinstance(decoded, list) else [decoded]
        if np.issubdtype(sequences.dtype, np.floating):
            # One-hot [N, L, V] or [L, V]
            if sequences.ndim == 2:
                sequences = sequences[None, ...]
            ids = np.argmax(sequences, axis=-1)
            decoded = decode_token_ids(ids)
            return decoded if isinstance(decoded, list) else [decoded]
    if isinstance(sequences, (list, tuple)):
        if len(sequences) == 0:
            return []
        if isinstance(sequences[0], str):
            return list(sequences)
        return _as_nucleotide_strings(np.asarray(sequences))
    raise TypeError(f"Cannot convert type {type(sequences)} to nucleotide strings")


# ---------------------------------------------------------------------------
# Property change
# ---------------------------------------------------------------------------

def property_change_metrics(
    start_scores: np.ndarray,
    generated_scores: np.ndarray,
    direction: float,
    start_tokens,
    generated_tokens,
) -> Dict[str, float]:
    """Property optimization metrics from oracle scores and sequence identity.

    Percentage metrics are on a 0-100 scale.
    pct_improved / pct_worse use oracle property change in the optimization direction.
    pct_identical_to_input uses exact sequence equality (token ids), not property equality.
    """
    start_scores = np.asarray(start_scores, dtype=np.float64).reshape(-1)
    generated_scores = np.asarray(generated_scores, dtype=np.float64).reshape(-1)
    if start_scores.shape != generated_scores.shape:
        raise ValueError(
            f"Score shape mismatch: start {start_scores.shape} vs gen {generated_scores.shape}"
        )
    start_ids = np.asarray(to_token_ids(start_tokens).cpu().numpy())
    gen_ids = np.asarray(to_token_ids(generated_tokens).cpu().numpy())
    if start_ids.shape != gen_ids.shape:
        raise ValueError(
            f"Sequence shape mismatch: start {start_ids.shape} vs gen {gen_ids.shape}"
        )

    delta = generated_scores - start_scores
    n = len(delta)
    identical = np.all(start_ids == gen_ids, axis=1)
    # direction > 0: improve means delta > 0; direction < 0: improve means delta < 0
    signed = direction * delta
    improved = signed > 0
    worse = signed < 0
    edited = ~identical
    property_unchanged = signed == 0
    edited_property_unchanged = edited & property_unchanged

    return {
        "mean_property_change": float(np.mean(delta)),
        "median_property_change": float(np.median(delta)),
        "pct_improved": float(100.0 * improved.sum() / n),
        "pct_worse": float(100.0 * worse.sum() / n),
        "pct_identical_to_input": float(100.0 * identical.sum() / n),
        "pct_edited_property_unchanged": float(100.0 * edited_property_unchanged.sum() / n),
        "n_sequences": float(n),
    }


# ---------------------------------------------------------------------------
# Diversity / novelty / distances
# ---------------------------------------------------------------------------

def uniqueness(sequences) -> float:
    """Percentage (0-100) of unique sequences in a batch of token ids [N, L]."""
    arr = np.asarray(to_token_ids(sequences).cpu().numpy())
    if arr.ndim != 2 or arr.shape[0] == 0:
        return 0.0
    unique = {tuple(row.tolist()) for row in arr}
    return 100.0 * len(unique) / arr.shape[0]


def novelty(sequences, reference_sequences) -> float:
    """Percentage (0-100) of sequences not exact-matching any reference (token ids)."""
    seqs = np.asarray(to_token_ids(sequences).cpu().numpy())
    refs = np.asarray(to_token_ids(reference_sequences).cpu().numpy())
    if seqs.shape[0] == 0:
        return 0.0
    ref_set = {tuple(row.tolist()) for row in refs}
    n_novel = sum(1 for row in seqs if tuple(row.tolist()) not in ref_set)
    return 100.0 * n_novel / seqs.shape[0]


def pairwise_edit_distance(start, generated) -> np.ndarray:
    """Per-pair Hamming edit counts between start and generated token ids."""
    a = np.asarray(to_token_ids(start).cpu().numpy())
    b = np.asarray(to_token_ids(generated).cpu().numpy())
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: start {a.shape} vs generated {b.shape}")
    return (a != b).sum(axis=-1).astype(np.int64)


def wasserstein_distance(seq_a, seq_b, metric: str = "hamming") -> float:
    a = np.asarray(seq_a, dtype=np.float64)
    b = np.asarray(seq_b, dtype=np.float64)
    cost = sc_dist.cdist(a, b, metric=metric)
    wa = np.ones(a.shape[0]) / a.shape[0]
    wb = np.ones(b.shape[0]) / b.shape[0]
    return float(ot.emd2(wa, wb, cost))


def manifold_distance(generated, reference) -> Dict[str, float]:
    """Mean / median Euclidean distance to nearest reference point.

    Within-run dispersion is not reported; seed-level mean +/- std is
    computed later across random seeds.
    """
    gen = np.asarray(generated, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    pairwise = sc_dist.cdist(gen, ref, metric="euclidean")
    closest = pairwise.min(axis=1)
    return {
        "manifold_distance_mean": float(np.mean(closest)),
        "manifold_distance_median": float(np.median(closest)),
    }


def nearest_hamming_distance(query, reference) -> np.ndarray:
    """For each query sequence, Hamming distance to the nearest reference sequence."""
    q = np.asarray(to_token_ids(query).cpu().numpy())
    r = np.asarray(to_token_ids(reference).cpu().numpy())
    if q.ndim != 2 or r.ndim != 2:
        raise ValueError(f"Expected 2D token ids, got query {q.shape}, reference {r.shape}")
    if r.shape[0] == 0:
        raise ValueError("reference set is empty")
    pairwise = sc_dist.cdist(q, r, metric="hamming")
    return pairwise.min(axis=1).astype(np.float64)


def select_elite_mask(
    scores: np.ndarray,
    direction: float,
    std_scale: float = 1.0,
) -> np.ndarray:
    """Elite = much better than median in the optimization direction.

    A point is elite if direction * (score - median) > std_scale * std(scores).
    For direction > 0 this is score > median + std_scale * std;
    for direction < 0 this is score < median - std_scale * std.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(scores))
    spread = float(np.std(scores))
    return (direction * (scores - med)) > (std_scale * spread)


def elite_distance_metrics(
    *,
    generated_tokens,
    start_tokens,
    pool_tokens,
    pool_scores: np.ndarray,
    direction: float,
    std_scale: float = 1.0,
) -> Dict[str, float]:
    """Distances from generated (and start) sequences to the held-out elite set."""
    pool_scores = np.asarray(pool_scores, dtype=np.float64).reshape(-1)
    elite_mask = select_elite_mask(pool_scores, direction=direction, std_scale=std_scale)
    n_elite = int(elite_mask.sum())
    med = float(np.median(pool_scores))
    out: Dict[str, float] = {
        "elite_n": float(n_elite),
        "elite_pool_n": float(pool_scores.size),
        "elite_median": med,
        "elite_std_scale": float(std_scale),
    }
    if n_elite == 0:
        out["elite_nn_hamming_gen_mean"] = float("nan")
        out["elite_nn_hamming_start_mean"] = float("nan")
        out["elite_w2_hamming"] = float("nan")
        return out

    pool_ids = to_token_ids(pool_tokens)
    elite_ids = pool_ids[elite_mask]
    gen_nn = nearest_hamming_distance(generated_tokens, elite_ids)
    start_nn = nearest_hamming_distance(start_tokens, elite_ids)
    out["elite_nn_hamming_gen_mean"] = float(np.mean(gen_nn))
    out["elite_nn_hamming_start_mean"] = float(np.mean(start_nn))
    out["elite_w2_hamming"] = float(
        wasserstein_distance(
            to_token_ids(generated_tokens).cpu().numpy(),
            elite_ids.cpu().numpy(),
            metric="hamming",
        )
    )
    return out


# ---------------------------------------------------------------------------
# Biological heuristics
# ---------------------------------------------------------------------------

def uorf_aug_content(sequences, subseq: str = "AUG") -> np.ndarray:
    strings = _as_nucleotide_strings(sequences)
    out = np.zeros(len(strings), dtype=np.float64)
    for i, seq in enumerate(strings):
        seq = seq.replace("T", "U")
        max_count = max(len(seq) // len(subseq), 1)
        out[i] = seq.count(subseq) / max_count
    return out


def uorf_oof_aug_content(sequences, subseq: str = "AUG") -> np.ndarray:
    strings = _as_nucleotide_strings(sequences)
    out = np.zeros(len(strings), dtype=np.float64)
    for i, seq in enumerate(strings):
        seq = seq.replace("T", "U")
        oof = 0
        for pos in range(len(seq) - len(subseq) + 1):
            if seq[pos : pos + len(subseq)] == subseq and pos % 3 != 0:
                oof += 1
        max_count = max((len(seq) + 1) // len(subseq), 1)
        out[i] = oof / max_count
    return out


def gc_content(sequences) -> np.ndarray:
    strings = _as_nucleotide_strings(sequences)
    out = np.zeros(len(strings), dtype=np.float64)
    for i, seq in enumerate(strings):
        if len(seq) == 0:
            out[i] = 0.0
        else:
            out[i] = (seq.count("G") + seq.count("C")) / len(seq)
    return out


_KOZAK_WEIGHTS = np.array(
    [
        [0.04210526, 0.0, 0.03157895, 0.05263158, 0.0],
        [0.04210526, 0.05263158, 0.10526316, 0.0625, 0.0],
        [0.03157895, 0.04210526, 0.05263158, 0.07368421, 0.0],
        [0.03157895, 0.01052632, 0.04210526, 0.05263158, 0.0],
        [0.08421053, 0.07368421, 0.18947368, 0.10526316, 0.0],
        [0.04210526, 0.05263158, 0.05263158, 0.08421053, 0.0],
        [0.12631579, 0.0625, 0.12631579, 0.21052632, 0.0],
        [0.83157895, 0.12631579, 0.65263158, 0.16842105, 0.0],
        [0.15789474, 0.06315789, 0.11578947, 0.2, 0.0],
        [0.21052632, 0.09473684, 0.31578947, 0.51578947, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.24210526, 0.16666667, 0.53684211, 0.13684211, 0.0],
        [0.15789474, 0.09473684, 0.09473684, 0.24210526, 0.0],
        [0.05263158, 0.08421053, 0.14736842, 0.09473684, 0.0],
        [0.07216495, 0.05263158, 0.10526316, 0.06315789, 0.0],
        [0.0, 0.0, 0.0, 0.05263158, 0.0],
        [0.05263158, 0.05263158, 0.10526316, 0.09473684, 0.0],
        [0.04210526, 0.03157895, 0.05263158, 0.04210526, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.04210526, 0.04210526, 0.08421053, 0.07368421, 0.0],
        [0.0625, 0.04210526, 0.09473684, 0.05263158, 0.0],
    ]
)


def kozak_similarity(sequences) -> np.ndarray:
    strings = _as_nucleotide_strings(sequences)
    scores = np.zeros(len(strings), dtype=np.float64)
    max_score = float(np.sum(_KOZAK_WEIGHTS.max(axis=1)))
    base_to_idx = {"A": 0, "U": 1, "G": 2, "C": 3}

    for i, seq in enumerate(strings):
        seq = seq.upper().replace("T", "U")
        aug_pos = seq.find("AUG")
        if aug_pos < 0:
            scores[i] = 0.0
            continue
        start = aug_pos - 10
        end = aug_pos + 13
        if start < 0 or end > len(seq):
            scores[i] = 0.0
            continue
        region = seq[start:end]
        score = 0.0
        for k, base in enumerate(region):
            if k >= len(_KOZAK_WEIGHTS):
                break
            score += _KOZAK_WEIGHTS[k][base_to_idx.get(base, 4)]
        scores[i] = score / max_score if max_score > 0 else 0.0
    return scores


def mfe_scores(sequences) -> np.ndarray:
    if ViennaRNA is None:
        raise ImportError("ViennaRNA is required for MFE scores")
    strings = _as_nucleotide_strings(sequences)
    out = np.zeros(len(strings), dtype=np.float64)
    for i, seq in enumerate(strings):
        out[i] = ViennaRNA.fold(seq)[1]
    return out


def heuristic_summary(sequences, prefix: str) -> Dict[str, float]:
    """Per-run means only; seed-level mean +/- std is aggregated later."""
    uorf = uorf_aug_content(sequences)
    oof = uorf_oof_aug_content(sequences)
    gc = gc_content(sequences)
    kozak = kozak_similarity(sequences)
    out = {
        f"{prefix}_uorf_aug_mean": float(np.mean(uorf)),
        f"{prefix}_uorf_oof_aug_mean": float(np.mean(oof)),
        f"{prefix}_gc_mean": float(np.mean(gc)),
        f"{prefix}_kozak_mean": float(np.mean(kozak)),
    }
    if ViennaRNA is not None:
        mfe = mfe_scores(sequences)
        out[f"{prefix}_mfe_mean"] = float(np.mean(mfe))
    return out


def subsample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = min(max_n, n)
    return rng.choice(n, size=size, replace=False)


def optimization_metrics(
    *,
    start_tokens,
    generated_tokens,
    start_scores: np.ndarray,
    generated_scores: np.ndarray,
    generated_embeddings: Optional[np.ndarray],
    reference_embeddings: Optional[np.ndarray],
    novelty_vs_train_reference,
    novelty_vs_test_reference,
    test_pool_tokens,
    test_pool_scores: np.ndarray,
    direction: float,
    seed: int,
    subsample: int = 1000,
    elite_std_scale: float = 1.0,
) -> Dict[str, Any]:
    """Full optimization metric dict for one seed (for CSV rows).

    Percentage metrics use a 0-100 scale.
    """
    metrics: Dict[str, Any] = {}
    metrics.update(
        property_change_metrics(
            start_scores,
            generated_scores,
            direction,
            start_tokens=start_tokens,
            generated_tokens=generated_tokens,
        )
    )

    gen_ids = to_token_ids(generated_tokens)
    start_ids = to_token_ids(start_tokens)
    n = gen_ids.shape[0]
    idx = subsample_indices(n, subsample, seed)

    metrics["uniqueness_pct"] = float(uniqueness(gen_ids[idx]))
    metrics["novelty_vs_train_pct"] = float(novelty(gen_ids[idx], novelty_vs_train_reference))
    metrics["novelty_vs_test_pct"] = float(novelty(gen_ids[idx], novelty_vs_test_reference))

    metrics["sequence_w2"] = float(
        wasserstein_distance(
            gen_ids.cpu().numpy(),
            start_ids.cpu().numpy(),
            metric="hamming",
        )
    )

    if generated_embeddings is not None and reference_embeddings is not None:
        metrics["latent_w2"] = float(
            wasserstein_distance(
                generated_embeddings, reference_embeddings, metric="euclidean"
            )
        )
        metrics.update(manifold_distance(generated_embeddings, reference_embeddings))

    edits = pairwise_edit_distance(start_ids, gen_ids)
    metrics["mean_edit_distance"] = float(np.mean(edits))
    metrics["median_edit_distance"] = float(np.median(edits))

    metrics.update(
        elite_distance_metrics(
            generated_tokens=generated_tokens,
            start_tokens=start_tokens,
            pool_tokens=test_pool_tokens,
            pool_scores=test_pool_scores,
            direction=direction,
            std_scale=elite_std_scale,
        )
    )

    metrics.update(heuristic_summary(gen_ids, prefix="generated"))
    metrics.update(heuristic_summary(start_ids, prefix="start"))
    return metrics
