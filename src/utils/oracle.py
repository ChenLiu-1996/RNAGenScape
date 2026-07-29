"""Oracle loading and sequence scoring."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

# Allow `python oracle.py` / `python src/utils/oracle.py`.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd
import torch

from utils.metrics import VOCAB_SIZE, decode_token_ids, to_token_ids
from utils.results import repo_root, results_root


# Dataset registry for evaluation (paths under data/).
DATASET_CONFIG: Dict[str, dict] = {
    "Zebrafish": {
        "relative_csv": "Zebrafish/MPRA_mean_translation_2hpf_pa_Fish5UTR.csv",
        "seq_key": "sequence",
        "seq_len": 124,
    },
    "OpenVaccine": {
        "relative_csv": "OpenVaccine/train.csv",
        "seq_key": "sequence",
        "seq_len": 107,
    },
    "Ribosome_loading": {
        "relative_csv": "Ribosome_loading/MRL_Random50Nuc_SynthesisLibrary_Sample/4.10_train_data_GSM3130438_egfp_pseudo_2.csv",
        "seq_key": "utr",
        "seq_len": 50,
    },
}

# Frozen zero-shot validators only — never enter train_oracle.py.
FROZEN_HF_ORACLES = frozenset({"UTRLM_TE", "UTRLM_MRL"})
# Architectures trained by train_oracle.py (pretrained UTRLM, Conv1d, ...).
# TE/MRL must never appear here.
TRAINABLE_ORACLES = frozenset({"UTRLM"})  # Conv1d added when implemented
SUPPORTED_ORACLES = FROZEN_HF_ORACLES | TRAINABLE_ORACLES


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_train_token_ids(dataset: str) -> torch.Tensor:
    """Load training sequences as token ids [N, L] for novelty reference."""
    if dataset not in DATASET_CONFIG:
        raise KeyError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_CONFIG)}")
    cfg = DATASET_CONFIG[dataset]
    csv_path = os.path.join(repo_root(), "data", cfg["relative_csv"])
    df = pd.read_csv(csv_path)
    seqs = df[cfg["seq_key"]].astype(str).tolist()
    return to_token_ids(seqs)


def oracle_checkpoint_path(dataset: str, oracle: str) -> str:
    return os.path.join(results_root(), dataset, oracle, "model.pt")


def load_oracle(oracle: str, dataset: str):
    """Load an oracle model. Frozen HF oracles need no local checkpoint."""
    if oracle not in SUPPORTED_ORACLES:
        raise ValueError(
            f"Unsupported oracle '{oracle}'. Supported: {sorted(SUPPORTED_ORACLES)}"
        )
    if dataset not in DATASET_CONFIG:
        raise KeyError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_CONFIG)}")

    device = resolve_device()
    seq_len = int(DATASET_CONFIG[dataset]["seq_len"])

    if oracle in TRAINABLE_ORACLES:
        ckpt = oracle_checkpoint_path(dataset, oracle)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)

    # Local import so metrics-only use does not require multimolecule.
    from models.utrlm import UTRLM, UTRLM_MRL, UTRLM_TE

    cls = {"UTRLM": UTRLM, "UTRLM_TE": UTRLM_TE, "UTRLM_MRL": UTRLM_MRL}[oracle]
    model = cls(device=device, seq_len=seq_len, vocab_size=VOCAB_SIZE, latent_dim=128)
    model = model.to(device)
    model.eval()

    if oracle in TRAINABLE_ORACLES:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

    return model


@torch.no_grad()
def score_sequences(
    model,
    sequences,
    *,
    oracle_name: str,
    batch_size: int = 128,
    return_embeddings: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Score token-id or string sequences. Returns (scores [N], embeddings or None)."""
    token_ids = to_token_ids(sequences)
    strings: List[str] = decode_token_ids(token_ids.cpu().numpy())  # type: ignore[assignment]
    if isinstance(strings, str):
        strings = [strings]

    scores: List[np.ndarray] = []
    embeds: List[np.ndarray] = []
    model.eval()

    for i in range(0, len(strings), batch_size):
        batch = strings[i : i + batch_size]
        if oracle_name in FROZEN_HF_ORACLES and hasattr(model, "score_sequences"):
            fitness = model.score_sequences(batch)
            z = model.encode(batch) if return_embeddings else None
        else:
            z = model.encode(batch)
            fitness = model.regress(z)
        scores.append(fitness.detach().cpu().numpy().reshape(-1))
        if return_embeddings:
            embeds.append(z.detach().cpu().numpy())

    score_arr = np.concatenate(scores, axis=0)
    embed_arr = np.concatenate(embeds, axis=0) if return_embeddings else None
    return score_arr, embed_arr


def _dummy_sequences(seq_len: int, n: int = 4) -> List[str]:
    bases = ["A", "G", "C", "U"]
    seqs = []
    for i in range(n):
        seqs.append("".join(bases[(i + j) % 4] for j in range(seq_len)))
    return seqs


if __name__ == "__main__":
    # Unit test: load each oracle and score dummy sequences.
    device = resolve_device()
    print(f"device={device}")

    # Frozen TE/MRL: zero-shot check. Trainable UTRLM needs results/.../model.pt;
    # if missing, fall back to official pretrained weights.
    checks = [
        ("UTRLM_TE", "Zebrafish"),
        ("UTRLM_MRL", "Ribosome_loading"),
        ("UTRLM", "OpenVaccine"),
    ]
    for oracle_name, dataset in checks:
        seq_len = int(DATASET_CONFIG[dataset]["seq_len"])
        dummy = _dummy_sequences(seq_len, n=4)
        print(f"\n=== {oracle_name} on {dataset} (seq_len={seq_len}) ===")
        try:
            model = load_oracle(oracle_name, dataset)
        except FileNotFoundError as exc:
            if oracle_name != "UTRLM":
                print(f"SKIP (missing checkpoint): {exc}")
                continue
            from models.utrlm import UTRLM, default_pretrained_pkl

            pretrained = default_pretrained_pkl()
            print(f"missing fine-tuned checkpoint: {exc}")
            print(f"loading official pretrained from: {pretrained}")
            model = UTRLM(
                device=device,
                seq_len=seq_len,
                vocab_size=VOCAB_SIZE,
                latent_dim=128,
                pretrained_path=pretrained,
            )
            model.eval()
        scores, embeds = score_sequences(
            model,
            dummy,
            oracle_name=oracle_name,
            batch_size=2,
            return_embeddings=True,
        )
        assert scores.shape == (len(dummy),), scores.shape
        assert embeds is not None and embeds.shape[0] == len(dummy), embeds.shape if embeds is not None else None
        print(f"scores={scores.tolist()}")
        print(f"embeddings.shape={tuple(embeds.shape)}")
        print("OK")
    print("\noracle unit test finished")
