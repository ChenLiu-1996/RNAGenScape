# RNAGenScape

**Property-Guided Optimized Generation of mRNA Sequences with Manifold Langevin Dynamics**

[![arXiv](https://img.shields.io/badge/arXiv-RNAGenScape-firebrick)](https://arxiv.org/pdf/2510.24736)

RNAGenScape learns a structured latent space over mRNA sequences (via a joint autoencoder–regressor), then optimizes a continuous property with **on-manifold Langevin dynamics**. Off-manifold steps are retracted with a denoising autoencoder (or kNN), optionally densified by SUGAR, and decoded back to sequences. Final scoring uses a separate oracle (UTR-LM or a Conv1d regressor), not the guidance head.

This repository is being rebuilt from the research codebase (`mRNA-translation`) with a cleaner layout, coherent experiment pipelines, and more rigorous evaluation. Source code will land in subsequent migration steps.

---

## Environment

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
cd /path/to/RNAGenScape

# Create / sync the virtualenv from the lockfile
uv sync --python 3.12

# Activate
source .venv/bin/activate
```

### Cluster note (McCleary / PyG)

`torch-scatter` is built from source on clusters with older glibc. Load a newer GCC **only while building**, then unload it before running Python (leaving GCC loaded can break the torch_scatter ABI):

```bash
module load GCC/12.2.0 CUDA/12.2.2
UV_HTTP_TIMEOUT=600 CXX=$(which g++) CC=$(which gcc) uv sync --python 3.12
module unload GCC
```

---

## Data

Datasets live under `data/` (gitignored; present on disk after setup).

| Directory | Contents | Primary use |
|-----------|----------|-------------|
| `data/Zebrafish/` | MPRA 5′ UTR translation CSVs | Translation efficiency |
| `data/OpenVaccine/` | OpenVaccine sequences + reactivity | Reactivity optimization |
| `data/Ribosome_loading/` | MRL / TE libraries (incl. fixed train–test splits) | Ribosome load / TE |

**Primary experiment files:**

- Zebrafish: `MPRA_mean_translation_2hpf_pa_Fish5UTR.csv` (and related MPRA variants)
- OpenVaccine: `train.csv`
- Ribosome loading: Mengdi GSM3130438 train/test CSVs (e.g. `egfp_pseudo` library)

---

## Method overview

1. **Train** an `OrganizedAE` (reconstruction + property regression).
2. **Fit** a manifold projector on AE latents (DAE or kNN), optionally with SUGAR densification.
3. **Walk** with fitness-guided manifold Langevin (MFD-ULA) and decode to sequences.
4. **Evaluate** generated sequences with a held-out-style oracle (UTR-LM or Conv1d).

Training, generation, and evaluation CLIs will be documented here once the package layout is in place.

---

## Citation

If you use RNAGenScape, please cite the paper:

```
https://arxiv.org/pdf/2510.24736
```
