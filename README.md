# RNAGenScape

**Property-Guided Optimized Generation of mRNA Sequences with Manifold Langevin Dynamics**

[![arXiv](https://img.shields.io/badge/arXiv-RNAGenScape-firebrick)](https://arxiv.org/pdf/2510.24736)

Generating property-optimized mRNA sequences is central to applications such as vaccine design and protein replacement therapy, but remains challenging: viable sequences occupy a narrow subset of sequence space, data are limited, and unconstrained edits often yield nonfunctional transcripts. RNAGenScape addresses this with property-guided manifold Langevin dynamics that optimize sequences while staying on a learned manifold of real data. It combines three components: (1) an organized autoencoder (OAE) that jointly learns sequence reconstruction and property prediction, (2) a manifold projector that maps updates back onto the data manifold, and (3) property-guided Langevin dynamics that refine latent embeddings under this constraint. The result is local, guided optimization that improves target properties while preserving biological viability.

---

## Environment

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.12.

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

**Primary experiment files:**

- OpenVaccine: ~2k samples
- Zebrafish: ~55k samples
- Ribosome loading: ~260k samples

---

## Method overview

The same procedure is used for each dataset (train / val / test split):

1. **Oracle.** Fine-tune (or load) a property predictor and check it on the held-out test set. This model is used only for final evaluation, not for guiding generation.
2. **Train RNAGenScape.** Fit the OAE on the training and validation sets (SUGAR to hole-fill the sparse manifolds). Train the manifold projector if it is a parameterized by a learnable DAE module (skip if using a kNN projector).
3. **Generate.** Encode unseen test sequences, run fitness-guided Langevin in latent space with periodic manifold projection, and decode to new sequences.
4. **Evaluate.** Score start vs. generated sequences with the frozen oracle (property improvement and related metrics).

---

## Citation

If you use RNAGenScape, please cite the paper:

```
@article{liao2025rnagenscape,
  title={RNAGenScape: Property-Guided, Optimized Generation of mRNA Sequences with Manifold Langevin Dynamics},
  author={Liao, Danqi and Liu, Chen and Sun, Xingzhi and Tang, Di{\'e} and Wang, Haochen and Youlten, Scott and Gopinath, Srikar Krishna and Lee, Haejeong and Strayer, Ethan C and Giraldez, Antonio J and others},
  journal={arXiv preprint arXiv:2510.24736},
  year={2025}
}
```
