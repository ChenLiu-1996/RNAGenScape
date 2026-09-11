#!/bin/bash
#SBATCH --job-name=train_projector
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --output=train_projector-%j.out

# Train / cache manifold projectors for OAE recon_w=5.0:
#   - DAE vs kNN (k chosen later at generation: knn:1 / knn:5 / knn:10)
#   - SUGAR weights in SUGAR_WS (each saved under ..._sugar{w}/ — no collisions)
#
# Torch in .venv ships its own CUDA/cuDNN - do not module-load system CUDA.
module purge
module load miniconda

set -euo pipefail

ROOT_DIR="/gpfs/radev/project/krishnaswamy_smita/cl2482/RNAGenScape"
cd "${ROOT_DIR}"
source .venv/bin/activate

DATASETS=(OpenVaccine Zebrafish RibosomeLoading)
SEEDS=(1 2 3)
# Must match the trained OAE ablation folder: d{LATENT_DIM}_recon{RECON_W}_reg{REG_W}
LATENT_DIM=128
RECON_W=5.0
REG_W=1.0

# dae: train weights; knn: cache train (+SUGAR) latents (k chosen at generation)
PROJECTORS=(dae knn)
# Match prior sugar ablation grid (mRNA-translation).
SUGAR_WS=(0.0 0.01 0.1 1.0 10.0)
SKIP_EXISTING="${SKIP_EXISTING:-0}"

float_tag() {
  PYTHONPATH="${ROOT_DIR}/src" python -c "from utils.results import float_tag; print(float_tag(float('${1}')))"
}
RECON_TAG="$(float_tag "${RECON_W}")"
REG_TAG="$(float_tag "${REG_W}")"
OAE_TAG="d${LATENT_DIM}_recon${RECON_TAG}_reg${REG_TAG}"

for data in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    oae_ckpt="${ROOT_DIR}/results/${data}/OAE/${OAE_TAG}/seed_${seed}/model.pt"
    if [[ ! -f "${oae_ckpt}" ]]; then
      echo "========== SKIP ${data} seed=${seed} (missing OAE ckpt: ${oae_ckpt}) =========="
      continue
    fi
    for sugar_w in "${SUGAR_WS[@]}"; do
      sugar_tag="$(float_tag "${sugar_w}")"
      for projector in "${PROJECTORS[@]}"; do
        if [[ "${projector}" == "dae" ]]; then
          out="${ROOT_DIR}/results/${data}/OAE/${OAE_TAG}/seed_${seed}/manifold_projector_dae_sugar${sugar_tag}/model_latentnorm_none.pt"
        else
          out="${ROOT_DIR}/results/${data}/OAE/${OAE_TAG}/seed_${seed}/manifold_projector_knn_sugar${sugar_tag}/latent_trainset.pt"
        fi
        if [[ "${SKIP_EXISTING}" == "1" && -f "${out}" ]]; then
          echo "========== SKIP ${projector} ${data} seed=${seed} sugar=${sugar_w} (exists) =========="
          continue
        fi

        echo "========== Projector ${projector} on ${data} seed=${seed} D=${LATENT_DIM} recon_w=${RECON_W} reg_w=${REG_W} sugar_w=${sugar_w} =========="
        python src/train_manifold_projector.py \
          --dataset "${data}" \
          --projector "${projector}" \
          --latent_dim "${LATENT_DIM}" \
          --recon_w "${RECON_W}" \
          --sugar_w "${sugar_w}" \
          --dae_lr 1e-3 \
          --latent_normalization none \
          --dae_epochs 100 \
          --dae_patience 20 \
          --batch_size 128 \
          --seed "${seed}"
      done
    done
  done
done

echo "Done."
echo "  DAE: results/<dataset>/OAE/${OAE_TAG}/seed_*/manifold_projector_dae_sugar*/"
echo "  kNN: results/<dataset>/OAE/${OAE_TAG}/seed_*/manifold_projector_knn_sugar*/"
echo "  Ablate k at generation via --projector knn --knn_k {1,5,10}."
