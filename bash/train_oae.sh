#!/bin/bash
#SBATCH --job-name=train_oae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --output=train_oae-%j.out

# Torch in .venv ships its own CUDA/cuDNN - do not module-load system CUDA.
module purge
module load miniconda

ROOT_DIR="/gpfs/radev/project/krishnaswamy_smita/cl2482/RNAGenScape"
cd "${ROOT_DIR}"
source .venv/bin/activate

DATASETS=(OpenVaccine Zebrafish RibosomeLoading)
SEEDS=(1 2 3)
# Ablation axes encoded in results path: d{LATENT_DIM}_recon{RECON_W}_reg{REG_W}
# Deterministic AE (no KL); manifold projector provides Langevin smoothing.
RECON_W=5.0
REG_W=1.0
LATENT_DIM=128

for data in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "========== Training OAE on ${data} (seed ${seed}, D=${LATENT_DIM}, recon_w=${RECON_W}, reg_w=${REG_W}) =========="
    python src/train_oae.py \
      --dataset "${data}" \
      --lr 1e-3 \
      --recon_w "${RECON_W}" \
      --reg_w "${REG_W}" \
      --latent_dim "${LATENT_DIM}" \
      --label_norm normal \
      --max_epochs 100 \
      --patience 20 \
      --batch_size 128 \
      --seed "${seed}"
  done
done

echo "Done. Checkpoints:"
echo "  ${ROOT_DIR}/results/<dataset>/OAE/d${LATENT_DIM}_recon$(PYTHONPATH="${ROOT_DIR}/src" python -c "from utils.results import float_tag; print(float_tag(float('${RECON_W}')))")_reg$(PYTHONPATH="${ROOT_DIR}/src" python -c "from utils.results import float_tag; print(float_tag(float('${REG_W}')))")/seed_*/model.pt"
