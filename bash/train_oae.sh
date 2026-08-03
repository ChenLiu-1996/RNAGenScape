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
# Ablation axes encoded in results path: d{LATENT_DIM}_recon{RECON_W}
# kl_w is fixed (not path-ablated): mild β-VAE so recon/regression dominate.
RECON_W=5.0
KL_W=1e-4
LATENT_DIM=128

for data in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "========== Training OAE on ${data} (seed ${seed}, D=${LATENT_DIM}, recon_w=${RECON_W}, kl_w=${KL_W}) =========="
    python src/train_oae.py \
      --dataset "${data}" \
      --lr 1e-3 \
      --recon_w "${RECON_W}" \
      --kl_w "${KL_W}" \
      --latent_dim "${LATENT_DIM}" \
      --label_norm normal \
      --max_epochs 100 \
      --patience 20 \
      --batch_size 128 \
      --seed "${seed}"
  done
done

echo "Done. Checkpoints:"
echo "  ${ROOT_DIR}/results/<dataset>/OAE/d${LATENT_DIM}_recon${RECON_W//./p}/seed_*/model.pt"
