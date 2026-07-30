#!/bin/bash
#SBATCH --job-name=train_projector
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --output=train_projector-%j.out

# Torch in .venv ships its own CUDA/cuDNN - do not module-load system CUDA.
module purge
module load miniconda

ROOT_DIR="/gpfs/radev/project/krishnaswamy_smita/cl2482/RNAGenScape"
cd "${ROOT_DIR}"
source .venv/bin/activate

DATASETS=(OpenVaccine Zebrafish RibosomeLoading)
SEEDS=(1 2 3)
# dae: train weights; knn: cache train latents (k chosen at generation)
PROJECTORS=(dae knn)
SUGAR_W=0.0

for data in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for projector in "${PROJECTORS[@]}"; do
      echo "========== Projector ${projector} on ${data} (seed ${seed}) =========="
      python src/train_manifold_projector.py \
        --dataset "${data}" \
        --projector "${projector}" \
        --sugar_w "${SUGAR_W}" \
        --dae_lr 1e-3 \
        --latent_normalization none \
        --dae_epochs 100 \
        --dae_patience 20 \
        --batch_size 128 \
        --seed "${seed}"
    done
  done
done

echo "Done. Checkpoints under results/<dataset>/OAE/seed_*/"
