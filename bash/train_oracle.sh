#!/bin/bash
#SBATCH --job-name=train_oracle
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --output=train_oracle-%j.out

# Torch in .venv ships its own CUDA/cuDNN - do not module-load system CUDA.
module purge
module load miniconda

ROOT_DIR="/gpfs/radev/project/krishnaswamy_smita/cl2482/RNAGenScape"
cd "${ROOT_DIR}"
source .venv/bin/activate

DATASETS=(OpenVaccine Zebrafish RibosomeLoading)
ORACLE=UTRLM
SEED=1

for data in "${DATASETS[@]}"; do
    echo "========== Training ${ORACLE} on ${data} (seed ${SEED}) =========="
    python src/train_oracle.py \
        --dataset "${data}" \
        --oracle "${ORACLE}" \
        --lr 1e-3 \
        --label_norm normal \
        --max_epochs 100 \
        --patience 20 \
        --batch_size 128 \
        --seed "${SEED}"
done

echo "Done. Checkpoints:"
echo "  ${ROOT_DIR}/results/<dataset>/${ORACLE}/model.pt"
