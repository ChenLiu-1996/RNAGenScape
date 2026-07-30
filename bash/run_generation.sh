#!/bin/bash
#SBATCH --job-name=run_generation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --output=run_generation-%j.out

# Torch in .venv ships its own CUDA/cuDNN - do not module-load system CUDA.
module purge
module load miniconda

ROOT_DIR="/gpfs/radev/project/krishnaswamy_smita/cl2482/RNAGenScape"
cd "${ROOT_DIR}"
source .venv/bin/activate

DATASETS=(OpenVaccine Zebrafish RibosomeLoading)
SEEDS=(1 2 3)
DIRECTIONS=(1 -1)
# projector configs: "dae" or "knn:<k>"
PROJECTORS=(dae knn:1 knn:5 knn:10)
SUGAR_W=0.0
SUBSAMPLE_SEED=42
ORACLE=UTRLM

for data in "${DATASETS[@]}"; do
  for direction in "${DIRECTIONS[@]}"; do
    if (( $(echo "${direction} > 0" | bc -l) )); then
      DIR_TAG=pos
    else
      DIR_TAG=neg
    fi
    for proj_cfg in "${PROJECTORS[@]}"; do
      if [[ "${proj_cfg}" == dae ]]; then
        PROJECTOR=dae
        KNN_K=1
        PROJ_TAG=dae
      else
        PROJECTOR=knn
        KNN_K="${proj_cfg#knn:}"
        PROJ_TAG="knn_k${KNN_K}"
      fi
      EXPERIMENT="${DIR_TAG}_samehyper_sugar0p0_${PROJ_TAG}"

      for seed in "${SEEDS[@]}"; do
        echo "========== Generate ${data} ${EXPERIMENT} seed=${seed} =========="
        python src/run_generation.py \
          --dataset "${data}" \
          --method rnagenscape \
          --model OAE \
          --experiment "${EXPERIMENT}" \
          --seed "${seed}" \
          --subsample_seed "${SUBSAMPLE_SEED}" \
          --direction "${direction}" \
          --projector "${PROJECTOR}" \
          --knn_k "${KNN_K}" \
          --sugar_w "${SUGAR_W}" \
          --latent_normalization none \
          --num_steps 100 \
          --step_size 5e-3 \
          --temperature 1e-3 \
          --use_fitness \
          --use_projector \
          --batch_size 128
      done

      echo "========== Eval optimization ${data} ${EXPERIMENT} oracle=${ORACLE} =========="
      python src/evaluation/eval_optimization.py \
        --dataset "${data}" \
        --model OAE \
        --experiment "${EXPERIMENT}" \
        --oracle "${ORACLE}" \
        --batch_size 128
    done
  done
done

echo "Done."
echo "  checkpoints: results/<dataset>/OAE/seed_*/model.pt (+ dae/knn projector)"
echo "  generation:  results/<dataset>/OAE/<experiment>/seed_*/generation.npz"
echo "  evaluation:  results/<dataset>/OAE/<experiment>/evaluation/"
