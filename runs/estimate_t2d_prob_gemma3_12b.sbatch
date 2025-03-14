#!/bin/sh
#SBATCH --job-name=estimate_t2d_prob_gemma3_12b
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL


module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate

python estimate_prob_given_drug.py \
--model_name google/gemma-3-12b-it \
--assessment diabetes \
--num_gpus 1 \
--batch_size 2 \
--seed 42
