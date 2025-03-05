#!/bin/sh
#SBATCH --job-name=estimate_brc_prob_llama3.1_8b_cot
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
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
--model_name meta-llama/Llama-3.1-8B-Instruct \
--assessment cancer_breast \
--num_gpus 1 \
--batch_size 4 \
--cot \
--seed 42
