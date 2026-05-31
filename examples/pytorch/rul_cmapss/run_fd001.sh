#!/bin/bash
#SBATCH --job-name=pat176_rul
#SBATCH --output=examples/pytorch/rul_cmapss/outputs/fd001/%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

cd /project/nlp-work5/hongyu-s/transformers
python examples/pytorch/rul_cmapss/run_fd001.py \
    --model path \
    --data_dir data/cmapss \
    --epochs 40
