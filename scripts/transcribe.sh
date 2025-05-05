#!/bin/bash
#SBATCH --job-name=transcribe
#SBATCH --output=logs/transcribe_%j.out
#SBATCH --error=logs/transcribe_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

source ~/.bashrc
module purge
module load python/3.10

python scripts/transcribe.py
