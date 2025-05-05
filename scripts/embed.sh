#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module purge
module load python/3.10
source ~/videoqa-rag/.venv/bin/activate

python scripts/embed.py
