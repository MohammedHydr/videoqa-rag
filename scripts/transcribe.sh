#!/bin/bash
#SBATCH --job-name=transcribe
#SBATCH --output=logs/transcribe_%j.out
#SBATCH --error=logs/transcribe_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal

source ~/.bashrc
module purge
module load python/3.10

source ~/videoqa-rag/.venv/bin/activate

python scripts/transcribe.py
