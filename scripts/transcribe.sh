#!/bin/bash
#SBATCH --job-name=transcribe
#SBATCH --output=logs/transcribe_%j.out
#SBATCH --error=logs/transcribe_%j.err
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Load necessary modules if needed
module purge
module load python/3.10

# Activate your venv
source ~/videoqa-rag/.venv/bin/activate

# Optional: print Python path to verify
which python

# Run transcription script
python scripts/transcribe.py
