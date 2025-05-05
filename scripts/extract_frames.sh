#!/bin/bash
#SBATCH --job-name=frames
#SBATCH --output=logs/frames_%j.out
#SBATCH --error=logs/frames_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/.venv/bin/activate
python scripts/extract_frames.py
