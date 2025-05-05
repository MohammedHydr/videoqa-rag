#!/bin/bash
#SBATCH --job-name=frames
#SBATCH --output=logs/frames_%j.out
#SBATCH --error=logs/frames_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8


# Load necessary modules
module purge
module load python/3.10
module load ffmpeg/4.2.2

# Activate virtual environment
source ~/videoqa-rag/.venv/bin/activate

# Run the frame extraction script
python scripts/extract_frames.py
