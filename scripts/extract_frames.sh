#!/bin/bash
#SBATCH --job-name=frames
#SBATCH --output=logs/frames_%j.out
#SBATCH --error=logs/frames_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8


source ~/.bashrc
source ~/.venv/bin/activate

module purge
module load FFmpeg/5.1.2-GCC-11.3.0

python scripts/extract_frames.py
