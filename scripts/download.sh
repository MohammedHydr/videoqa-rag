#!/bin/bash
#
#
#SBATCH --job-name=video_dl
#SBATCH --partition=normal          # ← from sinfo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2           # yt‑dlp & ffmpeg parallel
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/video_dl_%j.out

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load FFmpeg/5.1.2-GCC-11.3.0

# activate your virtual env
python scripts/download.py
