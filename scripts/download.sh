#!/bin/bash
#
#
#SBATCH --job-name=video_dl
#SBATCH --output=logs/video_dl_%j.out
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# ── module environment — use the names shown in the Octopus doc ──
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load FFmpeg/5.1.2-GCC-11.3.0

# ── activate project venv ─────────────────────────────────────────
source ~/videoqa/.venv/bin/activate          # adapt to your path

# ── run the downloader ────────────────────────────────────────────
python scripts/download.py
