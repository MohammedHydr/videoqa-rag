#!/bin/bash
#SBATCH --job-name=video-ui
#SBATCH --output=logs/ui_%j.out
#SBATCH --error=logs/ui_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=interactive

source ~/.bashrc
module purge
module load python/3.10
source ~/videoqa-rag/.venv/bin/activate

# Run streamlit UI
streamlit run app/streamlit_app.py --server.port=8501
