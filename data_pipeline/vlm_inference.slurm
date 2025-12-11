#!/bin/bash
#SBATCH -J vllm_caption         # Job Name
#SBATCH -o logs/vllm_%j.out     # Output log
#SBATCH -e logs/vllm_%j.err     # Error log
#SBATCH -p gh                   # Queue (gh-dev for testing, gh for production)
#SBATCH -N 2                    # Number of nodes
#SBATCH --ntasks-per-node=1     # One task per node (1 GPU per task)
#SBATCH --time=02:00:00         # Time limit
#SBATCH -A OTH25007

# --- Config ---
DATA_DIR="/work/09889/troydutton/datasets/data"
OUTPUT_DIR="/work/09889/troydutton/datasets/data/checkpoints"
HF_CACHE='/home1/09889/troydutton/.cache'

# Source
source ~/.bashrc

# Load modules
module load gcc/14.2.0 cuda/12.4 nvidia_math/12.4 cmake/3.29.5

# Activate environment
conda activate sight

cd /home1/09889/troydutton/work/sight/data_pipeline

# Set Cache to Scratch (Critical on TACC)
export HF_HOME="$HF_CACHE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Launching $SLURM_NTASKS tasks across $SLURM_NNODES nodes"

# Use srun to distribute work - SLURM_PROCID and SLURM_NTASKS are set automatically
srun python generate_captions.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --save_every 10

echo "Job Complete"
