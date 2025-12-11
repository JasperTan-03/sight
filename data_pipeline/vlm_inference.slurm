#!/bin/bash
#SBATCH -J vllm_shard           # Job Name
#SBATCH -o logs/vllm_%A_%a.out  # Output log
#SBATCH -e logs/vllm_%A_%a.err  # Error log
#SBATCH -p gh # Queue (gh-dev for testing, gh for production)
#SBATCH -N 2                   # Something else
#SBATCH -n 2                    # Total tasks
#SBATCH --time=02:00:00         # Time limit
#SBATCH --array=0-1             # Array range (0-9 = 10 chunks)
#SBATCH -A OTH25007

# --- Config ---
NUM_CHUNKS=2             # Must match array size
DATA_DIR="/work/09889/troydutton/datasets/data/renders/"
OUTPUT_FILE="captions.jsonl"
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

echo "Launching Shard $SLURM_ARRAY_TASK_ID of $NUM_CHUNKS"

python generate_captions.py \
    --data_dir "$DATA_DIR" \
    --output_file "$OUTPUT_FILE" \
    --num_chunks $NUM_CHUNKS \
    --chunk_idx $SLURM_ARRAY_TASK_ID \
    --loader_batch_size 20

echo "Shard $SLURM_ARRAY_TASK_ID Complete"
