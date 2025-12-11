#!/bin/bash
#SBATCH -J vllm_shard           # Job Name
#SBATCH -o logs/vllm_%A_%a.out  # Output log
#SBATCH -e logs/vllm_%A_%a.err  # Error log
#SBATCH -p gh-dev               # Queue (gh-dev for testing, gh for production)
#SBATCH -N 1                    # Nodes per task
#SBATCH -n 1                    # Tasks per node
#SBATCH --time=02:00:00         # Time limit
#SBATCH --array=0-9             # Array range (0-9 = 10 chunks)

# --- Config ---
NUM_CHUNKS=10                   # Must match array size
DATA_DIR="/scratch/01234/user/med_data"
OUTPUT_FILE="captions.jsonl"
HF_CACHE="/scratch/01234/user/.cache"

# Load Modules / Environment
module load python3
source activate my_vllm_env

# Set Cache to Scratch (Critical on TACC)
export HF_HOME="$HF_CACHE"

echo "Launching Shard $SLURM_ARRAY_TASK_ID of $NUM_CHUNKS"

python vllm_inference.py \
    --data_dir "$DATA_DIR" \
    --output_file "$OUTPUT_FILE" \
    --num_chunks $NUM_CHUNKS \
    --chunk_idx $SLURM_ARRAY_TASK_ID \
    --loader_batch_size 20

echo "Shard $SLURM_ARRAY_TASK_ID Complete"