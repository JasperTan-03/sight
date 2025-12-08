#!/bin/bash
# Setup script for TACC Lonestar6
# This script sets up the conda environment with proper modules loaded

# Load necessary modules for TACC
# CUDA module (choose version based on your needs: 11.4, 12.0, or 12.8)
module load cuda/12.0

# GCC compiler (optional, but recommended for building packages)
module load gcc

# Initialize conda/mamba (if not already initialized)
# Uncomment the line below if needed:
# source $WORK/miniconda3/etc/profile.d/conda.sh

# Use mamba instead of conda (faster and avoids solver issues)
# Create environment using mamba
mamba env create -f environment.yml

# Activate the environment
conda activate sight

# Verify installation
python --version
which python

echo "Setup complete! Activate the environment with: conda activate sight"

