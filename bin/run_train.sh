#!/bin/bash
# Wrapper script for training with proper logging

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4

# Run with unbuffered output
# Default: Run all folds (--fold -1)
# To run specific fold: add --fold N
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin
exec python -u train.py --config config.yaml --fold -1 --data_dir .. --continue-on-error 2>&1 | tee ../log/train_all_folds_$(date +%Y%m%d_%H%M%S).log
