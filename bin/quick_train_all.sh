#!/bin/bash
# Quick start script to run remaining folds (1-4)
# This is the simplest way to continue training after fold 0

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4

# Navigate to bin directory
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin

echo "========================================"
echo "Starting training for ALL FOLDS"
echo "This will run folds 0-4 sequentially"
echo "Started at: $(date)"
echo "========================================"
echo ""
echo "Note: If fold 0 checkpoints exist, it may skip or overwrite"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Run training with unbuffered output
python -u train.py \
    --config config.yaml \
    --fold -1 \
    --data_dir .. \
    --continue-on-error \
    2>&1 | tee ../log/train_all_folds_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "Training complete at: $(date)"
echo "========================================"
