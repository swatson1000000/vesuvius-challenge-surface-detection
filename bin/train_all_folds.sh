#!/bin/bash
# Train all 5 folds sequentially
# Usage: ./train_all_folds.sh [start_fold] [end_fold]
# Example: ./train_all_folds.sh 1 4  (runs folds 1-4)

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4

# Default to folds 1-4 if no arguments provided
START_FOLD=${1:-1}
END_FOLD=${2:-4}

# Navigate to bin directory
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin

echo "========================================"
echo "Training folds $START_FOLD to $END_FOLD"
echo "Started at: $(date)"
echo "========================================"

# Run each fold
for fold in $(seq $START_FOLD $END_FOLD); do
    echo ""
    echo "========================================"
    echo "Starting Fold $fold at $(date)"
    echo "========================================"
    
    # Create log filename
    LOG_FILE="../log/train_fold${fold}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run training with unbuffered output
    python -u train.py --config config.yaml --fold $fold --data_dir .. 2>&1 | tee "$LOG_FILE"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Fold $fold completed successfully"
    else
        echo "✗ Fold $fold failed! Check $LOG_FILE"
        exit 1
    fi
    
    echo "Fold $fold finished at $(date)"
    echo ""
done

echo "========================================"
echo "All folds completed at: $(date)"
echo "========================================"
