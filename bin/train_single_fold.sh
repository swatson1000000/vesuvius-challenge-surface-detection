#!/bin/bash
# Train a single fold in background
# Usage: ./train_single_fold.sh <fold_number>
# Example: ./train_single_fold.sh 1

if [ $# -eq 0 ]; then
    echo "Error: Please provide fold number"
    echo "Usage: $0 <fold_number>"
    echo "Example: $0 1"
    exit 1
fi

FOLD=$1

# Validate fold number
if [ $FOLD -lt 0 ] || [ $FOLD -gt 4 ]; then
    echo "Error: Fold number must be between 0 and 4"
    exit 1
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4

# Navigate to bin directory
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin

# Create log filename
LOG_FILE="../log/train_fold${FOLD}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training for Fold $FOLD"
echo "Log file: $LOG_FILE"
echo "Running in background with nohup..."

# Run in background with nohup
nohup python -u train.py --config config.yaml --fold $FOLD --data_dir .. > "$LOG_FILE" 2>&1 &

# Get process ID
PID=$!
echo "Training started with PID: $PID"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Check process with: ps -p $PID"
echo ""
echo "To stop training: kill $PID"
