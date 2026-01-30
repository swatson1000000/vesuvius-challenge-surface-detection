#!/bin/bash
# Wrapper script for running monitor_training.py from cron
# Properly sources conda environment and exports required variables

# Set up conda
eval "$(conda shell.bash hook)"
conda activate phi4

# Load Gmail password from secure environment file
if [ -f .monitor_env ]; then
    source .monitor_env
else
    echo "⚠️ Warning: .monitor_env not found. Email sending will be disabled."
fi

# Go to project directory
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection

# Run monitor script (auto-detects latest training log)
# Searches in priority order: train_value1_*.log, train_3class_*.log, train_v9_progressive.log, train_*.log
/home/swatson/miniconda3/envs/phi4/bin/python3 monitor_training.py >> log/monitor_cron.log 2>&1
