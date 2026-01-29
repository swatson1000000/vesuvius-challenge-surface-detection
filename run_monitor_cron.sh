#!/bin/bash
# Wrapper script for running monitor_training.py from cron
# Properly sources conda environment and exports required variables

# Set up conda
eval "$(conda shell.bash hook)"
conda activate phi4

# Export Gmail password for email sending
export GMAIL_APP_PASSWORD="vlgnlhkifapctzjg"

# Go to project directory
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection

# Run monitor script with proper Python path, reading log from the log directory
/home/swatson/miniconda3/envs/phi4/bin/python3 monitor_training.py --log-file log/train_v9_progressive.log >> log/monitor_cron.log 2>&1
