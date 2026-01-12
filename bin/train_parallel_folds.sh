#!/bin/bash
# Train multiple folds in parallel (requires multiple GPUs)
# Usage: ./train_parallel_folds.sh
# Runs folds 1-4 in parallel, each on a different GPU

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi4

# Navigate to bin directory
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin

echo "========================================"
echo "Starting parallel training for folds 1-4"
echo "Started at: $(date)"
echo "========================================"

# Check number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

if [ $NUM_GPUS -lt 4 ]; then
    echo "Warning: Only $NUM_GPUS GPU(s) available. Folds will share GPUs."
fi

# Array to store PIDs
declare -a PIDS

# Start each fold on a different GPU
for fold in {1..4}; do
    GPU_ID=$((fold % NUM_GPUS))
    LOG_FILE="../log/train_fold${fold}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting Fold $fold on GPU $GPU_ID (log: $LOG_FILE)"
    
    # Run in background with specific GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -u train.py \
        --config config.yaml \
        --fold $fold \
        --data_dir .. > "$LOG_FILE" 2>&1 &
    
    PIDS[$fold]=$!
    echo "  PID: ${PIDS[$fold]}"
    
    # Small delay to avoid race conditions
    sleep 2
done

echo ""
echo "========================================"
echo "All folds started!"
echo "Monitoring PIDs: ${PIDS[@]}"
echo "========================================"
echo ""
echo "Monitor individual folds:"
for fold in {1..4}; do
    echo "  Fold $fold: tail -f ../log/train_fold${fold}_*.log | grep -E 'Epoch|Loss|Val'"
done
echo ""
echo "Check running processes:"
echo "  ps -p ${PIDS[1]},${PIDS[2]},${PIDS[3]},${PIDS[4]}"
echo ""
echo "To stop all training:"
echo "  kill ${PIDS[@]}"
echo ""

# Wait for all processes to complete
echo "Waiting for all folds to complete..."
for fold in {1..4}; do
    wait ${PIDS[$fold]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Fold $fold completed successfully"
    else
        echo "✗ Fold $fold failed with exit code $EXIT_CODE"
    fi
done

echo ""
echo "========================================"
echo "All folds completed at: $(date)"
echo "========================================"
