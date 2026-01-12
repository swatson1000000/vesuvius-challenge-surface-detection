# Claude AI Instructions

## Critical Training Fix (Jan 2026)

**IMPORTANT**: Previous training had a fundamental bug causing uniform predictions. See `TRAINING_FIX.md` for details.

**Fixed issues**:
1. Added variance regularization loss (weight=0.2) to prevent uniform outputs
2. Increased model capacity (initial_filters: 16→32, ~22M parameters)
3. Rebalanced loss weights (dice=0.4, focal=0.2, variance=0.2)

**Monitor these during training**:
- Variance loss component (should decrease from >0.5 to <0.1)
- Prediction variance in logs (should be >0.01, not 0.00001)
- Training should NOT plateau at 0.072 like before

## Environment Setup

**ALWAYS** use the `phi4` conda environment for all training and inference operations.

## Running Training

**ALWAYS** run training with nohup in the background, logging to the `log/` directory with timestamp:

```bash
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin && python train.py --config config.yaml --fold 0 --data_dir .." > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Training Parameters

- `--config`: Path to config file (default: `config.yaml`)
- `--fold`: Fold number for cross-validation (0-4)
- `--data_dir`: Path to data directory (parent directory with `train_images/`, `train_labels/`)

### Monitoring Training

Check the latest training log:
```bash
tail -f log/train_*.log
```

View training progress:
```bash
ls -lth log/train_*.log | head -5
```

## Running Inference

**ALWAYS** run inference with nohup in the background, logging to the `log/` directory with timestamp:

```bash
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin && python inference.py --checkpoint checkpoints/fold_0/checkpoint_epoch_40.pth --input ../test_images/ --output ../predictions/ --device cuda" > log/inference_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Inference Parameters

- `--checkpoint`: Path to model checkpoint (e.g., `checkpoints/fold_0/checkpoint_epoch_40.pth`)
- `--input`: Path to input directory or file
- `--output`: Path to output directory
- `--device`: Device to use (`cuda` or `cpu`)
- `--patch_size`: Patch size for inference (default: `128 128 128`)
- `--overlap`: Overlap between patches (default: `32`)
- `--threshold`: Probability threshold (default: `0.5`)
- `--min_component_size`: Minimum component size in voxels (default: `100`)
- `--max_hole_size`: Maximum hole size to fill in voxels (default: `50`)
- `--kernel_size`: Morphological kernel size (default: `2`)
- `--separate_instances`: Apply instance separation (flag)

### Monitoring Inference

Check the latest inference log:
```bash
tail -f log/inference_*.log
```

View inference progress:
```bash
ls -lth log/inference_*.log | head -5
```

## Output Files

### Training Outputs
- **Checkpoints**: `bin/checkpoints/fold_0/checkpoint_epoch_N.pth`
- **Logs**: `log/train_YYYYMMDD_HHMMSS.log`
- **Training log file**: `bin/training_fold0.log`

### Inference Outputs
- **Binary predictions** (for submission): `predictions/FILENAME.tif` (values: 0, 1)
- **Visualization predictions**: `predictions/FILENAME_visualization.tif` (values: 0, 255)
- **Logs**: `log/inference_YYYYMMDD_HHMMSS.log`

### Prediction Format
- **Binary version** (`FILENAME.tif`): 
  - Values: 0 (background), 1 (foreground)
  - Use for competition submission
  - Appears black in viewers (1 is very dark)
  
- **Visualization version** (`FILENAME_visualization.tif`):
  - Values: 0 (black), 255 (white)
  - Use for viewing and analysis
  - Displays properly in image viewers

## Important Notes

### Timing Information
Both scripts now log:
- Start time: `YYYY-MM-DD HH:MM:SS`
- End time: `YYYY-MM-DD HH:MM:SS`
- Total time: `X.XX seconds (X.XX hours)`

### Incremental Logging
Both scripts output logs incrementally in real-time with:
- Automatic stdout flushing
- Line buffering enabled
- Progress updates at each stage

### Process Management

Check if process is running:
```bash
ps aux | grep "train.py\|inference.py" | grep -v grep
```

Kill a running process:
```bash
pkill -f "train.py"
# or
pkill -f "inference.py"
```

View all background jobs:
```bash
jobs -l
```

### GPU Usage

Check GPU usage:
```bash
nvidia-smi
```

Monitor GPU usage continuously:
```bash
watch -n 1 nvidia-smi
```

## Quick Reference Commands

### Start Training (Fold 0)
```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && cd bin && python train.py --fold 0 --data_dir .." > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Start Inference (Latest Checkpoint)
```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && cd bin && python inference.py --checkpoint checkpoints/fold_0/checkpoint_epoch_40.pth --input ../test_images/ --output ../predictions/" > log/inference_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Check Latest Log
```bash
tail -f log/*.log
```

### List All Logs
```bash
ls -lth log/
```

## Troubleshooting

### Conda Activation Issues
If conda activation fails, ensure the shell hook is properly initialized:
```bash
eval "$(conda shell.bash hook)"
conda activate phi4
```

### Log Not Updating
Logs update incrementally. If not seeing updates:
1. Check if process is still running: `ps aux | grep python`
2. Verify log file is being written: `ls -lh log/`
3. Try tailing with more context: `tail -100f log/latest.log`

### Out of Memory
If training/inference runs out of GPU memory:
- Reduce batch size in `config.yaml`
- Reduce patch size: `--patch_size 96 96 96`
- Use CPU: `--device cpu` (slower but more memory)

### Process Not Running in Background
Ensure proper nohup syntax:
```bash
nohup bash -c "COMMANDS" > log/file.log 2>&1 &
```

The `&` at the end runs in background.

## Directory Structure

```
kaggle/vesuvius-challenge-surface-detection/
├── bin/
│   ├── train.py              # Training script
│   ├── inference.py          # Inference script
│   ├── config.yaml           # Training configuration
│   └── checkpoints/          # Model checkpoints
│       └── fold_0/
├── log/                      # All logs with timestamps
│   ├── train_*.log
│   └── inference_*.log
├── predictions/              # Inference outputs
│   ├── *.tif                 # Binary (0,1) for submission
│   └── *_visualization.tif   # Scaled (0,255) for viewing
├── train_images/             # Training images
├── train_labels/             # Training labels
└── test_images/              # Test images
```

## Best Practices

1. **Always use phi4 environment** - Contains all required dependencies
2. **Always run with nohup** - Prevents termination on logout
3. **Always log with timestamp** - Easier to track runs
4. **Always log to log/ directory** - Centralized logging
5. **Monitor logs initially** - Verify process started correctly
6. **Check disk space** - Models and logs can be large
7. **Use visualization files for viewing** - Binary files appear black

## Competition Submission

For final submission, create a ZIP containing only the binary prediction files:
```bash
cd predictions
zip submission.zip *.tif  # Exclude *_visualization.tif
# Or explicitly:
zip submission.zip $(ls *.tif | grep -v visualization)
```

Submit `submission.zip` to Kaggle.
