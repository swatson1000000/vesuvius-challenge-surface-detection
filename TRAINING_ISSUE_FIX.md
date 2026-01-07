# Training Issue Analysis & Fix

## Problem Identified

Training was stuck with **no improvement** from epoch 0 to epoch 49:
- Training Loss: ~0.39-0.41 (flat)
- Validation Loss: ~0.40-0.42 (flat)

## Root Causes

### 1. **Severe Class Imbalance**
- **Dataset composition**: 66% foreground, 34% background
- **Variation**: 27%-92% foreground across samples
- Model can achieve ~0.40 loss by just predicting mostly foreground everywhere

### 2. **Learning Rate Too Low**
- **Current**: 0.001
- At this rate with the current loss landscape, gradients are too small to escape the plateau

### 3. **Loss Function Weights Not Optimal**
- Focal loss weight too low (0.2) - should be higher for imbalanced data
- Dice loss weight too high (0.4) - doesn't help with imbalance

### 4. **No Class Weighting**
- Equal treatment of foreground and background despite 2:1 ratio
- Majority class (foreground) dominates the learning signal

## Fixes Applied

### 1. **Increased Learning Rate**
```yaml
learning_rate: 0.01  # Was 0.001 (10x increase)
warmup_epochs: 5     # Added warmup for stability
```

### 2. **Adjusted Loss Weights**
```yaml
loss_weights:
  dice_weight: 0.3     # Reduced from 0.4
  focal_weight: 0.4    # Increased from 0.2 (now primary loss)
  boundary_weight: 0.15
  cldice_weight: 0.10
  connectivity_weight: 0.05
```

### 3. **Added Focal Loss Parameters**
```yaml
focal_gamma: 2.0      # Focus on hard-to-classify examples
focal_alpha: 0.34     # Weight minority class (background)
```

### 4. **Added Class Weights**
```yaml
use_class_weights: true
background_weight: 1.94  # Upweight minority class
foreground_weight: 0.52  # Downweight majority class
```

Weights calculated as:
- `background_weight = 1 / (1 - 0.66) = 2.94` (normalized)
- `foreground_weight = 1 / 0.66 = 1.52` (normalized)

## Expected Results

With these fixes, you should see:

### Phase 1 (Epochs 1-10)
- Loss should **decrease noticeably** from ~0.40 to ~0.30
- Model starts learning meaningful features instead of predicting mean

### Phase 2 (Epochs 10-50)
- Training loss: 0.25-0.30
- Validation loss: 0.28-0.35  
- Clear separation between train/val indicates learning

### Phase 3 (Epochs 50-100)
- Training loss: 0.20-0.25
- Validation loss: 0.25-0.30
- Model converging to good solution

### Final Target
- Validation loss: <0.25
- Competition metrics: 0.70-0.75 (baseline without ensemble)

## How to Restart Training

### Option 1: Train from Scratch
```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
# Backup old checkpoints
mv bin/checkpoints bin/checkpoints_old_$(date +%Y%m%d)
# Start new training
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && cd bin && python train.py --fold 0 --data_dir .." > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Option 2: Continue from Last Checkpoint (Not Recommended)
The model has already converged to a bad local minimum. Better to start fresh.

## Monitoring Training

Check logs to verify improvement:
```bash
# Watch live
tail -f log/train_*.log | grep "Loss:"

# Check last 50 epochs
grep "Epoch.*Train.*Loss\|Epoch.*Val.*Loss" log/train_*.log | tail -100
```

You should now see:
- ✓ Decreasing loss over epochs
- ✓ Learning rate adjustments by scheduler
- ✓ Training loss lower than validation loss (healthy overfitting)

## Additional Improvements (Future)

If training is still not improving well:

1. **Data Augmentation**: Increase augmentation strength to prevent overfitting
2. **Patch Sampling**: Sample patches with balanced foreground/background
3. **Hard Negative Mining**: Focus on difficult background regions
4. **Reduce Model Capacity**: Smaller model (initial_filters: 16 instead of 32)
5. **Different Architecture**: Try proven models (ResNet + UNet hybrid)

## Timeline Estimate

With current config:
- **Epochs 0-10**: ~2 hours (should show clear improvement)
- **Epochs 10-50**: ~10 hours (main learning phase)  
- **Epochs 50-100**: ~10 hours (refinement)
- **Total for 100 epochs**: ~22 hours

Stop early if validation loss plateaus for 50 epochs (early stopping configured).

## Success Criteria

Training is working if you see:
- ✓ Training loss decreases to <0.30 within 10 epochs
- ✓ Validation loss follows training loss with small gap (<0.05)
- ✓ Validation loss improves by at least 0.10 over 50 epochs
- ✓ Model learns to distinguish foreground/background (not just predicting mean)

## Diagnostic Commands

Check if new training is improving:
```bash
# Compare first 10 epochs to previous run
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
echo "=== OLD TRAINING ===" && grep "Epoch [0-9] " bin/training_fold0.log | head -20
echo "=== NEW TRAINING ===" && grep "Epoch [0-9] " log/train_*.log | tail -1 | head -20
```

Expected: New training should show loss decreasing more rapidly.

## Next Steps

1. **Start fresh training** with updated config
2. **Monitor first 10 epochs** to confirm improvement
3. **Let it run for 50-100 epochs** for proper convergence
4. **Evaluate on validation set** with competition metrics
5. **Tune hyperparameters** based on results
