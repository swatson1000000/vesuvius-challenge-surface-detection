# Comprehensive Training Guide
**Vesuvius Challenge Surface Detection**

Last Updated: January 11, 2026

---

## Quick Start

```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin
nohup ./quick_train_all.sh > /dev/null 2>&1 &

# Monitor progress
tail -f ../log/train_all_folds_*.log
```

**Current Status:**
- ✅ Fold 0: Completed (Val Dice: 0.5807, 10.83 hours)
- Training all 5 folds automatically with early stopping → next fold

---

## Table of Contents

1. [Critical Bug Fix](#critical-bug-fix)
2. [Recent Updates](#recent-updates)
3. [Training Configuration](#training-configuration)
4. [Understanding K-Fold Cross-Validation](#understanding-k-fold-cross-validation)
5. [Running Training](#running-training)
6. [Monitoring Progress](#monitoring-progress)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

---

## Critical Bug Fix

### Problem Identified (Jan 10, 2026)

Model was predicting nearly uniform values (~0.60) everywhere:
- **Training plateau at 0.072** - couldn't improve beyond this
- **99.88% false positives** when thresholded at 0.5
- **Prediction variance: 0.000008** (should be >> 0 for good segmentation)

### Root Cause

Loss function allowed trivial solution:
- Labels have 3 classes: 0 (background 24.6%), 1 (surface 4.3%), 2 (unknown 71.1%)
- Training correctly converts to binary: `label > 0` → 75.4% foreground
- **Problem:** Outputting uniform ~0.75 everywhere minimizes Dice + Focal loss
- No spatial structure penalty allowed model to collapse to constant output

### Fixes Applied

#### 1. Variance Regularization Loss (NEW)
```python
def variance_regularization(pred):
    """Penalize low variance predictions"""
    variance = pred.var(dim=(2, 3, 4)).mean()
    min_variance = 0.01
    var_loss = torch.clamp(min_variance - variance, min=0.0) / min_variance
    return var_loss
```

**Configuration:**
```yaml
loss_weights:
  dice_weight: 0.4      # Down from 0.5
  focal_weight: 0.2     # Down from 0.5  
  variance_weight: 0.2  # NEW - prevents uniform predictions
```

#### 2. Increased Model Capacity
- `initial_filters`: 16 → 32
- Parameters: 5.6M → 22.6M (4x increase)
- Same depth (4) to maintain training speed

#### 3. Rebalanced Loss Weights
See configuration above - total still sums to < 1.0 for numerical stability

### Expected Behavior After Fix

**Training Metrics:**
- Initial variance loss should be high (>0.5)
- Variance should stabilize around 0.01-0.05
- Loss should continue improving (not plateau at 0.07)
- Monitor: `grep "variance" log/*.log`

**Predictions:**
- Variance > 0.01 (100x improvement)
- Probability distribution bimodal (peaks near 0 and 1)
- Visual predictions show clear spatial structure

### Success Criteria

✅ Prediction variance > 0.01  
✅ Training loss continues below 0.07  
✅ Probabilities distributed across [0,1] range  
✅ Visual predictions show spatial structure  
✅ No plateau at 0.07

---

## Recent Updates

### Automatic Sequential Fold Training (Jan 11, 2026)

Training now automatically continues through all folds when early stopping triggers.

**Before:**
```bash
python train.py --fold 0  # Stop after fold 0
python train.py --fold 1  # Manually run fold 1
# etc...
```

**After:**
```bash
python train.py --fold -1  # Runs all folds automatically
```

**Features:**
- Early stopping → automatically moves to next fold
- `--continue-on-error` flag to keep going if one fold fails
- Single log file with all fold results
- Summary report at the end

### SWA Start Epoch: 50 → 40

**Rationale:**
- Fold 0 hit early stopping at epoch 49
- SWA was set to start at 50, so never ran
- Now starts at epoch 40, ensuring it gets used

**Configuration:**
```yaml
swa_enabled: true
swa_start_epoch: 40  # Changed from 50
swa_lr: 0.00005
swa_update_freq: 5
```

---

## Training Configuration

### Model Architecture

```yaml
in_channels: 1
out_channels: 1
initial_filters: 32    # Increased from 16
depth: 4               # 4-level U-Net
```

**Model size:** 22.6M parameters

### Training Parameters

```yaml
batch_size: 2
num_epochs: 300
learning_rate: 0.0001  # Conservative starting point
weight_decay: 0.01
warmup_epochs: 5       # Gradual LR warmup
```

### Data Configuration

```yaml
patch_size: [128, 128, 128]  # 3D patches
num_workers: 4
n_folds: 5                    # K-fold cross-validation
```

**Dataset:**
- 786 image-label pairs
- Each fold: 628 train, 158 validation (20%)

### Loss Configuration

```yaml
loss_weights:
  dice_weight: 0.4        # Overlap measure
  focal_weight: 0.2       # Handle class imbalance
  variance_weight: 0.2    # Prevent uniform predictions
  boundary_weight: 0.0    # Disabled (can enable later)
  cldice_weight: 0.0      # Disabled
  connectivity_weight: 0.0 # Disabled

# Focal loss parameters
focal_gamma: 2.0
focal_alpha: 0.25

# Class weights (corrected for imbalance)
use_class_weights: true
background_weight: 2.94  # 1 / (1 - 0.66)
foreground_weight: 1.52  # 1 / 0.66
```

### Learning Rate Schedule

**Cosine Annealing with Warm Restarts:**

```yaml
scheduler_type: "cosine_warm_restarts"
T_0: 10          # Initial cycle: 10 epochs
T_mult: 2        # Each cycle doubles (10, 20, 40, 80...)
eta_min: 0.000001 # Minimum LR
grad_clip_norm: 0.5  # Prevent gradient explosion
```

**Pattern:**
```
Epochs 0-9:   LR drops 0.0001 → 0.000001
Epoch 10:     LR jumps back to 0.0001 (warm restart)
Epochs 10-29: LR drops over 20 epochs
Epoch 30:     Another restart
...
```

**Benefits:**
- Periodic LR increases escape local minima
- Each cycle: explore (high LR) → exploit (low LR)
- Better than fixed LR or simple decay

### Stochastic Weight Averaging (SWA)

```yaml
swa_enabled: true
swa_start_epoch: 40   # Start averaging at epoch 40
swa_lr: 0.00005       # SWA learning rate
swa_update_freq: 5    # Update every 5 epochs
```

**How it works:**
- After epoch 40, maintains running average of model weights
- Averaging smooths out noise → better generalization
- Final SWA model often outperforms best single checkpoint

### Weight Noise Injection

```yaml
noise_enabled: true
noise_start_epoch: 30      # Add noise after epoch 30
noise_frequency: 10        # Every 10 epochs
noise_std: 0.001          # 0.1% of weights
noise_decay: 0.9          # Decay noise over time
noise_target_layers: ['decoders', 'output']
```

**Purpose:** Help escape plateaus by perturbing weights

### Early Stopping

```yaml
early_stopping_patience: 50
```

Stops training if no improvement for 50 consecutive epochs, then moves to next fold.

---

## Understanding K-Fold Cross-Validation

### What is a Fold?

A **fold** is a data split used to evaluate model performance more reliably than a single train/test split.

### Your 5-Fold Setup

786 images divided into 5 groups:

```
┌─────┬─────┬─────┬─────┬─────┐
│  A  │  B  │  C  │  D  │  E  │
│ 157 │ 157 │ 157 │ 157 │ 158 │
└─────┴─────┴─────┴─────┴─────┘

Fold 0: Train [B,C,D,E], Validate [A]
Fold 1: Train [A,C,D,E], Validate [B]
Fold 2: Train [A,B,D,E], Validate [C]
Fold 3: Train [A,B,C,E], Validate [D]
Fold 4: Train [A,B,C,D], Validate [E]
```

### Why Use Folds?

**1. More Reliable Performance**
- Single split: "Dice = 0.58" (could be lucky)
- 5-fold: "Dice = 0.58 ± 0.02" (statistically valid)

**2. Use All Data**
Every sample is validated exactly once

**3. Detect Overfitting**
Large variance between folds indicates poor generalization

**4. Ensemble Predictions**
Average predictions from all 5 models for better results:
```python
final = (pred_fold0 + pred_fold1 + ... + pred_fold4) / 5
```

### Changing Number of Folds

Edit `config.yaml`:
```yaml
n_folds: 3  # Faster: 3 × 11 hours = 33 hours
n_folds: 5  # Balanced: 5 × 11 hours = 55 hours (default)
n_folds: 10 # Slower but more reliable: 10 × 11 hours = 110 hours
```

---

## Running Training

### Method 1: Run All Folds (Recommended)

```bash
cd bin
nohup ./quick_train_all.sh > /dev/null 2>&1 &
```

**Estimated time:** ~55 hours for all 5 folds

### Method 2: Python Script Directly

```bash
cd bin
python -u train.py --config config.yaml --fold -1 --data_dir .. --continue-on-error
```

**Flags:**
- `--fold -1`: Run all folds
- `--fold N`: Run specific fold (0-4)
- `--continue-on-error`: Continue if one fold fails

### Method 3: Individual Folds

```bash
# Run specific fold
python train.py --fold 1 --data_dir ..
```

### Method 4: Parallel (Multiple GPUs)

```bash
cd bin
./train_parallel_folds.sh  # Requires 4+ GPUs
```

---

## Monitoring Progress

### Watch Live Training

```bash
tail -f ../log/train_all_folds_*.log
```

### Filter Specific Information

```bash
# See fold transitions
tail -f ../log/*.log | grep "FOLD\|complete"

# See validation results
tail -f ../log/*.log | grep "Val - Loss"

# Check variance (important for bug fix)
tail -f ../log/*.log | grep "variance"
```

### Check GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Check Running Processes

```bash
ps aux | grep train.py | grep -v grep
```

### View All Logs

```bash
ls -lht log/
```

---

## Advanced Features

### Plateau Escape Strategies

The model uses multiple strategies to avoid getting stuck:

#### 1. Cosine Warm Restarts
- Periodic LR increases help escape local minima
- Configured with `T_0=10`, `T_mult=2`

#### 2. Stochastic Weight Averaging (SWA)
- Averages model weights after epoch 40
- Smooths optimization trajectory
- Often yields best final model

#### 3. Weight Noise Injection
- Adds small noise to weights every 10 epochs after epoch 30
- Helps escape plateaus
- Noise decays over time (0.9 decay factor)

#### 4. Variance Regularization
- Critical fix to prevent uniform predictions
- Forces model to make discriminative predictions
- Weight: 0.2 (20% of total loss)

### Post-Processing

```yaml
postprocess:
  threshold: 0.5
  min_component_size: 100    # Remove small false positives
  max_hole_size: 50          # Fill small holes
  morphology_kernel_size: 2
  separate_instances: false
```

Applied during validation to clean up predictions.

---

## Troubleshooting

### Training Plateau (Loss not improving)

**Check variance loss:**
```bash
grep "variance" log/*.log
```

**Expected:**
- Variance loss high initially (>0.5)
- Should decrease to ~0.01-0.05
- If stays high: model struggling to find structure
- If stays near 0: model still predicting uniformly (bug not fixed)

**Solutions:**
- Increase `variance_weight` to 0.3-0.4
- Enable boundary loss: `boundary_weight: 0.1`
- Check if data augmentation is working

### Out of Memory

**Reduce batch size:**
```yaml
batch_size: 1  # Down from 2
```

**Or reduce patch size:**
```yaml
patch_size: [96, 96, 96]  # Down from 128
```

### Training Too Slow

**Reduce number of folds:**
```yaml
n_folds: 3  # Down from 5
```

**Or reduce epochs:**
```yaml
num_epochs: 100  # Down from 300
```

### Early Stopping Too Aggressive

**Increase patience:**
```yaml
early_stopping_patience: 75  # Up from 50
```

### One Fold Fails

If using `--continue-on-error`, training continues to next fold. Check error in log:
```bash
grep "ERROR\|Failed" log/*.log
```

### Predictions Look Uniform

**This was the bug!** Check that:
1. Variance weight is enabled (0.2)
2. Model has 32 initial filters (not 16)
3. Prediction variance > 0.01 in logs

### Want to Stop Training

```bash
# Find process
ps aux | grep train.py

# Kill gracefully
pkill -f train.py

# Or force kill
kill -9 <PID>
```

---

## Expected Results

### Per-Fold Performance (Based on Fold 0)

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Val Loss | 0.2142 | 0.0851 | -60.3% |
| Dice Score | 0.4736 | 0.5807 | +22.6% |
| Epochs | - | 50 (early stopped) | - |
| Time | - | 10.83 hours | - |

### Target Metrics

- **Dice > 0.55** on all folds for good performance
- **Fold variance < 0.03** for stable model
- **Prediction variance > 0.01** (confirms bug fix)

### After All Folds Complete

**Calculate ensemble:**
```python
avg_dice = (fold0 + fold1 + fold2 + fold3 + fold4) / 5
std_dice = std([fold0, fold1, fold2, fold3, fold4])
```

**Create ensemble predictions:**
```python
# Average predictions from all 5 models
final_pred = sum(model_outputs) / 5
```

Ensemble typically performs 2-5% better than single best fold.

---

## File Structure

```
kaggle/vesuvius-challenge-surface-detection/
├── bin/
│   ├── train.py                    # Main training script
│   ├── config.yaml                 # Configuration
│   ├── quick_train_all.sh         # Quick start script
│   ├── train_all_folds.sh         # Sequential fold training
│   ├── train_single_fold.sh       # Single fold script
│   └── checkpoints/
│       ├── fold_0/                # Fold 0 checkpoints
│       ├── fold_1/                # Fold 1 checkpoints
│       └── ...
├── log/
│   └── train_all_folds_*.log      # Training logs
├── train_images/                   # Training images (786)
├── train_labels/                   # Training labels (786)
├── TRAINING_GUIDE.md              # This file
├── CLAUDE.md                       # Claude AI instructions
├── QUICKSTART.txt                  # Quick reference
└── README.md                       # Project overview
```

---

## Quick Reference Commands

```bash
# Start training all folds
cd bin && nohup ./quick_train_all.sh > /dev/null 2>&1 &

# Monitor progress
tail -f ../log/train_all_folds_*.log

# Check fold results
grep "complete" ../log/train_all_folds_*.log

# Check GPU
nvidia-smi

# Kill training
pkill -f train.py

# View checkpoints
ls -lh checkpoints/fold_*/
```

---

## References

- **CLAUDE.md**: Instructions for running training with nohup
- **QUICKSTART.txt**: One-page quick reference
- **TRAINING_FOLDS_GUIDE.md**: Detailed fold training guide
- **config.yaml**: All training parameters

---

## Version History

- **Jan 11, 2026**: Added automatic fold progression, SWA epoch 40
- **Jan 10, 2026**: Critical variance regularization fix, model capacity increase
- **Jan 9, 2026**: Initial implementation with plateau escape strategies
