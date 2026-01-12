# Training Multiple Folds - Quick Reference

## Current Status
- ✅ **Fold 0**: Completed (50 epochs, 10.83 hours, Val Dice: 0.5807)
- ⏳ **Folds 1-4**: Pending

## ⚡ NEW: Automatic Sequential Fold Training

**The training script now automatically runs all folds sequentially!** When one fold completes (or early stopping triggers), it automatically moves to the next fold.

### Run All Remaining Folds (Recommended)

```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin

# Option 1: Run all 5 folds (including fold 0 again if desired)
python -u train.py --config config.yaml --fold -1 --data_dir .. --continue-on-error

# Option 2: Use the wrapper script
./run_train.sh
```

**Estimated Time**: ~11 hours × 5 folds = **55 hours total** (2.3 days)

**Features**:
- ✅ Automatically continues to next fold after early stopping
- ✅ `--continue-on-error` flag continues even if one fold fails
- ✅ Single log file with all folds
- ✅ Summary report at the end showing all fold results

---

## Training Options

### Option 1: All Folds at Once (NEW - Recommended)
Run all folds in a single command:

```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin
python -u train.py --config config.yaml --fold -1 --data_dir .. --continue-on-error 2>&1 | tee ../log/train_all_folds_$(date +%Y%m%d_%H%M%S).log
```

**Advantages**:
- One command for everything
- Automatic continuation after early stopping
- Single log file tracking all folds
- Summary report at the end

---

### Option 2: Single Fold Only
Run a specific fold:

```bash
# Run only fold 1
python -u train.py --config config.yaml --fold 1 --data_dir ..

# Run only fold 2
python -u train.py --config config.yaml --fold 2 --data_dir ..
```

---

### Option 3: Sequential Script (Legacy)
Use the shell script for sequential training:

```bash
./train_all_folds.sh 1 4
```

---

### Option 4: Parallel Training (Multiple GPUs)
Run all folds simultaneously if you have 4 GPUs:

```bash
./train_parallel_folds.sh
```

**Requirements**:
- 4 GPUs with ~16GB+ VRAM each
- Sufficient system memory (4 × batch_size × data)

---

## Key Changes

### 1. SWA Start Epoch: 50 → 40
Stochastic Weight Averaging now starts at epoch 40 instead of 50, allowing earlier model averaging benefits.

### 2. Automatic Fold Progression
Early stopping no longer ends the entire training run - it just moves to the next fold automatically.

### 3. Error Handling
Use `--continue-on-error` to keep training remaining folds even if one fails.

---

## Monitor Progress

```bash
# Watch latest training
tail -f ../log/train_all_folds_*.log | grep -E "FOLD|Epoch.*Val|complete"

# Check which fold is running
tail -f ../log/train_all_folds_*.log | grep "STARTING FOLD"

# View fold summary
tail -100 ../log/train_all_folds_*.log
```

---

## After Training Completes

### 1. Check All Fold Results
```bash
# View summary of all folds
for fold in {0..4}; do
    echo "=== Fold $fold ==="
    grep "Val - Loss" ../log/train_fold${fold}_*.log | tail -1
done
```

### 2. Model Checkpoints
Each fold saves checkpoints to:
```
bin/checkpoints/fold_0/
bin/checkpoints/fold_1/
bin/checkpoints/fold_2/
bin/checkpoints/fold_3/
bin/checkpoints/fold_4/
```

### 3. Ensemble Predictions
After all folds complete, create ensemble predictions:
```python
# Combine predictions from all 5 folds
# Average the outputs for better performance
```

---

## Troubleshooting

### Training Stops Early
Check if early stopping triggered:
```bash
grep "Early stopping" ../log/train_fold*_*.log
```

### Out of Memory
Reduce `batch_size` in `config.yaml`:
```yaml
batch_size: 1  # Reduce from 2
```

### Fold Fails
Rerun specific fold:
```bash
./train_single_fold.sh <fold_number>
```

---

## Expected Performance

Based on Fold 0 results:

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Val Loss | 0.2142 | 0.0851 | -60.3% |
| Dice Score | 0.4736 | 0.5807 | +22.6% |
| Training Time | - | 10.83 hrs | - |

**Target**: Dice > 0.55 on all folds for good cross-validation performance.

---

## Quick Commands

```bash
# Start sequential training (Recommended)
cd bin && ./train_all_folds.sh 1 4

# Monitor progress
tail -f ../log/train_fold*_*.log | grep "Epoch [0-9]* Val"

# Check if training is running
ps aux | grep train.py

# View GPU usage
watch -n 1 nvidia-smi

# Stop all training
pkill -f train.py
```
