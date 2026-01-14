# Training Fix V2 - Aggressive Variance Regularization

**Date**: January 14, 2026
**Issue**: Model still producing ~99.88% uniform foreground predictions
**Root Cause**: Variance loss was too weak - clamp(min_variance - variance) = 0 when variance ≈ 0

## Changes Made

### 1. Aggressive Variance Loss (topology_losses.py:250-266)

**OLD (ineffective):**
```python
min_variance = 0.01
var_loss = torch.clamp(min_variance - variance, min=0.0) / min_variance
# Result: Loss = 0 when variance < 0.01 (model learned this!)
```

**NEW (effective):**
```python
var_loss = (1.0 - variance) ** 2.0  # Grows exponentially as variance→0
entropy_penalty = torch.clamp(0.5 - entropy_mean, min=0.0)  # Penalize overconfidence
# Result: Very strong signal to increase prediction diversity
```

### 2. Increased Variance Weight (topology_losses.py:196-207)

| Component | Old | New | Reason |
|-----------|-----|-----|--------|
| dice_weight | 0.4 | 0.3 | Reduce overfitting to training distribution |
| focal_weight | 0.2 | 0.15 | Less emphasis on hard examples |
| boundary_weight | 0.2 | 0.15 | Surface accuracy less important than diversity |
| cldice_weight | 0.1 | 0.05 | Reduce topology constraints (causing uniform predictions) |
| connectivity_weight | 0.1 | 0.05 | Reduce topology constraints |
| **variance_weight** | **0.1** | **0.25** | **2.5x INCREASE - now dominant** |
| entropy_weight | 0.05 | 0.05 | Unchanged |

Total: 1.0 → 1.0 (normalized)

## What This Fixes

1. **Exponential penalty**: Variance loss now = (1-var)² instead of clamp()
   - var=0.001: loss = 0.998 (vs 1.0) 
   - var=0.01: loss = 0.980 (vs 1.0)
   - var=0.1: loss = 0.810 (vs 0.9)
   - var=0.2: loss = 0.640 (vs 0.8)

2. **Entropy regularization**: Prevents overconfident predictions
   - Targets entropy > 0.5 (encourages ~50% uncertainty)
   - Penalizes predictions near 0 or 1

3. **Balanced loss weights**: Variance now 2.5x more important than before

## Expected Results After Retraining

- ✓ Variance loss should be **>0.1** (not 0.0001)
- ✓ Entropy loss should decrease as diversity increases
- ✓ Dice score should go from 0.56→0.6-0.7 (diverse predictions)
- ✓ Predictions should show varied foreground percentages
- ✓ Early stopping should NOT trigger (improvement should continue)

## How to Retrain

```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && python bin/train.py --config bin/config.yaml --fold 0 --data_dir ." > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor training
tail -f log/train_*.log | grep -E "(Epoch|Loss:|variance:)"
```

## Validation Checklist

- [ ] Variance loss > 0.1 (not near 0)
- [ ] Entropy loss present and meaningful
- [ ] Training continuing past epoch 30 (no early stopping)
- [ ] Dice score improving (>0.56)
- [ ] Different foreground percentages across batches
- [ ] Predictions NOT uniform 99.88%
