# Model Bias Fix: 99.88% Foreground Overestimation

## Problem Analysis

**Investigation Result:**
- Model predictions: **99.88% foreground** (consistent across all test/train images)
- Training labels mean: **57.70% ± 19.62% foreground**
- **Discrepancy: +42.18%** (model predicts 73% MORE foreground than training data)

**Root Cause:**
The model learned a strong bias toward positive (foreground) predictions, likely due to:
1. Insufficient regularization against overconfident predictions
2. Dice loss (0.4 weight) encouraging high overlap without penalizing overconfidence
3. Weak focal loss (0.2 weight) not focusing enough on hard negatives (background)
4. Missing entropy regularization to discourage extreme predictions

## Solution Implemented

### 1. Loss Function Reweighting (config.yaml)

**Before:**
```yaml
loss_weights:
  dice_weight: 0.4
  focal_weight: 0.2
  variance_weight: 0.2
  boundary_weight: 0.0
  cldice_weight: 0.0
  entropy_weight: (none)
```

**After:**
```yaml
loss_weights:
  dice_weight: 0.2          # Reduced: Dice encourages high overlap
  focal_weight: 0.4         # Doubled: Focus more on hard negatives
  variance_weight: 0.3      # Increased: Force uncertainty
  boundary_weight: 0.05     # Enabled: Surface quality
  cldice_weight: 0.05       # Enabled: Topology awareness
  entropy_weight: 0.05      # NEW: Discourage overconfidence
```

**Rationale:**
- **Reduce Dice** (0.4 → 0.2): Dice naturally wants high overlap; reducing weight prevents it from dominating
- **Increase Focal** (0.2 → 0.4): Focal loss focuses on hard examples; increase weight to penalize background misses
- **Increase Variance** (0.2 → 0.3): Force model to output more varied predictions instead of ~0 or ~1
- **Add Entropy** (0 → 0.05): NEW regularization to penalize overconfident predictions

### 2. Focal Loss Alpha Adjustment (config.yaml)

**Before:**
```yaml
focal_alpha: 0.25  # Weight foreground
```

**After:**
```yaml
focal_alpha: 0.75  # Weight background (HARD class)
```

**Rationale:**
- Focal loss alpha should weight the MINORITY class
- Training data: ~42% background, ~58% foreground → Background is minority
- Alpha=0.75 focuses loss on hard negatives (background voxels)

### 3. New Entropy Regularization Loss (topology_losses.py)

Added `entropy_regularization()` function:
```python
def entropy_regularization(self, pred):
    """
    Penalize predictions that are too confident (entropy too low)
    Encourages the model to output less extreme values (not always 0 or 1)
    This addresses the 99.88% foreground bias
    """
    # Shannon entropy: -p*log(p) - (1-p)*log(1-p)
    # Maximum entropy = 0.693 at p=0.5 (most uncertain)
    # Minimum entropy = 0 at p=0 or p=1 (most confident)
    
    entropy = -(pred * log(pred) + (1 - pred) * log(1 - pred))
    
    # Target entropy > 0.3 to avoid overconfident predictions
    target_entropy = 0.3
    entropy_loss = clamp(target_entropy - entropy.mean(), min=0)
    
    # Loss is high when entropy < 0.3 (prediction too confident)
    return entropy_loss
```

## Expected Impact

With these changes:
1. **Bias reduction**: Model will be encouraged to output ~50-60% foreground instead of 99.88%
2. **Better calibration**: Entropy regularization prevents extreme confidence
3. **Improved topology**: Boundary and clDice losses now enabled
4. **Maintained accuracy**: Increased focal loss focuses on hard negatives without sacrificing overall Dice score

## Training Instructions

To retrain with these fixes:

```bash
cd bin
nohup python train.py \
  --config config.yaml \
  --epochs 300 \
  --learning_rate 0.0001 \
  --device cuda \
  > log/train_bias_fix_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

This will:
1. Load model architecture from config
2. Apply new loss weights from updated config.yaml
3. Use entropy regularization to discourage overconfident predictions
4. Train for 300 epochs with early stopping

## Files Modified

1. **config.yaml**
   - Updated loss_weights for better balance
   - Changed focal_alpha from 0.25 to 0.75
   - Added comments explaining the fixes

2. **topology_losses.py**
   - Added entropy_weight parameter to __init__
   - Implemented entropy_regularization() method
   - Integrated entropy loss into forward() pass
   - Updated loss component logging to include entropy

## Validation

After retraining, validate with:
```bash
# Check if predictions are now more balanced
python3 << 'EOF'
import tifffile
import numpy as np
from pathlib import Path

pred_dir = Path("predictions")
foreground_pcts = []

for pred_file in list(pred_dir.glob("*.tif"))[:10]:
    if "_visualization" not in pred_file.name:
        pred = tifffile.imread(pred_file)
        fg_pct = (pred > 0).sum() / pred.size * 100
        foreground_pcts.append(fg_pct)
        print(f"{pred_file.name}: {fg_pct:.2f}%")

print(f"\nMean: {np.mean(foreground_pcts):.2f}%")
print(f"Target: ~57.70% (from training labels)")
print(f"Previous: 99.88% (biased model)")
EOF
```

## Timeline

- **Identified problem**: Training data ~58% FG, predictions 99.88% FG
- **Root cause**: Loss function bias toward foreground
- **Solution**: Rebalanced loss weights + entropy regularization
- **Expected improvement**: Predictions closer to 50-70% foreground range

---

*Updated: 2026-01-13 22:32:00 UTC*
