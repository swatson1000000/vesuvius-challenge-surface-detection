# CRITICAL FIX IMPLEMENTED - January 26, 2026

## Problem Identified
Model was learning to output **constant values** (~0.627) across all inputs, regardless of input image or patch location.

**Evidence**:
- All 53 test predictions had **identical foreground ratios** (95.06% or 71.19%)
- Model probabilities: min=0.52, max=0.71, mean=0.627, std=0.069
- Same distribution on test AND training data
- **Verdict**: Model = constant output function, NOT learning

## Root Cause
**Loss weights were backwards**:
- `variance_weight: 0.3` (30% of total loss)
- `dice_weight: 0.2` (20% of total loss)

This forced the model to **minimize output variance** instead of learning spatial segmentation patterns. The variance regularization was so strong it **overconstrained the model** to uniform outputs.

## Solution Implemented

### 1. Rebalanced Loss Weights
```yaml
# OLD (BROKEN)
dice_weight: 0.2    # Only 20% of gradient signal
focal_weight: 0.4
variance_weight: 0.3   # TOO HEAVY - forces uniformity
boundary_weight: 0.05
cldice_weight: 0.05

# NEW (FIXED)
dice_weight: 0.5      # PRIMARY - spatial learning
focal_weight: 0.3     # Secondary - hard examples
variance_weight: 0.1  # Minimal - just regularization
boundary_weight: 0.05
cldice_weight: 0.0    # DISABLED - too complex
```

**Key Changes**:
- ✅ Dice: 0.2 → 0.5 (enable spatial feature learning)
- ✅ Variance: 0.3 → 0.1 (reduce constraint on output diversity)
- ✅ CL-Dice: 0.05 → 0.0 (disable complex topology losses)
- ✅ Focal Alpha: 0.75 → 0.5 (balanced class weighting)

### 2. Training Strategy Going Forward
- Start completely fresh training
- Monitor **output variance** at each epoch (should be >0.1)
- Monitor **per-image variation** in predictions
- Add debug outputs to catch learning failures early
- Use simpler architecture if needed (no CL-Dice, connectivity)

## Next Steps

1. ✅ Config updated: `bin/config.yaml`
2. ✅ Diagnosis documented: `TRAINING_DIAGNOSIS.md`
3. ⏭️ **RESTART TRAINING**:
   ```bash
   cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
   nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && cd bin && python train.py --fold 0 --data_dir .." > log/train_FIXED_$(date +%Y%m%d_%H%M%S).log 2>&1 &
   ```

4. Monitor for:
   - Varied foreground ratios across predictions (40-95% range)
   - Output variance > 0.15 (not <0.07)
   - Dice scores improving >0.5
   - Clear gradient signals in early epochs

## Verification Checkpoint

After 10-20 epochs, run test inference:
```python
# Should show varied output ranges
# NOT all 0.52-0.71
```

If outputs still uniform → **architecture issue** (needs redesign)
If varied outputs → **model is learning** (continue training)

---

**Status**: 🔧 Fixed and ready for restart
**Old checkpoint**: Corrupted - do not use
**Next run**: Clean start with corrected loss weights
