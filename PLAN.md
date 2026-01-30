# Vesuvius Challenge Training Plan - Surface Type 1 Only

## Problem Analysis (Jan 29, 2026)

### Critical Discovery: Wrong Label Target
The 3-class label preservation approach (preserving values 1, 2, and 0) was fundamentally flawed because:

- **Training data distribution**: 61% foreground (values 1+2 combined), 39% background
- **Model learned**: To output ~0.6 probability everywhere (matching data distribution)
- **Inference result**: 93-98% foreground predictions (WORSE than original 99.88%)
- **Root cause**: Treating BOTH surface types (thin + thick) as same "foreground" class

### Label Structure Analysis
```
Value 0: Background (~24%)
Value 1: Surface Type 1 - THIN papyrus (~5.28% mean, 0.87%-17.39% range)
Value 2: Surface Type 2 - THICK papyrus (~55.62% mean, 5.52%-93.62% range)
```

### Competition Specification
- **Goal**: Detect thin papyrus surfaces
- **Spec**: <1% foreground expected
- **Implication**: Should segment Value 1 ONLY, not both 1 and 2

## New Training Strategy: Surface Type 1 Only

### Core Change
Train model to segment **only Value 1** (thin surface) as foreground:
- Value 0 (background) → Label 0 during loss computation
- Value 1 (thin surface) → Label 1 during loss computation  
- Value 2 (thick surface) → Label 0 during loss computation (treat as background)

### Rationale
1. **Matches competition goal**: Looking for thin papyrus only
2. **Correct label distribution**: ~5% foreground (aligns with <1% spec)
3. **Model learns realistic probabilities**: Will output ~0.05 baseline instead of 0.6
4. **Inference at 0.5 threshold**: Should produce ~5% foreground (reasonable)

### Expected Behavior
- Baseline foreground: ~5% (from training data distribution)
- Model output: ~0.05 probability everywhere
- At threshold 0.5: ~5% foreground predictions
- Post-processing: May bring to <1% as intended

## Implementation Details

### Training Script Changes (`bin/train.py`)

**Current (Wrong)**:
```python
# Keep 3-class label (0=background, 1=surface1, 2=surface2)
label = label.astype(np.float32)
```

**New (Correct)**:
```python
# Convert to binary: 0=background, 1+=foreground (only surface type 1)
# Value 2 (thick surface) treated as background for loss computation
label = (label == 1).astype(np.float32)  # Only value 1 = foreground
```

### Loss Computation (`bin/nnunet_topo_wrapper.py`)

**No change needed** - the binary conversion already happens:
```python
labels_binary = (labels > 0).float()  # This will now be 0 for background + thick, 1 for thin
```

But we need to verify the dataset outputs the right label format.

## Execution Plan

### Step 1: Update Training Script
- Modify `bin/train.py` line ~64 to use: `label = (label == 1).astype(np.float32)`
- This ensures only value 1 becomes 1, everything else becomes 0

### Step 2: Start New Training
```bash
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin && python train.py --config config.yaml --fold 0 --data_dir .." > log/train_value1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Step 3: Monitor Training
- Expected best loss: Lower than 0.59 (since much less foreground)
- Expected convergence: Faster (class imbalance = 5% vs 39%, actually better)
- Expected inference: ~5% foreground on training samples

### Step 4: Validation
Run inference on 50 sample images and verify:
- Mean foreground: Should be ~5-10% (not 93%)
- Distribution: Most images <10% foreground
- Success criteria: Any value <50% is major improvement

## Key Differences from Previous Attempts

| Aspect | Previous (3-class) | New (Value 1 Only) |
|--------|------------------|-------------------|
| Foreground label | Values 1+2 | Value 1 only |
| Training % foreground | ~61% | ~5% |
| Model output baseline | ~0.6 | ~0.05 |
| Inference at 0.5 | ~60% | ~5% |
| Post-processing result | 93-98% | <10% (target) |
| Matches spec | ❌ | ✅ |

## Important Notes

1. **This is the fundamental fix** - not a tuning parameter change
2. **No hyperparameter changes needed** - architecture and loss functions stay the same
3. **Faster convergence expected** - 5% foreground is better defined than 61%
4. **Should reach spec** - <1% after post-processing is now achievable

## Files Modified

- `bin/train.py` - Line 64: Change label conversion to `(label == 1).astype(np.float32)`
- PLAN.md - This file, documenting the new strategy

## Timeline

- **Training start**: After script modification
- **Expected duration**: ~2-3 hours (50 epochs at 2m10s per epoch)
- **Next checkpoint**: Verify inference foreground % around 5-10%
- **Final step**: Run full validation on all data

---

**Status**: Ready to implement. Waiting for confirmation to modify training script.
