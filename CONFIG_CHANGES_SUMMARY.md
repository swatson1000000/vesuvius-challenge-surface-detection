# Configuration Changes Summary

**Date:** January 25, 2026 14:10 UTC  
**Reason:** Fix 34.75% foreground over-prediction issue  
**Status:** ✅ New config created and ready to use

## Files Created/Modified

### Created:
- ✅ `bin/config_v2_fixed_foreground_bias.yaml` (2.8 KB)

### Updated:
- ✅ `plan.md` - Added critical analysis section with timestamp

### Supporting Analysis Files:
- `analyze_predictions.py` - Analyzed 32 completed predictions
- `validate_predictions.py` - Validated 47 predictions vs training labels

---

## The Problem (In Numbers)

| Metric | Expected | Actual | Gap |
|--------|----------|--------|-----|
| Training Label FG % | 60.90% | N/A | Baseline |
| Model Prediction FG % | 60.90% | 95.65% | +34.75% ⚠️ |
| Model Precision | 0.75-0.85 | 0.6082 | -0.14 |
| Model Recall | 0.85-0.90 | 0.9649 | +0.06 |
| Dice Score | 0.80+ | 0.7293 | -0.07 |

**Root Cause:** Class weights were backwards, penalizing missing background more than predicting false foreground.

---

## The Solution (Configuration Changes)

### Loss Weight Rebalancing

```yaml
# BEFORE (v1) - Over-predicts foreground
dice_weight: 0.2          → INCREASED to 0.4
focal_weight: 0.4         → DECREASED to 0.25
variance_weight: 0.3      → DECREASED to 0.2
boundary_weight: 0.05     → INCREASED to 0.1

# AFTER (v2) - Precision-focused
# = Enforce correct overlap (dice)
# = Stop mining hard negatives (focal)
# = Allow confident predictions (variance)
# = Stricter boundaries (boundary)
```

### Class Weight Reversal

```yaml
# BEFORE (v1) - Backwards
background_weight: 2.94   # Penalizes missing BG more
foreground_weight: 1.52   # Penalizes false FG less

# AFTER (v2) - Correct
background_weight: 1.0    # Don't over-penalize
foreground_weight: 3.0    # Heavily penalize false FG (3x)
```

### Focal Loss Alpha Adjustment

```yaml
# BEFORE: focal_alpha: 0.75 (Focus on background minority)
# AFTER:  focal_alpha: 0.4  (Focus on FG false positives)
```

### Post-Processing Threshold

```yaml
# BEFORE: threshold: 0.5 (predict FG if >50% confident)
# AFTER:  threshold: 0.7 (predict FG if >70% confident)
# = More conservative, reduces false positives
```

---

## Expected Improvements

### Quantitative Targets

- **Precision:** 0.6082 → ~0.75-0.85 (+13-19%)
- **Recall:** 0.9649 → ~0.85-0.90 (-6-11%, acceptable tradeoff)
- **Dice:** 0.7293 → ~0.80+ (+7-10%)
- **Foreground %:** 95.65% → ~55-65% (match labels)

### Qualitative Improvements

1. **Fewer false positives** - Model won't over-predict foreground
2. **Better precision** - What it predicts as foreground is more likely correct
3. **Maintained recall** - Still catches most true foreground
4. **Better alignment** - Predictions match training label distribution

---

## How to Use the New Config

### Option 1: Use v2 directly
```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && cd bin && python train.py --config config_v2_fixed_foreground_bias.yaml --fold 0 --data_dir .." > log/train_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Option 2: Update default config
```bash
cp bin/config.yaml bin/config_backup_v1.yaml
cp bin/config_v2_fixed_foreground_bias.yaml bin/config.yaml
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate phi4 && cd bin && python train.py --fold 0 --data_dir .." > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## Validation Plan

After retraining with v2 config:

1. **Run inference on validation set** (158 images)
2. **Calculate foreground percentage** across all predictions
3. **Verify Dice/Precision/Recall** metrics improve
4. **Compare against training labels** with `validate_predictions.py`
5. **If successful:** Apply to folds 1-4, generate submission

---

## Key Insights

1. **Recall vs Precision:** High recall (96.49%) ≠ good predictions when precision is low (60.82%)
2. **Class weights are critical:** Direction matters as much as magnitude
3. **Post-processing threshold:** Another knob for precision control (0.5 → 0.7)
4. **Loss interactions:** Can't tune weights independently - system is interconnected
5. **Validation against labels:** Essential for detecting systematic biases

---

## References

- Previous config: `bin/config.yaml` (v1)
- New config: `bin/config_v2_fixed_foreground_bias.yaml` (v2)
- Analysis: `validate_predictions.py` (47 images analyzed)
- Updated plan: `plan.md` (section: "Critical Analysis Update - January 25, 2026")

