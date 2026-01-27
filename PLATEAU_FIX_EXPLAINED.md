# Post-Epoch 5 Plateau Analysis & Fix

## The Problem: Why Learning Stopped After Epoch 5

### What Happened in Training
- **Epoch 5 area**: Best validation loss reached (~0.5721)
- **Epoch 6+**: Model completely stuck - loss oscillates between 0.57-0.62
- **Epochs 48-60**: 18+ interventions triggered, LR scaled up to **92.5x** the original
- **Result**: Chaotic unstable training, no real learning after epoch 5

### Root Causes (3 issues combined)

#### 1. **Variance Loss Trap** ❌ 
`config_v5_variance_focused.yaml` had `variance_weight: 0.5`

- Variance loss **encourages uncertainty** in predictions
- Dice loss **encourages correct predictions**
- These conflicting objectives → local minimum with uniform-ish outputs
- Model gets stuck trying to satisfy both

**Evidence from logs:**
```
Variance: 0.90 (stuck high)
Dice: 0.42 (poor overlap)
→ Model can't improve on either front
```

#### 2. **Loss Function Imbalance** ⚖️
The loss weighting was:
- Dice: 0.3 (too weak)
- Focal: 0.15 (ok)
- **Variance: 0.5** (too strong - the culprit!)
- Boundary: 0.05

Result: Variance term dominates → pushes model away from confident predictions → poor segmentation

#### 3. **Runaway LR Scaling** 🚀
When gradients got small (0.04-0.06), the intervention logic did:
- Detected "small gradients" → scale LR by **25-35x**
- Each plateau → multiply by 1.15-3.7x
- Final: **LR × 92.5 = total chaos**

**Why this is wrong:**
- Small gradients in a local minimum ≠ need bigger LR
- Bigger LR just causes training to diverge/oscillate
- Should fix the loss function instead

### Evidence in Training Log

**First sign of trouble (epoch 6+):**
```
Epoch 5 Val - Loss: 0.5721  ← BEST
Epoch 6 Val - Loss: 0.5812  ← Starts degradation
Epoch 7 Val - Loss: 0.5923  ← Can't improve
```

**Interventions getting increasingly desperate:**
```
Intervention #15: LR scale 77.5x
Intervention #16: LR scale 81.2x  
Intervention #17: LR scale 85.0x
Intervention #18: LR scale 88.8x
Intervention #19: LR scale 92.5x  ← INSANE
```

**Result of high LR:**
```
Epoch 48 Val Loss: 0.5911 (worse)
Epoch 49 Val Loss: 0.5957 (worse)
Epoch 50 Val Loss: 0.5762 (worse)
→ Model degrading instead of improving
```

---

## The Fix

### Part 1: Fix Loss Function (`config_v6_fix_plateau.yaml`)
**Changes:**
```yaml
# OLD (broken)
dice_weight: 0.3
variance_weight: 0.5      ← Causes local minimum
batch_size: 2             ← Too small

# NEW (fixed)
dice_weight: 0.8          ← Strong primary objective
variance_weight: 0.0      ← DISABLED - was the trap
batch_size: 4             ← Larger = more stable gradients
learning_rate: 0.001      ← Normal starting LR
```

**Why this works:**
- **No variance loss** → Model can confidently output correct foreground
- **Strong Dice** → Primary objective is clear
- **Larger batch** → More stable gradient estimates
- **Normal LR** → No need for 92x scaling

### Part 2: Fix Intervention Logic (`nnunet_topo_wrapper.py`)
**Changed LR scaling from:**
```python
lr_scales = [25.0, 20.0, 15.0]  # 25-35x base
lr_scale *= 3.7  # multiply again
→ Final: up to 92.5x !!!
```

**Changed to:**
```python
lr_scales = [2.0, 1.5, 1.2]     # Conservative
lr_scale *= 1.3  # capped
→ Final: max 5.0x (absolute cap)
```

**Why:**
- If loss is stuck, **more aggressive LR makes it worse**
- Fix the loss function first (part 1)
- If LR scaling is needed, keep it conservative (2-5x max)

---

## What to Do Now

### Option A: Fresh Start (Recommended)
```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin

# Remove old checkpoints
rm -rf checkpoints/fold_0

# Train with new config (v6 - fixes plateau)
python train.py --config config_v6_fix_plateau.yaml --folds 0
```

**Expected behavior:**
- Epoch 1-5: Fast improvement (loss 0.6 → 0.55)
- Epoch 6-10: Continued improvement (loss 0.55 → 0.50)
- Epoch 11+: Gradual improvement or plateau
- **No interventions until epoch 15+** (model will actually be learning)

### Option B: Reduce Variance Weight Gradually
If you want to salvage v5:
```yaml
variance_weight: 0.1  # Down from 0.5
dice_weight: 0.7     # Up from 0.3
batch_size: 4        # Up from 2
```

Then resume training - model might escape plateau over next 20-30 epochs.

---

## Key Takeaway

**The lesson:** When your model plateaus early with small gradients:
1. **First suspect: Loss function imbalance** (not LR)
2. Don't increase LR aggressively - it's a sign you need to change the objective
3. Variance regularization can trap you in local minima if weighted too high
4. Always validate that your loss components aren't conflicting

**For this project:** Variance loss was well-intentioned (prevent uniform outputs) but WAY too strong. Dice loss alone should keep predictions diverse enough while forcing correctness.
