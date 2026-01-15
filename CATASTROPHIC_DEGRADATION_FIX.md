# Catastrophic Degradation Detection & Recovery Fix

## Problem
Training was achieving excellent validation loss (~0.3226 at epoch 10) but then experienced catastrophic degradation where loss jumped to ~0.65+ after interventions were applied. The model never recovered from this bad state.

## Root Cause
When aggressive interventions (LR increases, loss simplification) were applied to escape plateaus, they could cause the model to diverge into a worse local minimum. Once in this bad state, there was no mechanism to detect or recover from it.

## Solution
Added automatic catastrophic degradation detection and rollback mechanism:

### Key Features

#### 1. **Degradation Detection**
- Tracks best validation loss achieved (`best_loss_ever`)
- Monitors each epoch for loss increase > threshold (default: 15%)
- Calculates degradation fraction: `(current_loss - best_loss) / best_loss`

#### 2. **Automatic Rollback**
When catastrophic degradation is detected:
- Reloads the best checkpoint (`best_model.pth`)
- Restores the learning rate from when best loss was achieved
- Restores the original loss weights (before any interventions)
- Resets intervention counter to start fresh

#### 3. **Hyperparameter Preservation**
System now tracks and restores:
- Best learning rate
- Dice weight, Focal weight, Boundary weight, clDice weight, Connectivity weight
- Original loss configuration before interventions

## Changes Made

### File: `bin/nnunet_topo_wrapper.py`

**Added to `train()` method signature:**
```python
catastrophic_degradation_threshold: float = 0.15,  # 15% loss increase triggers rollback
```

**Added state tracking variables:**
```python
best_loss_ever = float('inf')  # Track absolute best loss across all epochs
best_lr = self.base_lr  # Store best learning rate
best_loss_weights = {...}  # Store best loss configuration
```

**Added degradation detection logic:**
```python
# ⚠️ CATASTROPHIC DEGRADATION DETECTION
if best_loss_ever != float('inf'):
    degradation_fraction = (current_val_loss - best_loss_ever) / best_loss_ever
    if degradation_fraction > catastrophic_degradation_threshold:
        # Rollback to best model
        self.load_checkpoint('best_model.pth')
        # Restore hyperparameters
        # Reset interventions
```

## Threshold Configuration

Default threshold: **15% increase** (`catastrophic_degradation_threshold=0.15`)

This means:
- If best loss was 0.3226, rollback triggers if loss reaches 0.371 or higher
- Prevents false positives from normal fluctuations
- Adjustable via config if needed

## Example Scenario

**Before (Old Behavior):**
```
Epoch 10: Val Loss = 0.3226 ✓ (best)
Epoch 11: Val Loss = 0.3900
Epoch 12: Val Loss = 0.4500
...
Epoch 42: Val Loss = 0.6500  (stuck in bad state - no recovery)
```

**After (New Behavior):**
```
Epoch 10: Val Loss = 0.3226 ✓ (best) [saved]
Epoch 11: Val Loss = 0.3900
Epoch 12: Val Loss = 0.4500
...
Epoch 20: Val Loss = 0.5850
🚨 CATASTROPHIC DEGRADATION DETECTED!
   Degradation: 81.2% (threshold: 15%)
   Rolling back to best checkpoint...
✓ Restored LR: 0.001000
✓ Restored loss weights
✓ Reset intervention counter
Epoch 21: Val Loss = 0.3226 [resumed from best state]
```

## Benefits

✅ **Prevents training collapse** - Automatically escapes bad local minima
✅ **Preserves best hyperparameters** - Never strays too far from working configuration
✅ **Transparent recovery** - Logs show exactly what happened and what was restored
✅ **Configurable** - Can adjust threshold for different use cases
✅ **Non-invasive** - Doesn't affect normal training when model is improving

## How to Use

The fix is automatic - no configuration changes needed. Just retrain:

```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
python bin/train.py --config config.yaml
```

To adjust the degradation threshold, modify `catastrophic_degradation_threshold` in the config or command line parameters.

## Testing

To verify the fix works, the model should now:
1. Train normally and improve loss
2. If loss suddenly increases by 15%+, it will automatically rollback
3. Continue training from the previous best state
4. Eventually find better minima or reach a stable state

## Log Output Example

```
Epoch 10 Val - Loss: 0.3226 (dice: 0.5860, focal: 0.0726, ...)
New best model saved! Score: -0.3226

Epoch 11 Val - Loss: 0.3234 (dice: 0.5658, focal: 0.0716, ...)

Epoch 12 Val - Loss: 0.4500 (dice: 0.4100, focal: 0.1200, ...)

🚨 CATASTROPHIC DEGRADATION DETECTED!
   Best loss: 0.3226
   Current loss: 0.4500
   Degradation: 39.4% (threshold: 15.0%)
   Rolling back to best checkpoint and restoring hyperparameters...
   ✓ Restored LR: 0.001000
   ✓ Restored loss weights (Dice=0.40, Focal=0.20)
   ✓ Reset intervention counter
   ✓ Continuing training from best state...
```
