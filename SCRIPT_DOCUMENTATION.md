# Comprehensive Script Documentation
**Vesuvius Challenge Surface Detection**

Detailed explanations of `train.py` and `inference.py`

---

## Table of Contents

1. [train.py Overview](#trainpy-overview)
2. [train.py Components](#trainpy-components)
3. [train.py Usage](#trainpy-usage)
4. [inference.py Overview](#inferencepy-overview)
5. [inference.py Components](#inferencepy-components)
6. [inference.py Usage](#inferencepy-usage)
7. [Data Flow](#data-flow)
8. [Error Handling](#error-handling)

---

## train.py Overview

### Purpose

`train.py` is the **main training orchestrator** for the topology-aware 3D segmentation model. It handles:

- **Multi-fold cross-validation**: Automatically runs 5 folds sequentially
- **Data loading and preprocessing**: Loads TIFF images, normalizes, patches
- **Model training**: Uses nnU-Net with topology-aware losses
- **Learning rate scheduling**: Cosine annealing with warm restarts
- **Stochastic Weight Averaging**: Averaging weights after epoch 40 for better generalization
- **Early stopping**: Stops training if validation loss doesn't improve
- **Automatic fold progression**: When one fold completes, automatically trains next fold

### Key Features

| Feature | Details |
|---------|---------|
| **Cross-validation** | 5-fold split (628 train, 158 val per fold) |
| **Augmentation** | Random flip, rotation, noise, brightness/contrast |
| **Loss Function** | Combined: Dice (0.4) + Focal (0.2) + Variance (0.2) |
| **Scheduling** | CosineAnnealingWarmRestarts with periodic LR restarts |
| **SWA** | Starts at epoch 40, averages weights to epoch end |
| **Patch Size** | 128×128×128 voxels |
| **Batch Size** | 2 |
| **Max Epochs** | 300 |
| **Early Stopping** | 50 epoch patience |

---

## train.py Components

### 1. VesuviusDataset Class

**Location:** Lines 39-192

**Purpose:** Custom PyTorch Dataset class for loading and preprocessing Vesuvius Challenge data

#### Key Methods

**`__init__(image_paths, label_paths, patch_size, augment)`**
- Initializes dataset with image/label path pairs
- Validates that all labels exist
- Sets patch size and augmentation flag

**`__len__()`**
- Returns total number of image-label pairs (786)

**`__getitem__(idx)`**
- **Called by DataLoader** when fetching a sample
- **Process:**
  1. Load TIFF image and label (3D volumes)
  2. Normalize image to [0, 1] range
  3. Convert label to binary (0=background, 1=foreground)
  4. Extract patch from volume (balanced sampling)
  5. Apply augmentations if training
  6. Add channel dimension: (D,H,W) → (1,D,H,W)
  7. Return as PyTorch tensors

**`_normalize(image)`**
- Clips extremes at 0.5th and 99.5th percentile (removes outliers)
- Scales to [0, 1] range
- Why: Improves model robustness to intensity variations

```python
p1, p99 = np.percentile(image, (0.5, 99.5))  # Find bounds
image = np.clip(image, p1, p99)               # Remove outliers
image = (image - min) / (max - min)           # Scale to [0,1]
```

**`_extract_patch(image, label)`**
- **Problem:** Images are too large (3D volumes) to fit in GPU memory
- **Solution:** Extract 128×128×128 patches
- **Balanced Sampling:** Tries up to 10 random positions to find patch with:
  - Foreground ratio between 30-70%
  - Avoids all-background or all-foreground patches
- **Padding:** If patch at edge, pads with zeros

```
Example: 512×512×512 volume
→ Extract 128×128×128 patch
→ Balanced: ~50% foreground, 50% background
```

**`_augment(image, label)`**
- Randomly applied (50% chance per operation):
  1. **Flip**: Random axis (depth, height, width)
  2. **Rotate**: 90° rotation in random plane
  3. **Noise**: Gaussian noise (σ=0.01)
  4. **Brightness/Contrast**: Scale by 0.8-1.2

```python
# Example augmentation sequence
image = flip(image)                    # 50% chance
image = rotate_90deg(image)            # 50% chance
image = add_gaussian_noise(image)      # 50% chance
image = scale_brightness(image)        # 50% chance
```

**Why:** Prevents overfitting by creating artificial variation

### 2. load_dataset() Function

**Location:** Lines 195-238

**Purpose:** Load all images, split into k-fold cross-validation sets

**Process:**
```
786 total images
    ↓
Shuffle with random_state=42 (reproducible)
    ↓
K-fold split (n_splits=5)
    ↓
For fold N:
  - Validation: 1/5 of data (~158 images)
  - Training: 4/5 of data (~628 images)
```

**Example for fold 0:**
```
All 786 images: [image_0, image_1, ..., image_785]
    ↓
After KFold split:
- Train indices: [1, 2, 3, ..., 785] (628 samples)
- Val indices: [0, 157, 314, 471, 628] (158 samples)
    ↓
Train dataset: 628 images
Val dataset: 158 images
```

### 3. train_single_fold() Function

**Location:** Lines 241-410

**Purpose:** Train one fold from start to finish

#### Flow

```
train_single_fold(fold=0)
    ↓
1. Load dataset for fold 0
   - Train: 628 images
   - Val: 158 images
    ↓
2. Create PyTorch DataLoaders
   - DataLoader: batches (batch_size=2)
   - Shuffle training data
   - Pin memory for GPU speed
    ↓
3. Create model (22.6M parameters)
    ↓
4. Create optimizer & scheduler
   - Optimizer: Adam
   - Scheduler: CosineAnnealingWarmRestarts
    ↓
5. Setup SWA (Stochastic Weight Averaging)
    ↓
6. Train for up to 300 epochs
   - Forward pass through model
   - Compute combined loss
   - Backward pass
   - Update weights
   - Evaluate on validation set
   - If no improvement for 50 epochs → STOP
   - If at epoch ≥ 40 → Update SWA weights
    ↓
7. Update batch norm statistics for SWA
    ↓
8. Save SWA model checkpoint
    ↓
9. Log results and move to next fold
```

#### Key Training Details

**DataLoader Creation:**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=2,           # Process 2 images at a time
    shuffle=True,           # Randomize order each epoch
    num_workers=4,          # Parallel loading on 4 CPU cores
    pin_memory=True         # Keep data in GPU memory
)
```

**Model Configuration:**
```yaml
in_channels: 1              # Grayscale input
out_channels: 1             # Binary segmentation output
initial_filters: 32         # Base number of filters
depth: 4                    # 4-level U-Net pyramid
```

**Loss Computation:**
```
Total Loss = 0.4 × Dice_Loss 
           + 0.2 × Focal_Loss 
           + 0.2 × Variance_Loss

Dice Loss:       Measures overlap between prediction and label
Focal Loss:      Handles class imbalance (more weight to hard examples)
Variance Loss:   Prevents uniform predictions (CRITICAL bug fix)
```

**Scheduler (CosineAnnealingWarmRestarts):**
```
Epoch 0-9:   LR: 0.0001 → 0.000001 (cosine decay)
Epoch 10:    LR jumps to 0.0001 (warm restart)
Epoch 10-29: LR: 0.0001 → 0.000001 (cosine decay)
Epoch 30:    LR jumps to 0.0001 (another restart)
...

Purpose: Periodic restarts help escape local minima
```

**Stochastic Weight Averaging (SWA):**
```
Epoch 0-39:      Regular training, weights updated normally
Epoch 40-299:    Maintain running average of weights
                 
                 avg_weights = 0.1 × current_weights + 0.9 × avg_weights
                 
                 This smooths optimization trajectory
                 Result: Better generalization!
```

**Early Stopping:**
```
Monitor validation loss each epoch:
  - If improves:     Reset patience counter (reset to 50)
  - If doesn't:      Increment counter
  
  If patience reaches 50:
    - Stop training
    - Move to next fold (if --fold -1)
```

### 4. main() Function

**Location:** Lines 413-521

**Purpose:** Main orchestrator - runs multiple folds sequentially

#### Flow

```
main()
    ↓
1. Load config.yaml
    ↓
2. Setup logging to stdout with immediate flush
   (ensures we see training progress in real-time)
    ↓
3. Determine which folds to run:
   - If --fold -1: Run all 5 folds [0, 1, 2, 3, 4]
   - If --fold N:  Run only fold N
    ↓
4. For each fold:
   ├─ Call train_single_fold(fold)
   │  └─ Trains until early stopping
   ├─ Record result (success/failure, time)
   └─ If --continue-on-error: keep going even if fails
    ↓
5. Print summary:
   - Total time across all folds
   - Per-fold results
   - Status (completed/failed)
```

#### Example Output

```
================================================================================
STARTING MULTI-FOLD TRAINING - 2026-01-11 15:01:13
================================================================================
================================================================================
STARTING FOLD 0
================================================================================
Found 786 image-label pairs
Fold 0: 628 train, 158 val
Model parameters: 22,600,000
Using CosineAnnealingWarmRestarts scheduler (T_0=10, T_mult=2)
SWA enabled: will start at epoch 40 with LR 0.00005
Starting training for fold 0...
Epoch 1/300: Train Loss: 0.1542, Val Loss: 0.1203, Val Dice: 0.4156
Epoch 2/300: Train Loss: 0.1204, Val Loss: 0.0987, Val Dice: 0.4523
...
Epoch 50/300: Train Loss: 0.0653, Val Loss: 0.0851, Val Dice: 0.5807
Early stopping triggered at epoch 50 (patience: 50)
================================================================================
Fold 0 complete!
Fold 0 time: 38,988.51 seconds (10.83 hours)
================================================================================
✓ Fold 0 completed successfully in 10.83 hours

================================================================================
STARTING FOLD 1
================================================================================
...
```

---

## train.py Usage

### Command Line Arguments

```bash
python train.py [OPTIONS]

Options:
  --config CONFIG_PATH
      Path to config.yaml (default: config.yaml)
      
  --data_dir DATA_PATH
      Path to data directory
      (default: /home/swatson/.../vesuvius-challenge-surface-detection)
      
  --fold FOLD_NUMBER
      Which fold to train
      -1: Run all folds [0,1,2,3,4] sequentially (default)
       N: Run only fold N (e.g., --fold 2)
      
  --continue-on-error
      If a fold fails, continue to next fold
      (default: stop on error)
```

### Usage Examples

**1. Run all 5 folds (recommended)**
```bash
cd bin
python train.py --fold -1
```

**2. Run specific fold**
```bash
python train.py --fold 2      # Just fold 2
```

**3. Run with custom config**
```bash
python train.py --config my_config.yaml --fold -1
```

**4. Run in background with logging**
```bash
cd bin
nohup python -u train.py --fold -1 > ../log/train_$(date +%s).log 2>&1 &
# Or use the provided wrapper:
./quick_train_all.sh
```

**5. Continue if a fold fails**
```bash
python train.py --fold -1 --continue-on-error
```

### Expected Execution Time

- **Per fold:** ~11 hours (50 epochs average)
- **All 5 folds:** ~55 hours
- **Single GPU:** Sequential (one fold at a time)

---

## inference.py Overview

### Purpose

`inference.py` is the **prediction engine** for the trained model. It handles:

- **Model loading**: Loads trained checkpoint from disk
- **Volume-level inference**: Processes full-sized 3D volumes
- **Patch-based processing**: Uses overlapping patches to manage memory
- **Patch stitching**: Reconstructs full volume with blending
- **Post-processing**: Applies topology corrections (threshold, morphology, etc.)
- **Batch inference**: Processes multiple test images

### Key Features

| Feature | Details |
|---------|---------|
| **Patch Size** | 128×128×128 (same as training) |
| **Patch Overlap** | 32 voxels (25% overlap for smooth blending) |
| **Blending** | Gaussian weight map for smooth transitions |
| **Threshold** | 0.5 (converts probabilities to binary) |
| **Post-processing** | Remove small components, fill holes |
| **Batch Support** | Can process entire directories |
| **Visualization** | Saves both binary (0,1) and visualization (0,255) outputs |

---

## inference.py Components

### 1. InferencePipeline Class

**Location:** Lines 22-278

**Purpose:** End-to-end inference with model loading, volume prediction, post-processing

#### Key Methods

**`__init__(model_path, device, patch_size, overlap, postprocess_config)`**

Initializes the inference pipeline:
- Loads model checkpoint from disk
- Sets device (GPU or CPU)
- Configures patch-based processing
- Initializes post-processor

```python
pipeline = InferencePipeline(
    model_path="checkpoints/fold_0/best_model.pth",
    device='cuda',
    patch_size=(128, 128, 128),
    overlap=32,
    postprocess_config={'threshold': 0.5, 'min_component_size': 100}
)
```

**`_load_model(model_path)`**

Loads trained model checkpoint:
```python
1. Load checkpoint file (dict with model weights, metadata)
2. Create new model instance with same architecture as training
3. Load weights into model: model.load_state_dict(checkpoint)
4. Move model to GPU/CPU: model.to(device)
5. Set to eval mode: model.eval() (disables dropout, batch norm)
6. Log metadata (epoch, best_val_score)
```

**`normalize(image)`**

Same normalization as training:
```python
p1, p99 = np.percentile(image, (0.5, 99.5))  # Clip extremes
image = np.clip(image, p1, p99)
image = (image - min) / (max - min + eps)    # Scale to [0,1]
```

**Why consistent normalization is critical:**
- Training: Model learned to recognize normalized intensities
- Inference: Must apply identical normalization
- If different: Model gets unexpected input → poor predictions

**`extract_patches(volume)`**

Divides large volume into overlapping patches:

```
Input volume: 512 × 512 × 512
Patch size: 128 × 128 × 128
Overlap: 32

Stride = patch_size - overlap = 128 - 32 = 96

Positions in depth (D):
  D_start: 0, 96, 192, 288, 384, 480 (6 positions)
Positions in height (H):
  H_start: 0, 96, 192, 288, 384, 480 (6 positions)
Positions in width (W):
  W_start: 0, 96, 192, 288, 384, 480 (6 positions)

Total patches: 6 × 6 × 6 = 216 patches
```

**Why overlapping?**
- Non-overlapping: Artifacts at patch boundaries
- Overlapping: Smooth transitions, better predictions

**`stitch_patches(patches, positions, volume_shape)`**

Reconstructs full volume from patches with smooth blending:

```
1. Create output volume (all zeros)
2. Create weight map (all zeros)

3. For each patch:
   ├─ Create Gaussian weight map
   ├─ Add patch × weight_map to output
   └─ Add weight_map to weight_map total

4. Normalize: output / weight_map
   (Areas with more patches get averaged)
```

**Gaussian Weight Map:**
```
Creates smooth blending at patch edges

Visualization (2D slice):
   Center of patch:     ████████  (weight ≈ 1.0)
   Middle region:       ▓▓▓▓▓▓▓▓  (weight ≈ 0.7)
   Edges:               ░░░░░░░░  (weight ≈ 0.2)

Result: Smooth transitions between patches!
```

**`_create_gaussian_weight_map(size)`**

Creates weight map that is high in center, low at edges:
```python
# Calculate distance from nearest edge
# For 128x128x128 patch:
#   Center voxel: distance=64, weight≈1.0
#   Edge voxel:   distance=0, weight≈0.0

weight = np.minimum(np.minimum(dd, hh), ww)
weight = np.clip(weight, 0, 1)
```

**`predict_volume(volume)` - Main Inference Function**

**Flow:**

```
predict_volume(volume: 512×512×512)
    ↓
1. Normalize volume to [0,1]
    ↓
2. Extract overlapping patches
   Result: ~216 patches of 128×128×128
    ↓
3. Process each patch through model
   - Add batch+channel dims: (128³) → (1,1,128³)
   - Forward pass through model
   - Apply sigmoid: logits → probabilities [0,1]
   - Remove batch+channel dims: (1,1,128³) → (128³)
    ↓
4. Stitch patches with Gaussian blending
   Result: Full volume of probabilities [0,1]
    ↓
5. Apply post-processing:
   - Threshold at 0.5 → binary (0,1)
   - Remove small components (< 100 voxels)
   - Fill small holes (< 50 voxels)
   - Compute topology stats
    ↓
6. Return binary mask
```

**Time Complexity:**
- 216 patches × GPU inference ≈ 30-60 seconds (with progress bar)
- Stitching + post-processing ≈ 10 seconds
- Total per volume: ~1-2 minutes

**`predict_and_save(input_path, output_path)`**

End-to-end: load image → predict → save

**Outputs:**
```
For input: test_image.tif

Creates 2 files:
1. test_image.tif (binary: 0,1)
   - For model submission
   - Compressed size: small
   
2. test_image_visualization.tif (0,255)
   - For human viewing
   - 8-bit grayscale: 0=black, 255=white
```

### 2. main() Function

**Location:** Lines 281-370

**Purpose:** Command-line interface for inference

#### Flow

```
main()
    ↓
1. Parse command-line arguments
    ↓
2. Setup logging to stdout (incremental output)
    ↓
3. Create InferencePipeline with config
    ↓
4. Find input files
   - If input is directory: get all .tif files
   - If input is file: process single file
    ↓
5. For each input file:
   ├─ Load image
   ├─ Run prediction
   ├─ Save binary and visualization
   └─ Handle errors (continue to next file)
    ↓
6. Print summary:
   - Total time
   - Number of files processed
```

#### Example Output

```
================================================================================
Inference started at: 2026-01-12 10:30:00
================================================================================
Loaded model from checkpoints/fold_0/best_model.pth
  Epoch: 50
  Best val score: 0.5807
Found 10 files to process

Processing test_image_1.tif
  Loaded volume: (512, 512, 512), dtype: uint16
Predicting volume of shape (512, 512, 512)
Extracting patches...
  Extracted 216 patches
Running model inference...
Predicting: 100%|████████████| 216/216 [00:45<00:00, 4.80it/s]
Stitching patches...
Applying post-processing...
Prediction stats:
  foreground_voxels: 15234
  background_voxels: 118766
  num_components: 42
  max_component_size: 8521
  Saved binary prediction (0,1) to test_image_1.tif
  Saved visualization (0,255) to test_image_1_visualization.tif

Processing test_image_2.tif
...

================================================================================
Inference complete!
Start time:   2026-01-12 10:30:00
End time:     2026-01-12 10:55:30
Total time:   1530.00 seconds (0.42 hours)
================================================================================
```

---

## inference.py Usage

### Command Line Arguments

```bash
python inference.py [OPTIONS]

Required:
  --checkpoint PATH
      Path to model checkpoint (e.g., checkpoints/fold_0/best_model.pth)
      
  --input PATH
      Input image or directory
      
  --output PATH
      Output directory for predictions

Optional:
  --device {cuda,cpu}
      Device to use (default: cuda)
      
  --patch_size D H W
      Patch size (default: 128 128 128)
      
  --overlap N
      Patch overlap (default: 32)
      
  --threshold FLOAT
      Probability threshold (default: 0.5)
      
  --min_component_size N
      Minimum component size in voxels (default: 100)
      
  --max_hole_size N
      Maximum hole size to fill in voxels (default: 50)
      
  --kernel_size N
      Morphological kernel size (default: 2)
      
  --separate_instances
      Apply instance separation (default: off)
```

### Usage Examples

**1. Run inference on single image**
```bash
cd bin
python inference.py \
  --checkpoint checkpoints/fold_0/best_model.pth \
  --input ../test_images/image_001.tif \
  --output ../predictions/
```

**2. Process entire test directory**
```bash
python inference.py \
  --checkpoint checkpoints/fold_0/best_model.pth \
  --input ../test_images/ \
  --output ../predictions/
```

**3. Custom thresholding and post-processing**
```bash
python inference.py \
  --checkpoint checkpoints/fold_0/best_model.pth \
  --input ../test_images/ \
  --output ../predictions/ \
  --threshold 0.4 \
  --min_component_size 50 \
  --max_hole_size 100
```

**4. Ensemble predictions (average all 5 folds)**
```bash
# After training all 5 folds, run inference for each:
for fold in 0 1 2 3 4; do
  python inference.py \
    --checkpoint checkpoints/fold_$fold/best_model.pth \
    --input ../test_images/ \
    --output ../predictions/fold_$fold/
done

# Then average predictions from all folds (using separate Python script)
python ensemble_predictions.py \
  --fold_dirs ../predictions/fold_0 ../predictions/fold_1 ... \
  --output ../predictions/ensemble/
```

**5. Run on CPU (slower but uses less memory)**
```bash
python inference.py \
  --checkpoint checkpoints/fold_0/best_model.pth \
  --input ../test_images/ \
  --output ../predictions/ \
  --device cpu
```

### Expected Execution Time

- **Per image (512×512×512):** 60-120 seconds (GPU)
- **Per image with 5 folds ensemble:** 5-10 minutes
- **CPU mode:** 5-10× slower

---

## Data Flow

### Training Flow

```
Raw TIFF Images (786)
        ↓
    Normalize [0,1]
        ↓
    K-fold Split
    ├─ Fold 0: Train (628), Val (158)
    ├─ Fold 1: Train (628), Val (158)
    ├─ ...
    └─ Fold 4: Train (628), Val (158)
        ↓
    For each fold:
    ├─ Extract 128³ patches (balanced sampling)
    ├─ Apply augmentations (flip, rotate, noise)
    ├─ Load into GPU (batches of 2)
    ├─ Forward pass → model output
    ├─ Compute combined loss
    │  ├─ Dice (0.4)
    │  ├─ Focal (0.2)
    │  └─ Variance (0.2)
    ├─ Backward pass + optimizer step
    ├─ Evaluate on validation set
    ├─ Update SWA weights (epoch ≥ 40)
    └─ Save best checkpoint
        ↓
    Result: 5 trained models (one per fold)
```

### Inference Flow

```
Raw Test TIFF (512³)
        ↓
    Normalize [0,1] (same as training!)
        ↓
    Extract overlapping patches
    (stride=96, overlap=32)
    Result: 216 patches
        ↓
    For each patch:
    ├─ Add batch+channel dims
    ├─ Load to GPU
    ├─ Forward pass → logits
    ├─ Apply sigmoid → probabilities [0,1]
    └─ Store prediction
        ↓
    Stitch patches
    ├─ Gaussian weight blending
    ├─ Average overlapping regions
    └─ Result: Full probability volume
        ↓
    Post-processing
    ├─ Threshold at 0.5 → binary
    ├─ Remove components < 100 voxels
    ├─ Fill holes < 50 voxels
    └─ Result: Final binary mask
        ↓
    Save outputs
    ├─ Binary (0,1) for submission
    └─ Visualization (0,255) for viewing
```

---

## Error Handling

### train.py Error Handling

**Early Stopping:**
```python
if not val_loss_improved:
    patience_counter += 1
    if patience_counter >= 50:
        logger.info(f"Early stopping at epoch {epoch}")
        break  # Exit training loop
```

**Fold Failure:**
```python
try:
    fold_time = train_single_fold(fold, config, ...)
    fold_results[fold] = {'status': 'completed', 'time': fold_time}
except Exception as e:
    fold_results[fold] = {'status': 'failed', 'error': str(e)}
    if not args.continue_on_error:
        break  # Stop all training
    else:
        continue  # Skip to next fold
```

**Common Errors & Solutions:**

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Batch too large | Reduce `batch_size` in config |
| `File not found` | Wrong data path | Check `--data_dir` path |
| `Model load failed` | Checkpoint corrupted | Retrain fold |
| `NaN loss` | Numerical instability | Reduce `learning_rate` |

### inference.py Error Handling

**File Processing:**
```python
try:
    pipeline.predict_and_save(input_path, output_path)
except Exception as e:
    logger.error(f"Error processing {input_path.name}: {e}")
    continue  # Skip to next file
```

**Model Loading:**
```python
try:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
except FileNotFoundError:
    logger.error(f"Model checkpoint not found: {model_path}")
    sys.exit(1)
except RuntimeError as e:
    logger.error(f"Model architecture mismatch: {e}")
    sys.exit(1)
```

**Common Errors & Solutions:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not found` | Wrong checkpoint path | Check `--checkpoint` path |
| `CUDA out of memory` | Volume too large | Reduce `patch_size` or increase `overlap` |
| `Input file error` | Corrupted TIFF | Verify TIFF file integrity |
| `Dimension mismatch` | Input not 3D | Verify volume is 3D (D,H,W) |

---

## Performance Tips

### Training Speedup

1. **Increase `num_workers`** in DataLoader
   ```yaml
   num_workers: 8  # More parallel data loading
   ```

2. **Reduce patch size** (trades accuracy for speed)
   ```yaml
   patch_size: [96, 96, 96]  # Down from 128
   ```

3. **Use multiple GPUs**
   ```bash
   python -m torch.distributed.launch --nproc_per_node=4 train.py
   ```

### Inference Speedup

1. **Larger patches** (if memory allows)
   ```bash
   python inference.py ... --patch_size 256 256 256
   ```

2. **Less overlap** (trades quality for speed)
   ```bash
   python inference.py ... --overlap 16  # Down from 32
   ```

3. **Batch processing** with multiple GPUs
   ```bash
   # Modify inference.py to load multiple patches at once
   ```

### Memory Optimization

1. **Training:**
   - Reduce `batch_size`: 2 → 1
   - Reduce `patch_size`: 128 → 96
   - Disable `pin_memory`

2. **Inference:**
   - Increase `--overlap` to reduce patches
   - Reduce `--patch_size`
   - Use smaller model (reduce `initial_filters`)

---

## File Outputs

### Training Outputs

```
checkpoints/
├── fold_0/
│   ├── best_model.pth          # Best checkpoint (lowest val loss)
│   ├── swa_model.pth           # SWA averaged model
│   ├── checkpoint_latest.pth   # Latest checkpoint
│   └── training_log.txt        # Per-epoch metrics
├── fold_1/
│   └── ...
└── ...

log/
└── train_all_folds_20260111_150113.log    # Training log
```

### Inference Outputs

```
predictions/
├── image_001.tif               # Binary (0,1) for submission
├── image_001_visualization.tif # Visual (0,255) for viewing
├── image_002.tif
├── image_002_visualization.tif
└── ...

Each .tif file has same dimensions as input volume
```

---

## References

- [train.py](bin/train.py): Main training script
- [inference.py](bin/inference.py): Inference script
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md): Training overview and configuration
- [config.yaml](bin/config.yaml): Training hyperparameters
- [CLAUDE.md](CLAUDE.md): AI instructions for running scripts
