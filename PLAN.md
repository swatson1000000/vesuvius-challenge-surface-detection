# Vesuvius Challenge - Surface Detection: Comprehensive Implementation Plan

## Executive Summary

This competition requires building a 3D segmentation model to detect papyrus surfaces in CT scans of carbonized Herculaneum scrolls from the Villa dei Papiri. The challenge is to trace scroll surfaces through complex folds, gaps, and distortions while maintaining topological integrity.

**Competition Deadline**: February 13, 2026  
**Prize Pool**: $200,000 (Top 10 places)  
**Computational Constraint**: 9 hours inference time (CPU or GPU)  
**Current Status**: ~1 month remaining  
**Last Updated**: January 5, 2026

**Downloaded Resources**:
- ✅ Competition data (25GB images + 691MB labels)
- ✅ Official metric demo notebook: `vesuvius-2025-metric-demo.ipynb`
- ✅ Topological metrics library: `topological-metrics-kaggle/`
- ✅ Competition documentation: `COMPETITION_INFO.md`

---

## 1. Problem Analysis

### 1.1 Core Challenge
- **Input**: 3D CT scan volumes (TIFF format) of carbonized scrolls
- **Output**: 3D binary segmentation masks identifying papyrus surfaces (recto layer preferred)
- **Key Difficulty**: Maintaining topological correctness while handling:
  - Tightly wound, compressed papyrus layers
  - Noise and artifacts from carbonization
  - Complex folds and gaps
  - Variable scroll conditions across different samples

### 1.2 Data Characteristics
- **Training Set**: 806 labeled 3D volumes across 6 scrolls
  - Scroll 34117: 382 samples (47.4%) - primary scroll
  - Scroll 35360: 176 samples (21.8%)
  - Scroll 26010: 130 samples (16.1%)
  - Scroll 26002: 88 samples (10.9%) - test scroll
  - Scroll 44430: 17 samples (2.1%)
  - Scroll 53997: 13 samples (1.6%)
  
- **Test Set**: 1 sample from scroll 26002 (ID: 1407735)
  
- **Volume Specifications**:
  - Typical dimensions: 320×320×320 voxels
  - Data type: 8-bit grayscale (0-255)
  - Image size: ~35MB compressed (TIFF/LZW)
  - Label size: ~660KB-1.8MB (sparse binary)
  - Total training data: ~25GB images + 691MB labels
  
- **Label Sparsity**: Highly sparse (typically <1% foreground voxels)
  - This indicates thin sheet-like structures in 3D space

### 1.3 Evaluation Metrics (Critical!)

The competition uses a **weighted combination** of three metrics:

```
Final Score = 0.30 × TopoScore + 0.35 × SurfaceDice@τ + 0.35 × VOI_score
```

#### A. Surface Dice @ τ=2.0 (35% weight)
- Measures surface proximity tolerance
- Points within τ=2.0 voxels of true surface count as correct
- Forgiving of slight boundary misalignments
- **Strategy**: Focus on getting the general surface position correct

#### B. VOI Score (35% weight)
- Variation of Information - penalizes splits and mergers
- VOI_split = H(GT | Pred) - over-segmentation penalty
- VOI_merge = H(Pred | GT) - under-segmentation penalty
- Converted to bounded score: VOI_score = 1 / (1 + 0.3 × VOI_total)
- **Critical**: Avoid false bridges between parallel layers (mergers)
- **Critical**: Avoid breaking single sheets into pieces (splits)

#### C. TopoScore (30% weight)
- Betti number matching from algebraic topology
- Evaluates three homology dimensions:
  - k=0: Connected components (should be few)
  - k=1: Tunnels/handles (should match ground truth)
  - k=2: Cavities (should be minimal)
- **Critical**: Penalizes spurious holes, artificial bridges, and disconnected pieces

### 1.4 Key Constraints & Competition Rules
- **Code Competition**: Notebook submission only
- **Runtime Limit**: 9 hours for full inference
- **No Internet**: Offline execution required
- **Pre-trained Models**: Allowed (must be publicly available)
- **External Data**: Allowed if freely & publicly available
- **Submission Format**: ZIP file containing .tif mask(s) matching input dimensions

---

## 2. Strategic Approach

### 2.1 Model Architecture Considerations

Given the problem requirements, we should consider:

#### Option A: 3D U-Net Variants (Recommended Primary)
**Pros**:
- Standard for 3D medical segmentation
- Proven track record on volumetric data
- Efficient with limited 3D data
- Good balance of accuracy and inference speed

**Cons**:
- May struggle with topology preservation
- Requires significant GPU memory

**Recommended**: nnU-Net framework
- Auto-configures preprocessing, architecture, and training
- State-of-the-art on medical segmentation benchmarks
- Handles variable input sizes
- Built-in cross-validation

#### Option B: Transformer-Based (e.g., UNETR, Swin-UNETR)
**Pros**:
- Better at capturing long-range dependencies
- Good for complex spatial relationships
- Recent SOTA on medical imaging tasks

**Cons**:
- Computationally expensive
- Requires more training data
- Slower inference (may hit 9-hour limit)

**Recommendation**: Use as ensemble candidate if time permits

#### Option C: nnU-Net + Topology-Aware Loss
**Pros**:
- Addresses metric requirements directly
- Can incorporate clDice, skeleton-based losses
- Forces topological consistency during training

**Cons**:
- More complex to implement
- Requires careful tuning

**Recommendation**: Primary approach - start with nnU-Net, add topology losses

### 2.2 Topology-Specific Strategies

Since 30-35% of the score depends on topology (TopoScore + part of VOI), we must:

1. **Morphological Post-Processing**:
   - Remove small isolated components
   - Fill small holes
   - Thin spurious bridges between layers
   
2. **Topology-Aware Loss Functions**:
   - clDice loss (centerline Dice) - proven for tubular structures
   - Persistent homology-based losses
   - Connectivity-preserving losses
   
3. **Instance Segmentation Approach**:
   - Treat each papyrus layer as separate instance
   - Use watershed or similar for layer separation
   - Prevents mergers between adjacent wraps

### 2.3 Training Strategy

#### Phase 1: Baseline Model (Days 1-5)
- Implement nnU-Net with default settings
- Train on full dataset with 5-fold cross-validation
- Establish baseline performance on all metrics
- Expected CV score: 0.60-0.70

#### Phase 2: Metric Optimization (Days 6-15)
- Add topology-aware losses:
  - clDice for connectivity
  - Boundary refinement losses for SurfaceDice
  - Connected component consistency losses for VOI
- Experiment with loss weights
- Test different post-processing pipelines
- Target CV score: 0.75-0.80

#### Phase 3: Ensemble & Refinement (Days 16-25)
- Train multiple models with different:
  - Random seeds
  - Augmentation strategies
  - Architecture variants
- Implement ensemble strategy (voting or averaging)
- Optimize inference pipeline for 9-hour limit
- Target CV score: 0.80-0.85

#### Phase 4: Final Optimization (Days 26-30)
- Incorporate any new data released by organizers
- Fine-tune on weakest scroll types
- Test submission pipeline thoroughly
- Perform final validation
- Target CV score: 0.85+

---

## 3. Technical Implementation Plan

### 3.1 Environment Setup

```bash
# Create conda environment
conda create -n vesuvius python=3.10 -y
conda activate vesuvius

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2
pip install tifffile imageio scikit-image scipy
pip install monai albumentations
pip install connected-components-3d topology
pip install gudhi  # For persistent homology if needed

# Monitoring & utilities
pip install wandb tensorboard tqdm
pip install pandas numpy matplotlib seaborn
```

### 3.2 Data Pipeline

#### Directory Structure
```
vesuvius-challenge-surface-detection/
├── train_images/          # 786 training volumes
├── train_labels/          # 786 label volumes  
├── test_images/           # 1 test volume
├── PLAN.md               # This file
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch dataset classes
│   │   ├── preprocessing.py     # Normalization, resampling
│   │   └── augmentation.py      # 3D augmentations
│   ├── models/
│   │   ├── unet3d.py           # 3D U-Net implementation
│   │   ├── nnunet_wrapper.py   # nnU-Net integration
│   │   └── ensemble.py         # Model ensembling
│   ├── losses/
│   │   ├── combined_loss.py    # Multi-component loss
│   │   ├── topo_losses.py      # Topology-aware losses
│   │   └── surface_dice.py     # Metric-aligned losses
│   ├── metrics/
│   │   ├── competition_metrics.py  # Official metric implementation
│   │   └── validation.py           # CV evaluation
│   ├── postprocessing/
│   │   ├── morphology.py       # Morphological operations
│   │   ├── topology_fix.py     # Topology correction
│   │   └── instance_sep.py     # Instance separation
│   └── utils/
│       ├── visualization.py    # 3D visualization
│       └── inference.py        # Submission generation
├── configs/
│   ├── base_config.yaml
│   └── experiment_configs/
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory analysis
│   ├── 02_metric_analysis.ipynb     # Understand metrics
│   ├── 03_baseline_validation.ipynb # Validate baseline
│   └── 04_visualization.ipynb       # Result visualization
├── experiments/
│   └── [timestamped runs]/
├── submissions/
│   └── [submission versions]/
└── checkpoints/
    └── [model weights]/
```

#### Data Preprocessing
1. **Intensity Normalization**:
   - Clip extreme values (0.5th and 99.5th percentiles)
   - Z-score normalization per volume
   - Or percentile normalization (0-255 → 0-1)

2. **Resampling** (if needed):
   - Volumes already 320³ - minimal resampling needed
   - Consider anisotropic spacing if provided in metadata

3. **Patching Strategy** (for memory efficiency):
   - Extract 128×128×128 or 160×160×160 patches
   - Overlapping patches with stride=96 or 128
   - Reconstruct full volume with weighted averaging

### 3.3 Augmentation Strategy

Critical for generalization with limited data:

```python
augmentations = {
    'spatial': [
        'RandomRotation90(axes=(0,1,2))',
        'RandomFlip(axes=(0,1,2))',
        'ElasticDeformation(sigma=10, points=3)',
        'Affine(scale=0.9-1.1, rotate=(-15, 15))',
    ],
    'intensity': [
        'GaussianNoise(sigma=0.01)',
        'GaussianBlur(sigma=0.5-1.5)',
        'BrightnessContrast(brightness=0.2, contrast=0.2)',
        'GammaTransform(gamma=0.7-1.5)',
    ],
    'artifact': [
        'SimulatePartialVolume',  # Simulate missing slices
        'AddGaussianNoise',
        'MultiplicativeNoise',
    ]
}
```

### 3.4 Loss Function Design

Proposed multi-component loss:

```python
total_loss = (
    0.4 * dice_loss +              # Basic region overlap
    0.2 * focal_loss +             # Handle class imbalance
    0.2 * boundary_loss +          # Sharp boundaries (for SurfaceDice)
    0.1 * clDice_loss +           # Topology/connectivity (for TopoScore)
    0.1 * hausdorff_loss          # Extreme distance penalty
)
```

**Rationale**:
- Dice: Core segmentation quality
- Focal: Handle extreme foreground/background imbalance
- Boundary: Optimize for SurfaceDice metric
- clDice: Preserve connectivity, reduce mergers/splits
- Hausdorff: Prevent outlier predictions

### 3.5 Training Configuration

```yaml
# Base training config
model:
  architecture: "nnUNet"
  encoder_depth: 5
  decoder_depth: 5
  initial_filters: 32
  
training:
  batch_size: 2  # Adjust based on GPU memory
  patch_size: [128, 128, 128]
  num_epochs: 300
  optimizer: "AdamW"
  learning_rate: 1e-3
  lr_scheduler: "ReduceLROnPlateau"
  weight_decay: 1e-5
  gradient_clip: 1.0
  
  # Early stopping
  patience: 50
  min_delta: 0.001
  
validation:
  n_folds: 5
  stratify_by: "scroll_id"  # Ensure each fold has samples from all scrolls
  val_interval: 5  # Validate every N epochs
  
augmentation:
  probability: 0.8
  spatial_prob: 0.5
  intensity_prob: 0.5
```

### 3.6 Post-Processing Pipeline

Critical for topology optimization:

```python
def postprocess_prediction(pred_volume, config):
    """
    Multi-stage post-processing for topology optimization
    """
    # Stage 1: Thresholding
    binary_mask = (pred_volume > config.threshold).astype(np.uint8)
    
    # Stage 2: Remove small components (reduce k=0 false positives)
    labeled = cc3d.connected_components(binary_mask, connectivity=26)
    sizes = np.bincount(labeled.ravel())
    mask = sizes > config.min_component_size
    binary_mask = mask[labeled]
    
    # Stage 3: Fill small holes (reduce k=2 cavities)
    from scipy.ndimage import binary_fill_holes
    for z in range(binary_mask.shape[0]):
        binary_mask[z] = binary_fill_holes(binary_mask[z])
    
    # Stage 4: Morphological operations
    from skimage.morphology import remove_small_holes, closing, opening
    binary_mask = opening(binary_mask, ball(2))  # Remove thin bridges
    binary_mask = closing(binary_mask, ball(2))  # Close small gaps
    
    # Stage 5: Watershed for instance separation (if mergers detected)
    if config.separate_instances:
        binary_mask = watershed_instance_separation(binary_mask)
    
    # Stage 6: Final smoothing
    binary_mask = gaussian_filter(binary_mask.astype(float), sigma=0.5) > 0.5
    
    return binary_mask
```

### 3.7 Validation Strategy

**K-Fold Cross-Validation** (K=5):
- Stratified by scroll_id to ensure representation
- Monitor all three metrics during training:
  - SurfaceDice@2.0
  - VOI_score  
  - TopoScore
- Track weighted competition score

**Validation Considerations**:
- Test scroll (26002) is underrepresented (88 samples, 10.9%)
- Ensure sufficient validation on scroll 26002
- Consider scroll-specific validation folds

**Metric Monitoring**:
```python
metrics_to_track = {
    'surface_dice_2.0': weight=0.35,
    'voi_score': weight=0.35,
    'topo_score': weight=0.30,
    'competition_score': weight=1.0,  # Weighted average
    
    # Additional diagnostics
    'dice_coefficient': None,
    'hausdorff_95': None,
    'num_components': None,  # k=0
    'num_handles': None,     # k=1
    'num_cavities': None,    # k=2
}
```

---

## 4. Inference & Submission Pipeline

### 4.1 Inference Optimization

Given 9-hour time limit:

1. **Model Selection**:
   - Use efficient architectures (nnU-Net > transformers)
   - Consider quantization (FP16 or INT8)
   - Profile inference time during development

2. **Batching**:
   - Process volumes in patches
   - Optimize patch size vs. batch size
   - Use sliding window with overlap

3. **Test-Time Augmentation (TTA)**:
   - If time permits: 8-fold TTA (rotations + flips)
   - Average predictions for robustness
   - May improve by 1-2% but costs 8× time

4. **Ensemble**:
   - If using ensemble, limit to 3-5 models max
   - Simple averaging or weighted voting
   - Test full ensemble time on local machine

### 4.2 Submission Format

```python
def generate_submission(model, test_loader, output_dir):
    """
    Generate competition submission
    """
    for volume_id, volume in test_loader:
        # Inference
        prediction = model.predict(volume)
        
        # Post-processing
        prediction = postprocess_prediction(prediction)
        
        # Save as TIFF
        output_path = os.path.join(output_dir, f"{volume_id}.tif")
        tifffile.imwrite(output_path, prediction.astype(np.uint8))
    
    # Create ZIP
    shutil.make_archive('submission', 'zip', output_dir)
```

---

## 5. Risk Analysis & Mitigation

### 5.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Insufficient GPU memory | High | Medium | Use patching, reduce batch size, gradient checkpointing |
| Overfitting to training scrolls | High | High | Strong augmentation, scroll-stratified CV, regularization |
| Topology metric optimization | High | Medium | Implement topology-aware losses, extensive post-processing |
| 9-hour inference timeout | High | Low | Profile early, optimize inference, limit ensemble size |
| New training data release | Medium | High | Modular pipeline for quick retraining |
| Label quality issues | Medium | Medium | Manual inspection, outlier detection, weighted samples |

### 5.2 Data Risks

**Training Data Issues**:
- Competition mentions new data will be "less curated"
- Need strategy to handle noisy labels
- Consider confidence-based sample weighting

**Test Data Risks**:
- Only 1 test sample (scroll 26002)
- Private leaderboard may have different scrolls
- Must generalize across all scroll types

**Mitigation**:
- Validate thoroughly on scroll 26002
- Ensure models work well on all training scrolls
- Use scroll-agnostic augmentation

### 5.3 Timeline Risks

**Current Status**: ~30 days remaining

**Compressed Timeline Issues**:
- Limited time for experimentation
- May need to skip certain approaches
- Reduced ensemble diversity

**Mitigation**:
- Prioritize high-impact components (nnU-Net first)
- Parallel development where possible
- Daily progress checkpoints
- Have submission ready 2-3 days before deadline

---

## 6. Detailed Timeline (30 Days)

### Week 1 (Days 1-7): Foundation
- **Day 1**: Environment setup, data loading, EDA
- **Day 2**: Implement metric calculation code, validate against official
- **Day 3**: Baseline nnU-Net setup and preprocessing
- **Day 4**: Training pipeline implementation
- **Day 5**: First baseline model training
- **Day 6**: Validation framework, CV setup
- **Day 7**: Baseline results analysis, identify weak points

**Deliverable**: Working baseline with CV score ~0.60-0.65

### Week 2 (Days 8-14): Optimization
- **Day 8**: Implement topology-aware losses (clDice)
- **Day 9**: Add boundary/surface losses
- **Day 10**: Post-processing pipeline v1
- **Day 11**: Hyperparameter tuning (learning rate, loss weights)
- **Day 12**: Augmentation optimization
- **Day 13**: Train improved models
- **Day 14**: Mid-point evaluation and strategy adjustment

**Deliverable**: Improved models with CV score ~0.70-0.75

### Week 3 (Days 15-21): Advanced Methods
- **Day 15**: Instance segmentation approach
- **Day 16**: Advanced post-processing (watershed, morphology)
- **Day 17**: Second architecture (UNETR or Swin-UNETR)
- **Day 18**: Ensemble strategy implementation
- **Day 19**: Test-time augmentation
- **Day 20**: Inference optimization and profiling
- **Day 21**: Weekly evaluation, adjust priorities

**Deliverable**: Multiple strong models, CV score ~0.75-0.80

### Week 4 (Days 22-28): Finalization
- **Day 22**: Incorporate any new released data
- **Day 23**: Final model training with best configurations
- **Day 24**: Ensemble selection and optimization
- **Day 25**: Full inference pipeline testing (9-hour check)
- **Day 26**: Generate and validate submission
- **Day 27**: Final post-processing tuning
- **Day 28**: Emergency buffer / final improvements

**Deliverable**: Competition-ready submission, CV score 0.80+

### Days 29-30: Submission & Buffer
- **Day 29**: Final submission generation and upload
- **Day 30**: Backup day for any issues

---

## 7. Success Metrics & Goals

### Primary Goal: Top 10 Finish ($5,000+)
- **Target CV Score**: 0.78-0.82
- **Required Leaderboard Position**: Top 10 (out of ~631 participants)

### Stretch Goal: Top 5 Finish ($15,000+)
- **Target CV Score**: 0.82-0.85
- **Required**: Novel approach or exceptional optimization

### Minimum Goal: Top 25%
- **Target CV Score**: 0.72-0.75
- **Focus**: Solid baseline with good engineering

### Component Targets
- SurfaceDice@2.0: > 0.85
- VOI_score: > 0.80
- TopoScore: > 0.75
- Combined: > 0.80

---

## 8. Key Technical Decisions

### Decision 1: Model Architecture
**Choice**: nnU-Net as primary, consider UNETR as secondary
**Rationale**: Proven reliability, auto-configuration, good inference speed

### Decision 2: Training Data Strategy
**Choice**: Use all training data, weighted by quality
**Rationale**: Limited data requires using everything available

### Decision 3: Topology Optimization
**Choice**: Combined training-time losses + post-processing
**Rationale**: Addresses 30-35% of competition score directly

### Decision 4: Validation Strategy
**Choice**: 5-fold CV stratified by scroll_id
**Rationale**: Ensures generalization, particularly to test scroll

### Decision 5: Inference Strategy
**Choice**: Single best model or small ensemble (≤3)
**Rationale**: Balance performance vs. 9-hour time limit

---

## 9. Resource Requirements

### Computational Resources
- **GPU**: NVIDIA RTX 3090 / 4090 or A100 (24GB+ VRAM)
- **CPU**: 16+ cores for data loading
- **RAM**: 64GB+ recommended
- **Storage**: 100GB for data + experiments

### Time Investment
- **Development**: ~6-8 hours/day × 28 days = 168-224 hours
- **Training**: ~48-72 hours total GPU time
- **Experimentation**: ~30-40 iterations

### Tools & Frameworks
- PyTorch 2.0+
- nnU-Net v2
- MONAI
- scikit-image, scipy
- tifffile, cc3d
- WandB or TensorBoard for tracking

---

## 10. Knowledge Gaps & Learning Needs

### Areas Requiring Research:
1. **Persistent Homology**: For TopoScore optimization
2. **3D Morphological Operations**: For post-processing
3. **Instance Segmentation in 3D**: For layer separation
4. **Topology-Aware Losses**: Implementation details

### Reference Papers:
1. "clDice - Topology-Preserving Loss" (CVPR 2021)
2. "Efficient Betti Matching" (2024) - for TopoScore
3. "nnU-Net" (Nature Methods 2021)
4. "Pitfalls of topology-aware segmentation" (IPMI 2025)

### Kaggle Resources:
1. ✅ Official metric notebook (downloaded): `vesuvius-2025-metric-demo.ipynb`
   - Online: https://www.kaggle.com/code/sohier/vesuvius-2025-metric-demo/
2. ✅ Metric resources (downloaded): `topological-metrics-kaggle/`
   - Online: https://www.kaggle.com/datasets/sohier/vesuvius-metric-resources
   - Includes: TopoScore, SurfaceDice, VOI implementations + 22 wheel packages
3. Previous competition: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection
4. ✅ Competition info (saved): `COMPETITION_INFO.md`

---

## 11. Contingency Plans

### If Behind Schedule:
- Skip secondary architecture (UNETR)
- Reduce ensemble size to 1-2 models
- Simplify post-processing
- Focus on one strong baseline

### If Overfitting:
- Increase augmentation strength
- Add more regularization (dropout, weight decay)
- Reduce model capacity
- Use pseudo-labeling on test data (if beneficial)

### If Inference Too Slow:
- Reduce patch size
- Use FP16 precision
- Skip TTA
- Use single model instead of ensemble

### If Metrics Not Improving:
- Focus on weakest metric component
- Add metric-specific losses
- Analyze failure cases visually
- Consult competition discussions/notebooks

---

## 12. Next Immediate Actions

### Priority 1 (This Week):
1. ✅ Read competition details and create plan
2. ✅ Download competition data and metric implementation
3. ⬜ Set up development environment (build topological-metrics library)
4. ⬜ Implement data loader and visualization
5. ⬜ Verify metric implementation against official scorer
6. ⬜ Create EDA notebook

### Priority 2 (Next Week):
1. ⬜ Implement baseline nnU-Net
2. ⬜ Set up training pipeline
3. ⬜ Implement CV framework
4. ⬜ First baseline training run
5. ⬜ Validate metrics against competition scorer

### Priority 3 (Week 3):
1. ⬜ Implement topology losses
2. ⬜ Optimize post-processing
3. ⬜ Train improved models
4. ⬜ Start ensemble experiments

---

## 13. Conclusion

This competition presents a unique challenge at the intersection of computer vision, medical imaging, and algebraic topology. Success requires:

1. **Strong 3D Segmentation**: Base capability to detect surfaces
2. **Topology Awareness**: Preventing mergers, splits, and holes
3. **Efficient Inference**: Meeting 9-hour time constraint
4. **Robust Generalization**: Working across different scroll conditions

The plan prioritizes proven approaches (nnU-Net) while incorporating competition-specific optimizations (topology losses, instance separation). With disciplined execution and daily progress, a top-10 finish is achievable.

**Key to Success**: 
- Fast iteration cycles
- Early metric validation
- Continuous monitoring of all three metric components
- Focus on topology from day one
- Don't neglect post-processing

**Philosophy**: 
"Perfect is the enemy of good. Ship a working solution early, then iterate."

---

## Appendix A: Useful Code Snippets

### Loading Competition Data
```python
import tifffile
import numpy as np
import pandas as pd

# Load training data
train_df = pd.read_csv('train.csv')
image_id = train_df.iloc[0]['id']
scroll_id = train_df.iloc[0]['scroll_id']

# Load volume and label
image = tifffile.imread(f'train_images/{image_id}.tif')
label = tifffile.imread(f'train_labels/{image_id}.tif')

print(f"Image: {image.shape}, Label: {label.shape}")
print(f"Scroll: {scroll_id}, Foreground: {100*label.sum()/label.size:.3f}%")
```

### Metric Calculation Template
```python
# Use the official implementation from topological-metrics-kaggle
from topometrics.leaderboard import compute_leaderboard_score
import tifffile

def calculate_competition_score(pred, gt, spacing=(1,1,1)):
    """Calculate official competition metric using downloaded library"""
    score_report = compute_leaderboard_score(
        predictions=pred,
        labels=gt,
        dims=(0, 1, 2),
        spacing=spacing,  # (z, y, x)
        surface_tolerance=2.0,  # in spacing units
        voi_connectivity=26,
        voi_transform="one_over_one_plus",
        voi_alpha=0.3,
        combine_weights=(0.3, 0.35, 0.35),  # (Topo, SurfaceDice, VOI)
        fg_threshold=None,  # None => legacy "!= 0"
        ignore_label=2,  # voxels with this GT label are ignored
        ignore_mask=None,
    )
    return score_report.score, score_report
```

### Simple 3D Visualization
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_mask(volume, slice_axis=0, slice_idx=None):
    """Visualize 3D volume"""
    if slice_idx is None:
        slice_idx = volume.shape[slice_axis] // 2
    
    if slice_axis == 0:
        slice_data = volume[slice_idx, :, :]
    elif slice_axis == 1:
        slice_data = volume[:, slice_idx, :]
    else:
        slice_data = volume[:, :, slice_idx]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')
    plt.show()
```

---

**Document Version**: 1.0  
**Last Updated**: January 5, 2026  
**Next Review**: After baseline results (Day 7)
