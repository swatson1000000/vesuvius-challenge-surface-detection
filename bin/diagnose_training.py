"""Diagnostic script to understand the training problem"""
import torch
import numpy as np
import tifffile
from pathlib import Path
import sys
sys.path.append('/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin')
from nnunet_topo_wrapper import TopologyAwareUNet3D
from topology_losses import CombinedTopologyLoss
import yaml

print("="*80)
print("DIAGNOSTIC: Investigating Training Failure")
print("="*80)

# Load config
with open('/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check if training data exists
data_dir = Path('/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection')
image_files = sorted((data_dir / 'train_images').glob('*.tif'))
label_files = sorted((data_dir / 'train_labels').glob('*.tif'))

print(f"\n1. Dataset Check:")
print(f"   Images found: {len(image_files)}")
print(f"   Labels found: {len(label_files)}")

if len(image_files) > 0 and len(label_files) > 0:
    # Load first training sample
    image = tifffile.imread(image_files[0])
    label = tifffile.imread(label_files[0])
    
    print(f"\n2. Sample Data:")
    print(f"   Image: {image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
    print(f"   Label: {label.shape}, dtype={label.dtype}, range=[{label.min()}, {label.max()}]")
    print(f"   Label unique values: {np.unique(label)}")
    
    # Check label distribution
    fg_ratio = (label > 0).mean()
    print(f"   Foreground ratio: {fg_ratio:.4f} ({fg_ratio*100:.2f}%)")
    print(f"   Background ratio: {1-fg_ratio:.4f} ({(1-fg_ratio)*100:.2f}%)")
    
    # Normalize image (same as training)
    p1, p99 = np.percentile(image, (0.5, 99.5))
    image_norm = np.clip(image, p1, p99)
    image_norm = (image_norm - image_norm.min()) / (image_norm.max() - image_norm.min() + 1e-8)
    print(f"   Normalized image range: [{image_norm.min():.4f}, {image_norm.max():.4f}]")
    
    # Create model
    device = torch.device('cuda')
    model = TopologyAwareUNet3D(
        in_channels=1,
        out_channels=1,
        initial_filters=config['initial_filters'],
        depth=config['depth']
    ).to(device)
    
    # Create loss function
    loss_fn = CombinedTopologyLoss(
        dice_weight=config['loss_weights']['dice_weight'],
        focal_weight=config['loss_weights']['focal_weight'],
        boundary_weight=config['loss_weights']['boundary_weight'],
        cldice_weight=config['loss_weights']['cldice_weight'],
        connectivity_weight=config['loss_weights']['connectivity_weight'],
        focal_gamma=config['focal_gamma'],
        focal_alpha=config['focal_alpha']
    ).to(device)
    
    print(f"\n3. Model Architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Initial filters: {config['initial_filters']}")
    print(f"   Depth: {config['depth']}")
    
    # Test with a patch
    patch_size = 128
    d, h, w = image_norm.shape
    d_start = d//2 - patch_size//2
    h_start = h//2 - patch_size//2
    w_start = w//2 - patch_size//2
    
    image_patch = image_norm[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]
    label_patch = (label[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size] > 0).astype(np.float32)
    
    print(f"\n4. Test Patch:")
    print(f"   Image patch: {image_patch.shape}, range=[{image_patch.min():.4f}, {image_patch.max():.4f}]")
    print(f"   Label patch: {label_patch.shape}, foreground={label_patch.mean():.4f}")
    
    # Convert to tensors
    image_tensor = torch.from_numpy(image_patch[None, None, ...]).float().to(device)
    label_tensor = torch.from_numpy(label_patch[None, None, ...]).float().to(device)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        
        print(f"\n5. Model Output (untrained):")
        print(f"   Logits: range=[{logits.min().item():.4f}, {logits.max().item():.4f}], mean={logits.mean().item():.4f}")
        print(f"   Probs: range=[{probs.min().item():.4f}, {probs.max().item():.4f}], mean={probs.mean().item():.4f}")
        
    # Test loss with uniform predictions
    print(f"\n6. Loss Function Tests:")
    
    # Test 1: Perfect prediction
    perfect_pred = label_tensor.clone()
    loss_result = loss_fn(perfect_pred, label_tensor)
    loss_perfect = loss_result[0] if isinstance(loss_result, tuple) else loss_result
    print(f"   Perfect prediction loss: {loss_perfect.item():.6f}")
    
    # Test 2: All zeros
    zero_pred = torch.zeros_like(label_tensor)
    loss_result = loss_fn(zero_pred, label_tensor)
    loss_zero = loss_result[0] if isinstance(loss_result, tuple) else loss_result
    print(f"   All zeros loss: {loss_zero.item():.6f}")
    
    # Test 3: All ones
    one_pred = torch.ones_like(label_tensor)
    loss_result = loss_fn(one_pred, label_tensor)
    loss_one = loss_result[0] if isinstance(loss_result, tuple) else loss_result
    print(f"   All ones loss: {loss_one.item():.6f}")
    
    # Test 4: Uniform 0.6 (what our model outputs)
    uniform_pred = torch.full_like(label_tensor, 0.6)
    loss_result = loss_fn(uniform_pred, label_tensor)
    loss_uniform = loss_result[0] if isinstance(loss_result, tuple) else loss_result
    print(f"   Uniform 0.6 loss: {loss_uniform.item():.6f}")
    
    # Test 5: Uniform at foreground ratio
    uniform_fg_pred = torch.full_like(label_tensor, label_patch.mean())
    loss_result = loss_fn(uniform_fg_pred, label_tensor)
    loss_uniform_fg = loss_result[0] if isinstance(loss_result, tuple) else loss_result
    print(f"   Uniform {label_patch.mean():.2f} loss: {loss_uniform_fg.item():.6f}")
    
    # Compare to actual training loss
    print(f"\n7. Loss Analysis:")
    print(f"   Training plateaued at: 0.072")
    print(f"   Uniform prediction gives: {loss_uniform.item():.6f}")
    print(f"   Ratio match prediction gives: {loss_uniform_fg.item():.6f}")
    if abs(loss_uniform_fg.item() - 0.072) < 0.01:
        print(f"   >>> FOUND IT! Model learned to output foreground ratio everywhere!")
    
    # Check label values issue
    print(f"\n8. CRITICAL ISSUE DETECTED:")
    print(f"   Labels contain values: {np.unique(label)}")
    print(f"   But model expects binary (0, 1)!")
    print(f"   Label value 2 count: {(label == 2).sum()} voxels ({(label == 2).mean()*100:.2f}%)")
    print(f"   This could be causing confusion in the loss function!")
    
    # Load trained model and check
    checkpoint = torch.load('/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/bin/checkpoints/fold_0/checkpoint_epoch_40.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        trained_logits = model(image_tensor)
        trained_probs = torch.sigmoid(trained_logits)
        loss_result = loss_fn(trained_probs, label_tensor)
        trained_loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
        
        print(f"\n9. Trained Model on Same Patch:")
        print(f"   Logits: range=[{trained_logits.min().item():.4f}, {trained_logits.max().item():.4f}]")
        print(f"   Probs: range=[{trained_probs.min().item():.4f}, {trained_probs.max().item():.4f}], mean={trained_probs.mean().item():.4f}, std={trained_probs.std().item():.4f}")
        print(f"   Loss: {trained_loss.item():.6f}")
        print(f"   Prediction variance: {trained_probs.var().item():.6f} (should be >> 0 for good segmentation)")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
