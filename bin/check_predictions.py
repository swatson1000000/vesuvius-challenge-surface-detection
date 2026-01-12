"""Quick diagnostic to check raw model predictions"""
import torch
import numpy as np
import tifffile
from nnunet_topo_wrapper import TopologyAwareUNet3D
from pathlib import Path

# Load model
device = torch.device('cuda')
checkpoint = torch.load('checkpoints/fold_0/checkpoint_epoch_40.pth', map_location=device)

model = TopologyAwareUNet3D(
    in_channels=1,
    out_channels=1,
    initial_filters=16,
    depth=4
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load test image
image = tifffile.imread('../test_images/1407735.tif')
print(f"Image shape: {image.shape}, dtype: {image.dtype}")
print(f"Image range: [{image.min()}, {image.max()}]")

# Normalize (same as inference.py)
p1, p99 = np.percentile(image, (0.5, 99.5))
image = np.clip(image, p1, p99)
image = (image - image.min()) / (image.max() - image.min() + 1e-8)
image = image.astype(np.float32)
print(f"Normalized range: [{image.min():.4f}, {image.max():.4f}]")

# Take a center patch
d, h, w = image.shape
patch_size = 128
d_start = d//2 - patch_size//2
h_start = h//2 - patch_size//2
w_start = w//2 - patch_size//2
patch = image[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]

print(f"\nPatch shape: {patch.shape}")
print(f"Patch range: [{patch.min():.4f}, {patch.max():.4f}]")

# Predict
with torch.no_grad():
    patch_tensor = torch.from_numpy(patch[None, None, ...]).float().to(device)
    print(f"Input tensor shape: {patch_tensor.shape}")
    
    # Get logits
    logits = model(patch_tensor)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
    
    # Apply sigmoid
    probs = torch.sigmoid(logits)
    print(f"\nProbabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"Probabilities mean: {probs.mean().item():.4f}, std: {probs.std().item():.4f}")
    
    # Check distribution
    probs_np = probs[0, 0].cpu().numpy()
    print(f"\nProbability distribution:")
    print(f"  < 0.1: {(probs_np < 0.1).sum() / probs_np.size * 100:.2f}%")
    print(f"  0.1-0.3: {((probs_np >= 0.1) & (probs_np < 0.3)).sum() / probs_np.size * 100:.2f}%")
    print(f"  0.3-0.5: {((probs_np >= 0.3) & (probs_np < 0.5)).sum() / probs_np.size * 100:.2f}%")
    print(f"  0.5-0.7: {((probs_np >= 0.5) & (probs_np < 0.7)).sum() / probs_np.size * 100:.2f}%")
    print(f"  0.7-0.9: {((probs_np >= 0.7) & (probs_np < 0.9)).sum() / probs_np.size * 100:.2f}%")
    print(f"  > 0.9: {(probs_np > 0.9).sum() / probs_np.size * 100:.2f}%")
