"""Visualize predictions to understand what model learned"""
import torch
import numpy as np
import tifffile
from nnunet_topo_wrapper import TopologyAwareUNet3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Normalize
p1, p99 = np.percentile(image, (0.5, 99.5))
image = np.clip(image, p1, p99)
image = (image - image.min()) / (image.max() - image.min() + 1e-8)
image = image.astype(np.float32)

# Take center slice
d, h, w = image.shape
center_slice = image[d//2, :, :]

# Predict on the slice (with some depth context)
patch_depth = 32
d_start = d//2 - patch_depth//2
patch = image[d_start:d_start+patch_depth, :, :]

with torch.no_grad():
    patch_tensor = torch.from_numpy(patch[None, None, ...]).float().to(device)
    logits = model(patch_tensor)
    probs = torch.sigmoid(logits)
    
    # Get center slice from prediction
    pred_slice = probs[0, 0, patch_depth//2, :, :].cpu().numpy()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(center_slice, cmap='gray')
axes[0].set_title('Input Image (center slice)')
axes[0].axis('off')

axes[1].imshow(pred_slice, cmap='jet', vmin=0, vmax=1)
axes[1].set_title('Model Prediction (probabilities)')
axes[1].colorbar = plt.colorbar(axes[1].images[0], ax=axes[1])
axes[1].axis('off')

axes[2].imshow(pred_slice > 0.5, cmap='gray')
axes[2].set_title('Binary Mask (threshold=0.5)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('../predictions/visualization.png', dpi=150, bbox_inches='tight')
print("Saved visualization to ../predictions/visualization.png")

# Print statistics
print(f"\nPrediction statistics for center slice:")
print(f"  Mean: {pred_slice.mean():.4f}")
print(f"  Std: {pred_slice.std():.4f}")
print(f"  Min: {pred_slice.min():.4f}")
print(f"  Max: {pred_slice.max():.4f}")
print(f"  Median: {np.median(pred_slice):.4f}")
print(f"  Pixels > 0.5: {(pred_slice > 0.5).sum() / pred_slice.size * 100:.2f}%")
