 #!/usr/bin/env python3
"""
Evaluate the class distribution (foreground/background percentage) of model predictions.
This is the TRUE metric we care about - not the loss value.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from nnunet_wrapper import nnUNetWrapper
from data_loader import load_validation_data

def evaluate_class_distribution(model_path, val_data_dir, output_dir="analysis"):
    """
    Load model and evaluate foreground/background distribution on validation set.
    
    Args:
        model_path: Path to best_model.pth
        val_data_dir: Directory with validation data
        output_dir: Where to save analysis plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = nnUNetWrapper(
        in_channels=1,
        out_channels=1,
        initial_filters=32,
        depth=4,
        batch_size=2,
        learning_rate=0.0005
    )
    model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Load validation data
    print(f"Loading validation data from: {val_data_dir}")
    val_images, val_labels = load_validation_data(val_data_dir, fold=0)
    print(f"Loaded {len(val_images)} validation images")
    
    # Run inference and collect predictions
    foreground_percentages = []
    all_predictions = []
    
    with torch.no_grad():
        for idx, (img_path, label_path) in enumerate(tqdm(zip(val_images, val_labels), total=len(val_images))):
            # Load image
            img = np.load(img_path).astype(np.float32)
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict
            pred = model(img)
            pred = torch.sigmoid(pred)  # Convert to probability
            pred = pred > 0.5  # Threshold at 0.5
            
            # Calculate foreground percentage
            pred_np = pred.cpu().numpy().astype(np.uint8)
            foreground_pct = np.sum(pred_np) / pred_np.size * 100
            foreground_percentages.append(foreground_pct)
            all_predictions.append(pred_np)
    
    # Statistics
    fg_array = np.array(foreground_percentages)
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Total predictions: {len(fg_array)}")
    print(f"\nForeground percentage statistics:")
    print(f"  Mean:     {fg_array.mean():.2f}%")
    print(f"  Median:   {np.median(fg_array):.2f}%")
    print(f"  Std Dev:  {fg_array.std():.2f}%")
    print(f"  Min:      {fg_array.min():.2f}%")
    print(f"  Max:      {fg_array.max():.2f}%")
    
    print(f"\nBackground percentage statistics:")
    bg_array = 100 - fg_array
    print(f"  Mean:     {bg_array.mean():.2f}%")
    print(f"  Median:   {np.median(bg_array):.2f}%")
    print(f"  Std Dev:  {bg_array.std():.2f}%")
    print(f"  Min:      {bg_array.min():.2f}%")
    print(f"  Max:      {bg_array.max():.2f}%")
    
    # Check if it meets 30-70% goal
    target_met = (30 <= bg_array.mean() <= 70)
    print(f"\n{'✅' if target_met else '❌'} TARGET (30-70% background): {bg_array.mean():.2f}% background")
    
    # Save statistics
    stats = {
        "total_predictions": len(fg_array),
        "foreground": {
            "mean_pct": float(fg_array.mean()),
            "median_pct": float(np.median(fg_array)),
            "std_pct": float(fg_array.std()),
            "min_pct": float(fg_array.min()),
            "max_pct": float(fg_array.max())
        },
        "background": {
            "mean_pct": float(bg_array.mean()),
            "median_pct": float(np.median(bg_array)),
            "std_pct": float(bg_array.std()),
            "min_pct": float(bg_array.min()),
            "max_pct": float(bg_array.max())
        },
        "target_met": target_met,
        "model_path": str(model_path)
    }
    
    stats_path = output_dir / "class_distribution_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")
    
    # Create histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(fg_array, bins=30, edgecolor='black', color='coral', alpha=0.7)
    axes[0].axvline(fg_array.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {fg_array.mean():.1f}%")
    axes[0].set_xlabel("Foreground Percentage (%)")
    axes[0].set_ylabel("Number of Images")
    axes[0].set_title("Foreground Distribution (Target: 30-70%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(bg_array, bins=30, edgecolor='black', color='skyblue', alpha=0.7)
    axes[1].axvline(bg_array.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {bg_array.mean():.1f}%")
    axes[1].axvline(30, color='green', linestyle=':', linewidth=2, label="Target: 30-70%")
    axes[1].axvline(70, color='green', linestyle=':', linewidth=2)
    axes[1].set_xlabel("Background Percentage (%)")
    axes[1].set_ylabel("Number of Images")
    axes[1].set_title("Background Distribution (Target: 30-70%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "class_distribution_histogram.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    return stats, fg_array, bg_array

if __name__ == "__main__":
    model_path = Path(__file__).parent / "checkpoints" / "fold_0" / "best_model.pth"
    val_data_dir = Path(__file__).parent.parent / "data" / "train_images"
    
    stats, fg, bg = evaluate_class_distribution(
        str(model_path),
        str(val_data_dir),
        output_dir="analysis"
    )
