#!/usr/bin/env python3
"""
Quick evaluation of class distribution on validation set using existing model.
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import tifffile

sys.path.insert(0, str(Path(__file__).parent))

from nnunet_topo_wrapper import TopologyAwareUNet3D

def evaluate_model():
    """Load model and evaluate on validation data"""
    
    model_path = Path(__file__).parent / "checkpoints" / "fold_0" / "best_model.pth"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = TopologyAwareUNet3D(
        in_channels=1,
        out_channels=1,
        initial_filters=32,
        depth=4
    )
    model.to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ Model loaded successfully\n")
    
    # Load validation data
    data_root = Path(__file__).parent.parent
    
    # Find all image files for fold 0 validation
    train_images_dir = data_root / "train_images"
    if not train_images_dir.exists():
        print(f"❌ Data directory not found: {train_images_dir}")
        return None
    
    # Get list of images
    image_files = sorted(list(train_images_dir.glob("*.tif")))
    print(f"Found {len(image_files)} images")
    
    # Split into train/val (fold 0: ~80% train, ~20% val)
    # Assuming 5-fold CV means ~157 val images out of ~786 total
    val_start = int(len(image_files) * 0.8)
    val_images = image_files[val_start:]
    
    print(f"Using {len(val_images)} validation images (fold 0, indices {val_start}-{len(image_files)})\n")
    
    # Inference
    foreground_percentages = []
    
    print("Running inference...")
    with torch.no_grad():
        for img_path in tqdm(val_images[:50]):  # Limit to 50 for quick evaluation
            try:
                # Load image (TIFF format)
                img = tifffile.imread(img_path).astype(np.float32)
                
                # Normalize
                if img.max() > 0:
                    img = img / img.max()
                
                # Prepare for model (add batch and channel dims)
                img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
                
                # Predict
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    pred = model(img_tensor)
                
                # Sigmoid + threshold
                pred = torch.sigmoid(pred) > 0.5
                
                # Calculate foreground percentage
                fg_pct = (pred.sum().float() / pred.numel() * 100).item()
                foreground_percentages.append(fg_pct)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    if not foreground_percentages:
        print("❌ No predictions generated")
        return None
    
    # Analysis
    fg_array = np.array(foreground_percentages)
    bg_array = 100 - fg_array
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"\nForground Percentage (what we DON'T want too much of):")
    print(f"  Mean:     {fg_array.mean():.2f}%")
    print(f"  Median:   {np.median(fg_array):.2f}%")
    print(f"  Std:      {fg_array.std():.2f}%")
    print(f"  Range:    {fg_array.min():.2f}% - {fg_array.max():.2f}%")
    
    print(f"\nBackground Percentage (TARGET: 30-70%):")
    print(f"  Mean:     {bg_array.mean():.2f}%")
    print(f"  Median:   {np.median(bg_array):.2f}%")
    print(f"  Std:      {bg_array.std():.2f}%")
    print(f"  Range:    {bg_array.min():.2f}% - {bg_array.max():.2f}%")
    
    # Check target
    target_met = (30 <= bg_array.mean() <= 70)
    status = "✅ TARGET MET" if target_met else "❌ TARGET MISSED"
    print(f"\n{status}: Background mean = {bg_array.mean():.2f}% (need 30-70%)")
    
    # Summary
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    if fg_array.mean() > 95:
        print("⚠️  Model predicts >95% foreground (almost all positive)")
        print("    This is the ORIGINAL PROBLEM we were trying to fix!")
    elif 30 <= bg_array.mean() <= 70:
        print("✅ Model predictions show BALANCED class distribution")
        print("    This model is likely usable for submission!")
    elif bg_array.mean() < 30:
        print("⚠️  Model predicts >70% foreground (too much positive)")
        print("    Need more regularization or training adjustment")
    else:
        print("⚠️  Model predicts <30% foreground (too much negative)")
        print("    May need to reduce background weight")
    
    return {
        "foreground_mean": float(fg_array.mean()),
        "background_mean": float(bg_array.mean()),
        "target_met": bool(target_met),
        "model_path": str(model_path)
    }

if __name__ == "__main__":
    result = evaluate_model()
    if result:
        print(f"\nResult: {json.dumps(result, indent=2)}")
