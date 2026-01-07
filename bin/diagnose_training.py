"""
Diagnostic script to analyze training issues
"""

import torch
import numpy as np
import tifffile
from pathlib import Path
from collections import Counter

def analyze_dataset():
    """Analyze the training dataset for potential issues"""
    
    train_labels_dir = Path("/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/train_labels")
    label_files = sorted(list(train_labels_dir.glob("*.tif")))
    
    print(f"Analyzing {len(label_files)} label files...")
    print("="*80)
    
    all_stats = []
    value_counts = Counter()
    
    # Sample 50 files for quick analysis
    sample_files = label_files[::len(label_files)//50] if len(label_files) > 50 else label_files
    
    for i, label_file in enumerate(sample_files):
        label = tifffile.imread(label_file)
        
        # Convert to binary as the training code does
        label_binary = (label > 0).astype(np.uint8)
        
        fg_count = np.sum(label_binary == 1)
        bg_count = np.sum(label_binary == 0)
        fg_ratio = fg_count / label.size
        
        all_stats.append({
            'file': label_file.name,
            'shape': label.shape,
            'fg_ratio': fg_ratio,
            'unique_values': np.unique(label).tolist()
        })
        
        if i < 5:  # Print first 5 in detail
            print(f"\nFile: {label_file.name}")
            print(f"  Shape: {label.shape}")
            print(f"  Original unique values: {np.unique(label)}")
            print(f"  After binarization: {np.unique(label_binary)}")
            print(f"  Foreground: {fg_count:,} ({fg_ratio*100:.2f}%)")
            print(f"  Background: {bg_count:,} ({(1-fg_ratio)*100:.2f}%)")
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    fg_ratios = [s['fg_ratio'] for s in all_stats]
    print(f"Average foreground ratio: {np.mean(fg_ratios)*100:.2f}%")
    print(f"Median foreground ratio: {np.median(fg_ratios)*100:.2f}%")
    print(f"Min foreground ratio: {np.min(fg_ratios)*100:.2f}%")
    print(f"Max foreground ratio: {np.max(fg_ratios)*100:.2f}%")
    print(f"Std dev: {np.std(fg_ratios)*100:.2f}%")
    
    # Check if data is too imbalanced
    avg_fg = np.mean(fg_ratios)
    if avg_fg > 0.7:
        print(f"\n⚠️  WARNING: Dataset is highly imbalanced (75%+ foreground)")
        print("   This can cause training issues. Consider:")
        print("   1. Using weighted loss functions")
        print("   2. Focal loss with proper gamma")
        print("   3. Sampling strategies to balance classes")
    
    if avg_fg < 0.05:
        print(f"\n⚠️  WARNING: Dataset has very sparse labels (<5% foreground)")
        print("   This is extremely challenging. Consider:")
        print("   1. Cropping to regions with more foreground")
        print("   2. Using weighted sampling")
        print("   3. Hard negative mining")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if 0.4 < avg_fg < 0.6:
        print("✓ Dataset appears reasonably balanced")
    else:
        print("Current issues detected:")
        print(f"  - Foreground ratio: {avg_fg*100:.1f}% (target: 40-60%)")
        print("\nSuggested fixes:")
        print("  1. Use focal loss with gamma=2.0 (emphasizes hard examples)")
        print("  2. Use class weights inversely proportional to frequency")
        print("  3. Sample patches with balanced foreground/background")
        print("  4. Increase learning rate to 0.01 (currently might be stuck)")
        print("  5. Add learning rate warmup for first few epochs")
        print("  6. Check if model is actually updating (gradient flow)")

if __name__ == "__main__":
    analyze_dataset()
