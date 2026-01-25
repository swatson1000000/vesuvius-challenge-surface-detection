#!/usr/bin/env python3
"""Analyze foreground/background statistics in prediction files."""

import os
import glob
import numpy as np
import tifffile
from pathlib import Path
import json
from datetime import datetime

def analyze_prediction(filepath):
    """Analyze a single prediction file."""
    try:
        # Load prediction
        pred = tifffile.imread(filepath)
        
        # Calculate statistics
        total_voxels = pred.size
        foreground_voxels = np.sum(pred > 0)
        background_voxels = total_voxels - foreground_voxels
        
        foreground_pct = (foreground_voxels / total_voxels) * 100
        background_pct = (background_voxels / total_voxels) * 100
        
        return {
            'filename': Path(filepath).name,
            'shape': pred.shape,
            'dtype': str(pred.dtype),
            'total_voxels': int(total_voxels),
            'foreground_voxels': int(foreground_voxels),
            'background_voxels': int(background_voxels),
            'foreground_pct': float(foreground_pct),
            'background_pct': float(background_pct),
            'min_value': int(pred.min()),
            'max_value': int(pred.max()),
        }
    except Exception as e:
        return {
            'filename': Path(filepath).name,
            'error': str(e)
        }

def main():
    pred_dir = '/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/predictions/train_sample'
    
    # Find all prediction files (excluding visualization)
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.tif')))
    pred_files = [f for f in pred_files if '_visualization' not in f]
    
    print(f"\n{'='*80}")
    print(f"FOREGROUND/BACKGROUND ANALYSIS - {len(pred_files)} Predictions")
    print(f"{'='*80}\n")
    
    results = []
    for filepath in pred_files:
        result = analyze_prediction(filepath)
        results.append(result)
    
    # Print detailed results
    print(f"{'Filename':<20} {'Foreground %':<15} {'Background %':<15} {'FG Voxels':<15}")
    print(f"{'-'*65}")
    
    foreground_pcts = []
    background_pcts = []
    
    for result in results:
        if 'error' not in result:
            fg_pct = result['foreground_pct']
            bg_pct = result['background_pct']
            fg_voxels = result['foreground_voxels']
            filename = result['filename'][:19]
            
            print(f"{filename:<20} {fg_pct:>13.2f}% {bg_pct:>13.2f}% {fg_voxels:>14,}")
            
            foreground_pcts.append(fg_pct)
            background_pcts.append(bg_pct)
    
    # Summary statistics
    if foreground_pcts:
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Total predictions analyzed: {len(foreground_pcts)}")
        print(f"\nForeground Coverage:")
        print(f"  Mean:    {np.mean(foreground_pcts):.2f}%")
        print(f"  Median:  {np.median(foreground_pcts):.2f}%")
        print(f"  Std Dev: {np.std(foreground_pcts):.2f}%")
        print(f"  Min:     {np.min(foreground_pcts):.2f}%")
        print(f"  Max:     {np.max(foreground_pcts):.2f}%")
        
        print(f"\nBackground Coverage:")
        print(f"  Mean:    {np.mean(background_pcts):.2f}%")
        print(f"  Median:  {np.median(background_pcts):.2f}%")
        print(f"  Std Dev: {np.std(background_pcts):.2f}%")
        print(f"  Min:     {np.min(background_pcts):.2f}%")
        print(f"  Max:     {np.max(background_pcts):.2f}%")
        
        # Distribution analysis
        print(f"\nForeground Distribution:")
        ranges = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100)]
        for low, high in ranges:
            count = sum(1 for pct in foreground_pcts if low <= pct < high)
            print(f"  {low:>2}% - {high:<2}%: {count:>2} predictions ({count/len(foreground_pcts)*100:.1f}%)")
        
        # Count extreme cases
        extreme_high = sum(1 for pct in foreground_pcts if pct > 95)
        extreme_low = sum(1 for pct in foreground_pcts if pct < 5)
        print(f"\nExtreme Cases:")
        print(f"  >95% foreground: {extreme_high} predictions ({extreme_high/len(foreground_pcts)*100:.1f}%)")
        print(f"  <5% foreground:  {extreme_low} predictions ({extreme_low/len(foreground_pcts)*100:.1f}%)")

if __name__ == '__main__':
    main()
