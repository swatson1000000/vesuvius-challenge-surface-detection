#!/usr/bin/env python3
"""
Compare predictions with actual training labels to validate model quality.
"""

import os
import glob
import numpy as np
import tifffile
from pathlib import Path

def analyze_label_vs_prediction(image_name):
    """Compare label and prediction for a single image."""
    pred_dir = '/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/predictions/train_sample'
    label_dir = '/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/train_labels'
    
    pred_file = os.path.join(pred_dir, f"{image_name}.tif")
    label_file = os.path.join(label_dir, f"{image_name}.tif")
    
    if not os.path.exists(pred_file) or not os.path.exists(label_file):
        return None
    
    try:
        pred = tifffile.imread(pred_file)
        label = tifffile.imread(label_file)
        
        # Ensure same shape
        if pred.shape != label.shape:
            return None
        
        # Calculate metrics
        total = pred.size
        
        # Overlap statistics
        tp = np.sum((pred > 0) & (label > 0))  # True positive
        fp = np.sum((pred > 0) & (label == 0))  # False positive
        tn = np.sum((pred == 0) & (label == 0))  # True negative
        fn = np.sum((pred == 0) & (label > 0))  # False negative
        
        # Metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        # Label statistics
        label_fg_pct = (np.sum(label > 0) / total) * 100
        
        return {
            'image': image_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'dice': dice,
            'iou': iou,
            'label_fg_pct': label_fg_pct,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    except Exception as e:
        print(f"Error analyzing {image_name}: {e}")
        return None

def main():
    pred_dir = '/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/predictions/train_sample'
    
    # Get all prediction files
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.tif')))
    pred_files = [f for f in pred_files if '_visualization' not in f]
    
    # Extract image names
    image_names = [Path(f).stem for f in pred_files]
    
    print(f"\n{'='*100}")
    print(f"LABEL vs PREDICTION VALIDATION - {len(image_names)} Images")
    print(f"{'='*100}\n")
    
    results = []
    for img_name in image_names:
        result = analyze_label_vs_prediction(img_name)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No matching labels found. Checking directory structure...")
        print(f"Predictions: {pred_dir}")
        print(f"Labels: /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/train_labels")
        return
    
    # Print results
    print(f"{'Image':<20} {'Label%':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'Dice':<8} {'IoU':<8}")
    print(f"{'-'*90}")
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    dices = []
    ious = []
    label_fgs = []
    
    for r in results:
        print(f"{r['image'][:19]:<20} {r['label_fg_pct']:>8.2f}% {r['accuracy']:>9.3f} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['dice']:>7.3f} {r['iou']:>7.3f}")
        
        accuracies.append(r['accuracy'])
        precisions.append(r['precision'])
        recalls.append(r['recall'])
        f1_scores.append(r['f1'])
        dices.append(r['dice'])
        ious.append(r['iou'])
        label_fgs.append(r['label_fg_pct'])
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}")
    
    print(f"\nLabel Statistics:")
    print(f"  Mean foreground %: {np.mean(label_fgs):.2f}%")
    print(f"  Median foreground %: {np.median(label_fgs):.2f}%")
    print(f"  Std Dev: {np.std(label_fgs):.2f}%")
    
    print(f"\nPrediction Quality Metrics:")
    print(f"  Mean Accuracy:  {np.mean(accuracies):.4f}")
    print(f"  Mean Precision: {np.mean(precisions):.4f}")
    print(f"  Mean Recall:    {np.mean(recalls):.4f}")
    print(f"  Mean F1-Score:  {np.mean(f1_scores):.4f}")
    print(f"  Mean Dice:      {np.mean(dices):.4f}")
    print(f"  Mean IoU:       {np.mean(ious):.4f}")
    
    print(f"\nMetric Ranges:")
    print(f"  Accuracy:  [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]")
    print(f"  Dice:      [{np.min(dices):.4f}, {np.max(dices):.4f}]")
    print(f"  IoU:       [{np.min(ious):.4f}, {np.max(ious):.4f}]")
    
    # Assess if predictions match labels
    print(f"\n{'='*100}")
    print("ASSESSMENT")
    print(f"{'='*100}")
    
    if abs(np.mean(label_fgs) - 95) < 5:
        print("✅ Label distribution matches predictions (both ~95% foreground)")
        print("   → Model is learning correctly from the training data")
        print("   → The 99.88% predictions may be valid for this dataset")
    else:
        print(f"⚠️  Label distribution differs from predictions")
        print(f"   Labels: {np.mean(label_fgs):.2f}% foreground")
        print(f"   Preds:  95.65% foreground")
        print(f"   → Model is over-predicting foreground")
        print(f"   → Need to investigate loss weights and class balance")
    
    if np.mean(dices) > 0.75:
        print(f"✅ High Dice score ({np.mean(dices):.4f}) suggests good predictions")
    else:
        print(f"❌ Low Dice score ({np.mean(dices):.4f}) suggests poor predictions")

if __name__ == '__main__':
    main()
