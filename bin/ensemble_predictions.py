#!/usr/bin/env python3
"""Create ensemble predictions by averaging all 5 folds."""

import numpy as np
import tifffile
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
log_dir = Path("../logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = log_dir / f"ensemble_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_ensemble():
    """Average predictions from all 5 folds."""
    
    logger.info("=" * 80)
    logger.info("Creating ensemble predictions (5-fold average)")
    logger.info("=" * 80)
    
    predictions_dirs = [
        Path("../predictions"),           # Fold 4
        Path("../predictions_fold_3"),
        Path("../predictions_fold_0"),
        Path("../predictions_fold_2"),
        Path("../predictions_fold_1"),
    ]
    
    # Find all test images from first fold
    test_images = sorted(predictions_dirs[0].glob("*.tif"))
    test_images = [f for f in test_images if "_visualization" not in f.name]
    
    logger.info(f"Found {len(test_images)} test images")
    
    for image_path in test_images:
        image_path = Path(image_path)
        image_name = image_path.name
        logger.info(f"\nProcessing {image_name}")
        
        # Load predictions from all folds
        fold_predictions = []
        for fold_dir in predictions_dirs:
            pred_path = fold_dir / image_name
            if pred_path.exists():
                pred = tifffile.imread(pred_path).astype(np.float32)
                fold_predictions.append(pred)
                logger.info(f"  Loaded from {fold_dir.name}: shape {pred.shape}, dtype {pred.dtype}")
            else:
                logger.warning(f"  Missing from {fold_dir.name}")
        
        if len(fold_predictions) < 5:
            logger.warning(f"  Skipping {image_name} - only {len(fold_predictions)}/5 folds available")
            continue
        
        # Average predictions
        ensemble_pred = np.stack(fold_predictions, axis=0).mean(axis=0)
        
        # Save soft predictions (0-1 range) for reference
        stem = Path(image_name).stem
        soft_path = Path("../predictions") / f"{stem}_soft_ensemble.tif"
        tifffile.imwrite(soft_path, (ensemble_pred * 255).astype(np.uint8))
        logger.info(f"  Saved soft ensemble: {soft_path}")
        
        # Apply threshold to get binary prediction
        threshold = 0.5
        binary_pred = (ensemble_pred >= threshold).astype(np.uint8)
        
        # Overwrite binary prediction with ensemble
        ensemble_binary_path = Path("../predictions") / f"{stem}_ensemble.tif"
        tifffile.imwrite(ensemble_binary_path, binary_pred)
        logger.info(f"  Saved binary ensemble (threshold={threshold}): {ensemble_binary_path}")
        
        # Stats
        num_comps = np.unique(binary_pred[binary_pred > 0]).size
        foreground_pct = (binary_pred > 0).sum() / binary_pred.size * 100
        logger.info(f"  Ensemble stats: {foreground_pct:.2f}% foreground")
    
    logger.info("\n" + "=" * 80)
    logger.info("Ensemble creation complete!")
    logger.info("=" * 80)

if __name__ == "__main__":
    create_ensemble()
