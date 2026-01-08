"""
Training Script for Topology-Aware nnU-Net

Main training script that ties everything together
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import tifffile
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import KFold
from datetime import datetime
import time

from nnunet_topo_wrapper import create_model_and_trainer, TopologyAwareTrainer
from topology_losses import CombinedTopologyLoss
from morphology_postprocess import TopologyPostProcessor


class VesuviusDataset(Dataset):
    """
    Dataset for Vesuvius Challenge
    """
    
    def __init__(self, 
                 image_paths: List[Path],
                 label_paths: List[Path],
                 patch_size: Tuple[int, int, int] = (128, 128, 128),
                 augment: bool = False):
        """
        Args:
            image_paths: List of paths to image TIFF files
            label_paths: List of paths to label TIFF files
            patch_size: Size of patches to extract
            augment: Whether to apply augmentations
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.augment = augment
        
        assert len(image_paths) == len(label_paths)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and label
        image = tifffile.imread(self.image_paths[idx])
        label = tifffile.imread(self.label_paths[idx])
        
        # Normalize image
        image = self._normalize(image)
        
        # Ensure binary label
        label = (label > 0).astype(np.float32)
        
        # Extract patch if needed
        if image.shape != self.patch_size:
            image, label = self._extract_patch(image, label)
        
        # Augment
        if self.augment:
            image, label = self._augment(image, label)
        
        # Add channel dimension
        image = image[np.newaxis, ...]
        label = label[np.newaxis, ...]
        
        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label).float(),
            'id': self.image_paths[idx].stem
        }
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1]"""
        # Clip extremes
        p1, p99 = np.percentile(image, (0.5, 99.5))
        image = np.clip(image, p1, p99)
        
        # Normalize
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image.astype(np.float32)
    
    def _extract_patch(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract patch with balanced foreground/background sampling"""
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # Try multiple times to get a balanced patch (if training/augmenting)
        max_attempts = 10 if self.augment else 1
        best_patch = None
        best_balance_score = float('inf')
        
        for _ in range(max_attempts):
            # Random starting position
            d_start = np.random.randint(0, max(1, d - pd))
            h_start = np.random.randint(0, max(1, h - ph))
            w_start = np.random.randint(0, max(1, w - pw))
            
            # Extract patch
            label_patch = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            
            # Score patch balance (prefer 30-70% foreground)
            fg_ratio = label_patch.mean()
            balance_score = abs(fg_ratio - 0.5)  # Lower is better
            
            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_patch = (d_start, h_start, w_start)
                
                # If we found a reasonably balanced patch, use it
                if 0.3 <= fg_ratio <= 0.7:
                    break
        
        # Extract best patch
        d_start, h_start, w_start = best_patch
        image_patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label_patch = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        # Pad if necessary
        if image_patch.shape != self.patch_size:
            image_patch = self._pad_to_size(image_patch, self.patch_size)
            label_patch = self._pad_to_size(label_patch, self.patch_size)
        
        return image_patch, label_patch
    
    def _pad_to_size(self, array: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Pad array to target size"""
        padding = []
        for i in range(3):
            diff = target_size[i] - array.shape[i]
            padding.append((0, max(0, diff)))
        
        return np.pad(array, padding, mode='constant', constant_values=0)
    
    def _augment(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations"""
        # Random flip
        if np.random.rand() > 0.5:
            axis = np.random.choice([0, 1, 2])
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        
        # Random rotation (90 degrees)
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])
            axes = np.random.choice(3)  # Choose which plane to rotate in: 0=(0,1), 1=(0,2), 2=(1,2)
            axes_pairs = [(0, 1), (0, 2), (1, 2)]
            axes = axes_pairs[axes]
            image = np.rot90(image, k=k, axes=axes).copy()
            label = np.rot90(label, k=k, axes=axes).copy()
        
        # Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        # Brightness/contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-0.1, 0.1)  # Brightness
            image = np.clip(alpha * image + beta, 0, 1)
        
        return image, label


def load_dataset(data_dir: Path, fold: int = 0, n_folds: int = 5) -> Tuple[List, List]:
    """
    Load dataset and split into train/val folds
    
    Args:
        data_dir: Root directory containing train_images/ and train_labels/
        fold: Which fold to use for validation
        n_folds: Total number of folds
        
    Returns:
        train_paths, val_paths (each is tuple of image_paths, label_paths)
    """
    image_dir = data_dir / "train_images"
    label_dir = data_dir / "train_labels"
    
    # Get all image files
    image_paths = sorted(list(image_dir.glob("*.tif")))
    label_paths = [label_dir / f"{img.stem}.tif" for img in image_paths]
    
    # Verify all labels exist
    label_paths = [lbl for lbl in label_paths if lbl.exists()]
    image_paths = [image_dir / f"{lbl.stem}.tif" for lbl in label_paths]
    
    print(f"Found {len(image_paths)} image-label pairs")
    
    # K-fold split
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(image_paths))
    
    train_idx, val_idx = splits[fold]
    
    train_images = [image_paths[i] for i in train_idx]
    train_labels = [label_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_labels = [label_paths[i] for i in val_idx]
    
    print(f"Fold {fold}: {len(train_images)} train, {len(val_images)} val")
    
    return (train_images, train_labels), (val_images, val_labels)


def main(args):
    """Main training function"""
    
    # Record start time
    start_time = datetime.now()
    start_timestamp = time.time()
    
    # Setup logging with flush enabled for incremental display
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_fold{args.fold}.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    # Ensure unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration: {config}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    data_dir = Path(args.data_dir)
    (train_images, train_labels), (val_images, val_labels) = load_dataset(
        data_dir, 
        fold=args.fold, 
        n_folds=config.get('n_folds', 5)
    )
    
    # Create datasets
    train_dataset = VesuviusDataset(
        train_images,
        train_labels,
        patch_size=tuple(config.get('patch_size', [128, 128, 128])),
        augment=True
    )
    
    val_dataset = VesuviusDataset(
        val_images,
        val_labels,
        patch_size=tuple(config.get('patch_size', [128, 128, 128])),
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 2),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 2),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model and trainer
    model_config = {
        'in_channels': 1,
        'out_channels': 1,
        'initial_filters': config.get('initial_filters', 32),
        'depth': config.get('depth', 5),
        'learning_rate': config.get('learning_rate', 1e-3),
        'weight_decay': config.get('weight_decay', 1e-5),
        'warmup_epochs': config.get('warmup_epochs', 0),
        'checkpoint_dir': f"checkpoints/fold_{args.fold}",
        **config.get('loss_weights', {})
    }
    
    model, trainer = create_model_and_trainer(model_config)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='min',
        factor=config.get('scheduler_factor', 0.5),
        patience=config.get('scheduler_patience', 10)
    )
    
    # Train
    logger.info("Starting training...")
    sys.stdout.flush()
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 300),
        scheduler=scheduler,
        early_stopping_patience=config.get('early_stopping_patience', 50)
    )
    
    # Record end time
    end_time = datetime.now()
    end_timestamp = time.time()
    elapsed_time = end_timestamp - start_timestamp
    
    logger.info("="*80)
    logger.info("Training complete!")
    logger.info(f"Start time:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time:   {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
    logger.info("="*80)
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train topology-aware model")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection',
                       help='Path to data directory')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number for cross-validation')
    
    args = parser.parse_args()
    main(args)
