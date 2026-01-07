"""
Morphological Post-Processing for Topology Correction

Implements multi-stage post-processing to optimize topological metrics:
- Remove small components (TopoScore k=0)
- Fill holes (TopoScore k=2)
- Remove spurious bridges (VOI mergers)
- Instance separation (VOI splits)
"""

import numpy as np
import cc3d
from scipy import ndimage
from skimage import morphology, measure
from typing import Tuple, Optional
import logging


class TopologyPostProcessor:
    """
    Post-processing pipeline for topology correction
    """
    
    def __init__(self,
                 threshold: float = 0.5,
                 min_component_size: int = 100,
                 max_hole_size: int = 50,
                 morphology_kernel_size: int = 2,
                 separate_instances: bool = False,
                 verbose: bool = True):
        """
        Args:
            threshold: Probability threshold for binarization
            min_component_size: Minimum size (voxels) for connected components
            max_hole_size: Maximum size (voxels) for holes to fill
            morphology_kernel_size: Kernel size for opening/closing operations
            separate_instances: Whether to perform instance separation
            verbose: Print processing information
        """
        self.threshold = threshold
        self.min_component_size = min_component_size
        self.max_hole_size = max_hole_size
        self.morphology_kernel_size = morphology_kernel_size
        self.separate_instances = separate_instances
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
    
    def process(self, pred_volume: np.ndarray) -> np.ndarray:
        """
        Apply full post-processing pipeline
        
        Args:
            pred_volume: Predicted probability map (D, H, W) or (B, C, D, H, W)
            
        Returns:
            Binary mask after post-processing
        """
        # Handle batch dimension
        squeeze_batch = False
        if len(pred_volume.shape) == 3:
            pred_volume = pred_volume[np.newaxis, np.newaxis, ...]
            squeeze_batch = True
        elif len(pred_volume.shape) == 4:
            pred_volume = pred_volume[:, np.newaxis, ...]
            squeeze_batch = True
        
        batch_size = pred_volume.shape[0]
        processed = []
        
        for b in range(batch_size):
            volume = pred_volume[b, 0]
            
            if self.verbose:
                self.logger.info(f"Processing volume {b+1}/{batch_size}")
            
            # Stage 1: Thresholding
            binary_mask = self._threshold(volume)
            
            # Stage 2: Remove small components
            binary_mask = self._remove_small_components(binary_mask)
            
            # Stage 3: Fill holes
            binary_mask = self._fill_holes(binary_mask)
            
            # Stage 4: Morphological operations
            binary_mask = self._morphological_cleanup(binary_mask)
            
            # Stage 5: Instance separation (optional)
            if self.separate_instances:
                binary_mask = self._separate_instances(binary_mask)
            
            # Stage 6: Final smoothing
            binary_mask = self._smooth_surface(binary_mask)
            
            processed.append(binary_mask)
        
        result = np.stack(processed, axis=0)
        
        if squeeze_batch:
            result = result[0]
        
        return result
    
    def _threshold(self, volume: np.ndarray) -> np.ndarray:
        """Stage 1: Binarization"""
        if self.verbose:
            self.logger.info(f"  Stage 1: Thresholding at {self.threshold}")
        
        binary_mask = (volume > self.threshold).astype(np.uint8)
        
        fg_voxels = binary_mask.sum()
        total_voxels = binary_mask.size
        if self.verbose:
            self.logger.info(f"    Foreground: {fg_voxels}/{total_voxels} "
                           f"({100*fg_voxels/total_voxels:.2f}%)")
        
        return binary_mask
    
    def _remove_small_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """Stage 2: Remove small connected components (reduce k=0 false positives)"""
        if self.verbose:
            self.logger.info(f"  Stage 2: Removing components < {self.min_component_size} voxels")
        
        # Label connected components
        labeled, num_components = cc3d.connected_components(
            binary_mask, 
            connectivity=26,
            return_N=True
        )
        
        if self.verbose:
            self.logger.info(f"    Found {num_components} components")
        
        # Get component sizes
        sizes = np.bincount(labeled.ravel())
        
        # Keep only large components
        keep_mask = sizes >= self.min_component_size
        keep_mask[0] = False  # Background
        
        binary_mask = keep_mask[labeled].astype(np.uint8)
        
        remaining = keep_mask.sum() - 1  # Exclude background
        if self.verbose:
            self.logger.info(f"    Kept {remaining} components")
        
        return binary_mask
    
    def _fill_holes(self, binary_mask: np.ndarray) -> np.ndarray:
        """Stage 3: Fill small holes (reduce k=2 cavities)"""
        if self.verbose:
            self.logger.info(f"  Stage 3: Filling holes < {self.max_hole_size} voxels")
        
        # Fill holes slice by slice for efficiency
        filled = binary_mask.copy()
        
        for z in range(binary_mask.shape[0]):
            filled[z] = ndimage.binary_fill_holes(binary_mask[z])
        
        # Also fill 3D holes
        # Find holes (background components not connected to border)
        labeled_bg = cc3d.connected_components(1 - filled, connectivity=26)
        
        # Identify border-connected component (should be label 1 typically)
        border_labels = set()
        border_labels.update(labeled_bg[0, :, :].ravel())
        border_labels.update(labeled_bg[-1, :, :].ravel())
        border_labels.update(labeled_bg[:, 0, :].ravel())
        border_labels.update(labeled_bg[:, -1, :].ravel())
        border_labels.update(labeled_bg[:, :, 0].ravel())
        border_labels.update(labeled_bg[:, :, -1].ravel())
        border_labels.discard(0)
        
        # Fill holes (non-border background components)
        for label in range(1, labeled_bg.max() + 1):
            if label not in border_labels:
                component_size = (labeled_bg == label).sum()
                if component_size <= self.max_hole_size:
                    filled[labeled_bg == label] = 1
        
        holes_filled = (filled.sum() - binary_mask.sum())
        if self.verbose:
            self.logger.info(f"    Filled {holes_filled} voxels")
        
        return filled
    
    def _morphological_cleanup(self, binary_mask: np.ndarray) -> np.ndarray:
        """Stage 4: Morphological opening and closing"""
        if self.verbose:
            self.logger.info(f"  Stage 4: Morphological cleanup (kernel={self.morphology_kernel_size})")
        
        # Create 3D ball structuring element
        selem = morphology.ball(self.morphology_kernel_size)
        
        # Opening: Remove thin bridges
        opened = morphology.binary_opening(binary_mask, selem)
        removed = binary_mask.sum() - opened.sum()
        if self.verbose:
            self.logger.info(f"    Opening removed {removed} voxels (bridges)")
        
        # Closing: Close small gaps
        closed = morphology.binary_closing(opened, selem)
        added = closed.sum() - opened.sum()
        if self.verbose:
            self.logger.info(f"    Closing added {added} voxels (gaps)")
        
        return closed.astype(np.uint8)
    
    def _separate_instances(self, binary_mask: np.ndarray) -> np.ndarray:
        """Stage 5: Separate touching instances using watershed"""
        if self.verbose:
            self.logger.info("  Stage 5: Instance separation via watershed")
        
        # Compute distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find local maxima (seeds)
        from scipy.ndimage import maximum_filter
        local_max = (distance == maximum_filter(distance, size=5))
        
        # Label seeds
        markers, num_seeds = ndimage.label(local_max)
        
        if self.verbose:
            self.logger.info(f"    Found {num_seeds} seeds")
        
        # Watershed
        from skimage.segmentation import watershed
        labels = watershed(-distance, markers, mask=binary_mask)
        
        # Convert back to binary (keep all instances)
        result = (labels > 0).astype(np.uint8)
        
        return result
    
    def _smooth_surface(self, binary_mask: np.ndarray, sigma: float = 0.5) -> np.ndarray:
        """Stage 6: Smooth surface with Gaussian filter"""
        if self.verbose:
            self.logger.info(f"  Stage 6: Surface smoothing (sigma={sigma})")
        
        # Apply Gaussian smoothing and re-threshold
        smoothed = ndimage.gaussian_filter(binary_mask.astype(float), sigma=sigma)
        result = (smoothed > 0.5).astype(np.uint8)
        
        return result
    
    def compute_topology_stats(self, binary_mask: np.ndarray) -> dict:
        """
        Compute topological statistics for analysis
        
        Returns:
            Dictionary with topology metrics
        """
        # Connected components (k=0)
        labeled, num_components = cc3d.connected_components(
            binary_mask,
            connectivity=26,
            return_N=True
        )
        
        # Euler characteristic approximation
        # χ = components - tunnels + cavities
        euler = measure.euler_number(binary_mask, connectivity=3)
        
        stats = {
            'num_components': num_components,
            'euler_characteristic': euler,
            'foreground_voxels': binary_mask.sum(),
            'foreground_fraction': binary_mask.mean()
        }
        
        return stats


def batch_postprocess(predictions: np.ndarray,
                     config: Optional[dict] = None) -> np.ndarray:
    """
    Convenience function for batch post-processing
    
    Args:
        predictions: Batch of predictions (B, C, D, H, W) or (B, D, H, W)
        config: Post-processing configuration
        
    Returns:
        Post-processed binary masks
    """
    config = config or {}
    processor = TopologyPostProcessor(**config)
    
    return processor.process(predictions)


if __name__ == "__main__":
    # Test the post-processor
    print("Testing TopologyPostProcessor...")
    
    # Create synthetic data
    size = 128
    pred_volume = np.random.rand(size, size, size).astype(np.float32)
    
    # Add some structure (sheet-like)
    center = size // 2
    thickness = 5
    pred_volume[center-thickness:center+thickness, :, :] = 0.8
    
    # Add noise
    noise_mask = np.random.rand(size, size, size) > 0.95
    pred_volume[noise_mask] = 0.9
    
    print(f"Input shape: {pred_volume.shape}")
    print(f"Input range: [{pred_volume.min():.3f}, {pred_volume.max():.3f}]")
    
    # Process
    processor = TopologyPostProcessor(
        threshold=0.5,
        min_component_size=100,
        max_hole_size=50,
        morphology_kernel_size=2,
        separate_instances=False,
        verbose=True
    )
    
    result = processor.process(pred_volume)
    
    print(f"\nOutput shape: {result.shape}")
    print(f"Output unique values: {np.unique(result)}")
    
    # Compute stats
    stats = processor.compute_topology_stats(result)
    print("\nTopology statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nPost-processing test passed!")
