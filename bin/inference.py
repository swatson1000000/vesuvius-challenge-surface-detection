"""
Inference Script with Topology Post-Processing

Generates predictions for test set with topology correction
"""

import argparse
import sys
import torch
import numpy as np
import tifffile
from pathlib import Path
from typing import List, Tuple
import logging
from tqdm import tqdm
from datetime import datetime
import time

from nnunet_topo_wrapper import TopologyAwareUNet3D
from morphology_postprocess import TopologyPostProcessor


class InferencePipeline:
    """
    End-to-end inference pipeline with topology post-processing
    """
    
    def __init__(self,
                 model_path: str,
                 device: str = 'cuda',
                 patch_size: Tuple[int, int, int] = (128, 128, 128),
                 overlap: int = 32,
                 postprocess_config: dict = None):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            patch_size: Size of patches for inference
            overlap: Overlap between patches for stitching
            postprocess_config: Configuration for post-processing
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup post-processor
        postprocess_config = postprocess_config or {}
        self.postprocessor = TopologyPostProcessor(**postprocess_config)
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model (should match training configuration)
        model = TopologyAwareUNet3D(
            in_channels=1,
            out_channels=1,
            initial_filters=32,
            depth=5
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        self.logger.info(f"Loaded model from {model_path}")
        self.logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        self.logger.info(f"  Best val score: {checkpoint.get('best_val_score', 'N/A'):.4f}")
        
        return model
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image"""
        # Clip extremes
        p1, p99 = np.percentile(image, (0.5, 99.5))
        image = np.clip(image, p1, p99)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image.astype(np.float32)
    
    def extract_patches(self, volume: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:
        """
        Extract overlapping patches from volume
        
        Returns:
            patches, positions
        """
        d, h, w = volume.shape
        pd, ph, pw = self.patch_size
        stride = [p - self.overlap for p in self.patch_size]
        
        patches = []
        positions = []
        
        for d_start in range(0, d - pd + 1, stride[0]):
            for h_start in range(0, h - ph + 1, stride[1]):
                for w_start in range(0, w - pw + 1, stride[2]):
                    patch = volume[
                        d_start:d_start+pd,
                        h_start:h_start+ph,
                        w_start:w_start+pw
                    ]
                    
                    patches.append(patch)
                    positions.append((d_start, h_start, w_start))
        
        # Handle edges if volume is not evenly divisible
        # (simplified - in production, handle this more carefully)
        
        return patches, positions
    
    def stitch_patches(self, 
                      patches: List[np.ndarray],
                      positions: List[Tuple],
                      volume_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Stitch patches back into full volume with weighted averaging
        """
        d, h, w = volume_shape
        pd, ph, pw = self.patch_size
        
        # Output volume and weight map
        output = np.zeros((d, h, w), dtype=np.float32)
        weights = np.zeros((d, h, w), dtype=np.float32)
        
        # Create Gaussian weight map for blending
        weight_map = self._create_gaussian_weight_map(self.patch_size)
        
        for patch, (d_start, h_start, w_start) in zip(patches, positions):
            # Add patch with weights
            output[
                d_start:d_start+pd,
                h_start:h_start+ph,
                w_start:w_start+pw
            ] += patch * weight_map
            
            weights[
                d_start:d_start+pd,
                h_start:h_start+ph,
                w_start:w_start+pw
            ] += weight_map
        
        # Normalize by weights
        output = output / (weights + 1e-8)
        
        return output
    
    def _create_gaussian_weight_map(self, size: Tuple[int, int, int]) -> np.ndarray:
        """Create Gaussian weight map for patch blending"""
        from scipy.ndimage import gaussian_filter
        
        # Create centered weight map
        weight = np.ones(size, dtype=np.float32)
        
        # Fade edges
        sigma = [s / 8 for s in size]  # Adjust for desired fade width
        
        # Create distance from edge map
        d, h, w = size
        dd = np.minimum(np.arange(d), np.arange(d)[::-1])
        hh = np.minimum(np.arange(h), np.arange(h)[::-1])
        ww = np.minimum(np.arange(w), np.arange(w)[::-1])
        
        dd = dd[:, None, None]
        hh = hh[None, :, None]
        ww = ww[None, None, :]
        
        # Normalize
        dd = dd / (d / 2)
        hh = hh / (h / 2)
        ww = ww / (w / 2)
        
        # Combine
        weight = np.minimum(np.minimum(dd, hh), ww)
        weight = np.clip(weight, 0, 1)
        
        return weight
    
    @torch.no_grad()
    def predict_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Predict segmentation for full volume
        
        Args:
            volume: Input volume (D, H, W)
            
        Returns:
            Binary prediction mask (D, H, W)
        """
        self.logger.info(f"Predicting volume of shape {volume.shape}")
        sys.stdout.flush()
        
        # Normalize
        volume_norm = self.normalize(volume)
        
        # Extract patches
        self.logger.info("Extracting patches...")
        patches, positions = self.extract_patches(volume_norm)
        self.logger.info(f"  Extracted {len(patches)} patches")
        sys.stdout.flush()
        
        # Predict patches
        self.logger.info("Running model inference...")
        sys.stdout.flush()
        pred_patches = []
        
        for patch in tqdm(patches, desc="Predicting"):
            # Add batch and channel dimensions
            patch_tensor = torch.from_numpy(patch[None, None, ...]).float().to(self.device)
            
            # Predict
            pred = self.model(patch_tensor)
            
            # Remove batch and channel dimensions
            pred = pred[0, 0].cpu().numpy()
            pred_patches.append(pred)
        
        # Stitch patches
        self.logger.info("Stitching patches...")
        pred_volume = self.stitch_patches(pred_patches, positions, volume.shape)
        sys.stdout.flush()
        
        # Post-process
        self.logger.info("Applying post-processing...")
        binary_mask = self.postprocessor.process(pred_volume)
        sys.stdout.flush()
        
        # Compute stats
        stats = self.postprocessor.compute_topology_stats(binary_mask)
        self.logger.info(f"Prediction stats:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
        sys.stdout.flush()
        
        return binary_mask
    
    def predict_and_save(self, 
                        input_path: Path,
                        output_path: Path):
        """
        Load image, predict, and save result
        
        Args:
            input_path: Path to input TIFF
            output_path: Path to save output TIFF
        """
        self.logger.info(f"Processing {input_path.name}")
        
        # Load image
        volume = tifffile.imread(input_path)
        self.logger.info(f"  Loaded volume: {volume.shape}, dtype: {volume.dtype}")
        
        # Predict
        prediction = self.predict_volume(volume)
        
        # Save binary version (0,1) for submission
        output_path.parent.mkdir(exist_ok=True, parents=True)
        tifffile.imwrite(output_path, prediction.astype(np.uint8))
        self.logger.info(f"  Saved binary prediction (0,1) to {output_path}")
        
        # Save visualization version (0,255) for viewing
        vis_path = output_path.parent / f"{output_path.stem}_visualization{output_path.suffix}"
        prediction_vis = (prediction * 255).astype(np.uint8)
        tifffile.imwrite(vis_path, prediction_vis)
        self.logger.info(f"  Saved visualization (0,255) to {vis_path}")
        sys.stdout.flush()


def main(args):
    """Main inference function"""
    
    # Record start time
    start_time = datetime.now()
    start_timestamp = time.time()
    
    # Setup logging with flush enabled for incremental display
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    
    # Ensure unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info(f"Inference started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    sys.stdout.flush()
    
    # Post-processing configuration
    postprocess_config = {
        'threshold': args.threshold,
        'min_component_size': args.min_component_size,
        'max_hole_size': args.max_hole_size,
        'morphology_kernel_size': args.kernel_size,
        'separate_instances': args.separate_instances,
        'verbose': True
    }
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        model_path=args.checkpoint,
        device=args.device,
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        postprocess_config=postprocess_config
    )
    
    # Find input files
    input_dir = Path(args.input)
    if input_dir.is_dir():
        input_files = sorted(list(input_dir.glob("*.tif")))
    else:
        input_files = [input_dir]
    
    logger.info(f"Found {len(input_files)} files to process")
    sys.stdout.flush()
    
    # Process each file
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for input_path in input_files:
        output_path = output_dir / input_path.name
        
        try:
            pipeline.predict_and_save(input_path, output_path)
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error processing {input_path.name}: {e}")
            sys.stdout.flush()
            continue
    
    # Record end time
    end_time = datetime.now()
    end_timestamp = time.time()
    elapsed_time = end_timestamp - start_timestamp
    
    logger.info("="*80)
    logger.info("Inference complete!")
    logger.info(f"Start time:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time:   {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
    logger.info("="*80)
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with topology post-processing")
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    # Input/output arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input directory or file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output directory')
    
    # Inference arguments
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128],
                       help='Patch size for inference')
    parser.add_argument('--overlap', type=int, default=32,
                       help='Overlap between patches')
    
    # Post-processing arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold')
    parser.add_argument('--min_component_size', type=int, default=100,
                       help='Minimum component size (voxels)')
    parser.add_argument('--max_hole_size', type=int, default=50,
                       help='Maximum hole size to fill (voxels)')
    parser.add_argument('--kernel_size', type=int, default=2,
                       help='Morphological kernel size')
    parser.add_argument('--separate_instances', action='store_true',
                       help='Apply instance separation')
    
    args = parser.parse_args()
    main(args)
