"""
Batch Inference on Training Images

Runs inference on multiple training images for validation
"""

import argparse
import sys
import torch
import numpy as np
import tifffile
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm
from datetime import datetime
import time

from nnunet_topo_wrapper import TopologyAwareUNet3D
from morphology_postprocess import TopologyPostProcessor


class BatchInferencePipeline:
    """
    Batch inference pipeline for multiple images
    """
    
    def __init__(self,
                 model_path: str,
                 device: str = 'cuda',
                 patch_size: tuple = (128, 128, 128),
                 overlap: int = 32):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            patch_size: Size of patches for inference
            overlap: Overlap between patches for stitching
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        self.model = TopologyAwareUNet3D(in_channels=1, out_channels=1, 
                                          initial_filters=32, depth=4)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"Model loaded successfully (Epoch {checkpoint.get('epoch', 'unknown')})")
        
        # Post-processor
        self.postprocessor = TopologyPostProcessor()
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to model input format"""
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        return image
    
    def infer_patch(self, patch: np.ndarray) -> np.ndarray:
        """Infer on a single patch"""
        with torch.no_grad():
            # Add batch and channel dimensions
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
            patch_tensor = patch_tensor.to(self.device)
            
            # Forward pass
            output = self.model(patch_tensor)
            
            # Remove batch and channel dimensions
            pred = output.squeeze(0).squeeze(0).cpu().numpy()
            return pred
    
    def stitch_patches(self, 
                      patches: List[np.ndarray],
                      positions: List[tuple],
                      image_shape: tuple) -> np.ndarray:
        """Stitch patches with overlapping averaging"""
        output = np.zeros(image_shape, dtype=np.float32)
        weights = np.zeros(image_shape, dtype=np.float32)
        
        for patch, (z_start, y_start, x_start) in zip(patches, positions):
            z_end = min(z_start + patch.shape[0], image_shape[0])
            y_end = min(y_start + patch.shape[1], image_shape[1])
            x_end = min(x_start + patch.shape[2], image_shape[2])
            
            output[z_start:z_end, y_start:y_end, x_start:x_end] += patch[
                :z_end-z_start, :y_end-y_start, :x_end-x_start
            ]
            weights[z_start:z_end, y_start:y_end, x_start:x_end] += 1.0
        
        # Avoid division by zero
        weights[weights == 0] = 1.0
        output = output / weights
        return output
    
    def infer_image(self, image_path: str, output_path: str) -> dict:
        """Run inference on a single image"""
        try:
            start_time = time.time()
            
            # Load image
            image = tifffile.imread(image_path)
            original_shape = image.shape
            
            # Preprocess
            image = self.preprocess(image)
            
            # Extract patches with stride
            stride = self.patch_size[0] - self.overlap
            patches = []
            positions = []
            
            for z in range(0, image.shape[0] - self.patch_size[0] + 1, stride):
                for y in range(0, image.shape[1] - self.patch_size[1] + 1, stride):
                    for x in range(0, image.shape[2] - self.patch_size[2] + 1, stride):
                        patch = image[z:z+self.patch_size[0], 
                                     y:y+self.patch_size[1], 
                                     x:x+self.patch_size[2]]
                        if patch.shape == self.patch_size:
                            patches.append(patch)
                            positions.append((z, y, x))
            
            # Handle edges (add final patches if needed)
            if image.shape[0] % stride != 0:
                z_end = image.shape[0] - self.patch_size[0]
                for y in range(0, image.shape[1] - self.patch_size[1] + 1, stride):
                    for x in range(0, image.shape[2] - self.patch_size[2] + 1, stride):
                        patch = image[z_end:z_end+self.patch_size[0],
                                     y:y+self.patch_size[1],
                                     x:x+self.patch_size[2]]
                        if patch.shape == self.patch_size and (z_end, y, x) not in positions:
                            patches.append(patch)
                            positions.append((z_end, y, x))
            
            # Infer patches
            predictions = []
            for patch in tqdm(patches, desc=f"Inferring patches", leave=False):
                pred = self.infer_patch(patch)
                predictions.append(pred)
            
            # Stitch patches
            output = self.stitch_patches(predictions, positions, image.shape)
            
            # Post-process
            output_binary = (output > 0.5).astype(np.uint8)
            output_processed = self.postprocessor.process(output_binary)
            
            # Save output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(output_path, output_processed.astype(np.uint8))
            
            # Save visualization (0-255 grayscale for viewing)
            vis_path = Path(output_path).parent / f"{Path(output_path).stem}_viz.tif"
            output_vis = (output_processed * 255).astype(np.uint8)
            tifffile.imwrite(vis_path, output_vis)
            
            elapsed = time.time() - start_time
            
            # Compute metrics
            foreground_pct = 100 * np.sum(output_processed == 1) / output_processed.size
            
            return {
                'status': 'success',
                'time': elapsed,
                'patches': len(patches),
                'foreground_pct': foreground_pct,
                'output_path': output_path
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def infer_batch(self, image_dir: str, output_dir: str, num_images: int = 100) -> dict:
        """Run inference on batch of images"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image list
        image_files = sorted(image_dir.glob('*.tif'))[:num_images]
        
        self.logger.info(f"Processing {len(image_files)} images from {image_dir}")
        
        results = {
            'total': len(image_files),
            'successful': 0,
            'failed': 0,
            'images': [],
            'start_time': datetime.now().isoformat(),
            'total_time': 0
        }
        
        batch_start = time.time()
        
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"[{i}/{len(image_files)}] Processing {image_file.name}")
            
            output_path = output_dir / image_file.name
            result = self.infer_image(str(image_file), str(output_path))
            
            if result['status'] == 'success':
                results['successful'] += 1
                self.logger.info(f"  ✓ Completed in {result['time']:.2f}s "
                               f"({result['foreground_pct']:.2f}% foreground)")
            else:
                results['failed'] += 1
                self.logger.error(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
            
            results['images'].append(result)
        
        results['total_time'] = time.time() - batch_start
        results['end_time'] = datetime.now().isoformat()
        results['avg_time'] = results['total_time'] / len(image_files)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Batch inference on training images')
    parser.add_argument('--checkpoint', type=str, default='bin/checkpoints/fold_0/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, default='train_images',
                       help='Directory containing training images')
    parser.add_argument('--output', type=str, default='predictions_train',
                       help='Output directory for predictions')
    parser.add_argument('--num', type=int, default=100,
                       help='Number of images to process')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    pipeline = BatchInferencePipeline(args.checkpoint, device=args.device)
    results = pipeline.infer_batch(args.input, args.output, num_images=args.num)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH INFERENCE SUMMARY")
    print("="*60)
    print(f"Total images: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f}s")
    print(f"Average time per image: {results['avg_time']:.2f}s")
    print(f"Output directory: {args.output}")
    
    foreground_pcts = [img.get('foreground_pct', 0) for img in results['images'] 
                       if img['status'] == 'success']
    if foreground_pcts:
        print(f"Average foreground: {np.mean(foreground_pcts):.2f}% "
              f"(min: {np.min(foreground_pcts):.2f}%, max: {np.max(foreground_pcts):.2f}%)")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
