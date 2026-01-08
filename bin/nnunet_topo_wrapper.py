"""
nnU-Net Wrapper with Topology-Aware Training

Integrates nnU-Net with topology-preserving losses for the Vesuvius Challenge
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import logging

from topology_losses import CombinedTopologyLoss


class TopologyAwareUNet3D(nn.Module):
    """
    3D U-Net with topology-aware architecture
    
    This is a simplified version for demonstration. 
    In production, use nnUNetv2's dynamic architecture.
    """
    
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1,
                 initial_filters=32,
                 depth=5):
        super().__init__()
        
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        channels = initial_filters
        in_ch = in_channels
        
        for i in range(depth):
            self.encoders.append(
                self._conv_block(in_ch, channels)
            )
            self.pools.append(nn.MaxPool3d(2))
            in_ch = channels
            channels *= 2
        
        # Bottleneck
        self.bottleneck = self._conv_block(in_ch, channels)
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth):
            self.upconvs.append(
                nn.ConvTranspose3d(channels, channels // 2, kernel_size=2, stride=2)
            )
            channels = channels // 2
            self.decoders.append(
                self._conv_block(channels * 2, channels)
            )
        
        # Output
        self.output = nn.Conv3d(channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _conv_block(self, in_channels, out_channels):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, D, H, W)
        Returns:
            Output probability map (B, 1, D, H, W)
        """
        # Encoder
        encoder_outputs = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encoder_outputs.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            # Skip connection
            skip = encoder_outputs[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Output
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x


class TopologyAwareTrainer:
    """
    Training wrapper for topology-aware segmentation
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 loss_config: Optional[Dict] = None,
                 checkpoint_dir: Optional[Path] = None,
                 warmup_epochs: int = 0):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize loss
        loss_config = loss_config or {}
        self.criterion = CombinedTopologyLoss(**loss_config).to(device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.warmup_epochs = warmup_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': [],
            'dice': [],
            'focal': [],
            'boundary': [],
            'cldice': [],
            'connectivity': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss, loss_components = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (more aggressive for stability)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            
            self.optimizer.step()
            
            # Log gradient norm periodically
            if batch_idx % 50 == 0:
                self.logger.info(f"Gradient norm: {grad_norm:.4f}")
            
            # Log losses
            for key, value in loss_components.items():
                epoch_losses[key].append(value)
            
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader, metric_fn=None) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        val_losses = {
            'total': [],
            'dice': [],
            'focal': [],
            'boundary': [],
            'cldice': [],
            'connectivity': []
        }
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss, loss_components = self.criterion(outputs, labels)
                
                # Log losses
                for key, value in loss_components.items():
                    val_losses[key].append(value)
                
                # Store for metric computation
                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        # Compute competition metrics if provided
        if metric_fn:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = metric_fn(all_predictions, all_labels)
            avg_losses.update(metrics)
        
        return avg_losses
    
    def train(self, 
              train_loader,
              val_loader,
              num_epochs: int,
              scheduler=None,
              metric_fn=None,
              early_stopping_patience: int = 50):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            scheduler: Learning rate scheduler
            metric_fn: Function to compute competition metrics
            early_stopping_patience: Epochs to wait before early stopping
        """
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Apply warmup learning rate schedule
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.base_lr * warmup_factor
                self.logger.info(f"Warmup LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            elif epoch == self.warmup_epochs:
                # Reset to base LR after warmup
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.base_lr
                self.logger.info(f"Warmup complete. LR reset to: {self.base_lr:.6f}")
            
            # Train
            train_losses = self.train_epoch(train_loader)
            self.logger.info(f"Epoch {epoch} Train - Loss: {train_losses['total']:.4f}")
            
            # Validate
            val_losses = self.validate(val_loader, metric_fn)
            self.logger.info(f"Epoch {epoch} Val - Loss: {val_losses['total']:.4f}")
            
            # Learning rate scheduling (only after warmup)
            if scheduler and epoch >= self.warmup_epochs:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_losses['total'])
                else:
                    scheduler.step()
            
            # Save best model
            val_score = -val_losses['total']  # Lower loss = better
            if 'competition_score' in val_losses:
                val_score = val_losses['competition_score']
            
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.save_checkpoint('best_model.pth')
                self.logger.info(f"New best model saved! Score: {val_score:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        self.logger.info(f"Training complete! Best score: {self.best_val_score:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_score = checkpoint['best_val_score']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")


def create_model_and_trainer(config: Dict) -> Tuple[nn.Module, TopologyAwareTrainer]:
    """
    Factory function to create model and trainer
    
    Args:
        config: Configuration dictionary with model and training parameters
        
    Returns:
        model, trainer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = TopologyAwareUNet3D(
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('out_channels', 1),
        initial_filters=config.get('initial_filters', 32),
        depth=config.get('depth', 5)
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Create trainer
    loss_config = {
        'dice_weight': config.get('dice_weight', 0.4),
        'focal_weight': config.get('focal_weight', 0.2),
        'boundary_weight': config.get('boundary_weight', 0.2),
        'cldice_weight': config.get('cldice_weight', 0.1),
        'connectivity_weight': config.get('connectivity_weight', 0.1)
    }
    
    trainer = TopologyAwareTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_config=loss_config,
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        warmup_epochs=config.get('warmup_epochs', 0)
    )
    
    return model, trainer


if __name__ == "__main__":
    # Test the model and trainer
    print("Testing TopologyAwareUNet3D...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = TopologyAwareUNet3D(
        in_channels=1,
        out_channels=1,
        initial_filters=16,  # Smaller for testing
        depth=3
    )
    
    # Test forward pass
    batch_size = 1
    size = 64
    x = torch.randn(batch_size, 1, size, size, size)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    print("\nModel test passed!")
