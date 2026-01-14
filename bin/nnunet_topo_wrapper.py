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
            'connectivity': [],
            'variance': []
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
            'connectivity': [],
            'variance': []
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
              early_stopping_patience: int = 50,
              plateau_patience: int = 3,
              plateau_threshold: float = 0.002,
              substantial_progress_threshold: float = 0.5,
              swa_model=None,
              swa_scheduler=None,
              swa_start_epoch: int = 50,
              noise_enabled: bool = False,
              noise_start_epoch: int = 30,
              noise_frequency: int = 10,
              noise_std: float = 0.001,
              noise_decay: float = 0.9,
              noise_target_layers: list = None):
        """
        Full training loop with adaptive plateau detection, SWA, and weight noise
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            scheduler: Learning rate scheduler
            metric_fn: Function to compute competition metrics
            early_stopping_patience: Epochs to wait before early stopping
            plateau_patience: Epochs to wait before declaring plateau
            plateau_threshold: Minimum improvement to avoid plateau detection
            substantial_progress_threshold: Fraction of improvement from baseline to disable interventions (default 0.5 = 50%)
            swa_model: Stochastic Weight Averaging model (optional)
            swa_scheduler: SWA learning rate scheduler (optional)
            swa_start_epoch: Epoch to start SWA
            noise_enabled: Whether to inject weight noise
            noise_start_epoch: Epoch to start noise injection
            noise_frequency: Apply noise every N epochs
            noise_std: Standard deviation of noise
            noise_decay: Decay factor for noise over time
            noise_target_layers: List of layer name patterns to add noise to
        """
        if noise_target_layers is None:
            noise_target_layers = ['decoders', 'output']
        
        patience_counter = 0
        plateau_counter = 0
        last_best_loss = float('inf')
        baseline_loss = None  # Track starting loss for substantial progress detection
        intervention_count = 0
        max_interventions = 3
        
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
            
            # Apply weight noise injection (Phase 3 - Constrained Exploration)
            if noise_enabled and epoch >= noise_start_epoch and epoch % noise_frequency == 0:
                current_noise_std = noise_std * (noise_decay ** ((epoch - noise_start_epoch) // noise_frequency))
                self.add_weight_noise(current_noise_std, noise_target_layers)
                self.logger.info(f"🎲 Applied weight noise (std={current_noise_std:.6f}) to escape plateau")
            
            # Train
            train_losses = self.train_epoch(train_loader)
            self.logger.info(
                f"Epoch {epoch} Train - Loss: {train_losses['total']:.4f} "
                f"(dice: {train_losses['dice']:.4f}, focal: {train_losses['focal']:.4f}, "
                f"variance: {train_losses['variance']:.4f}, entropy: {train_losses.get('entropy', 0.0):.4f})"
            )
            
            # Validate
            val_losses = self.validate(val_loader, metric_fn)
            self.logger.info(
                f"Epoch {epoch} Val - Loss: {val_losses['total']:.4f} "
                f"(dice: {val_losses['dice']:.4f}, focal: {val_losses['focal']:.4f}, "
                f"variance: {val_losses['variance']:.4f}, entropy: {val_losses.get('entropy', 0.0):.4f})"
            )
            
            # Set baseline loss on first epoch for substantial progress detection
            current_val_loss = val_losses['total']
            if baseline_loss is None:
                baseline_loss = current_val_loss
                self.logger.info(f"📊 Baseline loss set: {baseline_loss:.4f}")
            
            # Check for substantial progress (disable interventions if achieved)
            improvement_fraction = (baseline_loss - current_val_loss) / baseline_loss
            has_substantial_progress = improvement_fraction > substantial_progress_threshold
            
            if has_substantial_progress and intervention_count == 0:
                self.logger.info(f"🎯 SUBSTANTIAL PROGRESS ACHIEVED: {improvement_fraction*100:.1f}% improvement from baseline!")
                self.logger.info(f"   Baseline: {baseline_loss:.4f} → Current: {current_val_loss:.4f}")
                self.logger.info(f"   Interventions disabled - model is learning well!")
            
            # Learning rate scheduling (only after warmup)
            if scheduler and epoch >= self.warmup_epochs:
                # Update SWA if enabled and past start epoch
                if swa_model is not None and epoch >= swa_start_epoch:
                    swa_model.update_parameters(self.model)
                    swa_scheduler.step()
                    if epoch == swa_start_epoch:
                        self.logger.info(f"🔄 Started SWA at epoch {epoch}")
                elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
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
            
            # Plateau detection
            improvement = last_best_loss - current_val_loss
            if improvement > plateau_threshold:
                last_best_loss = current_val_loss
                plateau_counter = 0
            else:
                plateau_counter += 1
            
            # Adaptive intervention when plateau detected
            # BUT: Skip if substantial progress already achieved (model is learning well!)
            if plateau_counter >= plateau_patience and intervention_count < max_interventions:
                if has_substantial_progress:
                    self.logger.info(f"ℹ️  Plateau detected but SKIPPING intervention (substantial progress: {improvement_fraction*100:.1f}%)")
                    plateau_counter = 0  # Reset counter to avoid repeated logging
                else:
                    self.logger.warning(f"⚠️  PLATEAU DETECTED after {plateau_counter} epochs with no improvement!")
                    self.logger.warning(f"   Current improvement from baseline: {improvement_fraction*100:.1f}% (threshold: {substantial_progress_threshold*100:.0f}%)")
                    self._handle_plateau(epoch, intervention_count)
                    intervention_count += 1
                    plateau_counter = 0  # Reset counter after intervention
            
            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        self.logger.info(f"Training complete! Best score: {self.best_val_score:.4f}")
    
    def _handle_plateau(self, epoch: int, intervention_num: int):
        """
        Handle training plateau with adaptive intervention strategies
        
        Intervention sequence:
        1. Increase LR + disable complex topology losses
        2. Further increase LR + simplify to focal only
        3. Reset optimizer + use dice only
        """
        self.logger.warning(f"🔧 Applying intervention #{intervention_num + 1}")
        
        if intervention_num == 0:
            # First intervention: Increase LR 10x, disable boundary/cldice/connectivity
            new_lr = self.base_lr * 10
            self.logger.warning(f"→ Increasing LR: {self.base_lr:.6f} → {new_lr:.6f}")
            self.logger.warning(f"→ Disabling complex topology losses (boundary, clDice, connectivity)")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.base_lr = new_lr
            
            # Disable complex losses
            if hasattr(self.criterion, 'boundary_weight'):
                self.criterion.boundary_weight = 0.0
                self.criterion.cldice_weight = 0.0
                self.criterion.connectivity_weight = 0.0
                self.logger.warning(f"→ Loss weights: Dice={self.criterion.dice_weight:.2f}, "
                                   f"Focal={self.criterion.focal_weight:.2f}")
        
        elif intervention_num == 1:
            # Second intervention: Further increase LR 5x, use focal loss only
            new_lr = self.base_lr * 5
            self.logger.warning(f"→ Further increasing LR: {self.base_lr:.6f} → {new_lr:.6f}")
            self.logger.warning(f"→ Switching to FOCAL LOSS ONLY")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.base_lr = new_lr
            
            # Focal loss only
            if hasattr(self.criterion, 'dice_weight'):
                self.criterion.dice_weight = 0.0
                self.criterion.focal_weight = 1.0
                self.criterion.boundary_weight = 0.0
                self.criterion.cldice_weight = 0.0
                self.criterion.connectivity_weight = 0.0
                self.logger.warning(f"→ Loss weights: Focal=1.0, all others=0.0")
        
        elif intervention_num == 2:
            # Third intervention: Reset optimizer with 3x LR boost, use dice loss only
            new_lr = self.base_lr * 3
            self.logger.warning(f"→ Resetting optimizer state with LR boost: {self.base_lr:.6f} → {new_lr:.6f}")
            self.logger.warning(f"→ Switching to DICE LOSS ONLY")
            
            # Reset optimizer with boosted LR
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=new_lr,
                weight_decay=0.01
            )
            self.base_lr = new_lr
            
            # Dice loss only
            if hasattr(self.criterion, 'dice_weight'):
                self.criterion.dice_weight = 1.0
                self.criterion.focal_weight = 0.0
                self.criterion.boundary_weight = 0.0
                self.criterion.cldice_weight = 0.0
                self.criterion.connectivity_weight = 0.0
                self.logger.warning(f"→ Loss weights: Dice=1.0, all others=0.0")
        
        self.logger.warning(f"✓ Intervention complete. Continuing training...")
    
    def add_weight_noise(self, noise_std=0.001, target_layers=['decoders', 'output']):
        """
        Add Gaussian noise to specified layers for constrained exploration
        
        Args:
            noise_std: Standard deviation of Gaussian noise
            target_layers: List of layer name patterns to add noise to
        """
        with torch.no_grad():
            noise_count = 0
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in target_layers):
                    noise = torch.randn_like(param) * noise_std
                    param.add_(noise)
                    noise_count += 1
        self.logger.info(f"   Added noise (std={noise_std:.6f}) to {noise_count} parameter tensors in {target_layers}")
    
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
        'connectivity_weight': config.get('connectivity_weight', 0.1),
        'focal_alpha': config.get('focal_alpha', 0.25),
        'focal_gamma': config.get('focal_gamma', 2.0),
        'use_class_weights': config.get('use_class_weights', False),
        'background_weight': config.get('background_weight', 1.0),
        'foreground_weight': config.get('foreground_weight', 1.0)
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
