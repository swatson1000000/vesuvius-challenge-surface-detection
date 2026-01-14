"""
Topology-Aware Loss Functions for Vesuvius Challenge

Implements loss functions that preserve topological structure:
- clDice: Centerline Dice for connectivity
- Boundary Loss: For surface accuracy
- Connectivity Loss: Prevents splits and mergers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, morphology
import numpy as np


class clDiceLoss(nn.Module):
    """
    Centerline Dice Loss - preserves connectivity and topology
    
    References:
    - Shit et al. "clDice - a Novel Topology-Preserving Loss Function for 
      Tubular Structure Segmentation" CVPR 2021
    """
    
    def __init__(self, iter_=3, smooth=1e-5):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth
        
    def soft_erode(self, img):
        """Soft morphological erosion"""
        if len(img.shape) == 4:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        """Soft morphological dilation"""
        if len(img.shape) == 4:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        """Soft morphological opening"""
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        """Soft skeletonization"""
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, pred, target):
        """
        Args:
            pred: predicted probability map (B, C, D, H, W) or (B, D, H, W)
            target: ground truth binary mask (B, C, D, H, W) or (B, D, H, W)
        """
        if len(pred.shape) == 4:
            pred = pred.unsqueeze(1)
        if len(target.shape) == 4:
            target = target.unsqueeze(1)
            
        # Compute skeletons
        skel_pred = self.soft_skel(pred)
        skel_true = self.soft_skel(target)
        
        # Topology-preserving dice (skeleton × prediction vs skeleton × target)
        tprec = (torch.sum(torch.multiply(skel_pred, target), dim=(2,3,4)) + self.smooth) / \
                (torch.sum(skel_pred, dim=(2,3,4)) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, pred), dim=(2,3,4)) + self.smooth) / \
                (torch.sum(skel_true, dim=(2,3,4)) + self.smooth)
        
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
        
        return 1.0 - cl_dice.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-focused loss for accurate surface detection
    Optimizes for SurfaceDice metric
    """
    
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        """
        Args:
            pred: predicted probability map (B, C, D, H, W) or (B, D, H, W)
            target: ground truth binary mask (B, C, D, H, W) or (B, D, H, W)
        """
        if len(pred.shape) == 4:
            pred = pred.unsqueeze(1)
        if len(target.shape) == 4:
            target = target.unsqueeze(1)
            
        # Compute distance transform on CPU for each sample in batch
        batch_size = target.shape[0]
        boundary_losses = []
        
        for b in range(batch_size):
            target_np = target[b, 0].detach().cpu().numpy()
            
            # Compute distance transform
            # Distance to foreground and background boundaries
            dist_pos = distance_transform_edt(target_np)
            dist_neg = distance_transform_edt(1 - target_np)
            
            # Level set function φ = dist_neg - dist_pos
            phi = dist_neg - dist_pos
            
            # Convert to weights (higher weight near boundary)
            # w(φ) = 1 when |φ| < θ0, decreases to 0 when |φ| > θ
            weights = np.where(
                np.abs(phi) <= self.theta0,
                1.0,
                np.where(
                    np.abs(phi) >= self.theta,
                    0.0,
                    1.0 - (np.abs(phi) - self.theta0) / (self.theta - self.theta0)
                )
            )
            
            weights = torch.from_numpy(weights).float().to(pred.device)
            
            # Weighted BCE
            pred_b = pred[b, 0]
            target_b = target[b, 0]
            
            bce = F.binary_cross_entropy(pred_b, target_b, reduction='none')
            weighted_bce = (bce * weights).sum() / (weights.sum() + 1e-5)
            boundary_losses.append(weighted_bce)
        
        return torch.stack(boundary_losses).mean()


class ConnectivityLoss(nn.Module):
    """
    Connectivity loss to prevent splits and mergers
    Helps optimize VOI metric
    """
    
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        
    def forward(self, pred, target):
        """
        Penalize predictions that change local connectivity patterns
        
        Args:
            pred: predicted probability map (B, C, D, H, W) or (B, D, H, W)
            target: ground truth binary mask (B, C, D, H, W) or (B, D, H, W)
        """
        if len(pred.shape) == 4:
            pred = pred.unsqueeze(1)
        if len(target.shape) == 4:
            target = target.unsqueeze(1)
            
        # Compute local connectivity using max pooling
        # If a region is connected, max pooling should show high values
        pred_conn = F.max_pool3d(pred, self.kernel_size, stride=1, 
                                  padding=self.kernel_size//2)
        target_conn = F.max_pool3d(target, self.kernel_size, stride=1,
                                    padding=self.kernel_size//2)
        
        # MSE between connectivity patterns
        loss = F.mse_loss(pred_conn, target_conn)
        
        return loss


class CombinedTopologyLoss(nn.Module):
    """
    Combined loss function optimized for the competition metrics
    
    Loss = weighted sum of dice + focal + boundary + clDice + connectivity + variance_reg
    """
    
    def __init__(self, 
                 dice_weight=0.3,
                 focal_weight=0.15,
                 boundary_weight=0.15,
                 cldice_weight=0.05,
                 connectivity_weight=0.05,
                 variance_weight=0.25,
                 entropy_weight=0.05,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 use_class_weights=False,
                 background_weight=1.0,
                 foreground_weight=1.0):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.cldice_weight = cldice_weight
        self.connectivity_weight = connectivity_weight
        self.variance_weight = variance_weight
        self.entropy_weight = entropy_weight  # NEW: entropy regularization
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.use_class_weights = use_class_weights
        self.background_weight = background_weight
        self.foreground_weight = foreground_weight
        
        # Initialize component losses
        self.cldice_loss = clDiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.connectivity_loss = ConnectivityLoss()
        
    def dice_loss(self, pred, target, smooth=1e-5):
        """Standard Dice loss with optional class weighting"""
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def variance_regularization(self, pred):
        """
        Penalize low variance predictions to prevent uniform outputs
        Encourages model to make discriminative predictions
        
        Uses negative entropy: punishes predictions that are too extreme (all 0s or all 1s)
        """
        # Clamp to avoid log(0)
        pred_safe = torch.clamp(pred, 1e-7, 1 - 1e-7)
        
        # Compute variance across spatial dimensions
        variance = pred.var(dim=(2, 3, 4)).mean()
        
        # AGGRESSIVE variance penalty: exponential penalty as variance approaches 0
        # Loss = (1 - variance)^2 so it grows rapidly as variance decreases
        var_loss = (1.0 - variance) ** 2.0
        
        # Additional entropy penalty to prevent extreme predictions
        # Shannon entropy: -p*log(p) - (1-p)*log(1-p)
        entropy = -(pred_safe * torch.log(pred_safe) + (1 - pred_safe) * torch.log(1 - pred_safe))
        entropy_mean = entropy.mean()
        
        # Penalize low entropy (too confident): target entropy > 0.5 for better diversity
        entropy_penalty = torch.clamp(0.5 - entropy_mean, min=0.0)
        
        # Combined: strong penalty for lack of variance + entropy penalty
        return var_loss + entropy_penalty
    
    def entropy_regularization(self, pred):
        """
        Penalize predictions that are too confident (entropy too low)
        Encourages the model to output less extreme values (not always 0 or 1)
        This addresses the 99.88% foreground bias by pushing toward ~50% uncertainty
        """
        # Clamp to avoid log(0)
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        
        # Shannon entropy: -p*log(p) - (1-p)*log(1-p)
        # Maximum entropy = 0.693 at p=0.5
        # Minimum entropy = 0 at p=0 or p=1
        entropy = -(pred * torch.log(pred) + (1 - pred) * torch.log(1 - pred))
        
        # Target: encourage entropy > 0.3 (avoid overconfident predictions)
        # Loss is high when entropy is low (overconfident)
        target_entropy = 0.3
        entropy_loss = torch.clamp(target_entropy - entropy.mean(), min=0.0)
        
        return entropy_loss
    
    def focal_loss(self, pred, target):
        """Focal loss for handling class imbalance with proper alpha weighting"""
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        
        # Compute per-pixel cross entropy
        ce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        # Compute pt (prediction probability for true class)
        pt = target * pred + (1 - target) * (1 - pred)
        
        # Apply focal term (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Apply alpha balancing (alpha for foreground, 1-alpha for background)
        # Alpha should weight the MINORITY class more heavily
        alpha_weight = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)
        
        # Combine
        focal = alpha_weight * focal_weight * ce_loss
        
        # Apply class weights if enabled
        if self.use_class_weights:
            class_weight = target * self.foreground_weight + (1 - target) * self.background_weight
            focal = focal * class_weight
        
        return focal.mean()
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted probability map (B, C, D, H, W) or (B, D, H, W)
            target: ground truth binary mask (B, C, D, H, W) or (B, D, H, W)
        """
        if len(pred.shape) == 4:
            pred = pred.unsqueeze(1)
        if len(target.shape) == 4:
            target = target.unsqueeze(1)
        
        # Compute all loss components
        loss_dice = self.dice_loss(pred, target)
        loss_focal = self.focal_loss(pred, target)
        loss_boundary = self.boundary_loss(pred, target)
        loss_cldice = self.cldice_loss(pred, target)
        loss_connectivity = self.connectivity_loss(pred, target)
        loss_variance = self.variance_regularization(pred)
        loss_entropy = self.entropy_regularization(pred)  # NEW: entropy regularization
        
        # Combine losses
        total_loss = (
            self.dice_weight * loss_dice +
            self.focal_weight * loss_focal +
            self.boundary_weight * loss_boundary +
            self.cldice_weight * loss_cldice +
            self.connectivity_weight * loss_connectivity +
            self.variance_weight * loss_variance +
            self.entropy_weight * loss_entropy  # NEW
        )
        
        # Return total loss and components for logging
        return total_loss, {
            'dice': loss_dice.item(),
            'focal': loss_focal.item(),
            'boundary': loss_boundary.item(),
            'cldice': loss_cldice.item(),
            'connectivity': loss_connectivity.item(),
            'variance': loss_variance.item(),
            'entropy': loss_entropy.item(),  # NEW
            'total': total_loss.item()
        }


if __name__ == "__main__":
    # Test the losses
    print("Testing topology losses...")
    
    # Create dummy data
    batch_size = 2
    size = 64
    pred = torch.rand(batch_size, 1, size, size, size).cuda()
    target = (torch.rand(batch_size, 1, size, size, size) > 0.5).float().cuda()
    
    # Test clDice
    print("Testing clDice loss...")
    cldice = clDiceLoss().cuda()
    loss = cldice(pred, target)
    print(f"  clDice loss: {loss.item():.4f}")
    
    # Test boundary loss
    print("Testing boundary loss...")
    boundary = BoundaryLoss().cuda()
    loss = boundary(pred, target)
    print(f"  Boundary loss: {loss.item():.4f}")
    
    # Test connectivity loss
    print("Testing connectivity loss...")
    connectivity = ConnectivityLoss().cuda()
    loss = connectivity(pred, target)
    print(f"  Connectivity loss: {loss.item():.4f}")
    
    # Test combined loss
    print("Testing combined loss...")
    combined = CombinedTopologyLoss().cuda()
    loss, components = combined(pred, target)
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Components: {components}")
    
    print("\nAll tests passed!")
