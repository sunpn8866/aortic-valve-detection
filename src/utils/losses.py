"""
Advanced Loss Functions for Medical Object Detection
Implements state-of-the-art losses for small object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Essential for small object detection in medical imaging
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss
        
        Args:
            inputs: Predicted logits [B, num_classes]
            targets: Ground truth labels [B]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL) for bounding box regression
    Key innovation from YOLOv8/v11 for precise boundaries
    """
    
    def __init__(self, loss_weight: float = 1.5, reg_max: int = 16):
        super().__init__()
        self.loss_weight = loss_weight
        self.reg_max = reg_max
        self.project = nn.Parameter(torch.linspace(0, reg_max - 1, reg_max), requires_grad=False)
        
    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate DFL
        
        Args:
            pred_dist: Predicted distribution [B, N, 4, reg_max]
            target: Target boxes [B, N, 4]
            
        Returns:
            DFL loss
        """
        # Normalize target to [0, reg_max-1]
        target = torch.clamp(target, min=0, max=self.reg_max - 1.01)
        
        # Convert target to discrete distribution
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()
        
        # Create target distribution
        target_dist = F.one_hot(target_left, self.reg_max).float() * weight_left.unsqueeze(-1)
        target_dist += F.one_hot(target_right, self.reg_max).float() * weight_right.unsqueeze(-1)
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(
            pred_dist.reshape(-1, self.reg_max),
            target_dist.reshape(-1, self.reg_max),
            reduction='none'
        )
        
        return (loss.reshape_as(target).mean(-1) * self.loss_weight).mean()


class CompleteIoULoss(nn.Module):
    """
    Complete IoU (CIoU) Loss
    Considers overlap, center distance, and aspect ratio
    """
    
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Calculate CIoU Loss
        
        Args:
            pred_boxes: Predicted boxes [B, N, 4] (x1, y1, x2, y2)
            target_boxes: Target boxes [B, N, 4]
            
        Returns:
            CIoU loss
        """
        # Calculate IoU
        inter_x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        inter_y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        inter_x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        inter_y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + self.eps)
        
        # Calculate center distance
        pred_cx = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
        pred_cy = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
        target_cx = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
        target_cy = (target_boxes[..., 1] + target_boxes[..., 3]) / 2
        
        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Calculate diagonal distance of enclosing box
        enc_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        enc_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        enc_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        enc_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
        
        enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
        
        # Calculate aspect ratio consistency
        pred_w = pred_boxes[..., 2] - pred_boxes[..., 0]
        pred_h = pred_boxes[..., 3] - pred_boxes[..., 1]
        target_w = target_boxes[..., 2] - target_boxes[..., 0]
        target_h = target_boxes[..., 3] - target_boxes[..., 1]
        
        v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(target_w / (target_h + self.eps)) - 
                                               torch.atan(pred_w / (pred_h + self.eps)), 2)
        
        alpha = v / (1 - iou + v + self.eps)
        
        # CIoU loss
        ciou_loss = 1 - iou + center_dist / (enc_diag + self.eps) + alpha * v
        
        return ciou_loss.mean()


class SoftCIoULoss(nn.Module):
    """
    Soft-CIoU Loss variant
    Applies square root to distances for better small object handling
    """
    
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Calculate Soft-CIoU Loss
        
        Args:
            pred_boxes: Predicted boxes [B, N, 4]
            target_boxes: Target boxes [B, N, 4]
            
        Returns:
            Soft-CIoU loss
        """
        # Similar to CIoU but with square root on distances
        # Calculate IoU first
        inter_x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        inter_y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        inter_x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        inter_y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + self.eps)
        
        # Center distance with square root (key difference)
        pred_cx = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
        pred_cy = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
        target_cx = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
        target_cy = (target_boxes[..., 1] + target_boxes[..., 3]) / 2
        
        center_dist = torch.sqrt((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2 + self.eps)
        
        # Diagonal distance with square root
        enc_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        enc_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        enc_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        enc_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
        
        enc_diag = torch.sqrt((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + self.eps)
        
        # Soft-CIoU loss
        soft_ciou_loss = 1 - iou + center_dist / (enc_diag + self.eps)
        
        return soft_ciou_loss.mean()


class PixelLevelBalancing(nn.Module):
    """
    Pixel-Level Balancing (PLB) for small object weighting
    Dynamically adjusts loss weights based on object size
    """
    
    def __init__(self, min_pixels: int = 100, balance_factor: float = 1.0):
        super().__init__()
        self.min_pixels = min_pixels
        self.balance_factor = balance_factor
        
    def forward(self, loss: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel-level balancing to loss
        
        Args:
            loss: Base loss values [B, N]
            target_boxes: Target boxes for size calculation [B, N, 4]
            
        Returns:
            Weighted loss
        """
        # Calculate object sizes (in pixels)
        widths = target_boxes[..., 2] - target_boxes[..., 0]
        heights = target_boxes[..., 3] - target_boxes[..., 1]
        areas = widths * heights
        
        # Apply weighting: 1/sqrt(pixel_count)
        # Clamp to minimum to avoid extreme weights
        areas_clamped = torch.clamp(areas, min=self.min_pixels)
        weights = 1.0 / torch.sqrt(areas_clamped)
        weights = weights * self.balance_factor
        
        # Normalize weights
        weights = weights / weights.mean()
        
        # Apply to loss
        weighted_loss = loss * weights
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function for YOLOv11 training
    Integrates all loss components with proper weighting
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Loss components
        self.box_loss = DistributionFocalLoss(
            loss_weight=config.get('box_weight', 7.5),
            reg_max=config.get('reg_max', 16)
        )
        
        self.cls_loss = FocalLoss(
            alpha=config.get('cls_alpha', 0.25),
            gamma=config.get('cls_gamma', 2.0)
        )
        
        self.obj_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config.get('obj_pos_weight', 1.0)])
        )
        
        # Optional components
        self.use_ciou = config.get('use_ciou', False)
        if self.use_ciou:
            self.ciou_loss = CompleteIoULoss()
        
        self.use_plb = config.get('use_plb', False)
        if self.use_plb:
            self.plb = PixelLevelBalancing(
                min_pixels=config.get('plb_min_pixels', 100),
                balance_factor=config.get('plb_factor', 1.0)
            )
        
        # Loss weights
        self.box_weight = config.get('box_weight', 7.5)
        self.cls_weight = config.get('cls_weight', 0.5)
        self.obj_weight = config.get('obj_weight', 1.0)
        self.ciou_weight = config.get('ciou_weight', 1.0)
        
    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Calculate combined loss
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth dictionary
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        # Box regression loss (DFL)
        if 'box_dist' in predictions:
            box_loss = self.box_loss(predictions['box_dist'], targets['boxes'])
            
            # Apply PLB if enabled
            if self.use_plb:
                box_loss = self.plb(box_loss.unsqueeze(0), targets['boxes']).squeeze()
            
            losses['box_loss'] = box_loss * self.box_weight
        
        # Classification loss (Focal)
        if 'cls_logits' in predictions:
            cls_loss = self.cls_loss(predictions['cls_logits'], targets['labels'])
            losses['cls_loss'] = cls_loss * self.cls_weight
        
        # Objectness loss
        if 'obj_logits' in predictions:
            obj_loss = self.obj_loss(predictions['obj_logits'], targets['obj_targets'])
            losses['obj_loss'] = obj_loss * self.obj_weight
        
        # Optional CIoU loss
        if self.use_ciou and 'pred_boxes' in predictions:
            ciou_loss = self.ciou_loss(predictions['pred_boxes'], targets['boxes'])
            losses['ciou_loss'] = ciou_loss * self.ciou_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation auxiliary task
    """
    
    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss
        
        Args:
            pred: Predicted masks [B, 1, H, W]
            target: Target masks [B, 1, H, W]
            
        Returns:
            Dice loss
        """
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss
    Better for highly imbalanced segmentation
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Tversky loss
        
        Args:
            pred: Predicted masks [B, 1, H, W]
            target: Target masks [B, 1, H, W]
            
        Returns:
            Tversky loss
        """
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        true_pos = (pred_flat * target_flat).sum()
        false_neg = ((1 - pred_flat) * target_flat).sum()
        false_pos = (pred_flat * (1 - target_flat)).sum()
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + 
                                              self.beta * false_pos + self.smooth)
        
        return 1 - tversky


# Configuration for loss functions
DEFAULT_LOSS_CONFIG = {
    # DFL settings
    'box_weight': 7.5,
    'reg_max': 16,
    
    # Focal Loss settings
    'cls_alpha': 0.25,
    'cls_gamma': 2.0,
    'cls_weight': 0.5,
    
    # Objectness settings
    'obj_pos_weight': 1.0,
    'obj_weight': 1.0,
    
    # Optional components
    'use_ciou': True,
    'ciou_weight': 1.0,
    
    'use_plb': True,
    'plb_min_pixels': 100,
    'plb_factor': 1.0
}
