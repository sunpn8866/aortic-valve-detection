"""
Two-Stage Detection Pipeline for Aortic Valve Detection
Based on RSNA 2023/2024 winning strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from ultralytics import YOLO
from torchvision.ops import roi_align


class TwoStageDetector(nn.Module):
    """
    Two-stage detection pipeline:
    Stage 1: Localization - Find cardiac region and valve candidates
    Stage 2: Refinement - Precise valve detection with auxiliary segmentation
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Stage 1: Localization Network (YOLOv11)
        self.stage1_detector = YOLO(config['stage1']['model'])
        
        # Stage 2: Refinement Network with Dual Decoder
        self.stage2_backbone = self._build_backbone(config['stage2']['backbone'])
        
        # Dual decoder architecture (RSNA 2023 innovation)
        self.detection_head = self._build_detection_head()
        self.segmentation_head = self._build_segmentation_head()
        
        # Auxiliary components
        self.attention_pool = AttentionPooling(
            in_channels=config['stage2']['features'],
            hidden_dim=256
        )
        
    def _build_backbone(self, backbone_config):
        """Build Stage 2 backbone (ResNet or EfficientNet)"""
        if backbone_config['type'] == 'resnet50':
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=True)
            # Remove final layers
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif backbone_config['type'] == 'efficientnet':
            import timm
            backbone = timm.create_model(
                'efficientnet_b4',
                pretrained=True,
                features_only=True,
                out_indices=[3, 4]
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_config['type']}")
        
        return backbone
    
    def _build_detection_head(self):
        """Detection decoder for precise bounding box regression"""
        return nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: [batch, 5, H, W] - (x, y, w, h, conf)
            nn.Conv2d(128, 5, 1)
        )
    
    def _build_segmentation_head(self):
        """Segmentation decoder for auxiliary loss"""
        return nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Binary segmentation for valve region
            nn.Conv2d(64, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Two-stage forward pass
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Dictionary containing:
            - stage1_boxes: Candidate regions from Stage 1
            - refined_boxes: Final refined detections
            - segmentation_masks: Auxiliary segmentation outputs
            - attention_weights: Attention maps for interpretability
        """
        batch_size = x.shape[0]
        
        # Stage 1: Localization (find cardiac region and candidates)
        with torch.no_grad():  # Freeze Stage 1 during training
            stage1_outputs = self.stage1_detector(x)
            candidate_boxes = self._extract_candidates(stage1_outputs)
        
        # Extract ROIs for Stage 2
        rois = self._extract_rois(x, candidate_boxes)
        
        # Stage 2: Refinement with dual decoder
        features = self.stage2_backbone(rois)
        
        # Get features from last two layers (RSNA 2023 technique)
        if isinstance(features, (list, tuple)):
            feat_last = features[-1]
            feat_second_last = features[-2]
        else:
            feat_last = features
            feat_second_last = features
        
        # Detection branch
        detection_output = self.detection_head(feat_last)
        refined_boxes = self._decode_boxes(detection_output)
        
        # Segmentation branch (auxiliary)
        # Combine features from last and second-last layers
        combined_features = torch.cat([
            feat_last,
            F.interpolate(feat_second_last, size=feat_last.shape[-2:], mode='bilinear')
        ], dim=1) if feat_second_last is not None else feat_last
        
        segmentation_output = self.segmentation_head(combined_features)
        segmentation_masks = torch.sigmoid(segmentation_output)
        
        # Attention pooling for additional context
        attention_weights = self.attention_pool(feat_last)
        
        return {
            'stage1_boxes': candidate_boxes,
            'refined_boxes': refined_boxes,
            'segmentation_masks': segmentation_masks,
            'attention_weights': attention_weights,
            'detection_logits': detection_output,
            'segmentation_logits': segmentation_output
        }
    
    def _extract_candidates(self, stage1_outputs) -> List[torch.Tensor]:
        """Extract candidate boxes from Stage 1 detector"""
        candidates = []
        
        for output in stage1_outputs:
            if hasattr(output, 'boxes'):
                # Extract boxes with confidence > threshold
                boxes = output.boxes
                conf_mask = boxes.conf > self.config['stage1']['conf_threshold']
                filtered_boxes = boxes.xyxy[conf_mask]
                candidates.append(filtered_boxes)
            else:
                candidates.append(torch.empty(0, 4))
        
        return candidates
    
    def _extract_rois(self, images: torch.Tensor, boxes: List[torch.Tensor]) -> torch.Tensor:
        """Extract Region of Interest features"""
        roi_size = self.config['stage2']['roi_size']
        all_rois = []
        
        for img, box_list in zip(images, boxes):
            if len(box_list) > 0:
                # Convert to ROI align format
                rois = roi_align(
                    img.unsqueeze(0),
                    [box_list],
                    output_size=(roi_size, roi_size),
                    spatial_scale=1.0,
                    aligned=True
                )
                all_rois.append(rois)
            else:
                # No candidates, use full image
                resized = F.interpolate(
                    img.unsqueeze(0),
                    size=(roi_size, roi_size),
                    mode='bilinear'
                )
                all_rois.append(resized)
        
        return torch.cat(all_rois, dim=0)
    
    def _decode_boxes(self, detection_output: torch.Tensor) -> torch.Tensor:
        """Decode detection output to bounding boxes"""
        # Apply sigmoid to get probabilities
        detection_output = torch.sigmoid(detection_output)
        
        # Extract box coordinates and confidence
        boxes = detection_output[:, :4]  # x, y, w, h
        confidence = detection_output[:, 4:5]  # confidence score
        
        # Convert to xyxy format if needed
        x, y, w, h = boxes.split(1, dim=1)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        refined_boxes = torch.cat([x1, y1, x2, y2, confidence], dim=1)
        
        return refined_boxes


class AttentionPooling(nn.Module):
    """
    Attention pooling module for Stage 2
    Based on RSNA 2024 2nd place solution
    """
    
    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-weighted pooling
        
        Args:
            x: Feature maps [B, C, H, W]
            
        Returns:
            Attention weights [B, 1, H, W]
        """
        # Calculate attention weights
        weights = self.attention(x)
        weights = torch.softmax(weights.flatten(2), dim=2).reshape_as(weights)
        
        # Apply weighted pooling
        pooled = torch.sum(x * weights, dim=(2, 3), keepdim=True)
        
        return weights


class AuxiliarySegmentationLoss(nn.Module):
    """
    Auxiliary segmentation loss from RSNA 2023
    Provides +0.01 to +0.03 performance boost
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred_masks: torch.Tensor, true_masks: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined Dice + BCE loss
        
        Args:
            pred_masks: Predicted segmentation masks [B, 1, H, W]
            true_masks: Ground truth masks [B, 1, H, W]
            
        Returns:
            Combined loss scalar
        """
        # Binary Cross-Entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, true_masks)
        
        # Dice loss
        pred_masks_sigmoid = torch.sigmoid(pred_masks)
        intersection = (pred_masks_sigmoid * true_masks).sum(dim=(2, 3))
        union = pred_masks_sigmoid.sum(dim=(2, 3)) + true_masks.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + 1e-7) / (union + 1e-7)
        dice_loss = dice_loss.mean()
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss


class TwoStageTrainer:
    """Training wrapper for two-stage detector"""
    
    def __init__(self, model: TwoStageDetector, config: Dict):
        self.model = model
        self.config = config
        
        # Loss functions
        self.detection_loss = DistributionFocalLoss(
            alpha=0.25,
            gamma=2.0
        )
        self.auxiliary_seg_loss = AuxiliarySegmentationLoss(
            dice_weight=0.5,
            bce_weight=0.5
        )
        
        # Loss weights
        self.detection_weight = config['loss']['detection_weight']
        self.segmentation_weight = config['loss']['segmentation_weight']
        
    def compute_loss(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with all components
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth dictionary
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Detection loss
        if 'refined_boxes' in outputs and 'boxes' in targets:
            det_loss = self.detection_loss(
                outputs['refined_boxes'],
                targets['boxes']
            )
            losses['detection'] = det_loss * self.detection_weight
        
        # Auxiliary segmentation loss (key innovation)
        if 'segmentation_logits' in outputs and 'masks' in targets:
            seg_loss = self.auxiliary_seg_loss(
                outputs['segmentation_logits'],
                targets['masks']
            )
            losses['segmentation'] = seg_loss * self.segmentation_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss for precise boundary regression
    Key component for small object detection
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reg_max: int = 16):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        
    def forward(self, pred_dist: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Calculate DFL loss
        
        Args:
            pred_dist: Predicted distribution [B, 4, reg_max]
            target_boxes: Target boxes [B, 4]
            
        Returns:
            DFL loss value
        """
        # Convert continuous targets to discrete distribution
        target_dist = self._label_to_distribution(target_boxes)
        
        # Calculate focal loss on distribution
        focal_weight = (1 - pred_dist) ** self.gamma
        focal_loss = -self.alpha * focal_weight * target_dist * torch.log(pred_dist + 1e-7)
        
        return focal_loss.mean()
    
    def _label_to_distribution(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert continuous labels to discrete distribution"""
        # Implementation details for distribution conversion
        # This is simplified - actual implementation would be more complex
        batch_size = labels.shape[0]
        distribution = torch.zeros(batch_size, 4, self.reg_max, device=labels.device)
        
        # Convert each coordinate to distribution
        for i in range(4):
            coord = labels[:, i] * (self.reg_max - 1)
            lower = coord.floor().long()
            upper = lower + 1
            
            # Linear interpolation for soft labels
            upper_weight = coord - lower.float()
            lower_weight = 1 - upper_weight
            
            # Assign weights
            valid_mask = (lower >= 0) & (upper < self.reg_max)
            distribution[valid_mask, i, lower[valid_mask]] = lower_weight[valid_mask]
            distribution[valid_mask, i, upper[valid_mask]] = upper_weight[valid_mask]
        
        return distribution


# Configuration for two-stage pipeline
DEFAULT_CONFIG = {
    'stage1': {
        'model': 'yolov11x.pt',
        'conf_threshold': 0.3,
        'iou_threshold': 0.5
    },
    'stage2': {
        'backbone': {
            'type': 'resnet50',
            'pretrained': True
        },
        'roi_size': 224,
        'features': 2048
    },
    'loss': {
        'detection_weight': 1.0,
        'segmentation_weight': 0.5  # Auxiliary loss weight
    }
}
