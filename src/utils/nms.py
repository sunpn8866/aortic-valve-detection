"""
Advanced Non-Maximum Suppression Techniques
Implements Soft-NMS, DIoU-NMS, and Weighted Boxes Fusion
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import torchvision


def soft_nms(boxes: torch.Tensor, scores: torch.Tensor, 
            sigma: float = 0.5, thresh: float = 0.001, 
            method: str = 'gaussian') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft-NMS implementation
    Better for overlapping objects than standard NMS
    
    Args:
        boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
        scores: Confidence scores [N]
        sigma: Gaussian sigma parameter
        thresh: Score threshold
        method: 'gaussian' or 'linear' decay
        
    Returns:
        Selected boxes and scores
    """
    device = boxes.device
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    N = boxes_np.shape[0]
    indexes = np.arange(N)
    
    # Areas of boxes
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by score
    order = scores_np.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU with all remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Apply soft suppression
        if method == 'gaussian':
            # Gaussian decay
            weight = np.exp(-(ovr * ovr) / sigma)
        elif method == 'linear':
            # Linear decay
            weight = np.ones_like(ovr)
            weight[ovr > thresh] = 1 - ovr[ovr > thresh]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scores_np[order[1:]] *= weight
        
        # Remove boxes with score below threshold
        inds = np.where(scores_np[order[1:]] > thresh)[0]
        order = order[inds + 1]
    
    keep = np.array(keep)
    return boxes[keep].to(device), scores[keep].to(device)


def diou_nms(boxes: torch.Tensor, scores: torch.Tensor,
            iou_threshold: float = 0.5, beta: float = 0.9) -> torch.Tensor:
    """
    DIoU-NMS: Distance-IoU based NMS
    Considers center distance in addition to IoU
    
    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
        beta: Weight for distance penalty
        
    Returns:
        Indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Sort boxes by score
    sorted_scores, order = scores.sort(descending=True)
    sorted_boxes = boxes[order]
    
    keep = []
    
    while sorted_boxes.shape[0] > 0:
        # Keep the box with highest score
        keep.append(order[0])
        
        if sorted_boxes.shape[0] == 1:
            break
        
        # Compute IoU with remaining boxes
        iou = box_iou(sorted_boxes[0:1], sorted_boxes[1:]).squeeze(0)
        
        # Compute center distance
        center_dist = compute_center_distance(sorted_boxes[0:1], sorted_boxes[1:]).squeeze(0)
        
        # DIoU = IoU - beta * center_distance
        diou = iou - beta * center_dist
        
        # Keep boxes with DIoU below threshold
        mask = diou < iou_threshold
        sorted_boxes = sorted_boxes[1:][mask]
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def compute_center_distance(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized center distance between boxes
    
    Args:
        boxes1: First set of boxes [N, 4]
        boxes2: Second set of boxes [M, 4]
        
    Returns:
        Center distances [N, M]
    """
    # Centers of boxes1
    cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
    cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    
    # Centers of boxes2
    cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
    cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    # Compute distances
    dist = torch.sqrt((cx1.unsqueeze(1) - cx2.unsqueeze(0)) ** 2 +
                     (cy1.unsqueeze(1) - cy2.unsqueeze(0)) ** 2)
    
    # Normalize by diagonal of enclosing box
    enc_x1 = torch.min(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    enc_y1 = torch.min(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    enc_x2 = torch.max(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    enc_y2 = torch.max(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    
    enc_diag = torch.sqrt((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2)
    
    return dist / (enc_diag + 1e-7)


def weighted_boxes_fusion(
    boxes_list: List[torch.Tensor],
    scores_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    weights: Optional[List[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0001,
    conf_type: str = 'avg'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted Boxes Fusion
    Superior to NMS for ensemble predictions
    
    Args:
        boxes_list: List of boxes from different models [M x [N, 4]]
        scores_list: List of scores [M x [N]]
        labels_list: List of labels [M x [N]]
        weights: Model weights [M]
        iou_thr: IoU threshold for matching boxes
        skip_box_thr: Minimum score threshold
        conf_type: Confidence aggregation method ('avg', 'max', 'box_and_model_avg')
        
    Returns:
        Fused boxes, scores, and labels
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)
    
    # Normalize weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    
    # Collect all boxes
    all_boxes = []
    all_scores = []
    all_labels = []
    all_weights = []
    
    for i, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        # Filter by threshold
        mask = scores > skip_box_thr
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        all_boxes.append(boxes)
        all_scores.append(scores * weights[i])
        all_labels.append(labels)
        all_weights.append(torch.full((len(boxes),), weights[i]))
    
    if not all_boxes:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.int64)
    
    # Concatenate all predictions
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_weights = torch.cat(all_weights, dim=0)
    
    # Sort by score
    sorted_scores, order = all_scores.sort(descending=True)
    sorted_boxes = all_boxes[order]
    sorted_labels = all_labels[order]
    sorted_weights = all_weights[order]
    
    # Fusion process
    fused_boxes = []
    fused_scores = []
    fused_labels = []
    
    while sorted_boxes.shape[0] > 0:
        # Take the top box
        current_box = sorted_boxes[0:1]
        current_score = sorted_scores[0:1]
        current_label = sorted_labels[0:1]
        current_weight = sorted_weights[0:1]
        
        # Find matching boxes
        if sorted_boxes.shape[0] > 1:
            ious = box_iou(current_box, sorted_boxes[1:]).squeeze(0)
            
            # Match by IoU and same label
            label_match = sorted_labels[1:] == current_label
            matches = (ious > iou_thr) & label_match
            
            if matches.any():
                # Get matched boxes
                matched_boxes = sorted_boxes[1:][matches]
                matched_scores = sorted_scores[1:][matches]
                matched_weights = sorted_weights[1:][matches]
                
                # Combine with current box
                all_matched_boxes = torch.cat([current_box, matched_boxes], dim=0)
                all_matched_scores = torch.cat([current_score, matched_scores], dim=0)
                all_matched_weights = torch.cat([current_weight, matched_weights], dim=0)
                
                # Weighted average for box coordinates
                weighted_boxes = all_matched_boxes * all_matched_weights.unsqueeze(1)
                fused_box = weighted_boxes.sum(dim=0) / all_matched_weights.sum()
                
                # Score aggregation
                if conf_type == 'avg':
                    fused_score = all_matched_scores.mean()
                elif conf_type == 'max':
                    fused_score = all_matched_scores.max()
                elif conf_type == 'box_and_model_avg':
                    fused_score = all_matched_scores.sum() / len(boxes_list)
                else:
                    raise ValueError(f"Unknown conf_type: {conf_type}")
                
                fused_boxes.append(fused_box)
                fused_scores.append(fused_score)
                fused_labels.append(current_label)
                
                # Remove processed boxes
                keep_mask = ~torch.cat([torch.tensor([False]), matches], dim=0)
                sorted_boxes = sorted_boxes[keep_mask]
                sorted_scores = sorted_scores[keep_mask]
                sorted_labels = sorted_labels[keep_mask]
                sorted_weights = sorted_weights[keep_mask]
            else:
                # No matches, keep the box as is
                fused_boxes.append(current_box.squeeze(0))
                fused_scores.append(current_score.squeeze(0))
                fused_labels.append(current_label.squeeze(0))
                
                # Remove processed box
                sorted_boxes = sorted_boxes[1:]
                sorted_scores = sorted_scores[1:]
                sorted_labels = sorted_labels[1:]
                sorted_weights = sorted_weights[1:]
        else:
            # Last box
            fused_boxes.append(sorted_boxes[0])
            fused_scores.append(sorted_scores[0])
            fused_labels.append(sorted_labels[0])
            break
    
    if not fused_boxes:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.int64)
    
    fused_boxes = torch.stack(fused_boxes, dim=0)
    fused_scores = torch.stack(fused_scores, dim=0)
    fused_labels = torch.stack(fused_labels, dim=0)
    
    return fused_boxes, fused_scores, fused_labels


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes
    
    Args:
        boxes1: First set of boxes [N, 4]
        boxes2: Second set of boxes [M, 4]
        
    Returns:
        IoU matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    inter_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    inter_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    
    return inter_area / (union_area + 1e-7)


class NMSManager:
    """
    Manager class for different NMS strategies
    """
    
    def __init__(self, method: str = 'soft_nms', **kwargs):
        """
        Initialize NMS manager
        
        Args:
            method: NMS method ('standard', 'soft_nms', 'diou_nms')
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.params = kwargs
        
    def apply(self, boxes: torch.Tensor, scores: torch.Tensor, 
             labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply NMS method
        
        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            labels: Optional class labels [N]
            
        Returns:
            Filtered boxes, scores, and labels
        """
        if self.method == 'standard':
            keep = torchvision.ops.nms(boxes, scores, self.params.get('iou_threshold', 0.5))
            return boxes[keep], scores[keep], labels[keep] if labels is not None else None
            
        elif self.method == 'soft_nms':
            filtered_boxes, filtered_scores = soft_nms(
                boxes, scores,
                sigma=self.params.get('sigma', 0.5),
                thresh=self.params.get('thresh', 0.001),
                method=self.params.get('decay_method', 'gaussian')
            )
            # Match labels if provided
            if labels is not None:
                # Find kept indices
                kept_mask = torch.isin(scores, filtered_scores)
                filtered_labels = labels[kept_mask]
            else:
                filtered_labels = None
            
            return filtered_boxes, filtered_scores, filtered_labels
            
        elif self.method == 'diou_nms':
            keep = diou_nms(
                boxes, scores,
                iou_threshold=self.params.get('iou_threshold', 0.5),
                beta=self.params.get('beta', 0.9)
            )
            return boxes[keep], scores[keep], labels[keep] if labels is not None else None
            
        else:
            raise ValueError(f"Unknown NMS method: {self.method}")


# Test Time Augmentation (TTA) utilities
class TTAWrapper:
    """
    Test Time Augmentation wrapper for inference
    """
    
    def __init__(self, model, scales: List[float] = [0.8, 1.0, 1.2],
                flips: List[str] = ['none', 'horizontal', 'vertical']):
        """
        Initialize TTA wrapper
        
        Args:
            model: Detection model
            scales: Scale factors for augmentation
            flips: Flip types to apply
        """
        self.model = model
        self.scales = scales
        self.flips = flips
        
    def __call__(self, image: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply TTA and aggregate predictions
        
        Args:
            image: Input image [B, C, H, W]
            
        Returns:
            Lists of boxes, scores, and labels from all augmentations
        """
        all_boxes = []
        all_scores = []
        all_labels = []
        
        original_size = image.shape[-2:]
        
        for scale in self.scales:
            for flip in self.flips:
                # Apply augmentation
                aug_image = self._apply_augmentation(image, scale, flip)
                
                # Get predictions
                with torch.no_grad():
                    predictions = self.model(aug_image)
                
                # Reverse augmentation on predictions
                boxes, scores, labels = self._reverse_augmentation(
                    predictions, scale, flip, original_size
                )
                
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)
        
        return all_boxes, all_scores, all_labels
    
    def _apply_augmentation(self, image: torch.Tensor, scale: float, flip: str) -> torch.Tensor:
        """Apply augmentation to image"""
        # Scale
        if scale != 1.0:
            size = (int(image.shape[-2] * scale), int(image.shape[-1] * scale))
            image = torch.nn.functional.interpolate(image, size=size, mode='bilinear')
        
        # Flip
        if flip == 'horizontal':
            image = torch.flip(image, dims=[-1])
        elif flip == 'vertical':
            image = torch.flip(image, dims=[-2])
        
        return image
    
    def _reverse_augmentation(self, predictions, scale: float, flip: str, 
                            original_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reverse augmentation on predictions"""
        # Extract predictions (simplified, actual implementation depends on model output format)
        boxes = predictions.get('boxes', torch.empty(0, 4))
        scores = predictions.get('scores', torch.empty(0))
        labels = predictions.get('labels', torch.empty(0, dtype=torch.int64))
        
        # Reverse scale
        if scale != 1.0:
            boxes = boxes / scale
        
        # Reverse flip
        if flip == 'horizontal':
            boxes[:, [0, 2]] = original_size[1] - boxes[:, [2, 0]]
        elif flip == 'vertical':
            boxes[:, [1, 3]] = original_size[0] - boxes[:, [3, 1]]
        
        return boxes, scores, labels
