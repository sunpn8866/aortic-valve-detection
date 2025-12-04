"""
Medical imaging dataset for aortic valve detection
Implements MONAI-based augmentations and CT-specific preprocessing
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import cv2
import json
from PIL import Image

# Medical imaging libraries
try:
    import monai
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
        Orientationd, ScaleIntensityRanged, CropForegroundd,
        RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
        RandGaussianSmoothd, RandShiftIntensityd, RandBiasFieldd,
        RandGhostingd, ToTensord
    )
    MONAI_AVAILABLE = True
except ImportError:
    print("Warning: MONAI not installed. Using basic augmentations.")
    MONAI_AVAILABLE = False

import albumentations as A
from albumentations.pytorch import ToTensorV2


class AorticValveDataset(Dataset):
    """Dataset for aortic valve detection in cardiac CT"""
    
    def __init__(
        self,
        data_path: Path,
        img_size: int = 640,
        augmentation: Optional[Dict[str, Any]] = None,
        preprocessing: Optional[Dict[str, Any]] = None,
        is_training: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to data directory
            img_size: Target image size
            augmentation: Augmentation configuration
            preprocessing: Preprocessing configuration
            is_training: Whether this is training dataset
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.augmentation = augmentation or {}
        self.preprocessing = preprocessing or {}
        self.is_training = is_training
        
        # Load image paths and annotations
        self.images, self.annotations = self._load_data()
        
        # Setup transforms
        self.transform = self._build_transforms()
        
        print(f"Dataset initialized with {len(self.images)} images")
    
    def _load_data(self) -> Tuple[List[Path], List[Dict]]:
        """Load image paths and annotations"""
        images = []
        annotations = []
        
        # Expected structure:
        # data_path/
        #   images/
        #     image1.png
        #   labels/
        #     image1.txt (YOLO format)
        
        image_dir = self.data_path / 'images'
        label_dir = self.data_path / 'labels'
        
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        # Supported image formats
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.dcm']
        
        for ext in extensions:
            for img_path in image_dir.glob(ext):
                # Check if corresponding label exists
                label_path = label_dir / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    images.append(img_path)
                    
                    # Load YOLO format annotations
                    with open(label_path, 'r') as f:
                        boxes = []
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:  # class x_center y_center width height
                                cls = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                boxes.append({
                                    'class': cls,
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'width': width,
                                    'height': height
                                })
                        
                        annotations.append(boxes)
                else:
                    # Image without annotation (for test set)
                    images.append(img_path)
                    annotations.append([])
        
        return images, annotations
    
    def _build_transforms(self):
        """Build augmentation pipeline"""
        
        if MONAI_AVAILABLE and self.is_training:
            # MONAI transforms for medical images
            transforms = self._build_monai_transforms()
        else:
            # Fallback to Albumentations
            transforms = self._build_albumentation_transforms()
        
        return transforms
    
    def _build_monai_transforms(self):
        """Build MONAI transformation pipeline"""
        transform_list = []
        
        # Basic loading and formatting
        transform_list.append(LoadImaged(keys=['image']))
        transform_list.append(EnsureChannelFirstd(keys=['image']))
        
        # Medical image preprocessing
        if self.preprocessing:
            # CT windowing
            if 'window_center' in self.preprocessing:
                transform_list.append(
                    ScaleIntensityRanged(
                        keys=['image'],
                        a_min=self.preprocessing['window_center'] - self.preprocessing['window_width'] / 2,
                        a_max=self.preprocessing['window_center'] + self.preprocessing['window_width'] / 2,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True
                    )
                )
        
        # Augmentations
        if self.is_training and self.augmentation:
            aug_config = self.augmentation
            
            # Spatial transforms
            if 'RandomFlip' in aug_config:
                transform_list.append(
                    RandFlipd(
                        keys=['image'],
                        prob=aug_config['RandomFlip']['prob'],
                        spatial_axis=aug_config['RandomFlip']['spatial_axis']
                    )
                )
            
            if 'RandomRotate90' in aug_config:
                transform_list.append(
                    RandRotate90d(
                        keys=['image'],
                        prob=aug_config['RandomRotate90']['prob'],
                        spatial_axis=aug_config['RandomRotate90']['spatial_axis']
                    )
                )
            
            if 'RandomAffine' in aug_config:
                transform_list.append(
                    RandAffined(
                        keys=['image'],
                        prob=aug_config['RandomAffine']['prob'],
                        rotate_range=aug_config['RandomAffine']['rotate_range'],
                        translate_range=aug_config['RandomAffine']['translate_range'],
                        scale_range=aug_config['RandomAffine']['scale_range']
                    )
                )
            
            # Intensity transforms
            if 'RandomGaussianNoise' in aug_config:
                transform_list.append(
                    RandGaussianNoised(
                        keys=['image'],
                        prob=aug_config['RandomGaussianNoise']['prob'],
                        mean=aug_config['RandomGaussianNoise']['mean'],
                        std=aug_config['RandomGaussianNoise']['std']
                    )
                )
            
            if 'RandomGaussianSmooth' in aug_config:
                transform_list.append(
                    RandGaussianSmoothd(
                        keys=['image'],
                        prob=aug_config['RandomGaussianSmooth']['prob'],
                        sigma_x=aug_config['RandomGaussianSmooth']['sigma'],
                        sigma_y=aug_config['RandomGaussianSmooth']['sigma']
                    )
                )
            
            # Medical-specific augmentations
            if 'RandomBiasField' in aug_config:
                transform_list.append(
                    RandBiasFieldd(
                        keys=['image'],
                        prob=aug_config['RandomBiasField']['prob'],
                        degree=aug_config['RandomBiasField']['degree']
                    )
                )
        
        # Convert to tensor
        transform_list.append(ToTensord(keys=['image']))
        
        return Compose(transform_list)
    
    def _build_albumentation_transforms(self):
        """Build Albumentations transformation pipeline (fallback)"""
        transform_list = []
        
        # Preprocessing
        if self.preprocessing and self.preprocessing.get('normalize', True):
            transform_list.append(
                A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0)
            )
        
        # Augmentations
        if self.is_training and self.augmentation:
            # Basic augmentations
            transform_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.2),
            ])
        
        # Resize to target size
        transform_list.append(A.Resize(self.img_size, self.img_size))
        
        # Convert to tensor
        transform_list.append(ToTensorV2())
        
        return A.Compose(
            transform_list,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ) if self.is_training else None
        )
    
    def apply_ct_windowing(self, image: np.ndarray) -> np.ndarray:
        """Apply CT windowing for soft tissue visualization"""
        if not self.preprocessing:
            return image
        
        center = self.preprocessing.get('window_center', 50)
        width = self.preprocessing.get('window_width', 350)
        
        # Apply windowing
        min_val = center - width / 2
        max_val = center + width / 2
        
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)
        
        return image
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset"""
        
        # Load image
        img_path = self.images[idx]
        image = self._load_image(img_path)
        
        # Get annotations
        annotations = self.annotations[idx] if idx < len(self.annotations) else []
        
        # Apply transforms
        if MONAI_AVAILABLE and isinstance(self.transform, monai.transforms.Compose):
            # MONAI transforms
            data_dict = {'image': image}
            transformed = self.transform(data_dict)
            image = transformed['image']
            
            # Convert annotations to tensor
            if annotations:
                boxes = torch.tensor([
                    [ann['x_center'], ann['y_center'], ann['width'], ann['height'], ann['class']]
                    for ann in annotations
                ], dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 5), dtype=torch.float32)
        else:
            # Albumentations transforms
            if annotations and self.is_training:
                # Convert to format expected by albumentations
                bboxes = [[ann['x_center'], ann['y_center'], ann['width'], ann['height']] 
                          for ann in annotations]
                class_labels = [ann['class'] for ann in annotations]
                
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                image = transformed['image']
                
                # Convert back to tensor format
                if transformed['bboxes']:
                    boxes = torch.tensor([
                        bbox + [cls] for bbox, cls in zip(transformed['bboxes'], transformed['class_labels'])
                    ], dtype=torch.float32)
                else:
                    boxes = torch.zeros((0, 5), dtype=torch.float32)
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
                boxes = torch.zeros((0, 5), dtype=torch.float32)
        
        return image, boxes
    
    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load image from path"""
        
        # Check if DICOM
        if img_path.suffix.lower() == '.dcm':
            try:
                import pydicom
                dcm = pydicom.dcmread(str(img_path))
                image = dcm.pixel_array.astype(np.float32)
                
                # Apply DICOM rescale
                if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                    image = image * dcm.RescaleSlope + dcm.RescaleIntercept
                
                # Apply CT windowing
                image = self.apply_ct_windowing(image)
                
                # Convert to 8-bit
                image = (image * 255).astype(np.uint8)
                
            except Exception as e:
                print(f"Error loading DICOM {img_path}: {e}")
                # Fallback to zeros
                image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            # Regular image file
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Error loading image {img_path}")
                image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Ensure single channel image has correct shape
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        return image
    
    def collate_fn(self, batch):
        """Custom collate function for batching"""
        images, targets = zip(*batch)
        
        # Stack images
        images = torch.stack(images, 0)
        
        # Handle variable number of boxes per image
        # Pad targets to same size
        max_boxes = max(len(t) for t in targets)
        
        if max_boxes > 0:
            padded_targets = []
            for i, target in enumerate(targets):
                if len(target) > 0:
                    # Add batch index as first column
                    batch_idx = torch.full((len(target), 1), i, dtype=torch.float32)
                    padded_target = torch.cat([batch_idx, target], dim=1)
                    
                    # Pad if necessary
                    if len(padded_target) < max_boxes:
                        padding = torch.zeros((max_boxes - len(padded_target), 6))
                        padded_target = torch.cat([padded_target, padding], dim=0)
                    
                    padded_targets.append(padded_target)
                else:
                    # No boxes in this image
                    padded_targets.append(torch.zeros((max_boxes, 6)))
            
            targets = torch.stack(padded_targets, 0)
        else:
            # No boxes in entire batch
            targets = torch.zeros((len(batch), 1, 6))
        
        return images, targets


def create_data_loaders(config: Dict[str, Any], num_workers: int = 4):
    """Helper function to create data loaders"""
    
    # Training dataset
    train_dataset = AorticValveDataset(
        data_path=Path(config['data']['train_path']),
        img_size=config['data']['img_size'],
        augmentation=config['data']['augmentation'],
        preprocessing=config['data']['preprocessing'],
        is_training=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation dataset
    val_dataset = AorticValveDataset(
        data_path=Path(config['data']['val_path']),
        img_size=config['data']['img_size'],
        preprocessing=config['data']['preprocessing'],
        is_training=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['val']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    import yaml
    
    # Load config
    with open('configs/yolov11_base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = AorticValveDataset(
        data_path=Path(config['data']['train_path']),
        img_size=config['data']['img_size'],
        augmentation=config['data']['augmentation'],
        preprocessing=config['data']['preprocessing'],
        is_training=True
    )
    
    # Test loading
    if len(dataset) > 0:
        image, boxes = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Boxes shape: {boxes.shape}")
        print(f"Sample boxes: {boxes[:5]}")
    else:
        print("No data found. Please check data path.")
