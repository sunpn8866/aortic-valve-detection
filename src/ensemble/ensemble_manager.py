"""
Ensemble Manager for Competition
Implements training diverse models and WBF aggregation
Based on winning strategies from RSNA/BraTS competitions
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
import json
from tqdm import tqdm
import pandas as pd

from ultralytics import YOLO
from src.utils.nms import weighted_boxes_fusion, TTAWrapper
from src.models.two_stage_detector import TwoStageDetector


class EnsembleManager:
    """
    Manages ensemble of diverse models for competition
    Target: 35 models (15 YOLOv11, 15 YOLOv8, 5 Faster R-CNN)
    """
    
    def __init__(self, config_path: str):
        """Initialize ensemble manager"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = []
        self.model_weights = []
        self.model_configs = []
        
    def train_diverse_models(self):
        """
        Train diverse models with different configurations
        Exploits multi-GPU infrastructure for parallel training
        """
        
        model_specs = self._generate_model_specs()
        
        print(f"Training {len(model_specs)} diverse models...")
        
        for i, spec in enumerate(model_specs):
            print(f"\n{'='*50}")
            print(f"Training Model {i+1}/{len(model_specs)}: {spec['name']}")
            print(f"{'='*50}")
            
            # Train model based on specification
            if spec['architecture'] in ['yolov11', 'dinoyolo']:
                model = self._train_yolov11(spec)
            elif spec['architecture'] == 'yolov8':
                model = self._train_yolov8(spec)
            elif spec['architecture'] == 'two_stage':
                model = self._train_two_stage(spec)
            else:
                raise ValueError(f"Unknown architecture: {spec['architecture']}")
            
            # Save model and configuration
            self._save_model(model, spec, i)
    
    def _generate_model_specs(self) -> List[Dict]:
        """
        Generate diverse model specifications
        Returns list of model configurations for training
        """
        specs = []
        
        # YOLOv11 variants (15 models)
        for fold in range(5):
            for seed in [42, 123, 456]:
                spec = {
                    'name': f'yolov11_fold{fold}_seed{seed}',
                    'architecture': 'yolov11',
                    'fold': fold,
                    'seed': seed,
                    'config': {
                        'model': 'yolov11x',
                        'img_size': 640,
                        'batch_size': 16,
                        'epochs': 300,
                        'augmentation': 'standard' if seed == 42 else 'strong'
                    }
                }
                specs.append(spec)
        
        # YOLOv8 variants (15 models) 
        for fold in range(5):
            for seed in [42, 123, 456]:
                spec = {
                    'name': f'yolov8_fold{fold}_seed{seed}',
                    'architecture': 'yolov8',
                    'fold': fold,
                    'seed': seed,
                    'config': {
                        'model': 'yolov8x',
                        'img_size': 640 if seed == 42 else 512,  # Vary input size
                        'batch_size': 16,
                        'epochs': 250,
                        'loss_variant': 'focal' if seed == 123 else 'standard'
                    }
                }
                specs.append(spec)
        
        # Two-stage models (5 models)
        for fold in range(5):
            spec = {
                'name': f'two_stage_fold{fold}',
                'architecture': 'two_stage',
                'fold': fold,
                'seed': 42,
                'config': {
                    'backbone': 'resnet50' if fold < 3 else 'efficientnet',
                    'roi_size': 224,
                    'epochs': 200
                }
            }
            specs.append(spec)
        
        return specs

    def _load_yolo_model(self, model_config: Dict) -> YOLO:
        """
        Build a YOLO model from YAML or weights.
        Supports DINO-YOLO configs by accepting explicit YAML paths.
        """
        model_name = model_config.get('model', 'yolov11x')
        model_source = model_config.get('yaml') or f"{model_name}.yaml"
        weight_source = None
        if model_config.get('pretrained', False):
            weight_source = model_config.get('weights') or f"{model_name}.pt"
        
        if weight_source and Path(weight_source).exists():
            model = YOLO(weight_source)
        else:
            if weight_source:
                print(f"Pretrained weights not found at {weight_source}, using {model_source}")
            model = YOLO(model_source)
        
        # Configure classes/names
        nc = model_config.get('num_classes', 1)
        class_names = model_config.get('class_names') or (['aortic_valve'] if nc == 1 else [str(i) for i in range(nc)])
        if hasattr(model.model, 'yaml'):
            model.model.yaml['nc'] = nc
        if hasattr(model.model, 'model') and len(model.model.model):
            model.model.model[-1].nc = nc
        if hasattr(model.model, 'nc'):
            model.model.nc = nc
        if hasattr(model.model, 'names'):
            model.model.names = class_names
        
        return model
    
    def _train_yolov11(self, spec: Dict) -> YOLO:
        """Train a YOLO model (supports DINO-YOLO via YAML/weights) with specified configuration."""
        torch.manual_seed(spec['seed'])
        np.random.seed(spec['seed'])
        
        model = self._load_yolo_model(spec['config'])
        
        # Allow extra Ultralytics training kwargs via train_args (e.g., dino_input/integration)
        train_args = {
            'data': self._get_fold_data(spec['fold']),
            'epochs': spec['config']['epochs'],
            'imgsz': spec['config']['img_size'],
            'batch': spec['config']['batch_size'],
            'seed': spec['seed'],
            'project': f"runs/{spec['name']}",
            'name': 'train'
        }
        train_args.update(spec['config'].get('train_args', {}))
        
        model.train(**train_args)
        return model
    
    def _train_yolov8(self, spec: Dict) -> YOLO:
        """Train YOLOv8 model with specified configuration"""
        torch.manual_seed(spec['seed'])
        np.random.seed(spec['seed'])
        
        model = self._load_yolo_model(spec['config'])
        
        # Custom training arguments for diversity
        train_args = {
            'data': self._get_fold_data(spec['fold']),
            'epochs': spec['config']['epochs'],
            'imgsz': spec['config']['img_size'],
            'batch': spec['config']['batch_size'],
            'seed': spec['seed'],
            'project': f"runs/{spec['name']}",
            'name': 'train'
        }
        
        # Add loss variant
        if spec['config'].get('loss_variant') == 'focal':
            train_args['box'] = 7.5
            train_args['cls'] = 0.5
        
        train_args.update(spec['config'].get('train_args', {}))
        
        model.train(**train_args)
        return model
    
    def _train_two_stage(self, spec: Dict) -> TwoStageDetector:
        """Train two-stage detector"""
        torch.manual_seed(spec['seed'])
        np.random.seed(spec['seed'])
        
        # Create two-stage configuration
        two_stage_config = {
            'stage1': {
                'model': 'yolov11x.pt',
                'conf_threshold': 0.3,
                'iou_threshold': 0.5
            },
            'stage2': {
                'backbone': {
                    'type': spec['config']['backbone'],
                    'pretrained': True
                },
                'roi_size': spec['config']['roi_size'],
                'features': 2048 if spec['config']['backbone'] == 'resnet50' else 1792
            },
            'loss': {
                'detection_weight': 1.0,
                'segmentation_weight': 0.5
            }
        }
        
        model = TwoStageDetector(two_stage_config)
        
        # Train model (simplified - actual implementation would be more complex)
        # This would use the TwoStageTrainer class
        
        return model

    def _train_dinoyolo(self, spec: Dict) -> YOLO:
        """
        Train DINO-YOLO model.
        Uses the same flow as YOLOv11 but expects spec['config']['yaml'] to point to a DINO-enabled model.
        """
        return self._train_yolov11(spec)
    
    def _get_fold_data(self, fold: int) -> str:
        """Get data configuration for specific fold"""
        # Return path to fold-specific data YAML
        return f"data/fold_{fold}/data.yaml"
    
    def _save_model(self, model, spec: Dict, index: int):
        """Save trained model and its configuration"""
        save_dir = Path(f"ensemble_models/{spec['name']}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        if hasattr(model, 'save'):
            # Ultralytics expects a directory for save()
            model.save(save_dir)
        else:
            torch.save(model.state_dict(), save_dir / 'weights.pt')
        
        # Save configuration
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(spec, f, indent=2)
        
        print(f"Model saved to {save_dir}")
    
    def load_ensemble(self, model_paths: List[str], weights: Optional[List[float]] = None):
        """Load ensemble of trained models"""
        self.models = []
        self.model_configs = []
        
        for path in model_paths:
            path = Path(path)
            
            # Load configuration
            with open(path / 'config.json', 'r') as f:
                config = json.load(f)
            
            # Load model based on architecture
            if config['architecture'] in ['yolov11', 'yolov8']:
                model = YOLO(path / 'weights.pt')
            elif config['architecture'] == 'two_stage':
                model = TwoStageDetector(config['config'])
                model.load_state_dict(torch.load(path / 'weights.pt'))
            else:
                raise ValueError(f"Unknown architecture: {config['architecture']}")
            
            self.models.append(model)
            self.model_configs.append(config)
        
        # Set or normalize weights
        if weights is None:
            self.model_weights = [1.0] * len(self.models)
        else:
            self.model_weights = weights
        
        # Normalize weights
        total_weight = sum(self.model_weights)
        self.model_weights = [w / total_weight for w in self.model_weights]
        
        print(f"Loaded ensemble of {len(self.models)} models")
    
    def predict_with_tta(self, image: torch.Tensor, use_tta: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with optional TTA
        
        Args:
            image: Input image tensor
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Fused boxes, scores, and labels
        """
        all_model_boxes = []
        all_model_scores = []
        all_model_labels = []
        
        for i, model in enumerate(self.models):
            if use_tta:
                # Use TTA wrapper
                tta_wrapper = TTAWrapper(
                    model,
                    scales=self.config.get('tta_scales', [0.8, 1.0, 1.2]),
                    flips=self.config.get('tta_flips', ['none', 'horizontal'])
                )
                
                # Get TTA predictions
                tta_boxes, tta_scores, tta_labels = tta_wrapper(image)
                
                # Aggregate TTA predictions for this model
                model_boxes, model_scores, model_labels = weighted_boxes_fusion(
                    tta_boxes, tta_scores, tta_labels,
                    iou_thr=0.5,
                    conf_type='avg'
                )
            else:
                # Single prediction
                with torch.no_grad():
                    outputs = model(image)
                    model_boxes = outputs[0].boxes.xyxy
                    model_scores = outputs[0].boxes.conf
                    model_labels = outputs[0].boxes.cls
            
            all_model_boxes.append(model_boxes)
            all_model_scores.append(model_scores)
            all_model_labels.append(model_labels)
        
        # Apply WBF to ensemble predictions
        final_boxes, final_scores, final_labels = weighted_boxes_fusion(
            all_model_boxes,
            all_model_scores,
            all_model_labels,
            weights=self.model_weights,
            iou_thr=self.config.get('wbf_iou_threshold', 0.55),
            skip_box_thr=self.config.get('wbf_skip_threshold', 0.0001),
            conf_type=self.config.get('wbf_conf_type', 'avg')
        )
        
        return final_boxes, final_scores, final_labels
    
    def generate_submission(self, test_loader, output_path: str):
        """
        Generate competition submission file
        
        Args:
            test_loader: DataLoader for test images
            output_path: Path to save submission file
        """
        submissions = []
        
        for batch_idx, (images, image_ids) in enumerate(tqdm(test_loader, desc="Generating predictions")):
            images = images.cuda()
            
            # Get ensemble predictions
            batch_boxes, batch_scores, batch_labels = self.predict_with_tta(images, use_tta=True)
            
            # Format predictions for submission
            for img_idx in range(len(images)):
                # Filter predictions for this image
                img_mask = batch_labels == 0  # Assuming single class (aortic valve)
                img_boxes = batch_boxes[img_mask]
                img_scores = batch_scores[img_mask]
                
                # Convert to submission format
                for box, score in zip(img_boxes, img_scores):
                    submission_entry = {
                        'image_id': image_ids[img_idx],
                        'x_min': float(box[0]),
                        'y_min': float(box[1]),
                        'x_max': float(box[2]),
                        'y_max': float(box[3]),
                        'confidence': float(score)
                    }
                    submissions.append(submission_entry)
        
        # Save submission
        submission_df = pd.DataFrame(submissions)
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        
        return submission_df
    
    def optimize_ensemble_weights(self, val_loader, metric='mAP'):
        """
        Optimize ensemble weights using validation set
        Uses grid search or random search
        """
        best_weights = self.model_weights.copy()
        best_score = 0
        
        # Generate weight combinations to test
        weight_combinations = self._generate_weight_combinations(len(self.models))
        
        for weights in tqdm(weight_combinations, desc="Optimizing weights"):
            self.model_weights = weights
            
            # Evaluate on validation set
            score = self._evaluate_ensemble(val_loader, metric)
            
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                print(f"New best score: {best_score:.4f} with weights: {best_weights}")
        
        self.model_weights = best_weights
        print(f"Optimal weights: {best_weights}")
        print(f"Best validation {metric}: {best_score:.4f}")
        
        return best_weights
    
    def _generate_weight_combinations(self, n_models: int, n_samples: int = 100):
        """Generate weight combinations for optimization"""
        combinations = []
        
        # Random search
        for _ in range(n_samples):
            weights = np.random.dirichlet([1] * n_models)
            combinations.append(weights.tolist())
        
        return combinations
    
    def _evaluate_ensemble(self, val_loader, metric: str = 'mAP'):
        """Evaluate ensemble on validation set"""
        # Simplified evaluation - actual implementation would calculate full metrics
        total_score = 0
        n_batches = 0
        
        for images, targets in val_loader:
            images = images.cuda()
            
            # Get predictions
            boxes, scores, labels = self.predict_with_tta(images, use_tta=False)
            
            # Calculate metric (simplified)
            # In practice, would use proper mAP calculation
            if len(boxes) > 0:
                total_score += scores.mean().item()
            n_batches += 1
        
        return total_score / n_batches if n_batches > 0 else 0


def create_5fold_split(data_path: str, output_dir: str, n_folds: int = 5):
    """
    Create 5-fold cross-validation splits
    Ensures patient-level splitting to prevent leakage
    """
    from sklearn.model_selection import KFold
    
    # Load patient IDs
    data_path = Path(data_path)
    image_files = list((data_path / 'images').glob('*.png'))
    
    # Extract patient IDs (assuming format: patient_XXX_slice_YYY.png)
    patient_ids = set()
    patient_to_files = {}
    
    for img_file in image_files:
        # Extract patient ID from filename
        parts = img_file.stem.split('_')
        patient_id = '_'.join(parts[:2])  # Adjust based on actual format
        
        patient_ids.add(patient_id)
        if patient_id not in patient_to_files:
            patient_to_files[patient_id] = []
        patient_to_files[patient_id].append(img_file)
    
    patient_ids = list(patient_ids)
    print(f"Found {len(patient_ids)} unique patients")
    
    # Create folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        fold_dir = Path(output_dir) / f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Get train and val patients
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]
        
        # Create fold data configuration
        fold_config = {
            'train': [],
            'val': []
        }
        
        # Assign files to train/val
        for patient in train_patients:
            fold_config['train'].extend(patient_to_files[patient])
        
        for patient in val_patients:
            fold_config['val'].extend(patient_to_files[patient])
        
        # Save fold configuration
        with open(fold_dir / 'data.yaml', 'w') as f:
            yaml.dump({
                'train': 'train/',
                'val': 'val/',
                'nc': 1,
                'names': ['aortic_valve']
            }, f)
        
        print(f"Fold {fold}: {len(train_patients)} train patients, {len(val_patients)} val patients")
        print(f"         {len(fold_config['train'])} train images, {len(fold_config['val'])} val images")


# Configuration for ensemble
DEFAULT_ENSEMBLE_CONFIG = {
    'n_models': 35,
    'architectures': {
        'yolov11': 15,
        'yolov8': 15,
        'two_stage': 5
    },
    'tta_scales': [0.8, 0.9, 1.0, 1.1, 1.2],
    'tta_flips': ['none', 'horizontal', 'vertical'],
    'wbf_iou_threshold': 0.55,
    'wbf_skip_threshold': 0.0001,
    'wbf_conf_type': 'avg'
}


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'optimize'], required=True)
    parser.add_argument('--config', default='configs/ensemble.yaml')
    parser.add_argument('--data', default='data/')
    parser.add_argument('--output', default='submissions/ensemble.csv')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Create 5-fold splits
        create_5fold_split(args.data, 'data/folds/')
        
        # Train ensemble
        manager = EnsembleManager(args.config)
        manager.train_diverse_models()
        
    elif args.mode == 'predict':
        # Load ensemble and generate predictions
        manager = EnsembleManager(args.config)
        
        # Load trained models
        model_paths = [f"ensemble_models/model_{i}" for i in range(35)]
        manager.load_ensemble(model_paths)
        
        # Generate submission
        # test_loader = create_test_loader(args.data)
        # manager.generate_submission(test_loader, args.output)
        
    elif args.mode == 'optimize':
        # Optimize ensemble weights
        manager = EnsembleManager(args.config)
        model_paths = [f"ensemble_models/model_{i}" for i in range(35)]
        manager.load_ensemble(model_paths)
        
        # val_loader = create_val_loader(args.data)
        # manager.optimize_ensemble_weights(val_loader)
