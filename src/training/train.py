#!/usr/bin/env python3
"""
Minimal YOLO training/validation script (DINO-YOLO compatible).
Uses Ultralytics' built-in training to avoid missing custom losses/callbacks.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure local ultralytics package is discoverable BEFORE import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from ultralytics import YOLO


def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def select_data_yaml(config: Dict[str, Any], override: str | None) -> Path:
    """Pick data yaml path: CLI override > config['data_yaml'] > config['data']['data_yaml']."""
    candidates = []
    if override:
        candidates.append(Path(override))
    if "data_yaml" in config:
        candidates.append(Path(config["data_yaml"]))
    if isinstance(config.get("data"), dict) and "data_yaml" in config["data"]:
        candidates.append(Path(config["data"]["data_yaml"]))

    for path in candidates:
        if path and path.exists():
            return path

    raise FileNotFoundError(
        "Data yaml not found. Pass --data /path/to/data.yaml or add data_yaml to the config."
    )


def build_model(model_cfg: Dict[str, Any]) -> YOLO:
    """Create YOLO model from YAML or weights and set class metadata."""
    model_source = model_cfg.get("yaml") or f"{model_cfg['name']}.yaml"
    weight_source = model_cfg.get("weights")

    if weight_source and Path(weight_source).exists():
        model = YOLO(weight_source)
        print(f"Loaded pretrained weights: {weight_source}")
    else:
        if weight_source:
            print(f"Pretrained weights not found at {weight_source}, using {model_source}")
        model = YOLO(model_source)

    nc = model_cfg.get("num_classes", 1)
    class_names = model_cfg.get("class_names") or (["aortic_valve"] if nc == 1 else [str(i) for i in range(nc)])
    if hasattr(model.model, "yaml"):
        model.model.yaml["nc"] = nc
    if hasattr(model.model, "model") and len(model.model.model):
        model.model.model[-1].nc = nc
    if hasattr(model.model, "nc"):
        model.model.nc = nc
    if hasattr(model.model, "names"):
        model.model.names = class_names

    return model


def train_and_val(config: Dict[str, Any], data_yaml: Path, device: str | None, resume: str | None):
    model = build_model(config["model"])

    project = Path(config["experiment"]["save_dir"])
    name = config["experiment"]["name"]

    train_args = {
        "data": str(data_yaml),
        "epochs": config["train"]["epochs"],
        "imgsz": config["data"]["img_size"],
        "batch": config["train"]["batch_size"],
        "seed": config.get("seed", 42),
        "project": str(project),
        "name": name,
    }
    if device:
        train_args["device"] = device
    if resume:
        train_args["resume"] = resume

    print(f"Starting training with model: {config['model'].get('yaml') or config['model'].get('weights')}")
    model.train(**train_args)

    print("Running validation...")
    model.val(data=str(data_yaml), imgsz=config["data"]["img_size"], batch=config["val"]["batch_size"])


def main():
    parser = argparse.ArgumentParser(description="Train YOLO (DINO backbone supported)")
    parser.add_argument("--config", type=str, default="configs/yolov12_dino_base.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to data YAML (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device string for Ultralytics (e.g., 0 or 0,1)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_yaml = select_data_yaml(cfg, args.data)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    train_and_val(cfg, data_yaml, args.device, args.resume)


if __name__ == "__main__":
    main()
