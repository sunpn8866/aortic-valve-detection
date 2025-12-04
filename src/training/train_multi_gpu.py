#!/usr/bin/env python3
"""
Multi-GPU launcher using Ultralytics' built-in DDP (via device string).
This is a thin wrapper around model.train() for DINO-YOLO compatibility.
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

# Propagate PYTHONPATH to spawned DDP workers so they see local ultralytics
existing_pp = os.environ.get("PYTHONPATH", "")
paths = [str(PROJECT_ROOT)] + ([existing_pp] if existing_pp else [])
os.environ["PYTHONPATH"] = ":".join([p for p in paths if p])

import yaml
from ultralytics import YOLO


def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def select_data_yaml(config: Dict[str, Any], override: str | None):
    if override:
        return Path(override)
    if "data_yaml" in config:
        return Path(config["data_yaml"])
    if isinstance(config.get("data"), dict) and "data_yaml" in config["data"]:
        return Path(config["data"]["data_yaml"])
    raise FileNotFoundError("Data yaml not found. Pass --data or add data_yaml to the config.")


def build_model(model_cfg: Dict[str, Any]) -> YOLO:
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


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU YOLO trainer (Ultralytics built-in DDP)")
    parser.add_argument("--config", type=str, default="configs/yolov12_dino_base.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to data YAML")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g., 0,1 or 0,1,2,3")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_yaml = select_data_yaml(cfg, args.data)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model = build_model(cfg["model"])

    train_args = {
        "data": str(data_yaml),
        "epochs": cfg["train"]["epochs"],
        "imgsz": cfg["data"]["img_size"],
        "batch": cfg["train"]["batch_size"],
        "seed": cfg.get("seed", 42),
        "project": str(Path(cfg["experiment"]["save_dir"])),
        "name": cfg["experiment"]["name"],
    }
    if args.device:
        train_args["device"] = args.device  # Ultralytics will spawn DDP if multiple devices are provided

    print(f"Starting multi-GPU training with model: {cfg['model'].get('yaml') or cfg['model'].get('weights')}")
    model.train(**train_args)


if __name__ == "__main__":
    main()
