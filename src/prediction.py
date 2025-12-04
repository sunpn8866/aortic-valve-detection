#!/usr/bin/env python3
"""
Generate competition submission from a trained YOLO/DINO model.
Scans all PNG images under the test root (recursively) and writes submission.txt.
"""

import argparse
import sys
from pathlib import Path
import csv

# Ensure local ultralytics is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run inference and create submission.txt")
    parser.add_argument(
        "--weights",
        type=str,
        default="experiments/yolov12_dino_baseline6/weights/best.pt",
        help="Path to trained weights",
    )
    parser.add_argument(
        "--test-root",
        type=str,
        default="/home/a110101162/ai_cup/dataset/testing_image",
        help="Root directory containing test images (can have nested patient folders)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.txt",
        help="Output submission file",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device string, e.g., 0 or 0,1")
    parser.add_argument("--batch", type=int, default=1, help="Batch size per predict call")
    parser.add_argument("--chunk", type=int, default=8, help="Number of images per predict call")
    parser.add_argument("--half", action="store_true", help="Use FP16 for inference (if supported)")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    test_root = Path(args.test_root).resolve()
    out_path = Path(args.output).resolve()

    images = sorted(test_root.rglob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found under {test_root}")

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"Found {len(images)} test images. Running inference in chunks (chunk={args.chunk}, batch={args.batch})...")

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(images), args.chunk):
            subset = [str(p) for p in images[i : i + args.chunk]]
            results = model.predict(
                source=subset,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=0.5,
                device=args.device,
                verbose=False,
                save=False,
                stream=True,
                batch=args.batch,
                half=args.half,
            )
            for res in results:
                img_name = Path(res.path).stem
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for j in range(len(boxes)):
                    cls_id = int(boxes.cls[j].item()) if boxes.cls is not None else 0
                    conf = boxes.conf[j].item()
                    x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                    line = f"{img_name} {cls_id} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                    f.write(line)
                    count += 1
            # Free CUDA cache between chunks
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"Wrote {count} detections to {out_path}")


if __name__ == "__main__":
    main()
