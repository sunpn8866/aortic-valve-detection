# Aortic Valve Detection

AI CUP 2025 秋季賽 - 電腦斷層主動脈瓣物件偵測競賽

## 方法概述

本專案使用 **YOLOv12 + RadioDINO** 混合架構進行主動脈瓣偵測：
- 基礎模型：YOLOv12m
- 骨幹網路：RadioDINO (ViT-B/16) 雙層整合 (P0 + P3)
- 最佳驗證成績：mAP@0.5 = 96.72%, mAP@0.5:0.95 = 70.74%

## 環境需求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+
- 修改版 Ultralytics (含 DINO 整合模組)

## 安裝

```bash
git clone https://github.com/your-repo/aortic-valve-detection.git
cd aortic-valve-detection
pip install -r requirements.txt
```

## 資料準備

將原始資料集轉換為 YOLO 格式：

```bash
python src/create_yolo_data.py \
    --src-root /path/to/dataset \
    --dst-root /path/to/yolo_dataset \
    --val-ratio 0.2
```

輸出結構：
```
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── aortic_valve.yaml
```

## 訓練

### 單 GPU 訓練

```bash
python src/training/train.py \
    --config configs/yolov12_dino_base.yaml \
    --data yolo_dataset/aortic_valve.yaml \
    --device 0
```

### 多 GPU 訓練

```bash
python src/training/train_multi_gpu.py \
    --config configs/yolov12_dino_base.yaml \
    --data yolo_dataset/aortic_valve.yaml \
    --device 0,1
```

### 使用 DINO-YOLO 訓練腳本

```bash
python dino-yolo/train_yolov12_dino.py \
    --data yolo_dataset/aortic_valve.yaml \
    --epochs 300 \
    --batch-size 16 \
    --imgsz 512 \
    --device 0,1
```

## 推論與提交

```bash
python src/prediction.py \
    --weights experiments/yolov12_dino_baseline13/weights/best.pt \
    --test-root /path/to/testing_image \
    --output submission.txt \
    --imgsz 512 \
    --conf 0.05
```

輸出格式：`{image_name} {class_id} {confidence} {x1} {y1} {x2} {y2}`

## 主要訓練參數

| 參數 | 值 |
|------|-----|
| 輸入尺寸 | 512x512 |
| Batch Size | 16 |
| Epochs | 300 |
| 優化器 | SGD (momentum=0.937) |
| 初始學習率 | 0.01 |
| 權重衰減 | 0.0005 |
| Mosaic | 1.0 |
| Copy-paste | 0.1 |
| AMP | True |

## 實驗結果

| 實驗 | mAP@0.5 | mAP@0.5:0.95 | 備註 |
|------|---------|--------------|------|
| yolov12_dino_baseline11 | 96.80% | 69.43% | DINO3-ViT-B/16 |
| yolov12_dino_baseline13 | 96.72% | 70.74% | RadioDINO-B/16 (Dual P0+P3) |

## 專案結構

```
aortic-valve-detection/
├── configs/                    # 配置文件
│   ├── yolov11_base.yaml
│   ├── yolov12_dino_base.yaml
│   └── yolov12m-dualp0p3-radiodino-b16.yaml
├── dino-yolo/
│   └── train_yolov12_dino.py   # DINO-YOLO 訓練腳本
├── experiments/                # 實驗結果
├── src/
│   ├── create_yolo_data.py     # 資料轉換
│   ├── prediction.py           # 推論腳本
│   ├── data/
│   │   └── medical_dataset.py
│   ├── ensemble/
│   │   └── ensemble_manager.py
│   ├── models/
│   │   └── two_stage_detector.py
│   ├── training/
│   │   ├── train.py
│   │   └── train_multi_gpu.py
│   └── utils/
│       ├── losses.py           # Focal Loss, CIoU, DFL
│       └── nms.py              # Soft-NMS, DIoU-NMS, WBF
├── ultralytics/                # 修改版 Ultralytics
└── yolo_dataset/
    └── aortic_valve.yaml
```

## 模型架構

YOLOv12m-DualP0P3-RadioDINO-B16：

1. **DINO3Preprocessor** (P0): 輸入層 RadioDINO 預處理
2. **YOLOv12 Backbone**: Conv → C3k2 → A2C2f 堆疊
3. **DINO3Backbone** (P3): P3 層級 RadioDINO 特徵增強
4. **YOLOv12 Head**: FPN + PAN 結構
5. **Detect**: 多尺度偵測頭 (P3/P4/P5)

## 外部資源

- [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics)
- [RadioDINO](https://huggingface.co/Snarcy/RadioDino-b16) - 醫學影像預訓練 ViT
- [DINOv2](https://github.com/facebookresearch/dinov2)

## License

MIT License
