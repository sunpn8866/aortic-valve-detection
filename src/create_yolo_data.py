import argparse
import random
import shutil
from pathlib import Path


def build_yolo_dataset(src_root: Path, dst_root: Path, val_ratio: float = 0.2, seed: int = 42):
    src_img_root = src_root / "training_image"
    src_lbl_root = src_root / "training_label"

    if not src_img_root.exists() or not src_lbl_root.exists():
        raise FileNotFoundError(
            "Cannot find 'training_image' or 'training_label'. "
            "Please make sure --src-root points to the 'Training Dataset' directory."
        )

    # Collect all patient folders, e.g. patient0001 ~ patient0050
    patients = sorted([p.name for p in src_img_root.iterdir() if p.is_dir()])
    if not patients:
        raise RuntimeError("No patient folders found under 'training_image'.")

    print(f"Found {len(patients)} patients: {patients[:5]} ...")

    rnd = random.Random(seed)
    rnd.shuffle(patients)

    num_val = max(1, int(len(patients) * val_ratio))
    val_patients = set(patients[:num_val])
    train_patients = set(patients[num_val:])

    print(f"Number of training patients: {len(train_patients)}, validation patients: {len(val_patients)}")
    print(f"Example validation patients: {sorted(list(val_patients))[:5]}")

    # Create YOLO target directories
    imgs_train = dst_root / "images" / "train"
    imgs_val = dst_root / "images" / "val"
    lbls_train = dst_root / "labels" / "train"
    lbls_val = dst_root / "labels" / "val"

    for p in [imgs_train, imgs_val, lbls_train, lbls_val]:
        p.mkdir(parents=True, exist_ok=True)

    # Iterate over all patients and PNG images
    train_img_count = val_img_count = 0
    train_lbl_count = val_lbl_count = 0

    for patient in patients:
        split = "val" if patient in val_patients else "train"
        src_img_dir = src_img_root / patient
        src_lbl_dir = src_lbl_root / patient

        for img_path in src_img_dir.glob("*.png"):
            stem = img_path.stem  # e.g. patient0001_0201
            lbl_path = src_lbl_dir / f"{stem}.txt"

            if split == "train":
                dst_img = imgs_train / img_path.name
                dst_lbl_dir = lbls_train
                train_img_count += 1
            else:
                dst_img = imgs_val / img_path.name
                dst_lbl_dir = lbls_val
                val_img_count += 1

            # Copy image
            shutil.copy2(img_path, dst_img)

            # If a label file exists, copy it as well; otherwise skip
            # (YOLO will treat this as an image with no objects).
            if lbl_path.exists():
                dst_lbl = dst_lbl_dir / lbl_path.name
                shutil.copy2(lbl_path, dst_lbl)
                if split == "train":
                    train_lbl_count += 1
                else:
                    val_lbl_count += 1

    print(f"✅ Train images: {train_img_count}, labeled slices: {train_lbl_count}")
    print(f"✅ Val   images: {val_img_count}, labeled slices: {val_lbl_count}")

    # Create dataset.yaml (the competition has only one class: aortic valve)
    dataset_yaml = f"""\
path: {dst_root}
train: images/train
val: images/val

nc: 1
names: ["aortic_valve"]
"""
    yaml_path = dst_root / "aortic_valve.yaml"
    yaml_path.write_text(dataset_yaml, encoding="utf-8")
    print(f"✅ YOLO dataset config written to: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert AICUP aortic valve competition dataset into YOLOv12 dataset structure"
    )
    parser.add_argument(
        "--src-root",
        type=str,
        required=True,
        help="Original 'Training Dataset' directory, e.g. /data/dataset_challenge_3/Training Dataset",
    )
    parser.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Output YOLO dataset root directory, e.g. /data/yolo_dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio (split by patient), default 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    build_yolo_dataset(
        src_root=Path(args.src_root),
        dst_root=Path(args.dst_root),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
