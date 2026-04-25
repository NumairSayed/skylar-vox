#!/usr/bin/env python3
"""
tools/prepare_dataset.py

Prepares a dataset for DEIMv2 training by converting annotations to COCO format
and creating an 80:20 train/val split.

Supported input formats (auto-detected):
  - GCP JSON   : dict keyed by relative image path, value has 'mark.x/y' and 'verified_shape'
  - COCO JSON  : already has 'images', 'annotations', 'categories' keys
  - YOLO       : images/ dir + labels/ dir with normalised cx cy w h .txt files
  - Pascal VOC : images + annotations/*.xml

Output layout (DEIMv2-ready):
  dataset/
  ├── images/
  │   ├── train/   (symlinked or copied)
  │   └── val/
  └── annotations/
      ├── instances_train.json
      └── instances_val.json

A ready-to-use DEIMv2 YAML config is written to
  configs/dataset/<dataset_name>_detection.yml

Usage examples:
  # Auto-detect from whatever is in dataset/ (images already placed there)
  python tools/prepare_dataset.py

  # Point at a source directory with GCP-style JSON
  python tools/prepare_dataset.py \\
      --source-dir /path/to/GCP_Assignment_Datasets/train_dataset \\
      --annotation-file /path/to/train_dataset/gcp_marks.json \\
      --format gcp_json \\
      --bbox-size 80

  # Specify custom output and split ratio
  python tools/prepare_dataset.py \\
      --dataset-dir dataset/ \\
      --train-ratio 0.8 \\
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _image_size(path: Path) -> Tuple[int, int]:
    """Return (width, height) without loading the full image."""
    try:
        import struct

        with open(path, "rb") as f:
            sig = f.read(8)

        if sig[:8] == b"\x89PNG\r\n\x1a\n":  # PNG
            with open(path, "rb") as f:
                f.seek(16)
                w = struct.unpack(">I", f.read(4))[0]
                h = struct.unpack(">I", f.read(4))[0]
            return w, h

        if sig[:2] == b"\xff\xd8":  # JPEG – need PIL or cv2
            try:
                from PIL import Image

                with Image.open(path) as img:
                    return img.size  # (w, h)
            except ImportError:
                pass
            try:
                import cv2

                img = cv2.imread(str(path))
                if img is not None:
                    h, w = img.shape[:2]
                    return w, h
            except ImportError:
                pass
        # Generic fallback: PIL
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        return 0, 0


def _center_to_bbox(
    cx: float, cy: float, bbox_size: int, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert a center point to a square bounding box clipped to image bounds.
    Returns (x_min, y_min, width, height) in COCO format.
    """
    half = bbox_size / 2
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(img_w, cx + half)
    y2 = min(img_h, cy + half)
    return x1, y1, x2 - x1, y2 - y1


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

FORMAT_GCP_JSON = "gcp_json"
FORMAT_COCO = "coco"
FORMAT_YOLO = "yolo"
FORMAT_VOC = "voc"


def detect_format(dataset_dir: Path, annotation_file: Optional[Path]) -> str:
    """Heuristically detect the annotation format."""

    # Explicit annotation file given
    if annotation_file and annotation_file.exists():
        with open(annotation_file) as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "images" in data and "annotations" in data and "categories" in data:
                return FORMAT_COCO
            # Check a handful of values
            sample = next(iter(data.values()), None)
            if isinstance(sample, dict) and (
                "mark" in sample or "verified_shape" in sample
            ):
                return FORMAT_GCP_JSON
        if isinstance(data, list) and data and "image_id" in data[0]:
            return FORMAT_COCO
        return FORMAT_GCP_JSON  # default for unknown JSON

    # Search dataset_dir for clues
    json_files = list(dataset_dir.rglob("*.json"))
    xml_files = list(dataset_dir.rglob("*.xml"))
    txt_files = [
        p
        for p in dataset_dir.rglob("*.txt")
        if p.parent.name in {"labels", "label"}
        or "label" in p.parent.name.lower()
    ]

    if json_files:
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                if (
                    isinstance(data, dict)
                    and "images" in data
                    and "annotations" in data
                ):
                    return FORMAT_COCO
                sample = next(iter(data.values()), None) if isinstance(data, dict) else None
                if isinstance(sample, dict) and "mark" in sample:
                    return FORMAT_GCP_JSON
            except Exception:
                continue

    if xml_files:
        return FORMAT_VOC

    if txt_files:
        return FORMAT_YOLO

    # Already COCO structured?
    if (dataset_dir / "annotations").exists() and list(
        (dataset_dir / "annotations").glob("instances_*.json")
    ):
        return FORMAT_COCO

    raise ValueError(
        "Could not auto-detect annotation format. "
        "Pass --format {gcp_json,coco,yolo,voc} explicitly."
    )


# ---------------------------------------------------------------------------
# Readers – each returns (records, categories)
#   record = {image_path: Path, file_name: str, width: int, height: int,
#              annotations: [{category_id, bbox, area}]}
# ---------------------------------------------------------------------------

Record = Dict
Category = Dict


def _read_gcp_json(
    annotation_file: Path,
    data_root: Path,
    bbox_size: int,
) -> Tuple[List[Record], List[Category]]:
    """Read GCP-style JSON: {rel_path: {mark: {x,y}, verified_shape: str}}"""
    classes = ["Cross", "Square", "L-Shape"]
    cls_map = {c.lower().replace("-", "").replace("_", ""): c for c in classes}
    # 0-indexed IDs: DEIMv2 with remap_mscoco_category=False uses category_id
    # directly as the label index, so IDs must start at 0 to match num_classes.
    cat_id = {c: i for i, c in enumerate(classes)}
    categories = [{"id": i, "name": c} for i, c in enumerate(classes)]

    with open(annotation_file) as f:
        raw = json.load(f)

    def _normalize(s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None
        key = s.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        return cls_map.get(key)

    records: List[Record] = []
    skipped_missing = 0
    skipped_bad = 0

    for rel_path, entry in raw.items():
        img_path = data_root / rel_path
        if not img_path.exists():
            skipped_missing += 1
            continue

        mark = entry.get("mark") if isinstance(entry, dict) else None
        shape = entry.get("verified_shape") if isinstance(entry, dict) else None
        if mark is None or shape is None:
            skipped_bad += 1
            continue

        cls_name = _normalize(shape)
        if cls_name is None:
            skipped_bad += 1
            continue

        try:
            cx = float(mark["x"])
            cy = float(mark["y"])
        except (KeyError, TypeError, ValueError):
            skipped_bad += 1
            continue

        w, h = _image_size(img_path)
        if w == 0 or h == 0:
            skipped_missing += 1
            continue

        x1, y1, bw, bh = _center_to_bbox(cx, cy, bbox_size, w, h)
        records.append(
            {
                "image_path": img_path,
                "file_name": rel_path.replace("/", "__"),  # flat filename
                "width": w,
                "height": h,
                "category_name": cls_name,
                "annotations": [
                    {
                        "category_id": cat_id[cls_name],
                        "category_name": cls_name,
                        "bbox": [x1, y1, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                ],
            }
        )

    log.info(
        "GCP JSON: loaded=%d  skipped(missing)=%d  skipped(bad)=%d",
        len(records),
        skipped_missing,
        skipped_bad,
    )
    return records, categories


def _read_coco(
    annotation_file: Path, data_root: Path
) -> Tuple[List[Record], List[Category]]:
    """Read a COCO JSON directly into records."""
    with open(annotation_file) as f:
        coco = json.load(f)

    categories = coco["categories"]
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    # Group annotations by image_id
    anns_by_img: Dict[int, List] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    records: List[Record] = []
    for img_info in coco["images"]:
        img_path = data_root / img_info["file_name"]
        if not img_path.exists():
            continue
        img_anns = anns_by_img.get(img_info["id"], [])
        for ann in img_anns:
            ann["category_name"] = cat_id_to_name.get(ann["category_id"], "unknown")
        records.append(
            {
                "image_path": img_path,
                "file_name": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
                "annotations": img_anns,
            }
        )

    log.info("COCO: loaded=%d images", len(records))
    return records, categories


def _read_yolo(
    dataset_dir: Path,
) -> Tuple[List[Record], List[Category]]:
    """Read YOLO format: images/ + labels/ with normalised bbox txt files."""
    # Try to load class names
    names_file = next(
        (p for p in dataset_dir.rglob("*.names") or dataset_dir.rglob("*.yaml")
         if p.exists()),
        None,
    )
    class_names: List[str] = []
    if names_file and names_file.suffix == ".yaml":
        import yaml  # type: ignore

        with open(names_file) as f:
            y = yaml.safe_load(f)
        class_names = y.get("names", [])
    elif names_file:
        class_names = names_file.read_text().strip().splitlines()

    if not class_names:
        log.warning("No class names file found; using numeric labels")

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    images_dir = next(
        (dataset_dir / d for d in ["images", "imgs", "JPEGImages"] if (dataset_dir / d).exists()),
        dataset_dir,
    )
    labels_dir = next(
        (dataset_dir / d for d in ["labels", "label", "annotations"] if (dataset_dir / d).exists()),
        None,
    )

    records: List[Record] = []
    categories_set: Dict[int, str] = {}

    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in img_exts:
            continue
        label_path = (
            (labels_dir / img_path.relative_to(images_dir)).with_suffix(".txt")
            if labels_dir
            else img_path.with_suffix(".txt")
        )
        if not label_path.exists():
            continue

        w, h = _image_size(img_path)
        if w == 0 or h == 0:
            continue

        anns = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = (float(x) for x in parts[1:5])
            # Denormalise
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            abs_w = bw * w
            abs_h = bh * h
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            categories_set[cls_id] = cls_name
            anns.append(
                {
                    "category_id": cls_id,
                    "category_name": cls_name,
                    "bbox": [x1, y1, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                }
            )
        if anns:
            records.append(
                {
                    "image_path": img_path,
                    "file_name": img_path.name,
                    "width": w,
                    "height": h,
                    "annotations": anns,
                }
            )

    categories = [{"id": k, "name": v} for k, v in sorted(categories_set.items())]
    log.info("YOLO: loaded=%d images", len(records))
    return records, categories


def _read_voc(dataset_dir: Path) -> Tuple[List[Record], List[Category]]:
    """Read Pascal VOC XML annotations."""
    ann_dir = next(
        (dataset_dir / d for d in ["Annotations", "annotations", "labels"] if (dataset_dir / d).exists()),
        dataset_dir,
    )
    img_dir = next(
        (dataset_dir / d for d in ["JPEGImages", "images", "imgs"] if (dataset_dir / d).exists()),
        dataset_dir,
    )

    class_to_id: Dict[str, int] = {}
    records: List[Record] = []

    for xml_path in sorted(ann_dir.rglob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename") or xml_path.stem
        img_path = next(
            (img_dir / filename for ext in ["", ".jpg", ".jpeg", ".png"]
             if (img_dir / (filename + ext)).exists()),
            None,
        )
        if img_path is None:
            continue

        size = root.find("size")
        w = int(size.findtext("width") or 0) if size is not None else 0
        h = int(size.findtext("height") or 0) if size is not None else 0
        if w == 0 or h == 0:
            w, h = _image_size(img_path)

        anns = []
        for obj in root.findall("object"):
            cls_name = obj.findtext("name") or "unknown"
            if cls_name not in class_to_id:
                class_to_id[cls_name] = len(class_to_id) + 1
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            x1 = float(bndbox.findtext("xmin") or 0)
            y1 = float(bndbox.findtext("ymin") or 0)
            x2 = float(bndbox.findtext("xmax") or 0)
            y2 = float(bndbox.findtext("ymax") or 0)
            anns.append(
                {
                    "category_id": class_to_id[cls_name],
                    "category_name": cls_name,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
        if anns:
            records.append(
                {
                    "image_path": img_path,
                    "file_name": img_path.name,
                    "width": w,
                    "height": h,
                    "annotations": anns,
                }
            )

    categories = [{"id": v, "name": k} for k, v in sorted(class_to_id.items(), key=lambda x: x[1])]
    log.info("VOC: loaded=%d images", len(records))
    return records, categories


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def stratified_split(
    records: List[Record], train_ratio: float, seed: int
) -> Tuple[List[Record], List[Record]]:
    """Stratified split by dominant category in each image."""
    rng = random.Random(seed)

    by_class: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        cats = [a.get("category_name", "unknown") for a in r["annotations"]]
        dominant = max(set(cats), key=cats.count) if cats else "unknown"
        by_class[dominant].append(r)

    train: List[Record] = []
    val: List[Record] = []

    for cls, items in by_class.items():
        rng.shuffle(items)
        n_train = max(1, round(len(items) * train_ratio))
        train.extend(items[:n_train])
        val.extend(items[n_train:])
        log.info(
            "  Class %-12s  total=%d  train=%d  val=%d",
            cls,
            len(items),
            n_train,
            len(items) - n_train,
        )

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# COCO JSON serialisation
# ---------------------------------------------------------------------------


def to_coco_json(
    records: List[Record],
    categories: List[Category],
    output_img_dir: Path,
    copy_images: bool,
) -> dict:
    """Build a COCO-format dict and optionally copy/symlink images."""
    output_img_dir.mkdir(parents=True, exist_ok=True)

    coco: dict = {
        "info": {"description": "Prepared by tools/prepare_dataset.py"},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    for img_id, r in enumerate(records, start=1):
        dst = output_img_dir / r["file_name"]
        if copy_images:
            if not dst.exists():
                shutil.copy2(r["image_path"], dst)
        else:
            _safe_symlink(r["image_path"].resolve(), dst)

        coco["images"].append(
            {
                "id": img_id,
                "file_name": r["file_name"],
                "width": r["width"],
                "height": r["height"],
            }
        )
        for ann in r["annotations"]:
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": [round(v, 4) for v in ann["bbox"]],
                    "area": round(ann["area"], 4),
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            ann_id += 1

    return coco


# ---------------------------------------------------------------------------
# Config YAML generation
# ---------------------------------------------------------------------------


def write_deimv2_config(
    dataset_name: str,
    dataset_dir: Path,
    num_classes: int,
    class_names: List[str],
) -> Path:
    """Write a DEIMv2-compatible dataset config YAML."""
    cfg_dir = REPO_ROOT / "configs" / "dataset"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{dataset_name}.yml"

    train_img = (dataset_dir / "images" / "train").resolve()
    val_img = (dataset_dir / "images" / "val").resolve()
    train_ann = (dataset_dir / "annotations" / "instances_train.json").resolve()
    val_ann = (dataset_dir / "annotations" / "instances_val.json").resolve()

    names_str = "\n".join(f"#   - {n}" for n in class_names)

    cfg_path.write_text(
        f"""\
# Auto-generated by tools/prepare_dataset.py
# Dataset: {dataset_name}
task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']

num_classes: {num_classes}
remap_mscoco_category: False

# Class names (for reference)
{names_str}

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {train_img}
    ann_file: {train_ann}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {val_img}
    ann_file: {val_ann}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
"""
    )
    return cfg_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a dataset for DEIMv2: convert to COCO + 80:20 split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(REPO_ROOT / "dataset"),
        help="Output dataset root (default: <repo>/dataset)",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Source directory containing raw images (for GCP/YOLO/VOC formats)",
    )
    parser.add_argument(
        "--annotation-file",
        default=None,
        help="Path to the annotation file (JSON/etc). Auto-searched if omitted.",
    )
    parser.add_argument(
        "--format",
        choices=[FORMAT_GCP_JSON, FORMAT_COCO, FORMAT_YOLO, FORMAT_VOC],
        default=None,
        help="Annotation format. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data used for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--bbox-size",
        type=int,
        default=80,
        help="[GCP JSON only] Side length in pixels for the synthetic bounding box "
             "placed around the annotated center point (default: 80)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking (required on some filesystems)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name used for the output YAML config (default: parent dir name)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    source_dir = Path(args.source_dir) if args.source_dir else dataset_dir
    annotation_file = Path(args.annotation_file) if args.annotation_file else None
    dataset_name = args.dataset_name or dataset_dir.name or "custom"

    # Auto-find annotation file inside source_dir if not given
    if annotation_file is None:
        candidates = sorted(source_dir.rglob("*.json"))
        if candidates:
            # Prefer files with 'marks', 'annotations', or 'instances' in name
            preferred = [
                p for p in candidates
                if any(kw in p.name.lower() for kw in ("mark", "annot", "instance", "label"))
            ]
            annotation_file = preferred[0] if preferred else candidates[0]
            log.info("Auto-detected annotation file: %s", annotation_file)

    # Detect format
    fmt = args.format or detect_format(source_dir, annotation_file)
    log.info("Using format: %s", fmt)

    # Read records
    if fmt == FORMAT_GCP_JSON:
        if annotation_file is None:
            sys.exit("ERROR: --annotation-file required for gcp_json format")
        records, categories = _read_gcp_json(annotation_file, source_dir, args.bbox_size)
    elif fmt == FORMAT_COCO:
        if annotation_file is None:
            sys.exit("ERROR: --annotation-file required for coco format")
        records, categories = _read_coco(annotation_file, source_dir)
    elif fmt == FORMAT_YOLO:
        records, categories = _read_yolo(source_dir)
    elif fmt == FORMAT_VOC:
        records, categories = _read_voc(source_dir)
    else:
        sys.exit(f"Unknown format: {fmt}")

    if not records:
        sys.exit("ERROR: No valid records found. Check --source-dir and --annotation-file.")

    log.info("Total valid records: %d", len(records))

    # Split
    log.info("Splitting %.0f%%/%.0f%% train/val (stratified by class)…", args.train_ratio * 100, (1 - args.train_ratio) * 100)
    train_records, val_records = stratified_split(records, args.train_ratio, args.seed)
    log.info("Split → train=%d  val=%d", len(train_records), len(val_records))

    # Output dirs
    ann_dir = dataset_dir / "annotations"
    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Build and write COCO JSONs
    log.info("Writing train split → %s", train_img_dir)
    train_coco = to_coco_json(train_records, categories, train_img_dir, args.copy_images)
    train_ann_path = ann_dir / "instances_train.json"
    with open(train_ann_path, "w") as f:
        json.dump(train_coco, f)
    log.info("  Wrote %s", train_ann_path)

    log.info("Writing val split → %s", val_img_dir)
    val_coco = to_coco_json(val_records, categories, val_img_dir, args.copy_images)
    val_ann_path = ann_dir / "instances_val.json"
    with open(val_ann_path, "w") as f:
        json.dump(val_coco, f)
    log.info("  Wrote %s", val_ann_path)

    # DEIMv2 YAML config
    class_names = [c["name"] for c in categories]
    cfg_path = write_deimv2_config(dataset_name, dataset_dir, len(categories), class_names)
    log.info("DEIMv2 config → %s", cfg_path)

    # Summary
    print("\n" + "=" * 60)
    print("Dataset preparation complete")
    print("=" * 60)
    print(f"  Format        : {fmt}")
    print(f"  Classes ({len(categories)})    : {', '.join(class_names)}")
    print(f"  Train images  : {len(train_records)}")
    print(f"  Val   images  : {len(val_records)}")
    print(f"  Train JSON    : {train_ann_path}")
    print(f"  Val   JSON    : {val_ann_path}")
    print(f"  DEIMv2 config : {cfg_path}")
    print()
    print("To train with DEIMv2:")
    print(f"  torchrun --nproc_per_node=<N> train.py \\")
    print(f"    -c configs/deimv2/deimv2_hgnetv2_n_coco.yml \\")
    print(f"    --dataset-config {cfg_path} \\")
    print(f"    --use-amp --seed=0")
    print("=" * 60)


if __name__ == "__main__":
    main()
