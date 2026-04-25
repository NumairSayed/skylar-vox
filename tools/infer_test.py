#!/usr/bin/env python3
"""
tools/infer_test.py

Run inference on a folder of images and save results to results/<run_name>/.

Outputs:
  results/<run_name>/predictions.json   COCO-format detections
  results/<run_name>/images/            visualised JPEGs

Usage:
  python tools/infer_test.py \
      -c configs/deimv2/finetune_gcp.yml \
      -r outputs/gcp_finetune/best_stg1.pth \
      -i dataset/images/val \
      --run-name my_eval \
      --thresh 0.45 \
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from engine.core import YAMLConfig

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
CLASS_NAMES = ["Cross", "Square", "L-Shape"]
PALETTE = [(255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 200, 56), (200, 56, 255)]


def color_for(label: int) -> tuple:
    return PALETTE[label % len(PALETTE)]


def visualize(im_pil: Image.Image, labels, boxes, scores, thresh: float) -> Image.Image:
    im = im_pil.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    mask = scores > thresh
    for label, box, score in zip(labels[mask], boxes[mask], scores[mask]):
        lbl = label.item()
        name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else str(lbl)
        color = color_for(lbl)
        b = box.tolist()
        draw.rectangle(b, outline=color, width=3)
        draw.text((b[0], max(0, b[1] - 20)), f"{name} {score.item():.2f}", fill=color, font=font)
    return im


def load_model(cfg_path: str, ckpt_path: str, device: str):
    cfg = YAMLConfig(cfg_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "ema" in ckpt:
        state = ckpt["ema"]["module"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    state = {k.replace("module.", ""): v for k, v in state.items()}
    cfg.model.load_state_dict(state, strict=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    model = _Model().to(device).eval()
    img_size = tuple(cfg.yaml_cfg.get("eval_spatial_size", [640, 640]))
    vit_backbone = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
    return model, img_size, vit_backbone


def build_transform(img_size: tuple, vit_backbone: bool) -> T.Compose:
    ops = [T.Resize(img_size), T.ToTensor()]
    if vit_backbone:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)


@torch.no_grad()
def run(args):
    device = args.device
    model, img_size, vit_backbone = load_model(args.config, args.resume, device)
    transform = build_transform(img_size, vit_backbone)

    img_dir = Path(args.input)
    image_paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
    if not image_paths:
        sys.exit(f"No images found in {img_dir}")

    run_name = args.run_name or Path(args.resume).stem
    out_dir = Path("results") / run_name
    vis_dir = out_dir / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Score threshold: {args.thresh}")
    print(f"Images found   : {len(image_paths)}\n")

    predictions, image_infos = [], []

    for img_id, path in enumerate(image_paths):
        im_pil = Image.open(path).convert("RGB")
        W, H = im_pil.size
        orig_size = torch.tensor([[W, H]], dtype=torch.float32).to(device)

        tensor = transform(im_pil).unsqueeze(0).to(device)
        labels, boxes, scores = model(tensor, orig_size)
        labels, boxes, scores = labels[0], boxes[0], scores[0]

        image_infos.append({"id": img_id, "file_name": path.name, "width": W, "height": H})

        mask = scores > args.thresh
        n_det = mask.sum().item()
        for label, box, score in zip(labels[mask], boxes[mask], scores[mask]):
            x1, y1, x2, y2 = box.tolist()
            lbl = label.item()
            predictions.append({
                "image_id": img_id,
                "file_name": path.name,
                "category_id": lbl,
                "category_name": CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else str(lbl),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": round(score.item(), 4),
            })

        vis = visualize(im_pil, labels, boxes, scores, args.thresh)
        vis.save(vis_dir / path.name)
        print(f"  [{img_id + 1:3d}/{len(image_paths)}] {path.name}  →  {n_det} detections")

    output = {"images": image_infos, "detections": predictions}
    json_path = out_dir / "predictions.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(image_paths)} images → {len(predictions)} total detections")
    print(f"  JSON  : {json_path}")
    print(f"  Images: {vis_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEIMv2 test-set inference")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--thresh", type=float, default=0.45)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()
    run(args)
