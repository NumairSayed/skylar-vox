#!/usr/bin/env python3
"""
tools/visualize_features.py

Visualize HybridEncoder feature maps (P3/P4/P5) for a folder of images.

For each image, saves:
  results/<run_name>/features/<image_name>/
    p3_mean.png      — mean activation across channels (stride-8, 80×80)
    p3_max.png       — max activation across channels
    p4_mean.png      — stride-16, 40×40
    p5_mean.png      — stride-32, 20×20
    overlay_p3.png   — P3 heatmap blended over original image

Usage:
  python tools/visualize_features.py \
      -c configs/deimv2/finetune_gcp_cnn.yml \
      -r outputs/gcp_cnn/best_stg1.pth \
      -i ~/Desktop/IISc/misc/skylark_assn/GCP_Assignment_Datasets/test_dataset/ \
      --run-name cnn_features \
      --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from engine.core import YAMLConfig

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


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
    model = cfg.model.to(device).eval()
    img_size = tuple(cfg.yaml_cfg.get("eval_spatial_size", [640, 640]))
    vit_backbone = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
    return model, img_size, vit_backbone


def build_transform(img_size, vit_backbone):
    ops = [T.Resize(img_size), T.ToTensor()]
    if vit_backbone:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)


def to_heatmap(tensor_2d: np.ndarray) -> np.ndarray:
    """Normalise a 2D array to [0,255] uint8."""
    t = tensor_2d - tensor_2d.min()
    if t.max() > 0:
        t = t / t.max()
    return (t * 255).astype(np.uint8)


def colorize(gray: np.ndarray) -> Image.Image:
    """Apply jet colormap to a uint8 grayscale array."""
    import matplotlib.cm as cm
    colored = cm.jet(gray / 255.0)          # RGBA float [0,1]
    return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))


def save_heatmap(arr2d: np.ndarray, path: Path, size=None):
    gray = to_heatmap(arr2d)
    img = colorize(gray)
    if size:
        img = img.resize(size, Image.BILINEAR)
    img.save(path)


def overlay(orig: Image.Image, feat_map: np.ndarray, alpha=0.5) -> Image.Image:
    """Blend heatmap over original image."""
    gray = to_heatmap(feat_map)
    heat = colorize(gray).resize(orig.size, Image.BILINEAR).convert("RGBA")
    base = orig.convert("RGBA")
    blended = Image.blend(base, heat, alpha=alpha)
    return blended.convert("RGB")


@torch.no_grad()
def run(args):
    device = args.device
    model, img_size, vit_backbone = load_model(args.config, args.resume, device)
    transform = build_transform(img_size, vit_backbone)

    img_dir = Path(args.input)
    image_paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
    if not image_paths:
        sys.exit(f"No images found in {img_dir}")

    out_dir = Path("results") / args.run_name / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images found: {len(image_paths)}")

    for idx, path in enumerate(image_paths):
        im_pil = Image.open(path).convert("RGB")
        tensor = transform(im_pil).unsqueeze(0).to(device)

        backbone_out = model.backbone(tensor)
        enc_out = model.encoder(backbone_out)   # [P3, P4, P5]

        # enc_out is the list [P3, P4, P5]
        feat_dir = out_dir / path.stem
        feat_dir.mkdir(exist_ok=True)

        level_names = ["p3_stride8", "p4_stride16", "p5_stride32"]
        for level_idx, (name, fmap) in enumerate(zip(level_names, enc_out)):
            # fmap: [1, C, H, W]
            f = fmap[0].float().cpu().numpy()    # [C, H, W]
            mean_map = f.mean(axis=0)            # [H, W]
            max_map  = f.max(axis=0)             # [H, W]

            save_heatmap(mean_map, feat_dir / f"{name}_mean.png", size=img_size)
            save_heatmap(max_map,  feat_dir / f"{name}_max.png",  size=img_size)

            # overlay only on P3 (most relevant for small objects)
            if level_idx == 0:
                ov = overlay(im_pil.resize(img_size), mean_map, alpha=0.55)
                ov.save(feat_dir / f"{name}_overlay.png")

                # also save at original resolution
                ov_orig = overlay(im_pil, mean_map, alpha=0.55)
                ov_orig.save(feat_dir / f"{name}_overlay_fullres.png")

        # save resized original for easy side-by-side comparison
        im_pil.resize(img_size).save(feat_dir / "original.jpg")

        print(f"  [{idx+1}/{len(image_paths)}] {path.name}  →  {feat_dir}/")

    print(f"\nDone. Results in results/{args.run_name}/features/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",  required=True)
    parser.add_argument("-r", "--resume",  required=True)
    parser.add_argument("-i", "--input",   required=True)
    parser.add_argument("--run-name", default="features")
    parser.add_argument("-d", "--device",  default="cuda")
    args = parser.parse_args()
    run(args)
