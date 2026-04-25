#!/usr/bin/env python3
"""
tools/train_finetune.py

Fine-tune DEIMv2 on a custom dataset (GCP markers by default).

What this script adds on top of the vanilla train.py:
  1. Loads a pretrained DEIMv2 checkpoint via --pretrained, with strict=False
     so head weights are re-initialised for the new class count.
  2. Freezes the DINOv3 backbone (requires_grad = False).
  3. Computes warmup_iter from the actual dataset size so the 5-epoch warmup
     is exact regardless of batch size.
  4. Initialises WandB logging when WANDB_PROJECT is set (or --wandb-project
     is passed).  Set WANDB_API_KEY before running.

Usage (single GPU):
  python tools/train_finetune.py \\
      -c configs/deimv2/finetune_gcp.yml \\
      --pretrained ~/Downloads/deimv2_dinov3_x_coco.pth \\
      --use-amp --seed 42

Usage (multi-GPU torchrun):
  torchrun --nproc_per_node=4 tools/train_finetune.py \\
      -c configs/deimv2/finetune_gcp.yml \\
      --pretrained ~/Downloads/deimv2_dinov3_x_coco.pth \\
      --use-amp --seed 42

WandB:
  export WANDB_API_KEY=<your-key>
  python tools/train_finetune.py ... --wandb-project gcp-detection
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import math
from pathlib import Path

import torch

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS


# ─────────────────────────────────────────────────────────────────────────────
# Backbone freezing
# ─────────────────────────────────────────────────────────────────────────────

_BACKBONE_PATTERNS = (
    "backbone",
    "dinov3",
    ".sta.",
    ".patch_embed",
    ".pos_embed",
    ".cls_token",
)


def freeze_backbone(model: torch.nn.Module) -> int:
    """Freeze backbone: requires_grad=False, eval mode, no_grad forward, BN→BatchNorm.

    Returns the number of frozen parameters.
    """
    module = getattr(model, "module", model)

    # 1. Freeze parameters
    frozen = 0
    for name, param in module.named_parameters():
        if any(pat in name.lower() for pat in _BACKBONE_PATTERNS):
            param.requires_grad = False
            frozen += param.numel()

    # 2. Find and configure the backbone sub-module
    backbone = getattr(module, "backbone", None)
    if backbone is not None:
        # Set to eval so BN uses running stats (not batch stats) and saves memory
        backbone.eval()

        # 3. Convert SyncBatchNorm → BatchNorm2d (SyncBN wastes memory on single GPU)
        backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm.__func__(
            torch.nn.SyncBatchNorm, backbone  # no-op if already BN
        ) if False else _replace_syncbn(backbone)

        # 4. Wrap backbone forward in torch.no_grad() so intermediate activations
        #    are not retained in the autograd graph (saves ~40% GPU memory)
        _wrap_no_grad(backbone)

    return frozen


def _replace_syncbn(module: torch.nn.Module) -> torch.nn.Module:
    """Recursively replace SyncBatchNorm with BatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.SyncBatchNorm):
            bn = torch.nn.BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            bn.weight = child.weight
            bn.bias = child.bias
            bn.running_mean = child.running_mean
            bn.running_var = child.running_var
            bn.num_batches_tracked = child.num_batches_tracked
            setattr(module, name, bn)
        else:
            _replace_syncbn(child)
    return module


def _wrap_no_grad(module: torch.nn.Module) -> None:
    """Patch module.forward so it always runs inside torch.no_grad()."""
    orig_forward = module.forward

    def _no_grad_forward(*args, **kwargs):
        with torch.no_grad():
            return orig_forward(*args, **kwargs)

    module.forward = _no_grad_forward


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained weight loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained(model: torch.nn.Module, path: str, num_classes: int) -> None:
    """Load weights from a DEIMv2 checkpoint, skipping head layers if the
    class count differs (so the model is re-initialised for the new dataset).
    """
    ckpt = torch.load(path, map_location="cpu")

    # Support both bare state-dicts and solver checkpoints
    state = ckpt
    for key in ("model", "ema", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            # For EMA, unwrap the inner model
            if key == "ema" and isinstance(state, dict) and "module" in state:
                state = state["module"]
            break

    # Strip DDP prefix
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    state = new_state

    module = getattr(model, "module", model)
    model_state = module.state_dict()

    # Drop keys whose shape doesn't match (e.g. classification heads)
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(f"{k}: ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}")
        else:
            skipped.append(f"{k}: not in model")

    if skipped:
        print(f"[pretrained] Skipped {len(skipped)} mismatched / extra keys:")
        for s in skipped[:10]:
            print(f"  {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    missing, unexpected = module.load_state_dict(filtered, strict=False)
    print(f"[pretrained] Loaded {len(filtered)} keys from {path}")
    print(f"[pretrained] Missing: {len(missing)}  Unexpected: {len(unexpected)}")


# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────

def setup_wandb(cfg, project: str | None, run_name: str | None) -> None:
    """Replace cfg.writer with a WandbWriter (keeps TensorBoard as secondary)."""
    if not dist_utils.is_main_process():
        return

    project = project or os.environ.get("WANDB_PROJECT")
    if not project:
        return

    try:
        from engine.misc.wandb_utils import WandbWriter
        from torch.utils.tensorboard import SummaryWriter
        from pathlib import Path as _Path
    except ImportError as e:
        print(f"[wandb] Skipping WandB setup: {e}")
        return

    # Build a parallel TensorBoard writer
    tb_dir = _Path(cfg.output_dir) / "summary"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(str(tb_dir))

    # Gather hyper-parameters from config for W&B config panel
    wb_config = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}

    writer = WandbWriter(
        project=project,
        name=run_name or _Path(cfg.output_dir).name,
        config=wb_config,
        tb_writer=tb_writer,
    )
    cfg._writer = writer  # bypass type-check; already relaxed in _config.py
    print(f"[wandb] Logging to project '{project}', run '{writer.run.name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not (args.tuning and args.resume), \
        "Only one of --resume / --tuning may be specified at a time"

    # Build config
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in vars(args).items()
                        if k not in ("update",) and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)

    # ── Instantiate model via cfg ──────────────────────────────────────
    # Accessing cfg.model triggers model construction
    model = cfg.model

    # ── Load pretrained weights ────────────────────────────────────────
    if args.pretrained:
        path = Path(args.pretrained).expanduser()
        if not path.exists():
            sys.exit(f"ERROR: pretrained checkpoint not found: {path}")
        num_classes = cfg.yaml_cfg.get("num_classes", 80)
        load_pretrained(model, str(path), num_classes)
    elif args.tuning:
        pass  # handled by the solver
    elif args.resume:
        pass

    # ── Freeze backbone ────────────────────────────────────────────────
    if not args.no_freeze_backbone:
        n_frozen = freeze_backbone(model)
        print(f"[freeze] Froze {n_frozen:,} backbone parameters")
    else:
        print("[freeze] Backbone NOT frozen (--no-freeze-backbone)")

    # ── Compute exact warmup_iter from dataset size ────────────────────
    # We need iter_per_epoch to turn "5 epochs" into iterations.
    # Use cfg.train_dataloader.dataset length + total_batch_size from config.
    try:
        train_ds = cfg.train_dataloader.dataset
        n_samples = len(train_ds)
        batch_size = getattr(cfg.train_dataloader, "total_batch_size",
                             getattr(cfg.train_dataloader, "batch_size", 2))
        # Account for gradient accumulation: effective batch = batch_size * accum
        accum = args.grad_accum
        effective_bs = batch_size * accum
        iter_per_epoch = math.ceil(n_samples / effective_bs)
        warmup_epochs = cfg.yaml_cfg.get("flat_epoch", 5)  # flat_epoch = warmup end
        warmup_iter = warmup_epochs * iter_per_epoch
        cfg.warmup_iter = warmup_iter
        print(f"[schedule] dataset={n_samples} batch={batch_size} "
              f"grad_accum={accum} effective_bs={effective_bs} "
              f"iter/epoch={iter_per_epoch} warmup_iter={warmup_iter} "
              f"(≈{warmup_epochs} epochs)")
    except Exception as e:
        print(f"[schedule] Could not compute warmup_iter dynamically: {e}. "
              f"Using YAML value ({cfg.warmup_iter}).")

    # ── WandB ──────────────────────────────────────────────────────────
    setup_wandb(cfg, args.wandb_project, args.wandb_run_name)

    # ── Run ───────────────────────────────────────────────────────────
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DEIMv2 with frozen backbone + WandB logging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Core (same as train.py) ────────────────────────────────────────
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to YAML config (e.g. configs/deimv2/finetune_gcp.yml)")
    parser.add_argument("-r", "--resume", type=str,
                        help="Resume from checkpoint (full solver state)")
    parser.add_argument("-t", "--tuning", type=str,
                        help="Tune from checkpoint (model weights only, via solver)")
    parser.add_argument("-d", "--device", type=str,
                        help="Device override (e.g. 'cuda:0')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true",
                        help="Enable automatic mixed precision")
    parser.add_argument("--output-dir", type=str,
                        help="Override output directory")
    parser.add_argument("--test-only", action="store_true", default=False)
    parser.add_argument("-u", "--update", nargs="+",
                        help="Override individual YAML keys (key=value)")

    # ── Fine-tuning extras ─────────────────────────────────────────────
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to DEIMv2 checkpoint to initialise from "
                             "(e.g. ~/Downloads/deimv2_dinov3_x_coco.pth)")
    parser.add_argument("--no-freeze-backbone", action="store_true",
                        help="Train the full model including backbone")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective_bs = batch_size × grad_accum). "
                             "Use to simulate larger batches on small GPUs. Default: 1 (no accum)")

    # ── WandB ─────────────────────────────────────────────────────────
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="WandB project name. Also reads $WANDB_PROJECT.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="WandB run name (defaults to output-dir basename)")

    # ── Distributed ───────────────────────────────────────────────────
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)

    args = parser.parse_args()
    main(args)
