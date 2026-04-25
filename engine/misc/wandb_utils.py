"""Thin WandB wrapper that mirrors the TensorBoard SummaryWriter interface.

This lets the rest of the codebase call writer.add_scalar / writer.add_text / writer.close
without knowing whether the backend is TensorBoard, WandB, or both.
"""
from __future__ import annotations

import os
import re
from typing import Optional

# Only these tags are forwarded to WandB. TensorBoard always receives everything.
#
# Eval tags (COCO stats array indices):
#   coco_eval_bbox_0  → mAP@0.50:0.95  (accuracy / IoU 50-95)
#   coco_eval_bbox_1  → AP@0.50        (IoU 50 / precision)
#   coco_eval_bbox_8  → AR@maxDets=100 (recall)
#
# Loss tags:
#   Loss/total              → total training loss
#   Loss/loss_vfl           → main decoder classification loss
#   Loss/loss_bbox          → main decoder box regression loss
#   Loss/loss_giou          → main decoder GIoU loss
#   Loss/loss_*_enc_*       → hybrid encoder losses (all layers)
_WANDB_EXACT = {
    "Loss/total",
    "Loss/loss_vfl",
    "Loss/loss_bbox",
    "Loss/loss_giou",
    "Test/coco_eval_bbox_0",
    "Test/coco_eval_bbox_1",
    "Test/coco_eval_bbox_8",
}
_WANDB_PATTERN = re.compile(r"^Loss/loss_\w+_enc_\d+$")


def _wandb_allowed(tag: str) -> bool:
    return tag in _WANDB_EXACT or bool(_WANDB_PATTERN.match(tag))


class WandbWriter:
    """Wraps a wandb Run with the SummaryWriter API used inside DEIMv2."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        tb_writer=None,
    ) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is not installed. Run: pip install wandb")

        self._run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            resume="allow",
        )
        self._tb = tb_writer  # optional parallel TensorBoard writer
        self._step_cache: dict = {}  # tag → last seen step (prevents duplicate logs)

    # ------------------------------------------------------------------
    # SummaryWriter-compatible API
    # ------------------------------------------------------------------

    def add_scalar(self, tag: str, value, global_step: int) -> None:
        if self._tb is not None:
            self._tb.add_scalar(tag, value, global_step)
        if self._run is not None and _wandb_allowed(tag):
            self._run.log({tag: value}, step=global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int) -> None:
        if self._tb is not None:
            self._tb.add_scalars(main_tag, tag_scalar_dict, global_step)
        if self._run is not None:
            filtered = {
                f"{main_tag}/{k}": v
                for k, v in tag_scalar_dict.items()
                if _wandb_allowed(f"{main_tag}/{k}")
            }
            if filtered:
                self._run.log(filtered, step=global_step)

    def add_text(self, tag: str, text_string: str, global_step: int = 0) -> None:
        if self._tb is not None:
            self._tb.add_text(tag, text_string, global_step)
        # WandB doesn't have a direct text API in the same form; log as a summary
        if self._run is not None:
            self._run.summary[tag] = text_string

    def add_image(self, tag: str, img_tensor, global_step: int = 0) -> None:
        if self._tb is not None:
            self._tb.add_image(tag, img_tensor, global_step)
        if self._run is not None:
            import wandb
            self._run.log({tag: wandb.Image(img_tensor)}, step=global_step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._run is not None:
            self._run.finish()
            self._run = None

    # ------------------------------------------------------------------
    # Passthrough so isinstance checks on SummaryWriter won't break
    # (we mark the class at module level after import)
    # ------------------------------------------------------------------

    @property
    def run(self):
        return self._run
