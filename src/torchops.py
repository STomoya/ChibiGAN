"""torch ops
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

__all__ = ["freeze", "unfreeze", "update_ema", "optimizer_step"]


@torch.no_grad()
def freeze(model: nn.Module):
    """freeze the model"""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


@torch.no_grad()
def unfreeze(model: nn.Module):
    """unfreeze the model"""
    for param in model.parameters():
        param.requires_grad = True
    model.train()


@torch.no_grad()
def update_ema(
    model: torch.nn.Module,
    model_ema: torch.nn.Module,
    decay: float = 0.999,
    copy_buffers: bool = False,
    force_cpu: bool = False,
) -> None:
    """Update exponential moving avg"""
    if force_cpu:
        org_device = next(model.parameters()).device
        model.to("cpu")
        model_ema.to("cpu")

    model.eval()
    param_ema = dict(model_ema.named_parameters())
    param = dict(model.named_parameters())
    for key in param_ema.keys():
        param_ema[key].data.mul_(decay).add_(param[key].data, alpha=(1 - decay))
    if copy_buffers:
        buffer_ema = dict(model_ema.named_buffers())
        buffer = dict(model.named_buffers())
        for key in buffer_ema.keys():
            buffer_ema[key].data.copy_(buffer[key].data)
    model.train()

    if force_cpu:
        model.to(org_device)


def optimizer_step(
    loss: torch.Tensor,
    optimizer: optim.Optimizer,
    scaler=None,
    zero_grad: bool = True,
    set_to_none: bool = True,
    update_scaler: bool = False,
) -> None:
    """optimization step which supports gradient scaling for AMP and other options."""
    assert scaler is None or isinstance(scaler, GradScaler)

    if zero_grad:
        optimizer.zero_grad(set_to_none)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        if update_scaler:
            scaler.update()
    else:
        loss.backward()
        optimizer.step()


def make_image_grid(*image_tensors, num_images=None):
    """align images"""

    def _split(x):
        return x.chunk(x.size(0), 0)

    image_tensor_lists = map(_split, image_tensors)
    images = []
    for index, image_set in enumerate(zip(*image_tensor_lists)):
        images.extend(list(image_set))
        if num_images is not None and index == num_images - 1:
            break
    return torch.cat(images, dim=0)
