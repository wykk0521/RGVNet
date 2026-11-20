"""GT-only boundary-neighborhood supervision from the archived HBGNet SR v2 design.

This module has no model, dataset, or training dependencies beyond PyTorch.
Image borders are not transitions: only differing in-image 4-neighbors seed
both sides of a boundary. The default dilation is the 13-point integer disk2.
"""

from numbers import Integral

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["region_boundary_band", "boundary_band_bce"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_TARGET_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
) + _FLOAT_DTYPES


def _validate_tensor(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided or tensor.is_nested:
        raise TypeError(f"{name} must be a dense strided tensor")
    if tensor.device.type == "meta":
        raise ValueError(f"{name} must have materialized data, not a meta device")
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape (B, 1, H, W)")
    if any(size == 0 for size in tensor.shape):
        raise ValueError(f"{name} must have positive B, H, and W")


def _validate_target(target: Tensor) -> None:
    _validate_tensor(target, "target")
    if target.dtype not in _TARGET_DTYPES:
        raise TypeError("target must have a supported boolean, integer, or real floating dtype")
    # One scalar validation read, never a per-pixel or per-image CPU transfer.
    # The binary predicate also rejects NaN and both infinities.
    if not ((target == 0) | (target == 1)).all().item():
        raise ValueError("target must contain only finite binary values (0 or 1)")


def _validate_radius(radius: int) -> int:
    # bool is an Integral in Python, but is not a meaningful pixel radius.
    if isinstance(radius, bool) or not isinstance(radius, Integral):
        raise TypeError("radius must be a nonnegative integer, not bool")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    return radius


def _band_from_validated_target(target: Tensor, radius: int) -> Tensor:
    transition = torch.zeros_like(target, dtype=torch.bool)
    horizontal = target[..., 1:] != target[..., :-1]
    transition[..., 1:] |= horizontal
    transition[..., :-1] |= horizontal
    vertical = target[..., 1:, :] != target[..., :-1, :]
    transition[..., 1:, :] |= vertical
    transition[..., :-1, :] |= vertical
    if radius == 0:
        return transition

    height, width = target.shape[-2:]
    # Once the disk covers the image diagonal, every seed reaches every pixel.
    # This also bounds kernel construction for arbitrarily large Python ints.
    if radius * radius >= (height - 1) ** 2 + (width - 1) ** 2:
        return transition.any(dim=-1, keepdim=True).any(dim=-2, keepdim=True).expand_as(transition).clone()

    # Offsets larger than the image cannot contribute, even at a large radius.
    ry, rx = min(radius, height - 1), min(radius, width - 1)
    dy = torch.arange(-ry, ry + 1, device=target.device, dtype=torch.int64)
    dx = torch.arange(-rx, rx + 1, device=target.device, dtype=torch.int64)
    disk = (dy[:, None].square() + dx[None, :].square()) <= radius * radius
    kernel = disk.to(dtype=torch.float32)[None, None]
    # Zero padding applies to transition seeds, NOT to the foreground labels.
    # The whole batch is dilated in a single convolution, with no image loops.
    return F.conv2d(transition.to(dtype=torch.float32), kernel, padding=(ry, rx)) > 0


def region_boundary_band(target: Tensor, radius: int = 2) -> Tensor:
    """Return a boolean (B, 1, H, W) GT band on ``target.device``.

    ``target`` must be a nonempty dense strided tensor containing exactly 0/1:
    bool, uint8, int8/16/32/64, or float16/bfloat16/float32/float64. Floating
    NaN/Inf, soft labels, complex/quantized/sparse tensors, and meta tensors are
    rejected. Noncontiguous strided tensors are accepted and never mutated.

    Both endpoints of each differing horizontal/vertical neighbor pair are
    marked, then dilated by offsets satisfying dy**2 + dx**2 <= radius**2.
    Dilation is clipped to the image; all-zero and all-one images have no band.
    ``radius`` is a nonnegative integer (not bool); zero returns just the seeds.
    Invalid types/dtypes raise TypeError; invalid shapes/values raise ValueError.
    """
    _validate_target(target)
    radius = _validate_radius(radius)
    return _band_from_validated_target(target, radius)


def boundary_band_bce(logits: Tensor, target: Tensor, radius: int = 2) -> Tensor:
    """Return class-balanced band BCE, averaged over ALL images in the batch.

    ``logits`` must be finite, dense, real float16/bfloat16/float32/float64 with
    exactly the same (B, 1, H, W) shape and device as ``target`` (no broadcasting).
    Target/radius validation is identical to :func:`region_boundary_band`.

    Each image contributes the mean of its nonempty background/foreground band
    BCE means. Empty bands contribute zero but remain in the batch denominator.
    The scalar stays connected to logits, including an entirely empty band,
    and differentiates only logits, not GT. No global loss coefficient is applied.
    Half/bfloat16 logits are promoted to float32; float32/float64 are preserved.
    Strict value validation requires one scalar device read per input tensor;
    all mask construction and loss reductions otherwise stay on the device.
    """
    _validate_target(target)
    _validate_tensor(logits, "logits")
    if logits.dtype not in _FLOAT_DTYPES:
        raise TypeError("logits must have dtype float16, bfloat16, float32, or float64")
    if logits.shape != target.shape:
        raise ValueError("logits and target must have exactly the same shape")
    if logits.device != target.device:
        raise ValueError("logits and target must be on the same device")
    radius = _validate_radius(radius)
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must contain only finite values, including outside the band")

    band = _band_from_validated_target(target, radius)
    foreground = target == 1
    class_masks = torch.cat((band & ~foreground, band & foreground), dim=1)
    counts = class_masks.sum(dim=(-2, -1))  # (B, 2), integer counts
    present = (counts > 0).sum(dim=1)  # (B,), empty images have zero classes

    work_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    work_logits = logits.to(dtype=work_dtype)
    labels = target.detach().to(dtype=work_dtype)
    pixel_bce = F.binary_cross_entropy_with_logits(work_logits, labels, reduction="none")

    # Broadcast B1HW BCE across the two disjoint B2HW class masks. Normalizing
    # before summation also avoids overflowing half-precision or large BCE sums.
    selected = torch.where(class_masks, pixel_bce, 0.0)
    per_class = (
        selected / counts.clamp_min(1).to(dtype=work_dtype)[..., None, None]
    ).sum(dim=(-2, -1))
    per_image = (per_class / present.clamp_min(1).to(dtype=work_dtype)[:, None]).sum(dim=1)
    # Do not return a detached literal zero or omit empty images from this mean.
    return (per_image / logits.shape[0]).sum()
