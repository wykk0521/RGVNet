"""rd_v8_grok: separate two-sided relational cut verifier with the v7 fill path.

The module keeps the rd_v7_grok_fullloss contract exactly: it cannot propose an
edit, it scores only the pixels where SR changes the raw binary Region decision,
its logits are ``[B,2,H,W]`` with channel 0 = cut and channel 1 = fill, the
accept gate is the strict ``logit > 0`` applied on candidate pixels only, every
verifier input is detached, and the objective is ordinary BCE averaged over the
candidate directions actually present in the batch (an absent direction is a
connected zero).

What v8 changes is only *where the cut decision is read from*:

* the **fill** branch is copied from ``region_discrimination_v7_grok_fullloss``
  verbatim, including the 78-channel descriptor (detached F64 + tanh(raw) +
  tanh(sr) + tanh(delta) + abs(tanh(delta)) + exp(edge_logp) both channels +
  cut/fill float masks + average-pooled union edits for kernels 3,5,7 +
  average-pooled signed edits (fill-cut) for kernels 3,5,7, all pooled with
  ``count_include_pad=False``), the bias-free 1x1 stem, the per-location channel
  LayerNorm and SiLU, the parallel depthwise 3x3 dilation-1/dilation-2 branches
  with channel LayerNorm, the deterministic mean fusion with stem residual and
  SiLU, and the zero-initialised 1x1 fill head. The attribute names are
  unchanged (``stem``, ``norm_stem``, ``depthwise_near``, ``depthwise_far``,
  ``norm_near``, ``norm_far``, ``fill_head``), so a v7 fill state loads into v8
  by name and reproduces the v7 fill logits bit for bit;
* the **cut** branch is a separate two-sided relational verifier. Its descriptor
  keeps the detached base (F64 + 4 margins + both edge probability channels +
  the cut mask = 71 channels) and adds, for every fixed ``(direction, offset)``
  pair with directions horizontal/vertical/diagonal and offsets 1/2/4, a
  two-sided sample read through zero-padded shifts that never wrap. Each pair
  contributes 6 relational channels: side feature absolute difference, side
  feature cosine similarity, side foreground support (mean of tanh(raw)) and its
  signed two-sided difference, and the same mean/signed pair for the foreground
  edge probability. Mean plus signed difference recover both sides exactly, so
  the pair encodes the two sides rather than only their average. The cut branch
  owns its own stem, LayerNorms, depthwise branches and zero-initialised 1x1 cut
  head, so no cut parameter is shared with the fill path.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Fill branch: rd_v7_grok_fullloss, preserved verbatim.
# ---------------------------------------------------------------------------
FILL_DESCRIPTOR_CHANNELS = 78
POOL_KERNELS = (3, 5, 7)
# v7-compatible alias for the preserved fill descriptor width.
DESCRIPTOR_CHANNELS = FILL_DESCRIPTOR_CHANNELS

# ---------------------------------------------------------------------------
# Cut branch: v8 fixed two-sided relational descriptor.
# ---------------------------------------------------------------------------
# (dy, dx) unit directions: horizontal, vertical, diagonal.
CUT_DIRECTIONS = ((0, 1), (1, 0), (1, 1))
CUT_OFFSETS = (1, 2, 4)
# Per-pair relational channels, in descriptor order.
CUT_PAIR_FEATURES = (
    "feature_absdiff",
    "feature_similarity",
    "foreground_mean",
    "foreground_signed",
    "edge_mean",
    "edge_signed",
)
CUT_PAIRS = len(CUT_DIRECTIONS) * len(CUT_OFFSETS)
CUT_PAIR_CHANNELS = len(CUT_PAIR_FEATURES)
# 64 detached features + 4 margins + 2 edge channels + 1 cut mask.
CUT_BASE_CHANNELS = 64 + 4 + 2 + 1
CUT_RELATIONAL_CHANNELS = CUT_PAIRS * CUT_PAIR_CHANNELS
CUT_DESCRIPTOR_CHANNELS = CUT_BASE_CHANNELS + CUT_RELATIONAL_CHANNELS

HIDDEN_CHANNELS = 24
VERIFIER_MODES = ("enabled", "always_reject")


def positive(logits: Tensor) -> Tensor:
    """Match formal sigmoid(z) > 0.5 semantics at the logit boundary."""
    return logits > 0


def candidate_masks(raw: Tensor, sr: Tensor, delta: Tensor) -> Tuple[Tensor, Tensor]:
    """Return cut (foreground->background) and fill (background->foreground) masks."""
    changed = delta != 0
    raw_positive = positive(raw)
    sr_positive = positive(sr)
    return raw_positive & ~sr_positive & changed, ~raw_positive & sr_positive & changed


def directional_labels(raw: Tensor, sr: Tensor, delta: Tensor, truth: Tensor) -> Dict[str, Tensor]:
    """Detached benefit/harm labels for the actual cut and fill candidate pixels.

    A candidate direction is beneficial exactly when SR's strict binary decision
    agrees with the strict binary truth; otherwise it is harmful. All returned
    masks are detached booleans and share the shape of ``raw``.
    """
    if not all(isinstance(tensor, Tensor) for tensor in (raw, sr, delta, truth)):
        raise ValueError("Directional labels require tensor inputs")
    raw = raw.detach()
    sr = sr.detach()
    delta = delta.detach()
    truth = truth.detach()
    if truth.shape != raw.shape:
        raise ValueError("Directional truth must match the raw Region shape")
    if sr.shape != raw.shape or delta.shape != raw.shape:
        raise ValueError("Directional sr and delta must match the raw Region shape")
    cut, fill = candidate_masks(raw, sr, delta)
    agree = positive(sr) == (truth > 0.5)
    beneficial_cut = cut & agree
    beneficial_fill = fill & agree
    return {
        "cut_edit": cut,
        "fill_edit": fill,
        "beneficial_cut": beneficial_cut,
        "harmful_cut": cut & ~agree,
        "beneficial_fill": beneficial_fill,
        "harmful_fill": fill & ~agree,
    }


def _directional_bce(logits: Tensor, edit: Tensor, target: Tensor) -> Tensor:
    """Ordinary BCE over one direction's actual candidate pixels.

    The returned empty term remains connected to logits so the training graph
    and optimizer behavior stay valid when a batch has no candidates.
    """
    edit = edit.to(dtype=torch.bool)
    target = target.to(dtype=torch.bool)
    if logits.shape != edit.shape or logits.shape != target.shape:
        raise ValueError("Verifier logits, edit mask, and target must have the same shape")
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, target.to(dtype=logits.dtype), reduction="none")
    selected = per_pixel[edit]
    if selected.numel() == 0:
        return logits.sum() * 0
    return selected.mean()


def verifier_loss(logits: Tensor, raw: Tensor, sr: Tensor, delta: Tensor, truth: Tensor) -> Tuple[Tensor, Dict[str, int]]:
    """Compute ordinary cut/fill BCE, averaging only present directions."""
    if not isinstance(logits, Tensor) or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Verifier logits must have shape [B,2,H,W]")
    for name, tensor in (("raw", raw), ("sr", sr), ("delta", delta), ("truth", truth)):
        if not isinstance(tensor, Tensor) or tensor.shape != logits[:, :1].shape:
            raise ValueError(f"{name} must have shape [B,1,H,W]")
    labels = directional_labels(raw, sr, delta, truth)
    cut = labels["cut_edit"]
    fill = labels["fill_edit"]
    # On either candidate direction, SR is beneficial exactly when it matches GT.
    target = labels["beneficial_cut"] | labels["beneficial_fill"]
    cut_loss = _directional_bce(logits[:, :1], cut, target)
    fill_loss = _directional_bce(logits[:, 1:2], fill, target)
    present = []
    if cut.any().item():
        present.append(cut_loss)
    if fill.any().item():
        present.append(fill_loss)
    loss = torch.stack(present).mean() if present else cut_loss + fill_loss
    stats = {
        "cut_candidates": int(cut.sum().item()),
        "fill_candidates": int(fill.sum().item()),
        "edit_candidates": int((cut | fill).sum().item()),
        "cut_positive": int((cut & target).sum().item()),
        "fill_positive": int((fill & target).sum().item()),
        "present_directions": len(present),
        "beneficial_cut": int(labels["beneficial_cut"].sum().item()),
        "harmful_cut": int(labels["harmful_cut"].sum().item()),
        "beneficial_fill": int(labels["beneficial_fill"].sum().item()),
        "harmful_fill": int(labels["harmful_fill"].sum().item()),
    }
    return loss, stats


def cut_pair_index(direction: int, offset: int) -> int:
    """Flat index of a (direction, offset) pair in the relational descriptor."""
    if not 0 <= direction < len(CUT_DIRECTIONS):
        raise ValueError("Cut direction index out of range")
    if offset not in CUT_OFFSETS:
        raise ValueError("Cut offset must be one of CUT_OFFSETS")
    return direction * len(CUT_OFFSETS) + CUT_OFFSETS.index(offset)


def cut_channel_index(pair_index: int, feature: str) -> int:
    """Descriptor channel index of one relational feature of one pair."""
    if type(pair_index) is not int or not 0 <= pair_index < CUT_PAIRS:
        raise ValueError("Cut pair index out of range")
    if feature not in CUT_PAIR_FEATURES:
        raise ValueError(f"Unknown cut pair feature: {feature!r}")
    return CUT_BASE_CHANNELS + CUT_PAIR_CHANNELS * pair_index + CUT_PAIR_FEATURES.index(feature)


def shift_padded(tensor: Tensor, dy: int, dx: int) -> Tensor:
    """Return ``tensor[..., y + dy, x + dx]`` using zero padding; never wraps.

    Out-of-range samples are zero instead of being rolled back to the opposite
    border, which is what a ``torch.roll`` based implementation would do.
    """
    if not isinstance(tensor, Tensor) or tensor.ndim < 2:
        raise ValueError("shift_padded requires a tensor of rank at least 2")
    if type(dy) is not int or type(dx) is not int:
        raise ValueError("shift_padded requires integer offsets")
    height, width = tensor.shape[-2], tensor.shape[-1]
    shifted = torch.zeros_like(tensor)
    target_y0, target_y1 = max(0, -dy), min(height, height - dy)
    target_x0, target_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = max(0, dy), min(height, height + dy)
    source_x0, source_x1 = max(0, dx), min(width, width + dx)
    if target_y1 > target_y0 and target_x1 > target_x0:
        shifted[..., target_y0:target_y1, target_x0:target_x1] = (
            tensor[..., source_y0:source_y1, source_x0:source_x1])
    return shifted


def side_similarity(first: Tensor, second: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-location cosine similarity of the two sides, guarded against zero norms."""
    numerator = (first * second).sum(dim=1, keepdim=True)
    denominator = (first.norm(dim=1, keepdim=True) * second.norm(dim=1, keepdim=True))
    return numerator / denominator.clamp_min(eps)


class RegionCandidateVerifier(nn.Module):
    """The v7 fill branch plus a separate two-sided relational cut branch."""

    def __init__(self, mode: str = "enabled") -> None:
        super().__init__()
        if type(mode) is not str or mode not in VERIFIER_MODES:
            raise ValueError(f"Unsupported verifier mode: {mode!r}")
        self.mode = mode
        # Fill path: identical attribute names, geometry and maths to rd_v7.
        self.stem = nn.Conv2d(FILL_DESCRIPTOR_CHANNELS, HIDDEN_CHANNELS, 1, bias=False)
        self.norm_stem = nn.LayerNorm(HIDDEN_CHANNELS)
        self.depthwise_near = nn.Conv2d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, 3, padding=1, dilation=1,
            groups=HIDDEN_CHANNELS, bias=False)
        self.depthwise_far = nn.Conv2d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, 3, padding=2, dilation=2,
            groups=HIDDEN_CHANNELS, bias=False)
        self.norm_near = nn.LayerNorm(HIDDEN_CHANNELS)
        self.norm_far = nn.LayerNorm(HIDDEN_CHANNELS)
        self.fill_head = nn.Conv2d(HIDDEN_CHANNELS, 1, 1, bias=True)
        # Cut path: fully separate relational encoder.
        self.cut_stem = nn.Conv2d(CUT_DESCRIPTOR_CHANNELS, HIDDEN_CHANNELS, 1, bias=False)
        self.cut_norm_stem = nn.LayerNorm(HIDDEN_CHANNELS)
        self.cut_depthwise_near = nn.Conv2d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, 3, padding=1, dilation=1,
            groups=HIDDEN_CHANNELS, bias=False)
        self.cut_depthwise_far = nn.Conv2d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, 3, padding=2, dilation=2,
            groups=HIDDEN_CHANNELS, bias=False)
        self.cut_norm_near = nn.LayerNorm(HIDDEN_CHANNELS)
        self.cut_norm_far = nn.LayerNorm(HIDDEN_CHANNELS)
        self.cut_head = nn.Conv2d(HIDDEN_CHANNELS, 1, 1, bias=True)
        for head in (self.cut_head, self.fill_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _check(features: Tensor, raw: Tensor, sr: Tensor, delta: Tensor, edge_logp: Tensor) -> None:
        tensors = {"features": features, "raw": raw, "sr": sr, "delta": delta, "edge_logp": edge_logp}
        for name, tensor in tensors.items():
            if not isinstance(tensor, Tensor) or tensor.ndim != 4 or not tensor.is_floating_point():
                raise ValueError(f"Verifier input must be floating rank-4: {name}")
            if not torch.isfinite(tensor).all().item():
                raise FloatingPointError(f"Non-finite verifier input: {name}")
        batch, channels, height, width = features.shape
        if channels != 64 or min(batch, height, width) < 1:
            raise ValueError("features must have shape [B,64,H,W]")
        if raw.shape != (batch, 1, height, width):
            raise ValueError("raw must have shape [B,1,H,W]")
        if sr.shape != raw.shape or delta.shape != raw.shape:
            raise ValueError("sr and delta must match raw shape")
        if edge_logp.shape != (batch, 2, height, width):
            raise ValueError("edge_logp must have shape [B,2,H,W]")
        device, dtype = features.device, features.dtype
        if any(t.device != device or t.dtype != dtype for t in tensors.values()):
            raise ValueError("Verifier inputs must share device and dtype")

    @staticmethod
    def _fill_descriptor(features: Tensor, raw: Tensor, sr: Tensor, delta: Tensor, edge_logp: Tensor) -> Tensor:
        """Build the detached 78-channel v7 local descriptor (unchanged)."""
        features = features.detach()
        raw = raw.detach()
        sr = sr.detach()
        delta = delta.detach()
        edge = torch.exp(edge_logp.detach())
        cut, fill = candidate_masks(raw, sr, delta)
        cut_float = cut.to(dtype=features.dtype)
        fill_float = fill.to(dtype=features.dtype)
        union = cut_float + fill_float
        signed = fill_float - cut_float
        tanh_delta = torch.tanh(delta)
        channels = [
            features,
            torch.tanh(raw),
            torch.tanh(sr),
            tanh_delta,
            tanh_delta.abs(),
            edge[:, :1],
            edge[:, 1:2],
            cut_float,
            fill_float,
        ]
        for kernel in POOL_KERNELS:
            channels.append(F.avg_pool2d(
                union, kernel, stride=1, padding=kernel // 2, count_include_pad=False))
        for kernel in POOL_KERNELS:
            channels.append(F.avg_pool2d(
                signed, kernel, stride=1, padding=kernel // 2, count_include_pad=False))
        descriptor = torch.cat(channels, dim=1)
        if not torch.isfinite(descriptor).all().item():
            raise FloatingPointError("Non-finite verifier descriptor")
        return descriptor

    # v7-compatible name for the preserved fill descriptor.
    _descriptor = _fill_descriptor

    @staticmethod
    def _cut_descriptor(features: Tensor, raw: Tensor, sr: Tensor, delta: Tensor, edge_logp: Tensor) -> Tensor:
        """Build the detached 125-channel two-sided relational cut descriptor.

        Layout: 64 detached features, tanh(raw), tanh(sr), tanh(delta),
        abs(tanh(delta)), exp(edge_logp)[:, :1], exp(edge_logp)[:, 1:2], the cut
        float mask, then 6 relational channels for each of the 9 fixed
        (direction, offset) pairs, in ``CUT_PAIR_FEATURES`` order.
        """
        features = features.detach()
        raw = raw.detach()
        sr = sr.detach()
        delta = delta.detach()
        edge = torch.exp(edge_logp.detach())
        foreground = torch.tanh(raw)
        edge_foreground = edge[:, 1:2]
        cut, _ = candidate_masks(raw, sr, delta)
        channels = [
            features,
            foreground,
            torch.tanh(sr),
            torch.tanh(delta),
            torch.tanh(delta).abs(),
            edge[:, :1],
            edge[:, 1:2],
            cut.to(dtype=features.dtype),
        ]
        for direction_y, direction_x in CUT_DIRECTIONS:
            for offset in CUT_OFFSETS:
                step_y, step_x = direction_y * offset, direction_x * offset
                feature_a = shift_padded(features, step_y, step_x)
                feature_b = shift_padded(features, -step_y, -step_x)
                foreground_a = shift_padded(foreground, step_y, step_x)
                foreground_b = shift_padded(foreground, -step_y, -step_x)
                edge_a = shift_padded(edge_foreground, step_y, step_x)
                edge_b = shift_padded(edge_foreground, -step_y, -step_x)
                channels.extend([
                    (feature_a - feature_b).abs().mean(dim=1, keepdim=True),
                    side_similarity(feature_a, feature_b),
                    (foreground_a + foreground_b) / 2,
                    foreground_a - foreground_b,
                    (edge_a + edge_b) / 2,
                    edge_a - edge_b,
                ])
        descriptor = torch.cat(channels, dim=1)
        if not torch.isfinite(descriptor).all().item():
            raise FloatingPointError("Non-finite cut relational descriptor")
        return descriptor

    @staticmethod
    def _channel_norm(norm: nn.LayerNorm, hidden: Tensor) -> Tensor:
        return norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _encode_fill(self, descriptor: Tensor) -> Tensor:
        hidden = F.silu(self._channel_norm(self.norm_stem, self.stem(descriptor)))
        near = self._channel_norm(self.norm_near, self.depthwise_near(hidden))
        far = self._channel_norm(self.norm_far, self.depthwise_far(hidden))
        return F.silu(hidden + (near + far) / 2)

    def _encode_cut(self, descriptor: Tensor) -> Tensor:
        hidden = F.silu(self._channel_norm(self.cut_norm_stem, self.cut_stem(descriptor)))
        near = self._channel_norm(self.cut_norm_near, self.cut_depthwise_near(hidden))
        far = self._channel_norm(self.cut_norm_far, self.cut_depthwise_far(hidden))
        return F.silu(hidden + (near + far) / 2)

    def forward(self, features: Tensor, raw: Tensor, sr: Tensor, delta: Tensor, edge_logp: Tensor) -> Tensor:
        self._check(features, raw, sr, delta, edge_logp)
        fill_hidden = self._encode_fill(
            self._fill_descriptor(features, raw, sr, delta, edge_logp))
        cut_hidden = self._encode_cut(
            self._cut_descriptor(features, raw, sr, delta, edge_logp))
        logits = torch.cat((self.cut_head(cut_hidden), self.fill_head(fill_hidden)), dim=1)
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Non-finite verifier logits")
        if self.mode == "always_reject":
            return logits * 0 - 1
        return logits


def hard_keep(raw: Tensor, sr: Tensor, delta: Tensor, logits: Tensor) -> Tuple[Tensor, Tensor]:
    """Return direction-specific accepted masks; verifier cannot create edits."""
    if not isinstance(logits, Tensor) or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Verifier logits must have shape [B,2,H,W]")
    if logits.shape[0] != raw.shape[0] or logits.shape[2:] != raw.shape[2:]:
        raise ValueError("Verifier logits must match the Region spatial shape")
    cut, fill = candidate_masks(raw, sr, delta)
    return cut & (logits[:, :1] > 0), fill & (logits[:, 1:2] > 0)


def gated_region(raw: Tensor, delta: Tensor, keep_cut: Tensor, keep_fill: Tensor) -> Tensor:
    keep = (keep_cut | keep_fill).to(dtype=raw.dtype)
    return raw + keep * delta
