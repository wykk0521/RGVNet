"""rd_v9_context: context-supported candidate verification on top of rd_v8_grok.

The module keeps every rd_v8_grok/rd_v7_grok contract that the formal training
budget depends on:

* it cannot propose an edit; it only scores the pixels where SR changes the raw
  binary Region decision;
* its logits are ``[B,2,H,W]`` with channel 0 = cut and channel 1 = fill, the
  accept gate is the strict ``logit > 0`` applied on candidate pixels only, and
  the gated result is exactly ``raw + keep * delta``;
* every verifier input is detached from the backbone/SR graph, ground truth is
  only ever used to build training labels, and the objective is ordinary BCE
  averaged over the candidate directions actually present in the batch (an
  absent direction is a connected zero);
* the fill branch (78 channels, kernels 3/5/7) and the two-sided relational cut
  branch (125 channels, directions horizontal/vertical/diagonal x offsets 1/2/4)
  are *imported* from ``tools.region_discrimination_v8_grok_fullloss`` and are
  therefore literally the same code, layer for layer and attribute name for
  attribute name.  ``RegionCandidateVerifier`` here is a thin subclass whose
  ``local_logits`` reproduces the v8 forward exactly, so a v8 state loads by name
  with ``strict=False`` (the v9-only aggregation sentinel is reported missing)
  and the v9 module with a zero context head is bit-for-bit v8.  The sentinel is
  a persistent buffer checked in ``_load_from_state_dict``: the two aggregation
  variants share every parameter name and shape, and loading one variant's state
  into the other now raises before any tensor is copied.

What v9 adds is a *context-supported additive residual* on the v8 score:

    z9,d = z_local,d + r_d ,   keep_d = candidate_d AND (z9,d > 0)

where ``z_local,d`` is the unchanged v8 cut/fill logit and ``r_d`` is produced by
a direction-private, lightweight context branch.  The branch reads

1. a shared learnable 1x1 projection ``E`` of the **detached** F64 features
   (``CONTEXT_EMBED_CHANNELS = 8``); only the verifier loss updates it;
2. a *center block* (``CENTER_CHANNELS = 14``): ``E`` (8), the center margins
   ``tanh(raw)``, ``tanh(sr)``, ``tanh(delta)``, ``|tanh(delta)|`` (4), the
   center foreground edge probability ``exp(edge_logp)[:, 1:2]`` (1) and the
   direction candidate flag ``C_d`` (1);
3. a *support block* (``SUPPORT_CHANNELS = 117``): for each of the three
   direction-specific support masks - the direction candidates ``C_d``, the
   raw foreground outside all candidates ``S_fg`` and the raw background outside
   all candidates ``S_bg`` - and each fixed window scale in
   ``SUPPORT_SCALES = (3, 7, 15)``, the 13 channels
   ``[mask-weighted E mean (8), mask density (1), mask-weighted tanh(raw) (1),
   mask-weighted tanh(sr) (1), mask-weighted tanh(delta) (1), mask-weighted
   foreground edge probability (1)]``.

The density denominator counts only in-image pixels of the (zero-padded) window:
``avg_pool2d(..., count_include_pad=False)`` never wraps to the opposite border.
An empty support has mean 0 and density 0, so the absence is explicit and never
silently read as a zero-valued measurement.  The support regions are the places
where raw and SR did not flip; they are not ground truth and not guaranteed
correct.  Ground truth, benefit labels, object ids and v8 acceptance labels are
never descriptor inputs.

Each direction owns a separate residual encoder of exactly the same skeleton:
``1x1 CONTEXT_DESCRIPTOR_CHANNELS -> 16`` (bias-free stem), channel LayerNorm,
SiLU, depthwise 3x3 (groups=16, bias-free) with channel LayerNorm and an
additive residual, and a final ``1x1 16 -> 1`` head whose weight *and* bias are
zero-initialised.  Therefore a freshly built v9 module computes ``r_d == 0``
exactly and its logits are bit-for-bit the v8 logits for the same inputs; the
new branch starts to learn only after the zero head receives its first
gradients.  A zero head necessarily gives the upstream context encoder a zero
gradient in that first step; that is a property of the zero initialisation, not
a missing gradient path, and the second step (after the head moved) must show a
non-zero gradient in the whole branch.

The module also ships the diagnostic control and the calibration utilities that
the design decision of 2026-10-03 requires:

* ``aggregation="center_only"`` builds the *center-only control*: the same
  center block plus, in the 117 support channels, the same 13-channel recipe
  evaluated at the pixel itself and gated by the mask value (window size 1).
  The control therefore keeps the identical network skeleton and parameter
  count without any dead zero input, and differs from the trained variant only
  in the absence of multi-scale spatial support aggregation.  The three scale
  slots of the control necessarily carry the same center statistics; that
  duplication is the price of the matched parameter budget and is documented
  here instead of being hidden behind zero padding.
* ``monotone_affine_calibration`` / ``calibration_logit_threshold`` provide the
  per-direction strictly increasing affine score calibration ``a_d > 0, b_d``
  used to *re-parameterise* a BR/HA curve.  ``br_ha_curve`` reports per-direction
  rejection/acceptance rates; it never claims that a monotone calibration is a
  ranking improvement.
"""
from __future__ import annotations

from numbers import Real
from typing import Dict, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# The whole v8 scoring path is imported, never copied: the fill descriptor,
# the relational cut descriptor, the two encoders, the heads, the BCE objective,
# the candidate masks and the gate are the identical objects.
from tools.region_discrimination_v8_grok_fullloss import (
    CUT_BASE_CHANNELS,
    CUT_DESCRIPTOR_CHANNELS,
    CUT_DIRECTIONS,
    CUT_OFFSETS,
    CUT_PAIR_CHANNELS,
    CUT_PAIR_FEATURES,
    CUT_PAIRS,
    CUT_RELATIONAL_CHANNELS,
    DESCRIPTOR_CHANNELS,
    FILL_DESCRIPTOR_CHANNELS,
    HIDDEN_CHANNELS,
    POOL_KERNELS,
    VERIFIER_MODES,
    RegionCandidateVerifier as _V8RegionCandidateVerifier,
    candidate_masks,
    cut_channel_index,
    cut_pair_index,
    directional_labels,
    gated_region,
    hard_keep,
    positive,
    shift_padded,
    side_similarity,
    verifier_loss,
)

# ---------------------------------------------------------------------------
# v9 context residual descriptor and encoder.
# ---------------------------------------------------------------------------
CONTEXT_EMBED_CHANNELS = 8
SUPPORT_SCALES = (3, 7, 15)
SUPPORT_MASKS = ("candidate", "foreground", "background")
# Per (mask, scale) channels, in order: 8 embedding means, density, three
# margin means, one edge mean.
SUPPORT_FEATURES = ("embedding_mean", "density", "raw_mean", "sr_mean",
                    "delta_mean", "edge_mean")
SUPPORT_GROUP_CHANNELS = CONTEXT_EMBED_CHANNELS + 5
SUPPORT_CHANNELS = len(SUPPORT_MASKS) * len(SUPPORT_SCALES) * SUPPORT_GROUP_CHANNELS
# Center block: 8 embedding channels, four margins, one edge channel, one flag.
CENTER_FEATURES = ("embedding", "raw_margin", "sr_margin", "delta_margin",
                   "delta_abs_margin", "edge_foreground", "candidate_flag")
CENTER_CHANNELS = CONTEXT_EMBED_CHANNELS + 4 + 1 + 1
CONTEXT_DESCRIPTOR_CHANNELS = CENTER_CHANNELS + SUPPORT_CHANNELS
CONTEXT_HIDDEN_CHANNELS = 16
AGGREGATION_MODES = ("multi_scale_support", "center_only")
# Directions of the residual branch, aligned with the logits channel order.
CONTEXT_DIRECTIONS = ("cut", "fill")
DIRECTION_COUNT = 2
# Floor used only to guard 0/0 in empty supports; empty supports are zeroed.
SUPPORT_DENSITY_EPS = 1e-6


# ---------------------------------------------------------------------------
# v9 context descriptor helpers.
# ---------------------------------------------------------------------------
def support_masks(raw: Tensor, sr: Tensor, delta: Tensor, direction: str) -> Dict[str, Tensor]:
    """Return the three support masks of one direction.

    ``candidate`` is the direction's own candidate mask; ``foreground`` and
    ``background`` are the raw foreground/background pixels outside *all*
    candidates.
    """
    if direction not in CONTEXT_DIRECTIONS:
        raise ValueError(f"Unknown context direction: {direction!r}")
    cut, fill = candidate_masks(raw, sr, delta)
    outside = ~(cut | fill)
    raw_positive = positive(raw)
    candidate = cut if direction == "cut" else fill
    return {
        "candidate": candidate,
        "foreground": raw_positive & outside,
        "background": (~raw_positive) & outside,
    }


def masked_window_statistics(mask: Tensor, values: Tensor, kernel: int) -> Tuple[Tensor, Tensor]:
    """Mask-weighted means and density over a ``kernel`` window; never wraps.

    ``mask`` is a float [B,1,H,W] indicator, ``values`` a finite [B,C,H,W]
    tensor.  Both pooled quantities use ``count_include_pad=False``, so the
    density denominator counts only in-image pixels of the window.  A window
    with zero mask mass returns a zero mean and a zero density.
    """
    if not isinstance(mask, Tensor) or not isinstance(values, Tensor):
        raise ValueError("masked_window_statistics requires tensors")
    if mask.ndim != 4 or mask.shape[1] != 1 or values.ndim != 4:
        raise ValueError("mask must be [B,1,H,W] and values [B,C,H,W]")
    if mask.shape[0] != values.shape[0] or mask.shape[2:] != values.shape[2:]:
        raise ValueError("mask and values must share batch and spatial shape")
    if type(kernel) is not int or kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel must be a positive odd integer")
    padding = kernel // 2
    mask_float = mask.to(values.dtype)
    density = F.avg_pool2d(mask_float, kernel, stride=1,
                           padding=padding, count_include_pad=False)
    weighted = F.avg_pool2d(mask_float * values, kernel, stride=1,
                            padding=padding, count_include_pad=False)
    mean = weighted / density.clamp_min(SUPPORT_DENSITY_EPS)
    mean = mean * (density > 0).to(mean.dtype)
    # Finiteness is enforced once per descriptor instead of once per block: a
    # host sync per (mask, scale) would cost 18 synchronisations per direction
    # for no extra evidence, because the descriptor is the only consumer.
    return mean, density


def center_channel_index(feature: str) -> int:
    """First channel index of one center-block feature (embedding spans 8)."""
    if feature not in CENTER_FEATURES:
        raise ValueError(f"Unknown center feature: {feature!r}")
    offset = 0
    for name in CENTER_FEATURES:
        if name == feature:
            return offset
        offset += CONTEXT_EMBED_CHANNELS if name == "embedding" else 1
    raise ValueError(f"Unknown center feature: {feature!r}")


def support_block_index(mask: str, scale: int) -> int:
    """First descriptor channel index of one (mask, scale) support block."""
    if mask not in SUPPORT_MASKS:
        raise ValueError(f"Unknown support mask: {mask!r}")
    if scale not in SUPPORT_SCALES:
        raise ValueError("Support scale must be one of SUPPORT_SCALES")
    block = SUPPORT_MASKS.index(mask) * len(SUPPORT_SCALES) + SUPPORT_SCALES.index(scale)
    return CENTER_CHANNELS + block * SUPPORT_GROUP_CHANNELS


def support_channel_index(mask: str, scale: int, feature: str) -> int:
    """Descriptor channel index of one support feature of one (mask, scale)."""
    if feature not in SUPPORT_FEATURES:
        raise ValueError(f"Unknown support feature: {feature!r}")
    offset = 0
    for name in SUPPORT_FEATURES:
        if name == feature:
            return support_block_index(mask, scale) + offset
        offset += CONTEXT_EMBED_CHANNELS if name == "embedding_mean" else 1
    raise ValueError(f"Unknown support feature: {feature!r}")


def context_center_block(embedding: Tensor, raw: Tensor, sr: Tensor, delta: Tensor,
                         edge_logp: Tensor, candidate: Tensor) -> Tensor:
    """Build the 14-channel center block shared by both aggregation variants."""
    edge_foreground = torch.exp(edge_logp.detach())[:, 1:2]
    tanh_delta = torch.tanh(delta.detach())
    return torch.cat([
        embedding,
        torch.tanh(raw.detach()),
        torch.tanh(sr.detach()),
        tanh_delta,
        tanh_delta.abs(),
        edge_foreground,
        candidate.to(dtype=embedding.dtype),
    ], dim=1)


def context_support_block(embedding: Tensor, raw: Tensor, sr: Tensor, delta: Tensor,
                          edge_logp: Tensor, masks: Dict[str, Tensor],
                          aggregation: str) -> Tensor:
    """Build the 117-channel support block for one direction.

    ``multi_scale_support`` applies the mask-weighted recipe at scales 3/7/15.
    ``center_only`` applies the same recipe to the center pixel alone (window
    size 1) and stores it in each scale slot: the control keeps the identical
    network skeleton and parameter count without any dead zero input.  The
    three slots are therefore duplicates of the center statistics, and the
    control has no spatial support aggregation.
    """
    if aggregation not in AGGREGATION_MODES:
        raise ValueError(f"Unsupported aggregation mode: {aggregation!r}")
    values = torch.cat([
        embedding,
        torch.tanh(raw.detach()),
        torch.tanh(sr.detach()),
        torch.tanh(delta.detach()),
        torch.exp(edge_logp.detach())[:, 1:2],
    ], dim=1)
    groups = []
    for name in SUPPORT_MASKS:
        mask = masks[name]
        if aggregation == "multi_scale_support":
            for scale in SUPPORT_SCALES:
                mean, density = masked_window_statistics(mask, values, scale)
                groups.append(torch.cat([
                    mean[:, :CONTEXT_EMBED_CHANNELS],
                    density,
                    mean[:, CONTEXT_EMBED_CHANNELS:],
                ], dim=1))
        else:
            gate = mask.to(dtype=values.dtype)
            gated = values * gate
            group = torch.cat([
                gated[:, :CONTEXT_EMBED_CHANNELS],
                gate,
                gated[:, CONTEXT_EMBED_CHANNELS:],
            ], dim=1)
            for _ in SUPPORT_SCALES:
                groups.append(group)
    block = torch.cat(groups, dim=1)
    if block.shape[1] != SUPPORT_CHANNELS:
        raise ValueError("Context support block has an unexpected channel count")
    return block


def context_descriptor(projection: nn.Module, features: Tensor, raw: Tensor, sr: Tensor,
                       delta: Tensor, edge_logp: Tensor, direction: str,
                       aggregation: str) -> Tensor:
    """Detached 131-channel context descriptor of one direction.

    Layout: center block (8 embedding + 4 margins + edge foreground + candidate
    flag = 14), then the support block in ``SUPPORT_MASKS`` x ``SUPPORT_SCALES``
    order (117).  Ground truth, benefit labels, object ids and v8 acceptance
    labels are never inputs.
    """
    if direction not in CONTEXT_DIRECTIONS:
        raise ValueError(f"Unknown context direction: {direction!r}")
    if aggregation not in AGGREGATION_MODES:
        raise ValueError(f"Unsupported aggregation mode: {aggregation!r}")
    embedding = projection(features.detach())
    if embedding.shape[1] != CONTEXT_EMBED_CHANNELS:
        raise ValueError("Context projection must emit CONTEXT_EMBED_CHANNELS channels")
    masks = support_masks(raw, sr, delta, direction)
    center = context_center_block(embedding, raw, sr, delta, edge_logp,
                                  masks["candidate"])
    support = context_support_block(embedding, raw, sr, delta, edge_logp, masks, aggregation)
    descriptor = torch.cat([center, support], dim=1)
    if descriptor.shape[1] != CONTEXT_DESCRIPTOR_CHANNELS:
        raise ValueError("Context descriptor has an unexpected channel count")
    if not torch.isfinite(descriptor).all().item():
        raise FloatingPointError("Non-finite context descriptor")
    return descriptor


# ---------------------------------------------------------------------------
# Diagnostic calibration: strictly increasing per-direction affine score maps.
# ---------------------------------------------------------------------------
def _calibration_values(value: Union[Real, Tensor, Sequence[Real]], name: str,
                        dtype: torch.dtype) -> Tensor:
    """Cast a calibration coefficient to ``dtype`` and validate it there.

    The validation happens *after* the cast, so the exact numbers the arithmetic
    will use are the numbers that were checked: a coefficient that overflows to
    ``inf`` or underflows to ``0`` in the working dtype is rejected instead of
    silently deforming (or disabling) the calibration.
    """
    if isinstance(value, Tensor):
        tensor = value.detach().to(dtype=torch.float64).reshape(-1)
    elif isinstance(value, Real) and not isinstance(value, bool):
        tensor = torch.tensor([float(value)], dtype=torch.float64)
    elif isinstance(value, (list, tuple)) and len(value) == DIRECTION_COUNT:
        tensor = torch.tensor([float(item) for item in value], dtype=torch.float64)
    else:
        raise ValueError(f"{name} must be a real scalar or a {DIRECTION_COUNT}-element sequence")
    if tensor.numel() == 1:
        tensor = tensor.expand(DIRECTION_COUNT)
    if tensor.numel() != DIRECTION_COUNT or not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain {DIRECTION_COUNT} finite values")
    cast = tensor.clone().to(dtype)
    if not torch.isfinite(cast).all().item():
        raise ValueError(f"{name} is not finite in dtype {dtype}")
    return cast


def monotone_affine_calibration(logits: Tensor,
                                a: Union[Real, Tensor, Sequence[Real]],
                                b: Union[Real, Tensor, Sequence[Real]]) -> Tensor:
    """Per-direction strictly increasing affine calibration ``a_d * z_d + b_d``.

    ``a_d > 0`` is enforced in the logits dtype: the map preserves the score
    order inside each direction, so a calibrated decision
    ``a_d * z_d + b_d > t`` is the same rule as ``z_d > (t - b_d) / a_d``.  A
    calibration therefore only moves the operating point along the existing
    BR/HA curve; it is never evidence of a ranking improvement.  Coefficients
    are validated after the cast to ``logits.dtype``, and a calibrated score
    that overflows that dtype raises instead of returning ``inf``.
    """
    if not isinstance(logits, Tensor) or logits.ndim != 4 or logits.shape[1] != DIRECTION_COUNT:
        raise ValueError("Calibration logits must have shape [B,2,H,W]")
    if not logits.is_floating_point():
        raise ValueError("Calibration logits must be floating point")
    dtype = logits.dtype
    slope = _calibration_values(a, "a", dtype)
    offset = _calibration_values(b, "b", dtype)
    if not bool((slope > 0).all().item()):
        raise ValueError(f"Calibration slope must be strictly positive (a > 0) in {dtype}")
    shape = (1, DIRECTION_COUNT, 1, 1)
    calibrated = (logits * slope.to(device=logits.device).view(shape) +
                  offset.to(device=logits.device).view(shape))
    if not torch.isfinite(calibrated).all().item():
        raise FloatingPointError(f"Non-finite calibrated scores in dtype {dtype}")
    return calibrated


def calibration_logit_threshold(a: Union[Real, Tensor, Sequence[Real]],
                                b: Union[Real, Tensor, Sequence[Real]],
                                dtype: Union[torch.dtype, None] = None) -> Tuple[float, float]:
    """Per-direction raw-logit threshold equivalent to calibrated score > 0.

    ``dtype`` is the dtype the calibration will actually be applied in (pass the
    logits dtype for half-precision runs); coefficients are validated after the
    cast to it and the returned thresholds must be finite.
    """
    if dtype is None:
        dtype = torch.float64
    if not isinstance(dtype, torch.dtype):
        raise ValueError("dtype must be a torch.dtype")
    slope = _calibration_values(a, "a", dtype)
    offset = _calibration_values(b, "b", dtype)
    if not bool((slope > 0).all().item()):
        raise ValueError(f"Calibration slope must be strictly positive (a > 0) in {dtype}")
    threshold = -offset / slope
    if not torch.isfinite(threshold).all().item():
        raise FloatingPointError(f"Non-finite calibrated threshold in dtype {dtype}")
    values = threshold.tolist()
    return (float(values[0]), float(values[1]))


def _rate(numerator: int, denominator: int) -> Union[float, None]:
    if denominator == 0:
        return None
    return numerator / denominator


def br_ha_curve(scores: Tensor, beneficial: Tensor, harmful: Tensor,
                thresholds: Sequence[float]) -> Dict[str, object]:
    """Per-direction rejected-beneficial and accepted-harmful rates over thresholds.

    ``scores`` is [B,2,H,W]; ``beneficial``/``harmful`` are boolean candidate
    masks of the same shape whose channels are never mixed.  A candidate is
    accepted at a threshold when ``score > threshold`` (strict, matching the
    formal gate at threshold 0).  Every point reports the raw counts next to the
    rates; a zero denominator is reported as ``None`` rather than 0.
    """
    if not isinstance(scores, Tensor) or scores.ndim != 4 or scores.shape[1] != DIRECTION_COUNT:
        raise ValueError("BR/HA scores must have shape [B,2,H,W]")
    if not scores.is_floating_point():
        raise ValueError("BR/HA scores must be floating point")
    if not torch.isfinite(scores).all().item():
        raise FloatingPointError("Non-finite BR/HA scores")
    for name, mask in (("beneficial", beneficial), ("harmful", harmful)):
        if not isinstance(mask, Tensor) or mask.shape != scores.shape:
            raise ValueError(f"{name} must be a boolean tensor matching the scores")
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must have boolean dtype")
    if bool((beneficial & harmful).any().item()):
        raise ValueError("beneficial and harmful must be disjoint candidate sets")
    curve = []
    for threshold in thresholds:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("Thresholds must be real numbers")
        point = {"threshold": float(threshold), "directions": {}}
        for index, name in enumerate(CONTEXT_DIRECTIONS):
            accepted = scores[:, index:index + 1] > float(threshold)
            good = beneficial[:, index:index + 1].to(dtype=torch.bool)
            bad = harmful[:, index:index + 1].to(dtype=torch.bool)
            accepted_good = int((accepted & good).sum().item())
            accepted_bad = int((accepted & bad).sum().item())
            total_good = int(good.sum().item())
            total_bad = int(bad.sum().item())
            point["directions"][name] = {
                "beneficial_total": total_good,
                "harmful_total": total_bad,
                "accepted_beneficial": accepted_good,
                "accepted_harmful": accepted_bad,
                "rejected_beneficial": total_good - accepted_good,
                "rejected_harmful": total_bad - accepted_bad,
                "beneficial_rejection_rate": _rate(total_good - accepted_good, total_good),
                "harmful_acceptance_rate": _rate(accepted_bad, total_bad),
                "acceptance_precision": _rate(accepted_good, accepted_good + accepted_bad),
            }
        curve.append(point)
    return {"direction_order": list(CONTEXT_DIRECTIONS), "points": curve}


# ---------------------------------------------------------------------------
# Verifier: the v8 module plus one zero-initialised context residual per branch.
# ---------------------------------------------------------------------------
class _ContextResidual(nn.Module):
    """Direction-private lightweight residual encoder (zero-initialised head)."""

    def __init__(self, hidden_channels: int = CONTEXT_HIDDEN_CHANNELS) -> None:
        super().__init__()
        if type(hidden_channels) is not int or hidden_channels != CONTEXT_HIDDEN_CHANNELS:
            raise ValueError(f"Context hidden channels must be {CONTEXT_HIDDEN_CHANNELS}")
        self.stem = nn.Conv2d(CONTEXT_DESCRIPTOR_CHANNELS, hidden_channels, 1, bias=False)
        self.norm_stem = nn.LayerNorm(hidden_channels)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                                   groups=hidden_channels, bias=False)
        self.norm_depthwise = nn.LayerNorm(hidden_channels)
        self.head = nn.Conv2d(hidden_channels, 1, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _channel_norm(norm: nn.LayerNorm, hidden: Tensor) -> Tensor:
        return norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, descriptor: Tensor) -> Tensor:
        hidden = F.silu(self._channel_norm(self.norm_stem, self.stem(descriptor)))
        near = self._channel_norm(self.norm_depthwise, self.depthwise(hidden))
        return self.head(F.silu(hidden + near))


class RegionCandidateVerifier(_V8RegionCandidateVerifier):
    """rd_v8_grok scoring plus a direction-private v9 context residual."""

    def __init__(self, mode: str = "enabled",
                 aggregation: str = "multi_scale_support") -> None:
        if type(aggregation) is not str or aggregation not in AGGREGATION_MODES:
            raise ValueError(f"Unsupported aggregation mode: {aggregation!r}")
        super().__init__(mode=mode)
        # Read-only after construction: the two variants share every parameter
        # name and shape, so a mutation after a load could silently reuse the
        # other variant's statistics.  Rebuild the module instead.
        self._aggregation = aggregation
        # Persistent sentinel: the two aggregation variants have the same
        # parameter names and shapes, so a state load would otherwise be silent.
        # `_load_from_state_dict` refuses a state whose sentinel disagrees, which
        # also forbids "load then mutate aggregation" from going unnoticed.
        self.register_buffer(
            "aggregation_code",
            torch.tensor(AGGREGATION_MODES.index(aggregation), dtype=torch.int64),
            persistent=True)
        # No v8 parameter is shared with the context branch; the projection only
        # sees detached F64 and is updated by the verifier loss alone.
        self.context_projection = nn.Conv2d(64, CONTEXT_EMBED_CHANNELS, 1, bias=False)
        self.cut_context = _ContextResidual()
        self.fill_context = _ContextResidual()

    @property
    def aggregation(self) -> str:
        """The aggregation mode this module was built with (never reassignable)."""
        return self._aggregation

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs) -> None:
        """Reject a state dict produced by the other aggregation variant.

        Checked before any tensor is copied, so a rejected load cannot leave a
        partially overwritten module behind.  A v8 state dict has no sentinel and
        is therefore only loadable with ``strict=False`` (a missing key), never
        as a silent same-shape substitute for a v9 state.
        """
        key = prefix + "aggregation_code"
        if key in state_dict:
            incoming = torch.as_tensor(state_dict[key]).reshape(-1).to(
                device=self.aggregation_code.device, dtype=self.aggregation_code.dtype)
            if incoming.numel() != 1 or int(incoming.item()) != int(self.aggregation_code.item()):
                raise RuntimeError(
                    "Refusing to load a state dict whose aggregation sentinel "
                    f"{int(incoming.item()) if incoming.numel() == 1 else 'shape ' + str(tuple(incoming.shape))} "
                    f"does not match this module's aggregation {self.aggregation!r} "
                    f"(sentinel {int(self.aggregation_code.item())}); rebuild the "
                    "module with the matching aggregation instead of loading "
                    "across variants")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def local_logits(self, features: Tensor, raw: Tensor, sr: Tensor, delta: Tensor,
                     edge_logp: Tensor) -> Tensor:
        """The preserved v8 cut/fill logits, without the v9 residual."""
        fill_hidden = self._encode_fill(
            self._fill_descriptor(features, raw, sr, delta, edge_logp))
        cut_hidden = self._encode_cut(
            self._cut_descriptor(features, raw, sr, delta, edge_logp))
        return torch.cat((self.cut_head(cut_hidden), self.fill_head(fill_hidden)), dim=1)

    def context_residual(self, features: Tensor, raw: Tensor, sr: Tensor, delta: Tensor,
                         edge_logp: Tensor, direction: str) -> Tensor:
        """The scalar residual ``r_d`` of one direction (exactly 0 at init)."""
        descriptor = context_descriptor(
            self.context_projection, features, raw, sr, delta, edge_logp,
            direction, self.aggregation)
        encoder = self.cut_context if direction == "cut" else self.fill_context
        return encoder(descriptor)

    def forward(self, features: Tensor, raw: Tensor, sr: Tensor, delta: Tensor,
                edge_logp: Tensor) -> Tensor:
        self._check(features, raw, sr, delta, edge_logp)
        local = self.local_logits(features, raw, sr, delta, edge_logp)
        cut_residual = self.context_residual(features, raw, sr, delta, edge_logp, "cut")
        fill_residual = self.context_residual(features, raw, sr, delta, edge_logp, "fill")
        logits = local + torch.cat((cut_residual, fill_residual), dim=1)
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Non-finite verifier logits")
        if self.mode == "always_reject":
            return logits * 0 - 1
        return logits


# ---------------------------------------------------------------------------
# Pinned self-description used by the verification scripts.
# ---------------------------------------------------------------------------
def descriptor_layout() -> Dict[str, object]:
    """Machine-readable v9 spec used by guards, tests and the decision record."""
    return {
        "context_embed_channels": CONTEXT_EMBED_CHANNELS,
        "support_scales": list(SUPPORT_SCALES),
        "support_masks": list(SUPPORT_MASKS),
        "support_features": list(SUPPORT_FEATURES),
        "support_group_channels": SUPPORT_GROUP_CHANNELS,
        "support_channels": SUPPORT_CHANNELS,
        "center_features": list(CENTER_FEATURES),
        "center_channels": CENTER_CHANNELS,
        "context_descriptor_channels": CONTEXT_DESCRIPTOR_CHANNELS,
        "context_hidden_channels": CONTEXT_HIDDEN_CHANNELS,
        "aggregation_modes": list(AGGREGATION_MODES),
        "aggregation_sentinel": "persistent buffer aggregation_code; "
                                "_load_from_state_dict raises on variant mismatch",
        "fill_descriptor_channels": FILL_DESCRIPTOR_CHANNELS,
        "cut_descriptor_channels": CUT_DESCRIPTOR_CHANNELS,
        "hidden_channels": HIDDEN_CHANNELS,
        "gate": "candidate_d AND z_local_d + r_d > 0",
        "final": "raw + keep * delta",
        "verifier_loss": "ordinary_bce_present_direction_average",
    }
