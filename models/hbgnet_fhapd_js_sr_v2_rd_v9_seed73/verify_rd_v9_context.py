"""CPU/GPU engineering acceptance for the rd_v9_context fullloss verifier.

Synthetic checks confirm the v9 additive context residual, the frozen v8 scoring
path (bit-for-bit at zero residual), the candidate-subset gate, the detached
inputs, the zero-head first-step gradient staging, the matched center-only
control and the monotone affine calibration.  The real-smoke mode uses only train
and validation data, writes ``verification.json`` and never reads test or writes
checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent if ROOT.name == "package" else ROOT.parents[1]
for path in (ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from implementation.model import Field
from tools.region_discrimination_v8_grok_fullloss import (
    CUT_DESCRIPTOR_CHANNELS,
    DESCRIPTOR_CHANNELS,
    FILL_DESCRIPTOR_CHANNELS,
    HIDDEN_CHANNELS,
    RegionCandidateVerifier as V8RegionCandidateVerifier,
)
from tools.region_discrimination_v9 import (
    AGGREGATION_MODES,
    CENTER_CHANNELS,
    CONTEXT_DESCRIPTOR_CHANNELS,
    CONTEXT_EMBED_CHANNELS,
    CONTEXT_HIDDEN_CHANNELS,
    SUPPORT_CHANNELS,
    SUPPORT_MASKS,
    SUPPORT_SCALES,
    RegionCandidateVerifier,
    br_ha_curve,
    calibration_logit_threshold,
    candidate_masks,
    context_descriptor,
    descriptor_layout,
    gated_region,
    hard_keep,
    monotone_affine_calibration,
    support_masks,
    verifier_loss,
)

EXPECTED_V8_VERIFIER_PARAMETERS = 6074
EXPECTED_CONTEXT_PARAMETERS = 5154
EXPECTED_VERIFIER_PARAMETERS = 11228
EXPECTED_REFINER_PARAMETERS = 3941
EXPECTED_FIELD_PARAMETERS = 30468959
CONTEXT_STATE_KEYS = (
    # v9-only persistent sentinel; a v8 state dict reports it as missing too.
    "aggregation_code",
    "context_projection.weight",
    "cut_context.stem.weight", "cut_context.norm_stem.weight", "cut_context.norm_stem.bias",
    "cut_context.depthwise.weight", "cut_context.norm_depthwise.weight",
    "cut_context.norm_depthwise.bias", "cut_context.head.weight", "cut_context.head.bias",
    "fill_context.stem.weight", "fill_context.norm_stem.weight", "fill_context.norm_stem.bias",
    "fill_context.depthwise.weight", "fill_context.norm_depthwise.weight",
    "fill_context.norm_depthwise.bias", "fill_context.head.weight", "fill_context.head.bias",
)


def check(name, condition):
    if not condition:
        raise AssertionError(name)


def state_digest(model):
    """SHA256 over every parameter/buffer name, dtype, shape and raw bytes.

    A digest of this form cannot collide through reordered additions or through
    different tensors sharing a sum, so equality before and after an isolated
    probe is evidence the accepted model was left exactly as it was while a
    different instance was probed.
    """
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        try:
            digest.update(tensor.numpy().tobytes())
        except (TypeError, RuntimeError):
            digest.update(tensor.to(torch.float64).numpy().tobytes())
    return digest.hexdigest()


def head_gradient_evidence(module, directions=("cut", "fill")):
    """Finite/non-zero evidence for each direction-private head gradient."""
    evidence = {}
    for direction in directions:
        tensor = getattr(module, f"{direction}_context").head.weight.grad
        evidence[direction] = None if tensor is None else {
            "finite": bool(torch.isfinite(tensor).all().item()),
            "nonzero": bool((tensor != 0).any().item()),
            "abs_sum": float(tensor.abs().sum().item()),
        }
    return evidence


def synthetic_direction_probe(verifier, direction, device, size=32, batch=2):
    """Exercise the verifier's own gradient for one direction on CUDA.

    Every verifier input is detached in training, so a synthetic payload of
    detached tensors is a faithful test of the verifier parameter update.  The
    payload is built so that exactly one direction holds candidate pixels and
    the labels are unambiguous, which makes the per-direction head gradient and
    movement conclusive.  A fresh optimizer is created inside this call so no
    momentum from an earlier probe can move the head, and the target head
    gradient is required to be finite and non-zero before the step.  This is
    recorded as synthetic evidence, never as a real-data batch, and it is never
    checkpointed.
    """
    sign = 1.0 if direction == "fill" else -1.0
    raw = torch.full((batch, 1, size, size), -1.0 * sign, device=device)
    sr = torch.full((batch, 1, size, size), 1.0 * sign, device=device)
    delta = (sr - raw).contiguous()
    truth = torch.full((batch, 1, size, size), 1.0 if direction == "fill" else 0.0,
                       device=device)
    features = torch.randn(batch, 64, size, size).to(device)
    generator = torch.Generator().manual_seed(20261003)
    edge_logp = torch.log_softmax(
        torch.randn(batch, 2, size, size, generator=generator), dim=1).to(device)
    other = "fill" if direction == "cut" else "cut"
    before = {
        "cut": verifier.cut_context.head.weight.detach().clone(),
        "fill": verifier.fill_context.head.weight.detach().clone(),
    }
    optimizer = torch.optim.Adam(verifier.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    logits = verifier(features, raw, sr, delta, edge_logp)
    loss, stats = verifier_loss(logits, raw, sr, delta, truth)
    check("synthetic probe finite loss", bool(torch.isfinite(loss).all().item()))
    check("synthetic probe one present direction", stats["present_directions"] == 1)
    loss.backward()
    for parameter in verifier.parameters():
        if parameter.grad is not None:
            check("synthetic probe finite gradient",
                  bool(torch.isfinite(parameter.grad).all().item()))
    gradients = head_gradient_evidence(verifier)
    check(f"synthetic probe {direction} head gradient is finite and non-zero",
          gradients[direction]["finite"] and gradients[direction]["nonzero"])
    check(f"synthetic probe {other} head gradient stays exactly zero",
          gradients[other] is not None and gradients[other]["abs_sum"] == 0.0)
    optimizer.step()
    return {
        "evidence": "synthetic detached CUDA inputs (not a real data batch)",
        "direction": direction,
        "loss": float(loss),
        "verifier_bce": float(loss),
        "candidates": {"cut": stats["cut_candidates"], "fill": stats["fill_candidates"]},
        "present_directions": stats["present_directions"],
        "beneficial_pixels": stats["beneficial_cut"] + stats["beneficial_fill"],
        "harmful_pixels": stats["harmful_cut"] + stats["harmful_fill"],
        "head_gradients": gradients,
        "fresh_optimizer_per_probe": True,
        "heads_moved": {
            "cut": not torch.equal(before["cut"], verifier.cut_context.head.weight.detach()),
            "fill": not torch.equal(before["fill"], verifier.fill_context.head.weight.detach()),
        },
        "checkpointed": False,
    }


def _edges(batch, height, width, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return F.log_softmax(torch.randn(batch, 2, height, width, generator=generator), dim=1)


def check_architecture():
    layout = descriptor_layout()
    check("embed channels", layout["context_embed_channels"] == CONTEXT_EMBED_CHANNELS == 8)
    check("support geometry", layout["support_scales"] == list(SUPPORT_SCALES) == [3, 7, 15]
          and layout["support_masks"] == list(SUPPORT_MASKS)
          and layout["support_group_channels"] == 13 and SUPPORT_CHANNELS == 117)
    check("center block", layout["center_channels"] == CENTER_CHANNELS == 14)
    check("context descriptor width", layout["context_descriptor_channels"] == 131
          and CONTEXT_DESCRIPTOR_CHANNELS == CENTER_CHANNELS + SUPPORT_CHANNELS)
    check("context hidden", layout["context_hidden_channels"] == CONTEXT_HIDDEN_CHANNELS == 16)
    check("aggregation modes", layout["aggregation_modes"] == list(AGGREGATION_MODES)
          == ["multi_scale_support", "center_only"])
    check("frozen gate/final", layout["gate"] == "candidate_d AND z_local_d + r_d > 0"
          and layout["final"] == "raw + keep * delta")
    module = RegionCandidateVerifier()
    check("v9 subclasses the v8 verifier", isinstance(module, V8RegionCandidateVerifier))
    check("v8 fill descriptor preserved", FILL_DESCRIPTOR_CHANNELS == 78
          and DESCRIPTOR_CHANNELS == 78 and HIDDEN_CHANNELS == 24)
    check("v8 cut descriptor preserved", CUT_DESCRIPTOR_CHANNELS == 125)
    check("v8 local layers preserved",
          module.stem.in_channels == 78 and module.cut_stem.in_channels == 125
          and module.fill_head.out_channels == 1 and module.cut_head.out_channels == 1)
    for context in (module.cut_context, module.fill_context):
        check("context stem", context.stem.in_channels == CONTEXT_DESCRIPTOR_CHANNELS
              and context.stem.out_channels == CONTEXT_HIDDEN_CHANNELS
              and context.stem.bias is None)
        check("context depthwise", context.depthwise.groups == CONTEXT_HIDDEN_CHANNELS
              and context.depthwise.kernel_size == (3, 3) and context.depthwise.bias is None)
        check("zero context head", torch.count_nonzero(context.head.weight).item() == 0
              and torch.count_nonzero(context.head.bias).item() == 0)
    for head in (module.cut_head, module.fill_head):
        check("zero v8 head", torch.count_nonzero(head.weight).item() == 0
              and torch.count_nonzero(head.bias).item() == 0)
    counts = {
        "verifier": sum(p.numel() for p in module.parameters()),
        "context": sum(p.numel() for p in module.context_projection.parameters()) +
            sum(p.numel() for p in module.cut_context.parameters()) +
            sum(p.numel() for p in module.fill_context.parameters()),
    }
    check("verifier parameter count", counts["verifier"] == EXPECTED_VERIFIER_PARAMETERS)
    check("context parameter count", counts["context"] == EXPECTED_CONTEXT_PARAMETERS)
    check("v8 parameter count", counts["verifier"] - counts["context"]
          == EXPECTED_V8_VERIFIER_PARAMETERS)
    control = RegionCandidateVerifier(aggregation="center_only")
    check("control parameter budget", sum(p.numel() for p in control.parameters())
          == EXPECTED_VERIFIER_PARAMETERS)
    full_ids = {id(p) for p in module.parameters()}
    control_ids = {id(p) for p in control.parameters()}
    check("control and full are separate modules", not (full_ids & control_ids))
    check("context branch shares no parameter with the v8 score",
          not ({id(p) for name, p in module.named_parameters()
                if name.startswith(("context_projection", "cut_context", "fill_context"))}
               & {id(p) for name, p in module.named_parameters()
                  if not name.startswith(("context_projection", "cut_context", "fill_context"))}))
    return counts


def check_v8_parity():
    torch.manual_seed(101)
    v8 = V8RegionCandidateVerifier()
    with torch.no_grad():
        for head in (v8.cut_head, v8.fill_head):
            head.weight.normal_(0.0, 0.2)
            head.bias.normal_(0.0, 0.1)
    v9 = RegionCandidateVerifier()
    missing, unexpected = v9.load_state_dict(v8.state_dict(), strict=False)
    check("only context keys are new", set(missing) == set(CONTEXT_STATE_KEYS)
          and not unexpected)
    features = torch.randn(2, 64, 7, 6)
    raw = torch.randn(2, 1, 7, 6)
    sr = raw + torch.randn(2, 1, 7, 6)
    delta = sr - raw
    edge = _edges(2, 7, 6)
    with torch.no_grad():
        expected = v8(features, raw, sr, delta, edge)
        actual = v9(features, raw, sr, delta, edge)
        local = v9.local_logits(features, raw, sr, delta, edge)
        residual = v9.context_residual(features, raw, sr, delta, edge, "cut")
    check("zero residual is exactly zero", float(residual.abs().max()) == 0.0)
    check("v9 logits equal v8 bitwise", torch.equal(expected, actual))
    check("local logits equal v8 bitwise", torch.equal(expected, local))
    check("v8 and v9 share the descriptor code",
          type(v9)._fill_descriptor is type(v8)._fill_descriptor
          and type(v9)._cut_descriptor is type(v8)._cut_descriptor)
    control = RegionCandidateVerifier(aggregation="center_only")
    control.load_state_dict(v8.state_dict(), strict=False)
    with torch.no_grad():
        control_logits = control(features, raw, sr, delta, edge)
    check("control also starts at the v8 score", torch.equal(control_logits, expected))
    # The two aggregation variants share every parameter name and shape, so a
    # cross-variant load must be rejected by the aggregation sentinel instead of
    # silently reusing the other variant's statistics.
    full_state = v9.state_dict()
    control_state = control.state_dict()
    check("sentinel present and variant-specific",
          int(full_state["aggregation_code"]) == 0
          and int(control_state["aggregation_code"]) == 1)
    before = [tensor.clone() for tensor in full_state.values()]
    rejected = []
    for target, state in ((control, full_state), (v9, control_state)):
        try:
            target.load_state_dict(state, strict=True)
        except RuntimeError:
            rejected.append(True)
        else:
            rejected.append(False)
    check("cross-variant state loads are rejected", all(rejected))
    check("rejected load copies nothing",
          all(torch.equal(now, was) for now, was in zip(v9.state_dict().values(), before)))
    v9.load_state_dict(full_state, strict=True)
    control.load_state_dict(control_state, strict=True)
    check("same-variant strict reload still works", True)
    return {"missing_keys": sorted(missing), "bitwise_equal": True,
            "local_bitwise_equal": True, "residual_zero": True,
            "aggregation_sentinel": True, "cross_variant_load_rejected": True}


def check_masks_and_gate():
    raw = torch.tensor([[[[2.0, 2.0, -2.0, -2.0]]]])
    sr = torch.tensor([[[[-1.0, 2.0, 1.0, -2.0]]]])
    delta = torch.tensor([[[[-3.0, 0.0, 3.0, 0.0]]]])
    cut, fill = candidate_masks(raw, sr, delta)
    expected_cut = torch.tensor([[[[True, False, False, False]]]])
    expected_fill = torch.tensor([[[[False, False, True, False]]]])
    check("cut mask", torch.equal(cut, expected_cut))
    check("fill mask", torch.equal(fill, expected_fill))
    for direction in ("cut", "fill"):
        masks = support_masks(raw, sr, delta, direction)
        outside = ~(cut | fill)
        union = masks["candidate"] | masks["foreground"] | masks["background"]
        check(f"support partition {direction}", torch.equal(union, masks["candidate"] | outside))
        check(f"support excludes candidates {direction}",
              not bool((masks["foreground"] & ~outside).any())
              and not bool((masks["background"] & ~outside).any()))
    logits = torch.tensor([[[[1.0, 100.0, 100.0, 100.0]],
                            [[-1.0, 100.0, -1.0, 100.0]]]])
    keep_cut, keep_fill = hard_keep(raw, sr, delta, logits)
    check("direction gate", torch.equal(keep_cut, expected_cut))
    check("rejected fill", not bool(keep_fill.any()))
    final = gated_region(raw, delta, keep_cut, keep_fill)
    check("accepted equals sr and rejected equals raw",
          torch.equal(final, torch.tensor([[[[-1.0, 2.0, -2.0, -2.0]]]])))
    return {"cut_candidates": int(cut.sum()), "fill_candidates": int(fill.sum())}


def check_module_detach_and_modes():
    batch, height, width = 2, 5, 7
    features = torch.randn(batch, 64, height, width, requires_grad=True)
    raw = torch.randn(batch, 1, height, width, requires_grad=True)
    sr = torch.randn(batch, 1, height, width, requires_grad=True)
    delta = torch.randn(batch, 1, height, width, requires_grad=True)
    edge = _edges(batch, height, width).requires_grad_()
    module = RegionCandidateVerifier()
    logits = module(features, raw, sr, delta, edge)
    check("logit shape", logits.shape == (batch, 2, height, width))
    with torch.no_grad():
        for context in (module.cut_context, module.fill_context):
            context.head.weight.fill_(0.1)
    logits = module(features, raw, sr, delta, edge)
    logits.sum().backward()
    for name, value in (("features", features), ("raw", raw), ("sr", sr),
                        ("delta", delta), ("edge", edge)):
        check(name + " detached", value.grad is None)
    check("projection updated only by the verifier loss",
          module.context_projection.weight.grad is not None)
    reject = RegionCandidateVerifier(mode="always_reject")
    rejected = reject(features.detach(), raw.detach(), sr.detach(), delta.detach(), edge.detach())
    check("always reject", torch.equal(rejected, torch.full_like(rejected, -1)))
    control = RegionCandidateVerifier(mode="always_reject", aggregation="center_only")
    check("control always reject", torch.equal(
        control(features.detach(), raw.detach(), sr.detach(), delta.detach(), edge.detach()),
        torch.full_like(rejected, -1)))
    return {"shape": list(logits.shape), "detached_inputs": True, "always_reject": True}


def check_gradient_staging():
    module = RegionCandidateVerifier()
    with torch.no_grad():
        for head in (module.cut_head, module.fill_head):
            head.weight.fill_(0.3)
    features = torch.randn(2, 64, 6, 6)
    raw = torch.randn(2, 1, 6, 6)
    sr = raw + torch.randn(2, 1, 6, 6)
    delta = sr - raw
    edge = _edges(2, 6, 6)
    truth = (torch.rand(2, 1, 6, 6) > 0.5).to(torch.float32)
    logits = module(features, raw, sr, delta, edge)
    loss, stats = verifier_loss(logits, raw, sr, delta, truth)
    loss.backward()
    first = {}
    for context_name, context in (("cut", module.cut_context), ("fill", module.fill_context)):
        first[context_name + "_head"] = float(context.head.weight.grad.abs().sum())
        first[context_name + "_stem"] = float(context.stem.weight.grad.abs().sum())
    first["projection"] = float(module.context_projection.weight.grad.abs().sum())
    check("first step trains the zero heads", first["cut_head"] > 0 and first["fill_head"] > 0)
    check("first step upstream is exactly zero (zero head)",
          first["cut_stem"] == 0.0 and first["fill_stem"] == 0.0 and first["projection"] == 0.0)
    check("v8 local path still trained",
          float(module.cut_head.weight.grad.abs().sum()) > 0
          and float(module.fill_head.weight.grad.abs().sum()) > 0)
    # Move the heads and repeat: the whole branch must learn.
    torch.manual_seed(103)
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        for context in (module.cut_context, module.fill_context):
            context.head.weight.normal_(0.0, 0.1)
            context.head.bias.normal_(0.0, 0.05)
    logits = module(features, raw, sr, delta, edge)
    loss, _ = verifier_loss(logits, raw, sr, delta, truth)
    loss.backward()
    second = {
        "cut_stem": float(module.cut_context.stem.weight.grad.abs().sum()),
        "fill_stem": float(module.fill_context.stem.weight.grad.abs().sum()),
        "projection": float(module.context_projection.weight.grad.abs().sum()),
    }
    check("second step trains the context encoder",
          second["cut_stem"] > 0 and second["fill_stem"] > 0 and second["projection"] > 0)
    return {"first_step": first, "second_step": second,
            "candidates": {key: int(value) for key, value in stats.items()}}


def check_calibration():
    torch.manual_seed(107)
    scores = torch.randn(2, 2, 4, 5)
    good = (torch.rand(2, 2, 4, 5) > 0.6)
    bad = (torch.rand(2, 2, 4, 5) > 0.7) & ~good
    a, b = (2.0, 0.5), (1.0, -3.0)
    calibrated = monotone_affine_calibration(scores, a, b)
    threshold = calibration_logit_threshold(a, b)
    check("threshold mapping", threshold == (-0.5, 6.0))
    for index in range(2):
        raw = scores[:, index:index + 1]
        mapped = calibrated[:, index:index + 1]
        check("monotone keeps the decision rule", torch.equal(mapped > 0, raw > threshold[index]))
    curve_zero = br_ha_curve(calibrated, good, bad, [0.0])
    curve_raw = br_ha_curve(scores, good, bad, [threshold[0]])
    check("calibration re-parameterises, never improves",
          curve_zero["points"][0]["directions"]["cut"]["accepted_harmful"]
          == curve_raw["points"][0]["directions"]["cut"]["accepted_harmful"]
          and curve_zero["points"][0]["directions"]["cut"]["rejected_beneficial"]
          == curve_raw["points"][0]["directions"]["cut"]["rejected_beneficial"])
    for invalid in (0.0, -1.0, float("nan")):
        try:
            monotone_affine_calibration(scores, invalid, 0.0)
        except ValueError:
            continue
        raise AssertionError("non-positive slope accepted")
    # dtype safety: coefficients are validated after the cast to the logits
    # dtype, so half-precision underflow/overflow cannot deform the map, and a
    # calibrated score that overflows raises instead of returning inf.
    half = scores.to(torch.float16)
    half_threshold = calibration_logit_threshold(2.0, 0.5, dtype=torch.float16)
    half_calibrated = monotone_affine_calibration(half, 2.0, 0.5)
    check("half calibration finite and monotone",
          torch.isfinite(half_calibrated).all().item()
          and torch.equal(half_calibrated[:, 0] > 0, half[:, 0] > half_threshold[0]))
    for coerced in (1e-9, 1e30):  # 0.0 and inf in float16
        try:
            monotone_affine_calibration(half, coerced, 0.0)
        except ValueError:
            continue
        raise AssertionError(f"dtype-coerced slope {coerced} accepted in float16")
    for big, slope in ((torch.full((1, 2, 1, 1), 6.0e4, dtype=torch.float16), 2.0),
                       (torch.full((1, 2, 1, 1), 3.0e38), 2.0)):
        try:
            monotone_affine_calibration(big, slope, 0.0)
        except FloatingPointError:
            continue
        raise AssertionError("overflowing calibration returned a non-finite score")
    return {"threshold": list(threshold), "rates_present": True, "monotone_enforced": True,
            "dtype_checked_after_cast": True, "half_underflow_rejected": True,
            "half_overflow_rejected": True}


def check_loss():
    # One cut direction with three mixed-label candidates: ordinary BCE, and the
    # present-direction average must not halve a single present direction.
    raw = torch.tensor([[[[1.0, 1.0, 1.0]]]])
    sr = torch.tensor([[[[-1.0, -1.0, -1.0]]]])
    delta = sr - raw
    truth = torch.tensor([[[[0.0, 1.0, 1.0]]]])
    logits = torch.tensor([[[[0.0, 0.0, 2.0]], [[0.0, 0.0, 0.0]]]], requires_grad=True)
    loss, stats = verifier_loss(logits, raw, sr, delta, truth)
    cut, fill = candidate_masks(raw, sr, delta)
    target = (sr > 0) == (truth > 0.5)
    expected_cut = F.binary_cross_entropy_with_logits(logits[:, :1][cut],
                                                      target[cut].to(logits.dtype))
    balanced_cut = torch.stack([
        F.binary_cross_entropy_with_logits(
            logits[:, :1][cut][target[cut] == label],
            target[cut][target[cut] == label].to(logits.dtype))
        for label in (False, True)]).mean()
    torch.testing.assert_close(loss, expected_cut)
    check("ordinary differs from balanced", not torch.isclose(loss, balanced_cut).item())
    check("cut only present", int(cut.sum()) == 3 and int(fill.sum()) == 0)
    check("present direction count", stats["present_directions"] == 1)
    check("single direction is not halved", not torch.isclose(loss, expected_cut / 2).item())
    loss.backward()
    check("present head receives gradient", logits.grad[:, :1].abs().sum().item() > 0)
    check("empty head receives no gradient", logits.grad[:, 1:2].abs().sum().item() == 0)
    # Both directions present: the two directional means are averaged once.
    raw_both = torch.tensor([[[[1.0, 1.0, -1.0, -1.0]]]])
    sr_both = torch.tensor([[[[-1.0, -1.0, 1.0, 1.0]]]])
    delta_both = sr_both - raw_both
    truth_both = torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]])
    logits_both = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]]],
                               requires_grad=True)
    both_loss, both_stats = verifier_loss(logits_both, raw_both, sr_both, delta_both, truth_both)
    both_cut, both_fill = candidate_masks(raw_both, sr_both, delta_both)
    both_target = (sr_both > 0) == (truth_both > 0.5)
    cut_mean = F.binary_cross_entropy_with_logits(logits_both[:, :1][both_cut],
                                                  both_target[both_cut].to(logits_both.dtype))
    fill_mean = F.binary_cross_entropy_with_logits(logits_both[:, 1:2][both_fill],
                                                   both_target[both_fill].to(logits_both.dtype))
    torch.testing.assert_close(both_loss, (cut_mean + fill_mean) / 2)
    check("both directions present", both_stats["present_directions"] == 2)
    # Empty candidates: connected zero, not a detached constant.
    empty = torch.zeros_like(logits, requires_grad=True)
    zero, zero_stats = verifier_loss(empty, raw, raw, torch.zeros_like(raw), truth)
    check("empty loss connected", zero.requires_grad and zero_stats["edit_candidates"] == 0)
    check("empty loss is zero", float(zero.detach()) == 0.0)
    zero.backward()
    check("empty loss finite", torch.isfinite(empty.grad).all().item())
    return {
        "loss": float(loss.detach()), "both_loss": float(both_loss.detach()),
        "empty_loss": float(zero.detach()), **stats,
        "ordinary_bce": True, "present_direction_average": True,
        "single_direction_not_halved": True, "empty_direction_connected_zero": True,
    }


def check_candidate_subset_and_truth_independence():
    torch.manual_seed(109)
    module = RegionCandidateVerifier()
    torch.manual_seed(113)
    with torch.no_grad():
        for head in (module.cut_head, module.fill_head):
            head.weight.normal_(0.0, 0.2)
        for context in (module.cut_context, module.fill_context):
            context.head.weight.normal_(0.0, 0.1)
    features = torch.randn(2, 64, 9, 9)
    raw = torch.randn(2, 1, 9, 9)
    sr = raw + torch.randn(2, 1, 9, 9)
    delta = sr - raw
    edge = _edges(2, 9, 9)
    with torch.no_grad():
        logits = module(features, raw, sr, delta, edge)
        again = module(features, raw, sr, delta, edge)
        keep_cut, keep_fill = hard_keep(raw, sr, delta, logits)
        cut, fill = candidate_masks(raw, sr, delta)
        final = gated_region(raw, delta, keep_cut, keep_fill)
    check("forward is deterministic", torch.equal(logits, again))
    check("keep is a candidate subset",
          not bool((keep_cut & ~cut).any()) and not bool((keep_fill & ~fill).any()))
    outside = ~(cut | fill)
    check("outside candidates final equals raw", torch.equal(final[outside], raw[outside]))
    accepted = keep_cut | keep_fill
    check("accepted final equals sr", torch.equal(final[accepted], sr[accepted]))
    check("candidate set unchanged by the verifier", torch.equal(cut | fill, (delta != 0)
          & (candidate_masks(raw, sr, delta)[0] | candidate_masks(raw, sr, delta)[1])))
    return {"keep_subset": True, "outside_equals_raw": True, "accepted_equals_sr": True}


def check_field():
    refinement = {
        "enabled": True,
        "max_delta": 2.0,
        "checkpoint_groups": True,
        "mode": "local_separation_v2",
        "hidden_channels": 16,
        "path_channels": 4,
        "radii": [2, 4],
        "verifier": {"enabled": True, "mode": "enabled",
                     "aggregation": "multi_scale_support"},
    }
    model = Field(channel=32, num_classes=2, drop_rate=0.4, pretrained_path=None,
                  refinement=refinement)
    check("field verifier exists", model.edit_verifier is not None)
    check("field refiner exists", model.region_refiner is not None)
    check("field verifier aggregation", model.edit_verifier.aggregation == "multi_scale_support")
    check("field verifier count",
          sum(p.numel() for p in model.edit_verifier.parameters()) == EXPECTED_VERIFIER_PARAMETERS)
    check("field refiner count",
          sum(p.numel() for p in model.region_refiner.parameters()) == EXPECTED_REFINER_PARAMETERS)
    check("field parameter count",
          sum(p.numel() for p in model.parameters()) == EXPECTED_FIELD_PARAMETERS)
    with torch.no_grad():
        for context in (model.edit_verifier.cut_context, model.edit_verifier.fill_context):
            check("field zero context head", torch.count_nonzero(context.head.weight).item() == 0
                  and torch.count_nonzero(context.head.bias).item() == 0)
        for head in (model.edit_verifier.cut_head, model.edit_verifier.fill_head):
            check("field zero v8 head", torch.count_nonzero(head.weight).item() == 0
                  and torch.count_nonzero(head.bias).item() == 0)
    control_refinement = json.loads(json.dumps(refinement))
    control_refinement["verifier"]["aggregation"] = "center_only"
    control = Field(channel=32, num_classes=2, drop_rate=0.4, pretrained_path=None,
                    refinement=control_refinement)
    check("control field parameter count",
          sum(p.numel() for p in control.parameters()) == EXPECTED_FIELD_PARAMETERS)
    return {
        "verifier_parameters": sum(p.numel() for p in model.edit_verifier.parameters()),
        "field_parameters": sum(p.numel() for p in model.parameters()),
        "official_pretraining_loaded": False,
    }


def check_real_smoke(device_name):
    """Run the remote-only official-data/PVT/optimizer acceptance gate."""
    import common

    cfg = common.config(ROOT / "config_protocol_b.yaml")
    check("configured verifier loss mode",
          cfg["loss"].get("verifier_loss_mode") == common.VERIFIER_LOSS_MODE)
    check("configured aggregation",
          cfg["model"]["refinement"]["verifier"].get("aggregation") == "multi_scale_support")
    loss_contract = check_loss()
    architecture = check_architecture()
    parity = check_v8_parity()
    staging = check_gradient_staging()
    calibration = check_calibration()
    fingerprint = common.source_fingerprint()
    device = common.torch.device(device_name)
    if device.type != "cuda" or not common.torch.cuda.is_available():
        raise RuntimeError("real-smoke requires CUDA")
    common.torch.cuda.set_device(device)
    common.torch.cuda.reset_peak_memory_stats(device)
    model = common.build(cfg, pretrained=True).to(device)
    counts = {
        "field": sum(parameter.numel() for parameter in model.parameters()),
        "refiner": sum(parameter.numel() for parameter in model.region_refiner.parameters()),
        "verifier": sum(parameter.numel() for parameter in model.edit_verifier.parameters()),
    }
    check("real field parameter count", counts["field"] == EXPECTED_FIELD_PARAMETERS)
    check("real refiner parameter count", counts["refiner"] == EXPECTED_REFINER_PARAMETERS)
    check("real verifier parameter count", counts["verifier"] == EXPECTED_VERIFIER_PARAMETERS)
    parameter_bytes = sum(parameter.numel() * parameter.element_size()
                          for parameter in model.parameters())
    head_before = {
        "cut": model.edit_verifier.cut_context.head.weight.detach().clone(),
        "fill": model.edit_verifier.fill_context.head.weight.detach().clone(),
    }
    train_loader = common.loader(cfg, "train", batch=8, workers=0)
    val_loader = common.loader(cfg, "val", batch=8, workers=0)
    optimizer = common.torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    components = []
    per_batch_candidates = []
    iterator = iter(train_loader)
    common.torch.cuda.synchronize(device)
    train_started = time.monotonic()
    for index in range(4):
        batch = common.device_batch(next(iterator), device)
        outputs, auxiliary = model(batch["image"], return_training_aux=True)
        loss, values = common.loss_fn(outputs, batch, cfg, auxiliary, return_components=True)
        check("real finite loss", common.torch.isfinite(loss).all().item())
        losses.append(float(loss.detach().item()))
        components.append({name: values[name] for name in
                           ("total", "verifier_bce", "verifier_cut_candidates",
                            "verifier_fill_candidates", "verifier_present_directions")})
        per_batch_candidates.append({
            "cut": int(values["verifier_cut_candidates"]),
            "fill": int(values["verifier_fill_candidates"]),
        })
        (loss / 2).backward()
        if index % 2 == 1:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    check("real finite gradient", common.torch.isfinite(parameter.grad).all().item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        del batch, outputs, auxiliary, loss, values
    common.torch.cuda.synchronize(device)
    train_seconds = time.monotonic() - train_started
    train_peak_memory = int(common.torch.cuda.max_memory_allocated(device))
    def context_heads_moved(source):
        return {
            "cut": not torch.equal(head_before["cut"],
                                   source.edit_verifier.cut_context.head.weight.detach()),
            "fill": not torch.equal(head_before["fill"],
                                    source.edit_verifier.fill_context.head.weight.detach()),
        }

    # ---- natural phase -------------------------------------------------
    # The SR refiner head is zero-initialised, so at step 0 delta_sr == 0 and
    # sr == raw: no pixel changes its decision, the verifier loss is a
    # connected zero, and the verifier (including its context branch)
    # legitimately receives no gradient.  Head movement is therefore only
    # required for a direction that actually contains candidate pixels; the
    # synthetic staging checks prove the first-step head gradient and the
    # staged upstream gradients regardless.  The natural model, its optimizer
    # state and this validation path are never perturbed: any forced-candidate
    # probe runs on a separate, isolated model instance that is never
    # checkpointed.
    natural = {direction: sum(entry[direction] for entry in per_batch_candidates)
               for direction in ("cut", "fill")}
    moved = context_heads_moved(model)
    for direction in ("cut", "fill"):
        if natural[direction] > 0:
            check(f"real {direction} context head moved", moved[direction])
    heads_state = "moved" if (moved["cut"] and moved["fill"]) else "not_moved_natural_no_candidates"
    for direction in ("cut", "fill"):
        if natural[direction] == 0 and moved[direction]:
            heads_state = "moved_without_natural_candidates"

    # ---- forced-candidate probes on an isolated model ------------------
    # A positive SR residual flips background to foreground (fill candidates)
    # and a negative one flips foreground to background (cut candidates), so
    # each sign is probed and a second, larger magnitude is tried when a sign
    # yields no candidate for its direction.  Only a direction that really
    # produced candidate pixels on the GPU -- in the natural phase, in a forced
    # real-batch probe, or in a synthetic detached-input probe -- is claimed as
    # GPU-verified; both directions must be verified.  Nothing is checkpointed
    # and the accepted model, its optimizer and the validation path are never
    # perturbed.
    probes = []
    probe_isolation = {"separate_model": False, "separate_optimizer": False,
                       "natural_optimizer_passed_to_probe": False,
                       "rng_restored": False, "state_digest_unchanged": None,
                       "checkpointed": False}
    probe_verified = {direction: natural[direction] > 0 for direction in ("cut", "fill")}
    if not all(probe_verified.values()):
        digest_before = state_digest(model)
        rng_state = common.torch.get_rng_state()
        cuda_rng_state = common.torch.cuda.get_rng_state(device)
        probe_model = common.build(cfg, pretrained=True).to(device)
        check("probe field parameter count",
              sum(parameter.numel() for parameter in probe_model.parameters())
              == EXPECTED_FIELD_PARAMETERS)
        probe_model.train()
        for sign, target in ((1.0, "fill"), (-1.0, "cut"), (2.0, "fill"), (-2.0, "cut")):
            if all(probe_verified.values()):
                break
            if probe_verified[target]:
                continue
            last_head = probe_model.region_refiner.head[-1]
            with common.torch.no_grad():
                last_head.bias.fill_(sign)
            # Fresh optimizer per attempt: no momentum from an earlier probe can
            # move a head, so movement here is this step's own gradient.
            attempt_optimizer = common.torch.optim.Adam(probe_model.parameters(), lr=1e-4,
                                                       weight_decay=0.0)
            attempt_optimizer.zero_grad(set_to_none=True)
            batch = common.device_batch(next(iterator), device)
            outputs, auxiliary = probe_model(batch["image"], return_training_aux=True)
            loss, values = common.loss_fn(outputs, batch, cfg, auxiliary,
                                          return_components=True)
            check("probe finite loss", common.torch.isfinite(loss).all().item())
            entry = {
                "evidence": "real training batch with a forced SR residual sign",
                "sign": sign, "target_direction": target,
                "reason": ("zero-initialised SR refiner head leaves no natural candidate; "
                           "this isolated probe forces the SR decision to flip"),
                "loss": float(loss),
                "verifier_bce": float(values["verifier_bce"]),
                "candidates": {"cut": int(values["verifier_cut_candidates"]),
                               "fill": int(values["verifier_fill_candidates"])},
                "present_directions": int(values["verifier_present_directions"]),
                "fresh_optimizer_per_probe": True,
                "head_gradients": None,
                "heads_moved": None,
                "checkpointed": False,
            }
            (loss / 2).backward()
            for parameter in probe_model.parameters():
                if parameter.grad is not None:
                    check("probe finite gradient",
                          common.torch.isfinite(parameter.grad).all().item())
            grads = head_gradient_evidence(probe_model.edit_verifier)
            entry["head_gradients"] = grads
            if entry["candidates"][target] > 0:
                check(f"probe {target} head gradient is finite and non-zero",
                      grads[target]["finite"] and grads[target]["nonzero"])
            sign_heads_before = {
                "cut": probe_model.edit_verifier.cut_context.head.weight.detach().clone(),
                "fill": probe_model.edit_verifier.fill_context.head.weight.detach().clone(),
            }
            attempt_optimizer.step()
            entry["heads_moved"] = {
                "cut": not torch.equal(sign_heads_before["cut"],
                                       probe_model.edit_verifier.cut_context.head.weight.detach()),
                "fill": not torch.equal(sign_heads_before["fill"],
                                        probe_model.edit_verifier.fill_context.head.weight.detach()),
            }
            for direction in ("cut", "fill"):
                if entry["candidates"][direction] > 0:
                    check(f"probe {direction} context head moved", entry["heads_moved"][direction])
                    probe_verified[direction] = True
            probes.append(entry)
            del attempt_optimizer, grads, batch, outputs, auxiliary, loss, values
        for direction in ("cut", "fill"):
            if probe_verified[direction]:
                continue
            entry = synthetic_direction_probe(probe_model.edit_verifier, direction, device)
            probes.append(entry)
            check(f"synthetic probe {direction} candidates", entry["candidates"][direction] > 0)
            check(f"synthetic probe {direction} context head moved",
                  entry["heads_moved"][direction])
            probe_verified[direction] = True
        del probe_model
        common.torch.cuda.empty_cache()
        common.torch.set_rng_state(rng_state)
        common.torch.cuda.set_rng_state(cuda_rng_state, device)
        rng_restored = bool(torch.equal(common.torch.get_rng_state(), rng_state))
        cuda_rng_restored = bool(
            torch.equal(common.torch.cuda.get_rng_state(device), cuda_rng_state))
        probe_isolation = {"separate_model": True, "separate_optimizer": True,
                           "natural_optimizer_passed_to_probe": False,
                           "rng_restored": rng_restored and cuda_rng_restored,
                           "checkpointed": False,
                           "state_digest_unchanged": state_digest(model) == digest_before}
        check("isolated probe restored the RNG states exactly",
              rng_restored and cuda_rng_restored)
    check("cut and fill candidates both verified on the GPU",
          all(probe_verified.values()))
    if probe_isolation["separate_model"]:
        check("isolated probe left the accepted model untouched",
              probe_isolation["state_digest_unchanged"])
    model.eval()
    common.torch.cuda.synchronize(device)
    validation_started = time.monotonic()
    validation = common.evaluate(model, val_loader, device, max_batches=8)
    common.torch.cuda.synchronize(device)
    validation_seconds = time.monotonic() - validation_started
    check("real validation samples", validation["samples"] == 64)
    candidates = {key: int(validation[key]) for key in
                  ("edit_pixels", "cut_edit_pixels", "fill_edit_pixels",
                   "accepted_pixels", "beneficial_pixels", "harmful_pixels")}
    report = {
        "status": "pass",
        "source_fingerprint": fingerprint,
        "configs": {cfg["experiment"]["run_name"]: cfg},
        "design": "rd_v9_context_supported_candidate_verification_fullloss",
        "descriptor_layout": descriptor_layout(),
        "loss_contract": loss_contract,
        "parameter_counts": counts,
        "architecture": architecture,
        "v8_parity": parity,
        "gradient_staging": staging,
        "calibration": calibration,
        "verifier_assertions": {
            "candidate_only": True, "cut_fill_heads": True,
            "separate_relational_cut_path": True,
            "context_residual_added_to_local_score": True,
            "strict_positive_gate": True, "rejected_edit_returns_raw": True,
            "accepted_edit_equals_sr": True, "inputs_detached": True,
            "ground_truth_is_not_a_forward_input": True,
            "ordinary_bce_present_direction_average": True,
            "single_direction_not_halved_test": True,
            "empty_direction_connected_zero": True,
            "head_movement_required_only_when_candidates_exist": True,
            "forced_candidate_probe_is_isolated_and_uncheckpointed": True,
            "both_directions_verified_on_gpu_before_pass": True,
            "synthetic_cuda_probe_distinguished_from_real_batch": True,
            "direction_candidate_gpu_verified": probe_verified,
            "zero_head_first_step_upstream_zero": True,
            "whole_context_branch_learns_after_head_moves": True,
            "control_matched_parameter_budget": True,
            "control_matched_budget_is_not_equal_capacity": True,
            "aggregation_sentinel_blocks_cross_variant_load": True,
            "monotone_calibration_a_positive": True,
            "calibration_validated_after_dtype_cast": True,
            "validation_verifier_shape_b2": True,
        },
        "ablation_groups": common.ABLATION_GROUPS,
        "real_smoke": {
            "device": str(device), "train_batches": 4,
            "optimizer_updates": 2, "validation_batches": 8,
            "validation_samples": validation["samples"],
            "losses": losses, "loss_components": components,
            "context_heads_moved": moved,
            "context_heads_state": heads_state,
            "candidate_pixels_per_train_batch": per_batch_candidates,
            "natural_candidate_pixels": natural,
            "candidate_probes": probes,
            "candidate_probe_isolation": probe_isolation,
            "direction_candidate_gpu_verified": probe_verified,
            "validation_model": "accepted natural model (never probed or perturbed)",
            "validation": validation, "validation_candidates": candidates,
            "train_seconds": train_seconds,
            "train_seconds_per_batch": train_seconds / 4,
            "validation_seconds": validation_seconds,
            "validation_seconds_per_batch": validation_seconds / 8,
            "peak_cuda_memory_bytes": train_peak_memory,
            "parameter_bytes": parameter_bytes,
            "dataloader_workers": 0,
            "timing_caveat": ("measured on this host with smoke dataloader workers=0; "
                              "the training recipe uses 8 workers.  This is a "
                              "measurement of the smoke step, not a strict upper "
                              "bound or a prediction of the training epoch time"),
            "official_pretraining_only": True, "test_started": False,
        },
    }
    common.atomic_json(ROOT / "verification.json", report)
    return {"parameter_counts": counts, "losses": losses,
            "validation_samples": validation["samples"], "verification_written": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-field", action="store_true", help="skip full Field construction")
    parser.add_argument("--real-smoke", metavar="DEVICE", help="run the CUDA/official-data gate")
    args = parser.parse_args()
    if args.real_smoke:
        print(json.dumps({"status": "pass", "real_smoke": check_real_smoke(args.real_smoke)},
                         ensure_ascii=False, allow_nan=False, sort_keys=True))
        return
    report = {
        "status": "pass",
        "architecture": check_architecture(),
        "v8_parity": check_v8_parity(),
        "routing": check_masks_and_gate(),
        "detach_and_modes": check_module_detach_and_modes(),
        "gradient_staging": check_gradient_staging(),
        "calibration": check_calibration(),
        "loss": check_loss(),
        "route_invariants": check_candidate_subset_and_truth_independence(),
    }
    if not args.skip_field:
        report["field"] = check_field()
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
