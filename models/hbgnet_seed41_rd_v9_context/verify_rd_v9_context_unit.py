"""Dependency-light CPU unittest suite for the rd_v9_context candidate verifier.

The suite imports only ``torch`` plus the two tools modules
(``tools/region_discrimination_v8_grok_fullloss.py`` and
``tools/region_discrimination_v9.py``); no rasterio, scipy, numpy, yaml, dataset
or model dependency is needed, so it runs on a plain CPU interpreter.  The v8
module is a hard dependency of v9 (v9 subclasses it), so its presence is required
rather than optional.

Run it either directly::

    PYTHONPATH=<staging> python <staging>/package/verify_rd_v9_context_unit.py

or through unittest discovery from the package directory::

    cd <staging>/package && python -m unittest verify_rd_v9_context_unit

Every reference value is recomputed by an independent construction: the v8 fill
and cut descriptors by explicit channel lists, the v9 masked-window statistics by
a per-pixel double loop, and the v9 support/center blocks by the same loop-based
recipe.  The suite never reads the dataset, CUDA, or the test split.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
# Staging layout: <root>/package/... with <root>/tools/...  Deployed layout:
# /root/wyk5/RGVNet/models/<package>/... with /root/wyk5/RGVNet/tools/...
if HERE.name == 'package':
    PROJECT_ROOT = HERE.parent
else:
    PROJECT_ROOT = HERE.parents[1]
for _path in (PROJECT_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tools.region_discrimination_v8_grok_fullloss import (  # noqa: E402
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
    RegionCandidateVerifier as V8RegionCandidateVerifier,
    candidate_masks as v8_candidate_masks,
    cut_channel_index,
    cut_pair_index,
    directional_labels as v8_directional_labels,
    gated_region,
    hard_keep,
    positive,
    shift_padded,
    side_similarity,
    verifier_loss as v8_verifier_loss,
)
from tools.region_discrimination_v9 import (  # noqa: E402
    AGGREGATION_MODES,
    CENTER_CHANNELS,
    CENTER_FEATURES,
    CONTEXT_DESCRIPTOR_CHANNELS,
    CONTEXT_DIRECTIONS,
    CONTEXT_EMBED_CHANNELS,
    CONTEXT_HIDDEN_CHANNELS,
    RegionCandidateVerifier,
    SUPPORT_CHANNELS,
    SUPPORT_FEATURES,
    SUPPORT_GROUP_CHANNELS,
    SUPPORT_MASKS,
    SUPPORT_SCALES,
    br_ha_curve,
    calibration_logit_threshold,
    candidate_masks,
    center_channel_index,
    context_center_block,
    context_descriptor,
    context_support_block,
    directional_labels,
    masked_window_statistics,
    monotone_affine_calibration,
    support_block_index,
    support_channel_index,
    support_masks,
    verifier_loss,
)

import tools.region_discrimination_v8_grok_fullloss as v8_tool  # noqa: E402
import tools.region_discrimination_v9 as v9_tool  # noqa: E402

VERIFIER_PARAMETERS = 11228
CONTEXT_PARAMETERS = 5154
V8_VERIFIER_PARAMETERS = 6074
CONTEXT_STATE_KEYS = (
    "aggregation_code",
    "context_projection.weight",
    "cut_context.stem.weight", "cut_context.norm_stem.weight", "cut_context.norm_stem.bias",
    "cut_context.depthwise.weight", "cut_context.norm_depthwise.weight",
    "cut_context.norm_depthwise.bias", "cut_context.head.weight", "cut_context.head.bias",
    "fill_context.stem.weight", "fill_context.norm_stem.weight", "fill_context.norm_stem.bias",
    "fill_context.depthwise.weight", "fill_context.norm_depthwise.weight",
    "fill_context.norm_depthwise.bias", "fill_context.head.weight", "fill_context.head.bias",
)
FILL_ATTRIBUTE_NAMES = (
    "stem.weight",
    "norm_stem.weight", "norm_stem.bias",
    "depthwise_near.weight",
    "depthwise_far.weight",
    "norm_near.weight", "norm_near.bias",
    "norm_far.weight", "norm_far.bias",
    "fill_head.weight", "fill_head.bias",
)
LN2 = 0.6931471805599453  # BCEWithLogits at logit 0.


def log_softmax_edges(batch: int, height: int, width: int, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.log_softmax(torch.randn(batch, 2, height, width, generator=generator), dim=1)


def channel_norm(norm, hidden):
    return norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def reference_shift(tensor, dy, dx):
    """Loop-based zero-padded shift, written independently of the module."""
    height, width = tensor.shape[-2], tensor.shape[-1]
    out = torch.zeros_like(tensor)
    for y in range(height):
        for x in range(width):
            source_y, source_x = y + dy, x + dx
            if 0 <= source_y < height and 0 <= source_x < width:
                out[..., y, x] = tensor[..., source_y, source_x]
    return out


def reference_masked_window(mask, values, kernel):
    """Per-pixel double loop over the in-image window; independent of pooling."""
    batch, channels, height, width = values.shape
    half = kernel // 2
    density = torch.zeros(batch, 1, height, width, dtype=values.dtype)
    mean = torch.zeros(batch, channels, height, width, dtype=values.dtype)
    for y in range(height):
        for x in range(width):
            rows = [yy for yy in range(y - half, y + half + 1) if 0 <= yy < height]
            cols = [xx for xx in range(x - half, x + half + 1) if 0 <= xx < width]
            area = len(rows) * len(cols)
            m = mask[:, :, rows][:, :, :, cols]
            v = values[:, :, rows][:, :, :, cols]
            density[:, :, y, x] = m.sum(dim=(2, 3)) / area
            total = (m * v).sum(dim=(2, 3))
            weight = m.sum(dim=(2, 3))
            # Only a window with mask mass has a defined weighted mean, and the
            # weighted mean divides by the mask mass itself, not by the density.
            cond = weight > 0
            mean[:, :, y, x] = torch.where(
                cond, total / weight.clamp_min(1e-12), torch.zeros_like(total))
    return mean, density


def reference_fill_descriptor(features, raw, sr, delta, edge_logp):
    """Independent construction of the preserved 78 v7 channels."""
    cut, fill = candidate_masks(raw, sr, delta)
    cut = cut.to(features.dtype)
    fill = fill.to(features.dtype)
    union = cut + fill
    signed = fill - cut
    tanh_delta = torch.tanh(delta)
    pooled_union = [F.avg_pool2d(union, k, stride=1, padding=k // 2, count_include_pad=False)
                    for k in POOL_KERNELS]
    pooled_signed = [F.avg_pool2d(signed, k, stride=1, padding=k // 2, count_include_pad=False)
                     for k in POOL_KERNELS]
    return torch.cat([
        features, torch.tanh(raw), torch.tanh(sr), tanh_delta, tanh_delta.abs(),
        torch.exp(edge_logp)[:, :1], torch.exp(edge_logp)[:, 1:2], cut, fill,
        *pooled_union, *pooled_signed,
    ], dim=1)


def reference_cut_descriptor(features, raw, sr, delta, edge_logp):
    """Independent construction of the 125 relational cut channels."""
    cut, _ = candidate_masks(raw, sr, delta)
    edge = torch.exp(edge_logp)
    foreground = torch.tanh(raw)
    edge_foreground = edge[:, 1:2]
    channels = [
        features,
        foreground,
        torch.tanh(sr),
        torch.tanh(delta),
        torch.tanh(delta).abs(),
        edge[:, :1],
        edge[:, 1:2],
        cut.to(features.dtype),
    ]
    for direction_y, direction_x in CUT_DIRECTIONS:
        for offset in CUT_OFFSETS:
            step_y, step_x = direction_y * offset, direction_x * offset
            feature_a = reference_shift(features, step_y, step_x)
            feature_b = reference_shift(features, -step_y, -step_x)
            foreground_a = reference_shift(foreground, step_y, step_x)
            foreground_b = reference_shift(foreground, -step_y, -step_x)
            edge_a = reference_shift(edge_foreground, step_y, step_x)
            edge_b = reference_shift(edge_foreground, -step_y, -step_x)
            numerator = (feature_a * feature_b).sum(dim=1, keepdim=True)
            denominator = (feature_a.norm(dim=1, keepdim=True)
                           * feature_b.norm(dim=1, keepdim=True)).clamp_min(1e-6)
            channels.extend([
                (feature_a - feature_b).abs().mean(dim=1, keepdim=True),
                numerator / denominator,
                (foreground_a + foreground_b) / 2,
                foreground_a - foreground_b,
                (edge_a + edge_b) / 2,
                edge_a - edge_b,
            ])
    return torch.cat(channels, dim=1)


def reference_center_block(embedding, raw, sr, delta, edge_logp, candidate):
    edge_foreground = torch.exp(edge_logp.detach())[:, 1:2]
    tanh_delta = torch.tanh(delta.detach())
    return torch.cat([
        embedding, torch.tanh(raw.detach()), torch.tanh(sr.detach()), tanh_delta,
        tanh_delta.abs(), edge_foreground, candidate.to(embedding.dtype),
    ], dim=1)


def reference_context_support(embedding, raw, sr, delta, edge_logp, masks, aggregation):
    values = torch.cat([
        embedding, torch.tanh(raw.detach()), torch.tanh(sr.detach()),
        torch.tanh(delta.detach()), torch.exp(edge_logp.detach())[:, 1:2],
    ], dim=1)
    groups = []
    for name in SUPPORT_MASKS:
        mask = masks[name]
        if aggregation == "multi_scale_support":
            for scale in SUPPORT_SCALES:
                mean, density = reference_masked_window(mask, values, scale)
                groups.append(torch.cat([mean[:, :CONTEXT_EMBED_CHANNELS], density,
                                         mean[:, CONTEXT_EMBED_CHANNELS:]], dim=1))
        else:
            gate = mask.to(values.dtype)
            gated = values * gate
            group = torch.cat([gated[:, :CONTEXT_EMBED_CHANNELS], gate,
                               gated[:, CONTEXT_EMBED_CHANNELS:]], dim=1)
            for _ in SUPPORT_SCALES:
                groups.append(group)
    return torch.cat(groups, dim=1)


def module_with_live_heads(seed: int = 7) -> RegionCandidateVerifier:
    """Deterministic v9 module whose v8 heads are live and context head is zero."""
    torch.manual_seed(seed)
    module = RegionCandidateVerifier()
    with torch.no_grad():
        for head in (module.cut_head, module.fill_head):
            head.weight.fill_(0.5)
            head.bias.zero_()
    return module


def module_with_live_context(seed: int = 7) -> RegionCandidateVerifier:
    """Built module whose context heads have been moved away from zero."""
    module = module_with_live_heads(seed)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for context in (module.cut_context, module.fill_context):
            context.head.weight.normal_(0.0, 0.2)
            context.head.bias.normal_(0.0, 0.1)
    return module


class ShiftTests(unittest.TestCase):
    """shift_padded must zero-pad and must never wrap around the border."""

    def test_positive_offset_never_wraps(self):
        tensor = torch.zeros(1, 1, 4, 4)
        tensor[0, 0, 0, 0] = 5.0
        shifted = shift_padded(tensor, 1, 1)
        self.assertEqual(float(shifted.abs().max()), 0.0)
        self.assertEqual(float(shifted[0, 0, 3, 3]), 0.0)

    def test_negative_offset_never_wraps(self):
        tensor = torch.zeros(1, 1, 4, 4)
        tensor[0, 0, 3, 3] = 5.0
        shifted = shift_padded(tensor, -1, -1)
        self.assertEqual(float(shifted.abs().max()), 0.0)

    def test_matches_loop_reference(self):
        generator = torch.Generator().manual_seed(5)
        tensor = torch.randn(2, 3, 5, 6, generator=generator)
        for dy, dx in ((0, 0), (1, 0), (0, -1), (2, 3), (-3, -2), (4, 4)):
            with self.subTest(offset=(dy, dx)):
                self.assertTrue(torch.equal(shift_padded(tensor, dy, dx),
                                            reference_shift(tensor, dy, dx)))


class SupportMaskTests(unittest.TestCase):
    """The three support masks of each direction partition the image."""

    def setUp(self):
        self.raw = torch.tensor([[[[2.0, -2.0, 2.0, -2.0, 2.0]]]])
        self.sr = torch.tensor([[[[-1.0, 1.0, 2.0, -2.0, 2.0]]]])
        self.delta = self.sr - self.raw

    def test_partition_and_direction_membership(self):
        cut, fill = candidate_masks(self.raw, self.sr, self.delta)
        # Pixel 0: raw +, sr - -> cut.  Pixel 1: raw -, sr + -> fill.
        # Pixels 2 and 3: no change.  Pixel 4: delta 0 -> no candidate.
        self.assertEqual(cut.tolist(), [[[[True, False, False, False, False]]]])
        self.assertEqual(fill.tolist(), [[[[False, True, False, False, False]]]])
        for direction in CONTEXT_DIRECTIONS:
            masks = support_masks(self.raw, self.sr, self.delta, direction)
            union = masks["candidate"] | masks["foreground"] | masks["background"]
            # The three masks cover this direction's candidates plus every pixel
            # outside all candidates; the other direction's candidates are not
            # support for this direction.
            outside = ~(cut | fill)
            self.assertTrue(torch.equal(union, masks["candidate"] | outside))
            self.assertFalse(bool((masks["candidate"] & masks["foreground"]).any()))
            self.assertFalse(bool((masks["candidate"] & masks["background"]).any()))
            self.assertFalse(bool((masks["foreground"] & masks["background"]).any()))
            expected = cut if direction == "cut" else fill
            self.assertTrue(torch.equal(masks["candidate"], expected))
            # Support regions never contain any candidate of either direction.
            self.assertFalse(bool((masks["foreground"] & ~outside).any()))
            self.assertFalse(bool((masks["background"] & ~outside).any()))
            other = fill if direction == "cut" else cut
            self.assertFalse(bool((union & other).any()))

    def test_foreground_follows_raw_not_truth(self):
        masks = support_masks(self.raw, self.sr, self.delta, "cut")
        raw_positive = positive(self.raw)
        self.assertTrue(torch.equal(masks["foreground"],
                                    raw_positive & ~(candidate_masks(self.raw, self.sr, self.delta)[0]
                                                     | candidate_masks(self.raw, self.sr, self.delta)[1])))
        self.assertFalse(bool((masks["background"] & raw_positive).any()))

    def test_bad_direction_rejected(self):
        with self.assertRaises(ValueError):
            support_masks(self.raw, self.sr, self.delta, "both")


class MaskedWindowTests(unittest.TestCase):
    """Mask-weighted window statistics: density, borders, empty supports."""

    def test_matches_loop_reference(self):
        generator = torch.Generator().manual_seed(17)
        shape = (2, 1, 7, 5)
        mask = (torch.rand(shape, generator=generator) > 0.6).to(torch.float32)
        values = torch.randn(2, 3, 7, 5, generator=generator)
        for kernel in SUPPORT_SCALES:
            with self.subTest(kernel=kernel):
                mean, density = masked_window_statistics(mask, values, kernel)
                ref_mean, ref_density = reference_masked_window(mask, values, kernel)
                torch.testing.assert_close(density, ref_density, rtol=0, atol=1e-6)
                torch.testing.assert_close(mean, ref_mean, rtol=0, atol=1e-6)

    def test_density_denominator_counts_only_in_image_pixels(self):
        # A fully masked image has density 1.0 at every location, corners
        # included: the zero padding is not part of the denominator.
        mask = torch.ones(1, 1, 4, 4)
        values = torch.ones(1, 1, 4, 4)
        for kernel in (3, 5, 15):
            with self.subTest(kernel=kernel):
                _, density = masked_window_statistics(mask, values, kernel)
                torch.testing.assert_close(density, torch.ones_like(density), rtol=0, atol=1e-6)

    def test_hot_pixel_mass_does_not_move_to_the_opposite_border(self):
        mask = torch.zeros(1, 1, 5, 5)
        mask[0, 0, 0, 0] = 1.0
        values = torch.ones(1, 2, 5, 5)
        mean, density = masked_window_statistics(mask, values, 3)
        self.assertAlmostEqual(float(density[0, 0, 0, 0]), 0.25, places=6)
        self.assertAlmostEqual(float(density[0, 0, 4, 4]), 0.0, places=6)
        self.assertAlmostEqual(float(mean[0, 0, 4, 4]), 0.0, places=6)
        self.assertAlmostEqual(float(mean[0, 0, 0, 0]), 1.0, places=6)

    def test_empty_support_is_zero_mean_and_zero_density(self):
        mask = torch.zeros(1, 1, 3, 3)
        values = torch.full((1, 4, 3, 3), 7.0)
        mean, density = masked_window_statistics(mask, values, 3)
        self.assertEqual(float(density.sum()), 0.0)
        self.assertEqual(float(mean.abs().sum()), 0.0)

    def test_mask_weighted_mean_ignores_unmasked_values(self):
        mask = torch.zeros(1, 1, 3, 3)
        mask[0, 0, 1, 1] = 1.0
        values = torch.zeros(1, 1, 3, 3)
        values[0, 0, 1, 1] = 4.0
        mean, density = masked_window_statistics(mask, values, 3)
        self.assertAlmostEqual(float(mean[0, 0, 1, 1]), 4.0, places=6)
        self.assertAlmostEqual(float(mean[0, 0, 0, 0]), 4.0, places=6)
        self.assertAlmostEqual(float(density[0, 0, 0, 0]), 1 / 4, places=6)

    def test_guards(self):
        mask = torch.ones(1, 1, 3, 3)
        values = torch.ones(1, 2, 3, 3)
        with self.assertRaises(ValueError):
            masked_window_statistics(mask, values, 4)
        with self.assertRaises(ValueError):
            masked_window_statistics(mask, values, 0)
        with self.assertRaises(ValueError):
            masked_window_statistics(mask, values, True)
        with self.assertRaises(ValueError):
            masked_window_statistics(mask, values[0], 3)
        with self.assertRaises(ValueError):
            masked_window_statistics(mask[0], values, 3)
        with self.assertRaises(ValueError):
            masked_window_statistics(torch.ones(1, 2, 3, 3), values, 3)
        with self.assertRaises(ValueError):
            masked_window_statistics([[0.0]], values, 3)

    def test_non_finite_values_reach_the_descriptor_guard(self):
        # The per-block cost is one pooling pair, so finiteness is enforced once
        # per descriptor rather than once per (mask, scale) block.
        projection = torch.nn.Conv2d(64, CONTEXT_EMBED_CHANNELS, 1, bias=False)
        features = torch.randn(1, 64, 4, 4)
        raw = torch.randn(1, 1, 4, 4)
        sr = raw + torch.randn(1, 1, 4, 4)
        delta = sr - raw
        edge = log_softmax_edges(1, 4, 4)
        raw[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            context_descriptor(projection, features, raw, sr, sr - raw, edge, "cut",
                               "multi_scale_support")


class DescriptorLayoutTests(unittest.TestCase):
    """Channel counts, index helpers and the full descriptor construction."""

    def setUp(self):
        torch.manual_seed(23)
        self.features = torch.randn(2, 64, 6, 5)
        self.raw = torch.randn(2, 1, 6, 5)
        self.sr = self.raw + torch.randn(2, 1, 6, 5)
        self.delta = self.sr - self.raw
        self.edge = log_softmax_edges(2, 6, 5)

    def test_channel_counts(self):
        self.assertEqual(CONTEXT_EMBED_CHANNELS, 8)
        self.assertEqual(SUPPORT_MASKS, ("candidate", "foreground", "background"))
        self.assertEqual(SUPPORT_SCALES, (3, 7, 15))
        self.assertEqual(SUPPORT_GROUP_CHANNELS, 13)
        self.assertEqual(SUPPORT_CHANNELS, 117)
        self.assertEqual(CENTER_CHANNELS, 14)
        self.assertEqual(CONTEXT_DESCRIPTOR_CHANNELS, 131)
        self.assertEqual(CENTER_FEATURES[-1], "candidate_flag")
        self.assertEqual(SUPPORT_FEATURES[1], "density")

    def test_index_helpers(self):
        self.assertEqual(center_channel_index("embedding"), 0)
        self.assertEqual(center_channel_index("raw_margin"), 8)
        self.assertEqual(center_channel_index("edge_foreground"), 12)
        self.assertEqual(center_channel_index("candidate_flag"), 13)
        self.assertEqual(support_block_index("candidate", 3), CENTER_CHANNELS)
        self.assertEqual(support_block_index("candidate", 7), CENTER_CHANNELS + 13)
        self.assertEqual(support_block_index("foreground", 3), CENTER_CHANNELS + 3 * 13)
        self.assertEqual(support_block_index("background", 15), CENTER_CHANNELS + 8 * 13)
        self.assertEqual(support_channel_index("candidate", 3, "density"), CENTER_CHANNELS + 8)
        self.assertEqual(support_channel_index("candidate", 3, "raw_mean"), CENTER_CHANNELS + 9)
        self.assertEqual(support_channel_index("background", 15, "edge_mean"),
                         CENTER_CHANNELS + 9 * 13 - 1)
        with self.assertRaises(ValueError):
            center_channel_index("truth")
        with self.assertRaises(ValueError):
            support_block_index("candidate", 5)
        with self.assertRaises(ValueError):
            support_channel_index("candidate", 3, "truth")

    def test_projection_is_detached_from_features(self):
        projection = torch.nn.Conv2d(64, CONTEXT_EMBED_CHANNELS, 1, bias=False)
        features = self.features.clone().requires_grad_(True)
        descriptor = context_descriptor(projection, features, self.raw, self.sr, self.delta,
                                        self.edge, "cut", "multi_scale_support")
        # The descriptor is connected to the projection parameters only.
        self.assertTrue(descriptor.requires_grad)
        descriptor.sum().backward()
        self.assertIsNone(features.grad)
        self.assertIsNotNone(projection.weight.grad)
        self.assertGreater(float(projection.weight.grad.abs().sum()), 0.0)

    def test_descriptor_matches_reference_for_both_variants(self):
        projection = torch.nn.Conv2d(64, CONTEXT_EMBED_CHANNELS, 1, bias=False)
        with torch.no_grad():
            projection.weight.normal_(0.0, 0.05)
        embedding = projection(self.features)
        for direction in CONTEXT_DIRECTIONS:
            for aggregation in AGGREGATION_MODES:
                with self.subTest(direction=direction, aggregation=aggregation):
                    masks = support_masks(self.raw, self.sr, self.delta, direction)
                    center = reference_center_block(embedding, self.raw, self.sr, self.delta,
                                                    self.edge, masks["candidate"])
                    support = reference_context_support(embedding, self.raw, self.sr, self.delta,
                                                        self.edge, masks, aggregation)
                    expected = torch.cat([center, support], dim=1)
                    actual = context_descriptor(projection, self.features, self.raw, self.sr,
                                                self.delta, self.edge, direction, aggregation)
                    self.assertEqual(actual.shape[1], CONTEXT_DESCRIPTOR_CHANNELS)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)

    def test_center_only_slots_are_center_gated_and_repeated(self):
        projection = torch.nn.Conv2d(64, CONTEXT_EMBED_CHANNELS, 1, bias=False)
        embedding = projection(self.features)
        masks = support_masks(self.raw, self.sr, self.delta, "cut")
        block = context_support_block(embedding, self.raw, self.sr, self.delta, self.edge,
                                      masks, "center_only")
        self.assertEqual(block.shape[1], SUPPORT_CHANNELS)
        for scale_index in range(1, len(SUPPORT_SCALES)):
            first = block[:, 0:SUPPORT_GROUP_CHANNELS]
            other = block[:, scale_index * SUPPORT_GROUP_CHANNELS:
                          (scale_index + 1) * SUPPORT_GROUP_CHANNELS]
            self.assertTrue(torch.equal(first, other))
        # Density channel equals the mask itself and gated embedding is masked E.
        density = block[:, 8:9]
        self.assertTrue(torch.equal(density, masks["candidate"].to(block.dtype)))
        gated_embedding = block[:, :CONTEXT_EMBED_CHANNELS]
        torch.testing.assert_close(gated_embedding,
                                   embedding * masks["candidate"].to(block.dtype),
                                   rtol=0, atol=1e-7)
        # The center-only block contains no zero-padding and cannot be all zero.
        self.assertGreater(float(block.abs().sum()), 0.0)

    def test_bad_aggregation_rejected(self):
        projection = torch.nn.Conv2d(64, CONTEXT_EMBED_CHANNELS, 1, bias=False)
        with self.assertRaises(ValueError):
            context_descriptor(projection, self.features, self.raw, self.sr, self.delta,
                               self.edge, "cut", "multi_scale")
        with self.assertRaises(ValueError):
            context_descriptor(projection, self.features, self.raw, self.sr, self.delta,
                               self.edge, "fill", "center")


class EncoderTests(unittest.TestCase):
    """The context residual encoder: skeleton, zero init, parameter budget."""

    def setUp(self):
        torch.manual_seed(31)
        self.module = RegionCandidateVerifier()
        self.features = torch.randn(2, 64, 5, 7)
        self.raw = torch.randn(2, 1, 5, 7)
        self.sr = self.raw + torch.randn(2, 1, 5, 7)
        self.delta = self.sr - self.raw
        self.edge = log_softmax_edges(2, 5, 7)

    def test_skeleton(self):
        for context in (self.module.cut_context, self.module.fill_context):
            self.assertEqual(context.stem.in_channels, CONTEXT_DESCRIPTOR_CHANNELS)
            self.assertEqual(context.stem.out_channels, CONTEXT_HIDDEN_CHANNELS)
            self.assertIsNone(context.stem.bias)
            self.assertEqual(context.depthwise.groups, CONTEXT_HIDDEN_CHANNELS)
            self.assertEqual(context.depthwise.kernel_size, (3, 3))
            self.assertEqual(context.depthwise.padding, (1, 1))
            self.assertIsNone(context.depthwise.bias)
            self.assertEqual(context.head.in_channels, CONTEXT_HIDDEN_CHANNELS)
            self.assertEqual(context.head.out_channels, 1)
            self.assertEqual(torch.count_nonzero(context.head.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(context.head.bias).item(), 0)
        self.assertEqual(self.module.context_projection.in_channels, 64)
        self.assertEqual(self.module.context_projection.out_channels, CONTEXT_EMBED_CHANNELS)
        self.assertIsNone(self.module.context_projection.bias)
        self.assertEqual(self.module.aggregation, "multi_scale_support")

    def test_parameter_counts_and_disjointness(self):
        counts = {
            "verifier": sum(p.numel() for p in self.module.parameters()),
            "context": (sum(p.numel() for p in self.module.context_projection.parameters())
                        + sum(p.numel() for p in self.module.cut_context.parameters())
                        + sum(p.numel() for p in self.module.fill_context.parameters())),
        }
        self.assertEqual(counts["verifier"], VERIFIER_PARAMETERS)
        self.assertEqual(counts["context"], CONTEXT_PARAMETERS)
        self.assertEqual(counts["verifier"] - counts["context"], V8_VERIFIER_PARAMETERS)
        control = RegionCandidateVerifier(aggregation="center_only")
        self.assertEqual(sum(p.numel() for p in control.parameters()), VERIFIER_PARAMETERS)
        v9_ids = {id(p) for p in self.module.cut_context.parameters()} | \
            {id(p) for p in self.module.fill_context.parameters()}
        # 8 named tensors per direction branch (stem, 2 norms, depthwise, head).
        self.assertEqual(len(v9_ids), 16)
        v8_ids = {id(p) for name, p in self.module.named_parameters()
                  if not name.startswith(("cut_context", "fill_context", "context_projection"))}
        self.assertFalse(v9_ids & v8_ids)
        self.assertNotIn(id(self.module.context_projection.weight), v8_ids)
        self.assertFalse({id(p) for p in self.module.cut_context.parameters()} &
                         {id(p) for p in self.module.fill_context.parameters()})

    def test_zero_residual_and_local_logits(self):
        with torch.no_grad():
            logits = self.module(self.features, self.raw, self.sr, self.delta, self.edge)
            local = self.module.local_logits(self.features, self.raw, self.sr, self.delta, self.edge)
            residual = self.module.context_residual(self.features, self.raw, self.sr,
                                                    self.delta, self.edge, "cut")
        self.assertTrue(torch.equal(logits, local))
        self.assertEqual(float(residual.abs().max()), 0.0)
        self.assertEqual(float(residual.abs().sum()), 0.0)

    def test_reference_hidden_for_live_head(self):
        module = RegionCandidateVerifier()
        with torch.no_grad():
            for context in (module.cut_context, module.fill_context):
                context.stem.weight.normal_(0.0, 0.1)
                context.depthwise.weight.normal_(0.0, 0.1)
                context.norm_stem.weight.fill_(1.0)
                context.norm_stem.bias.zero_()
                context.norm_depthwise.weight.fill_(1.0)
                context.norm_depthwise.bias.zero_()
                context.head.weight.normal_(0.0, 0.1)
                context.head.bias.zero_()
        descriptor = context_descriptor(module.context_projection, self.features, self.raw,
                                       self.sr, self.delta, self.edge, "fill",
                                       module.aggregation)
        hidden = F.silu(channel_norm(module.fill_context.norm_stem,
                                     module.fill_context.stem(descriptor)))
        near = channel_norm(module.fill_context.norm_depthwise,
                            module.fill_context.depthwise(hidden))
        expected = module.fill_context.head(F.silu(hidden + near))
        with torch.no_grad():
            actual = module.context_residual(self.features, self.raw, self.sr,
                                             self.delta, self.edge, "fill")
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_control_and_full_descriptors_differ_but_can_be_zero_residual(self):
        full = RegionCandidateVerifier(aggregation="multi_scale_support")
        control = RegionCandidateVerifier(aggregation="center_only")
        with torch.no_grad():
            for context in (full.cut_context, full.fill_context):
                context.head.weight.fill_(0.25)
            for context in (control.cut_context, control.fill_context):
                context.head.weight.fill_(0.25)
        with torch.no_grad():
            full_logits = full(self.features, self.raw, self.sr, self.delta, self.edge)
            control_logits = control(self.features, self.raw, self.sr, self.delta, self.edge)
        # With different random stems the residuals differ, so the control is a
        # genuinely different scorer and not a zeroed copy of the full branch.
        self.assertFalse(torch.allclose(full_logits, control_logits))


class V8ParityTests(unittest.TestCase):
    """A zero context head must make v9 bit-for-bit v8."""

    def setUp(self):
        torch.manual_seed(41)
        self.v8 = V8RegionCandidateVerifier()
        with torch.no_grad():
            for head in (self.v8.cut_head, self.v8.fill_head):
                head.weight.normal_(0.0, 0.2)
                head.bias.normal_(0.0, 0.1)
        self.v9 = RegionCandidateVerifier()
        self.v9.load_state_dict(self.v8.state_dict(), strict=False)
        torch.manual_seed(43)
        self.features = torch.randn(3, 64, 9, 6)
        self.raw = torch.randn(3, 1, 9, 6)
        self.sr = self.raw + torch.randn(3, 1, 9, 6)
        self.delta = self.sr - self.raw
        self.edge = log_softmax_edges(3, 9, 6)

    def test_only_context_keys_are_new(self):
        v8_keys = set(self.v8.state_dict())
        v9_keys = set(self.v9.state_dict())
        self.assertEqual(v9_keys - v8_keys, set(CONTEXT_STATE_KEYS))
        self.assertEqual(len(CONTEXT_STATE_KEYS), 18)
        for name in FILL_ATTRIBUTE_NAMES:
            self.assertIn(name, v9_keys)

    def test_logits_are_bitwise_equal(self):
        with torch.no_grad():
            expected = self.v8(self.features, self.raw, self.sr, self.delta, self.edge)
            actual = self.v9(self.features, self.raw, self.sr, self.delta, self.edge)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(float((actual - expected).abs().max()), 0.0)

    def test_local_logits_match_v8_forward(self):
        with torch.no_grad():
            expected = self.v8(self.features, self.raw, self.sr, self.delta, self.edge)
            actual = self.v9.local_logits(self.features, self.raw, self.sr, self.delta, self.edge)
        self.assertTrue(torch.equal(expected, actual))

    def test_fill_and_cut_descriptors_unchanged(self):
        actual_fill = self.v8._fill_descriptor(self.features, self.raw, self.sr, self.delta, self.edge)
        expected_fill = reference_fill_descriptor(self.features, self.raw, self.sr, self.delta, self.edge)
        self.assertEqual(actual_fill.shape[1], FILL_DESCRIPTOR_CHANNELS)
        torch.testing.assert_close(actual_fill, expected_fill, rtol=0, atol=1e-6)
        actual_cut = self.v8._cut_descriptor(self.features, self.raw, self.sr, self.delta, self.edge)
        expected_cut = reference_cut_descriptor(self.features, self.raw, self.sr, self.delta, self.edge)
        self.assertEqual(actual_cut.shape[1], CUT_DESCRIPTOR_CHANNELS)
        torch.testing.assert_close(actual_cut, expected_cut, rtol=0, atol=1e-6)
        self.assertEqual(DESCRIPTOR_CHANNELS, FILL_DESCRIPTOR_CHANNELS)
        self.assertEqual(CUT_BASE_CHANNELS, 71)
        self.assertEqual(CUT_RELATIONAL_CHANNELS, 54)
        self.assertEqual(CUT_PAIRS, 9)
        self.assertEqual(CUT_PAIR_CHANNELS, 6)
        self.assertEqual(HIDDEN_CHANNELS, 24)

    def test_shared_contract_functions_are_the_v8_objects(self):
        self.assertIs(candidate_masks, v8_candidate_masks)
        self.assertIs(directional_labels, v8_directional_labels)
        self.assertIs(verifier_loss, v8_verifier_loss)
        for name in ("candidate_masks", "directional_labels", "verifier_loss",
                     "hard_keep", "gated_region", "positive", "shift_padded",
                     "side_similarity"):
            with self.subTest(function=name):
                self.assertIs(getattr(v9_tool, name), getattr(v8_tool, name))

    def test_v8_state_loads_strictly_into_a_stripped_v9(self):
        # Removing the context branch keys leaves exactly the v8 state.
        stripped = {name: value for name, value in self.v9.state_dict().items()
                    if name not in set(CONTEXT_STATE_KEYS)}
        self.v8.load_state_dict(stripped, strict=True)


class AggregationGuardTests(unittest.TestCase):
    """The two aggregation variants share shapes; loads must not be silent."""

    def setUp(self):
        self.full = RegionCandidateVerifier()
        self.control = RegionCandidateVerifier(aggregation="center_only")

    def test_sentinel_is_an_integer_buffer(self):
        code = self.full.state_dict()["aggregation_code"]
        self.assertFalse(code.is_floating_point())
        self.assertEqual(code.dtype, torch.int64)
        self.assertEqual(int(code), 0)
        self.assertEqual(int(self.control.state_dict()["aggregation_code"]), 1)

    def test_state_scan_skips_the_integer_sentinel(self):
        # The frozen evaluate.py finiteness scan filters on is_floating_point();
        # an integer buffer must neither raise nor hide a non-finite float.
        for module in (self.full, self.control):
            with self.subTest(aggregation=module.aggregation):
                self.assertFalse(any(value.is_floating_point()
                                     and not torch.isfinite(value).all()
                                     for value in module.state_dict().values()))

    def test_cross_variant_loads_are_rejected_before_any_copy(self):
        for target, state in ((self.control, self.full.state_dict()),
                              (self.full, self.control.state_dict())):
            with self.subTest(aggregation=target.aggregation):
                before = [value.clone() for value in target.state_dict().values()]
                with self.assertRaises(RuntimeError):
                    target.load_state_dict(state, strict=True)
                for now, was in zip(target.state_dict().values(), before):
                    self.assertTrue(torch.equal(now, was))

    def test_same_variant_strict_reload_succeeds(self):
        self.full.load_state_dict(self.full.state_dict(), strict=True)
        self.control.load_state_dict(self.control.state_dict(), strict=True)

    def test_aggregation_is_read_only_after_construction(self):
        with self.assertRaises(AttributeError):
            self.full.aggregation = "center_only"
        self.assertEqual(self.full.aggregation, "multi_scale_support")
        self.assertEqual(self.control.aggregation, "center_only")

    def test_forward_uses_the_declared_mode(self):
        # Give both variants identical parameters (a legal same-variant load for
        # the control), so any difference is produced by the aggregation only.
        state = {name: value.clone() for name, value in self.full.state_dict().items()}
        state["aggregation_code"] = self.control.state_dict()["aggregation_code"]
        self.control.load_state_dict(state, strict=True)
        for module in (self.full, self.control):
            with torch.no_grad():
                for context in (module.cut_context, module.fill_context):
                    context.head.weight.fill_(0.25)
        torch.manual_seed(59)
        features = torch.randn(1, 64, 5, 5)
        raw = torch.randn(1, 1, 5, 5)
        sr = raw + torch.randn(1, 1, 5, 5)
        delta = sr - raw
        edge = log_softmax_edges(1, 5, 5)
        with torch.no_grad():
            full_logits = self.full(features, raw, sr, delta, edge)
            control_logits = self.control(features, raw, sr, delta, edge)
            full_local = self.full.local_logits(features, raw, sr, delta, edge)
            control_local = self.control.local_logits(features, raw, sr, delta, edge)
        self.assertEqual(full_logits.shape, (1, 2, 5, 5))
        self.assertEqual(control_logits.shape, (1, 2, 5, 5))
        self.assertTrue(torch.equal(full_local, control_local))
        self.assertFalse(torch.equal(full_logits, control_logits))


class GradientTests(unittest.TestCase):
    """Detached inputs, zero-head staging, and full-branch learning afterwards."""

    def setUp(self):
        self.module = RegionCandidateVerifier()
        with torch.no_grad():
            for head in (self.module.cut_head, self.module.fill_head):
                head.weight.fill_(0.3)
                head.bias.zero_()
        torch.manual_seed(53)
        self.features = torch.randn(2, 64, 6, 6)
        self.raw = torch.randn(2, 1, 6, 6)
        self.sr = self.raw + torch.randn(2, 1, 6, 6)
        self.delta = self.sr - self.raw
        self.edge = log_softmax_edges(2, 6, 6)
        self.truth = (torch.rand(2, 1, 6, 6) > 0.5).to(torch.float32)

    def test_inputs_are_detached(self):
        features = self.features.clone().requires_grad_(True)
        raw = self.raw.clone().requires_grad_(True)
        sr = self.sr.clone().requires_grad_(True)
        delta = self.delta.clone().requires_grad_(True)
        edge = self.edge.clone().requires_grad_(True)
        logits = self.module(features, raw, sr, delta, edge)
        logits.sum().backward()
        for name, tensor in (("features", features), ("raw", raw), ("sr", sr),
                             ("delta", delta), ("edge", edge)):
            with self.subTest(name=name):
                self.assertIsNone(tensor.grad)
        self.assertIsNotNone(self.module.context_projection.weight.grad)

    def test_first_step_touches_only_the_zero_heads(self):
        logits = self.module(self.features, self.raw, self.sr, self.delta, self.edge)
        loss, _ = verifier_loss(logits, self.raw, self.sr, self.delta, self.truth)
        loss.backward()
        for context in (self.module.cut_context, self.module.fill_context):
            self.assertGreater(float(context.head.weight.grad.abs().sum()), 0.0)
            self.assertGreater(float(context.head.bias.grad.abs().sum()), 0.0)
            # A zero head blocks the upstream gradient in the first step.
            self.assertEqual(float(context.stem.weight.grad.abs().sum()), 0.0)
            self.assertEqual(float(context.depthwise.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(self.module.context_projection.weight.grad.abs().sum()), 0.0)
        # The preserved v8 local path is trained by the same loss as before.
        self.assertGreater(float(self.module.cut_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(self.module.fill_head.weight.grad.abs().sum()), 0.0)

    def test_upstream_learns_after_the_head_moves(self):
        module = RegionCandidateVerifier()
        torch.manual_seed(59)
        with torch.no_grad():
            for head in (module.cut_head, module.fill_head):
                head.weight.fill_(0.3)
            for context in (module.cut_context, module.fill_context):
                context.head.weight.normal_(0.0, 0.1)
                context.head.bias.normal_(0.0, 0.05)
        logits = module(self.features, self.raw, self.sr, self.delta, self.edge)
        loss, _ = verifier_loss(logits, self.raw, self.sr, self.delta, self.truth)
        loss.backward()
        self.assertGreater(float(module.cut_context.stem.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.fill_context.stem.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.cut_context.depthwise.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.context_projection.weight.grad.abs().sum()), 0.0)

    def test_optimizer_step_updates_only_declared_parameters(self):
        module = RegionCandidateVerifier()
        with torch.no_grad():
            for head in (module.cut_head, module.fill_head):
                head.weight.fill_(0.3)
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        logits = module(self.features, self.raw, self.sr, self.delta, self.edge)
        loss, _ = verifier_loss(logits, self.raw, self.sr, self.delta, self.truth)
        loss.backward()
        before = {name: value.detach().clone() for name, value in module.named_parameters()}
        optimizer.step()
        moved = {name for name, value in module.named_parameters()
                 if not torch.equal(value.detach(), before[name])}
        self.assertTrue(moved <= set(dict(module.named_parameters())))
        self.assertTrue(any(name.startswith("cut_context.head") for name in moved))
        self.assertTrue(any(name.startswith("fill_context.head") for name in moved))
        self.assertFalse(any(name.startswith("context_projection") for name in moved))
        self.assertTrue(all(torch.isfinite(value).all().item()
                            for value in dict(module.named_parameters()).values()))


class LossTests(unittest.TestCase):
    """Ordinary BCE, present-direction averaging, connected zero; unchanged."""

    def test_single_direction_ordinary_bce(self):
        raw = torch.tensor([[[[1.0, 1.0, 1.0]]]])
        sr = torch.tensor([[[[-1.0, -1.0, -1.0]]]])
        delta = sr - raw
        truth = torch.tensor([[[[0.0, 1.0, 1.0]]]])
        logits = torch.tensor([[[[0.0, 0.0, 2.0]], [[0.0, 0.0, 0.0]]]], requires_grad=True)
        loss, stats = verifier_loss(logits, raw, sr, delta, truth)
        cut, fill = candidate_masks(raw, sr, delta)
        target = (sr > 0) == (truth > 0.5)
        expected = F.binary_cross_entropy_with_logits(logits[:, :1][cut],
                                                      target[cut].to(logits.dtype))
        balanced = torch.stack([
            F.binary_cross_entropy_with_logits(logits[:, :1][cut][target[cut] == label],
                                               target[cut][target[cut] == label].to(logits.dtype))
            for label in (False, True)]).mean()
        torch.testing.assert_close(loss, expected)
        self.assertFalse(torch.isclose(loss, balanced).item())
        self.assertEqual(stats["present_directions"], 1)
        self.assertEqual(stats["cut_candidates"], 3)
        self.assertEqual(stats["cut_positive"], 1)
        self.assertFalse(torch.isclose(loss, expected / 2).item())

    def test_both_directions_average_once(self):
        raw = torch.tensor([[[[1.0, 1.0, -1.0, -1.0]]]])
        sr = torch.tensor([[[[-1.0, -1.0, 1.0, 1.0]]]])
        delta = sr - raw
        truth = torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]])
        logits = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]]],
                              requires_grad=True)
        loss, stats = verifier_loss(logits, raw, sr, delta, truth)
        cut, fill = candidate_masks(raw, sr, delta)
        target = (sr > 0) == (truth > 0.5)
        cut_mean = F.binary_cross_entropy_with_logits(logits[:, :1][cut],
                                                      target[cut].to(logits.dtype))
        fill_mean = F.binary_cross_entropy_with_logits(logits[:, 1:2][fill],
                                                       target[fill].to(logits.dtype))
        torch.testing.assert_close(loss, (cut_mean + fill_mean) / 2)
        self.assertEqual(stats["present_directions"], 2)

    def test_empty_direction_is_connected_zero(self):
        raw = torch.ones(1, 1, 2, 2)
        delta = torch.zeros_like(raw)
        truth = torch.zeros_like(raw)
        logits = torch.randn(1, 2, 2, 2, requires_grad=True)
        loss, stats = verifier_loss(logits, raw, raw, delta, truth)
        self.assertEqual(stats["edit_candidates"], 0)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_ordinary_bce_is_not_class_balanced(self):
        # Four cut candidates with three positives: the ordinary pixel mean
        # differs from a class-balanced mean, and the single present direction
        # is not halved by the present-direction average.
        raw = torch.full((1, 1, 1, 4), 2.0)
        sr = torch.full((1, 1, 1, 4), -2.0)
        delta = sr - raw
        truth = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        logits = torch.full((1, 2, 1, 4), 0.5)
        loss, stats = verifier_loss(logits, raw, sr, delta, truth)
        positive = F.softplus(torch.tensor(-0.5))
        negative = F.softplus(torch.tensor(0.5))
        ordinary = (3 * positive + negative) / 4
        balanced = (positive + negative) / 2
        torch.testing.assert_close(loss, ordinary)
        self.assertFalse(torch.isclose(loss, balanced).item())
        self.assertFalse(torch.isclose(loss, ordinary / 2).item())
        self.assertEqual(stats["present_directions"], 1)
        self.assertEqual(stats["cut_candidates"], 4)
        self.assertEqual(stats["cut_positive"], 3)


class GateTests(unittest.TestCase):
    """Strict logit > 0 on candidates; the verifier cannot create edits."""

    def test_candidate_masks_and_strict_gate(self):
        raw = torch.tensor([[[[2.0, 2.0, -2.0, -2.0]]]])
        sr = torch.tensor([[[[-1.0, 2.0, 1.0, -2.0]]]])
        delta = torch.tensor([[[[-3.0, 0.0, 3.0, 0.0]]]])
        cut, fill = candidate_masks(raw, sr, delta)
        self.assertTrue(torch.equal(cut, torch.tensor([[[[True, False, False, False]]]])))
        self.assertTrue(torch.equal(fill, torch.tensor([[[[False, False, True, False]]]])))
        logits = torch.tensor([[[[1.0, 100.0, 100.0, 100.0]],
                                [[-1.0, 100.0, -1.0, 100.0]]]])
        keep_cut, keep_fill = hard_keep(raw, sr, delta, logits)
        self.assertTrue(torch.equal(keep_cut, cut))
        self.assertFalse(bool(keep_fill.any()))
        final = gated_region(raw, delta, keep_cut, keep_fill)
        self.assertTrue(torch.equal(final, torch.tensor([[[[-1.0, 2.0, -2.0, -2.0]]]])))

    def test_zero_logit_is_rejected(self):
        raw = torch.tensor([[[[2.0]]]])
        sr = torch.tensor([[[[-2.0]]]])
        delta = sr - raw
        logits = torch.zeros(1, 2, 1, 1)
        keep_cut, keep_fill = hard_keep(raw, sr, delta, logits)
        self.assertFalse(bool(keep_cut.any()))
        self.assertTrue(torch.equal(gated_region(raw, delta, keep_cut, keep_fill), raw))

    def test_keep_subset_and_route_invariants(self):
        torch.manual_seed(61)
        module = module_with_live_heads(61)
        features = torch.randn(2, 64, 9, 9)
        raw = torch.randn(2, 1, 9, 9)
        sr = raw + torch.randn(2, 1, 9, 9)
        delta = sr - raw
        edge = log_softmax_edges(2, 9, 9)
        with torch.no_grad():
            logits = module(features, raw, sr, delta, edge)
            keep_cut, keep_fill = hard_keep(raw, sr, delta, logits)
            cut, fill = candidate_masks(raw, sr, delta)
            final = gated_region(raw, delta, keep_cut, keep_fill)
        self.assertFalse(bool((keep_cut & ~cut).any()))
        self.assertFalse(bool((keep_fill & ~fill).any()))
        outside = ~(cut | fill)
        accepted = keep_cut | keep_fill
        rejected = (cut | fill) & ~accepted
        self.assertTrue(torch.equal(final[outside], raw[outside]))
        self.assertTrue(torch.equal(final[accepted], sr[accepted]))
        self.assertTrue(torch.equal(final[rejected], raw[rejected]))
        self.assertTrue(bool((cut | fill).any()))

    def test_ground_truth_is_not_a_forward_input(self):
        module = module_with_live_context(67)
        features = torch.randn(2, 64, 6, 6)
        raw = torch.randn(2, 1, 6, 6)
        sr = raw + torch.randn(2, 1, 6, 6)
        delta = sr - raw
        edge = log_softmax_edges(2, 6, 6)
        with torch.no_grad():
            first = module(features, raw, sr, delta, edge)
            second = module(features, raw, sr, delta, edge)
        self.assertTrue(torch.equal(first, second))


class GuardTests(unittest.TestCase):
    """Shape, dtype, rank, finiteness and mode guards on public entry points."""

    def setUp(self):
        self.features = torch.randn(2, 64, 5, 5)
        self.raw = torch.randn(2, 1, 5, 5)
        self.sr = self.raw + torch.randn(2, 1, 5, 5)
        self.delta = self.sr - self.raw
        self.edge = log_softmax_edges(2, 5, 5)

    def forward(self, **overrides):
        arguments = dict(features=self.features, raw=self.raw, sr=self.sr,
                         delta=self.delta, edge_logp=self.edge)
        arguments.update(overrides)
        return RegionCandidateVerifier(mode="always_reject")(**arguments)

    def test_non_finite_inputs_raise(self):
        for name in ("features", "raw", "sr", "delta", "edge_logp"):
            for bad in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=name, bad=bad):
                    tensor = {"features": self.features, "raw": self.raw, "sr": self.sr,
                              "delta": self.delta, "edge_logp": self.edge}[name].clone()
                    tensor.view(-1)[0] = bad
                    with self.assertRaises(FloatingPointError):
                        self.forward(**{name: tensor})

    def test_rank_channel_and_shape_guards(self):
        with self.assertRaises(ValueError):
            self.forward(features=self.features[0])
        with self.assertRaises(ValueError):
            self.forward(features=torch.randn(2, 63, 5, 5))
        with self.assertRaises(ValueError):
            self.forward(edge_logp=torch.randn(2, 1, 5, 5))
        with self.assertRaises(ValueError):
            self.forward(raw=torch.randn(2, 1, 4, 5))
        with self.assertRaises(ValueError):
            self.forward(sr=torch.randn(2, 1, 5, 4))
        with self.assertRaises(ValueError):
            self.forward(delta=torch.randn(2, 1, 5, 5, 1))
        with self.assertRaises(ValueError):
            self.forward(features=torch.randn(0, 64, 5, 5))
        with self.assertRaises(ValueError):
            self.forward(features=[[0.0]])

    def test_dtype_guard_and_double_support(self):
        with self.assertRaises(ValueError):
            self.forward(features=torch.randint(0, 4, (2, 64, 5, 5)))
        with self.assertRaises(ValueError):
            self.forward(raw=torch.randn(2, 1, 5, 5, dtype=torch.float64))
        logits = RegionCandidateVerifier(mode="always_reject").double()(
            self.features.double(), self.raw.double(), self.sr.double(),
            self.delta.double(), self.edge.double())
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(logits.shape, (2, 2, 5, 5))

    def test_mode_and_aggregation_guards(self):
        with self.assertRaises(ValueError):
            RegionCandidateVerifier(mode="enabled_typo")
        with self.assertRaises(ValueError):
            RegionCandidateVerifier(aggregation="support")
        with self.assertRaises(ValueError):
            RegionCandidateVerifier(mode="enabled", aggregation=None)
        self.assertEqual(VERIFIER_MODES, ("enabled", "always_reject"))

    def test_always_reject_mode(self):
        rejected = RegionCandidateVerifier(mode="always_reject")(
            self.features, self.raw, self.sr, self.delta, self.edge)
        self.assertTrue(torch.equal(rejected, torch.full_like(rejected, -1)))

    def test_loss_guards(self):
        truth = torch.rand(2, 1, 5, 5)
        logits = torch.randn(2, 2, 5, 5, requires_grad=True)
        with self.assertRaises(ValueError):
            verifier_loss(logits[:, :1], self.raw, self.sr, self.delta, truth)
        with self.assertRaises(ValueError):
            verifier_loss(logits, self.raw, self.sr, self.delta, truth[0])
        with self.assertRaises(ValueError):
            verifier_loss(logits, self.raw, self.sr, torch.randn(2, 1, 4, 5), truth)

    def test_finite_outputs_for_extreme_but_finite_inputs(self):
        features = torch.full((1, 64, 5, 5), 1e4)
        raw = torch.full((1, 1, 5, 5), 1e4)
        sr = torch.full((1, 1, 5, 5), -1e4)
        module = module_with_live_context(71)
        logits = module(features, raw, sr, sr - raw, torch.full((1, 2, 5, 5), -1e4))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_tiny_spatial_shapes(self):
        for batch, height, width in ((1, 1, 1), (1, 2, 3), (2, 16, 16)):
            with self.subTest(shape=(batch, height, width)):
                features = torch.randn(batch, 64, height, width)
                raw = torch.randn(batch, 1, height, width)
                sr = raw + torch.randn(batch, 1, height, width)
                logits = RegionCandidateVerifier()(
                    features, raw, sr, sr - raw, log_softmax_edges(batch, height, width))
                self.assertEqual(logits.shape, (batch, 2, height, width))
                self.assertTrue(torch.isfinite(logits).all().item())


class CalibrationTests(unittest.TestCase):
    """Monotone affine calibration and the BR/HA curve it re-parameterises."""

    def setUp(self):
        torch.manual_seed(73)
        self.scores = torch.randn(2, 2, 4, 5)
        self.beneficial = (torch.rand(2, 2, 4, 5) > 0.6)
        self.harmful = (torch.rand(2, 2, 4, 5) > 0.7) & ~self.beneficial

    def test_calibration_preserves_order_and_maps_the_threshold(self):
        a, b = (2.0, 0.5), (1.0, -3.0)
        calibrated = monotone_affine_calibration(self.scores, a, b)
        threshold = calibration_logit_threshold(a, b)
        self.assertEqual(threshold, (-0.5, 6.0))
        for index, direction in enumerate(CONTEXT_DIRECTIONS):
            with self.subTest(direction=direction):
                raw = self.scores[:, index:index + 1]
                mapped = calibrated[:, index:index + 1]
                self.assertTrue(torch.equal(mapped > 0, raw > threshold[index]))
                self.assertTrue(torch.equal(mapped > 1.5, raw > (1.5 - b[index]) / a[index]))

    def test_scalar_broadcast_and_tensor_inputs(self):
        for a, b in ((2.0, 1.0), (torch.tensor(2.0), torch.tensor([1.0, 1.0])),
                     ((1.5, 1.5), (0.0, 0.0)), (torch.tensor([2.0, 3.0]), -1.0)):
            with self.subTest(a=a, b=b):
                calibrated = monotone_affine_calibration(self.scores, a, b)
                self.assertEqual(calibrated.shape, self.scores.shape)
                self.assertTrue(torch.isfinite(calibrated).all().item())

    def test_non_positive_or_non_finite_slope_rejected(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), (1.0, 0.0), (-0.5, 2.0)):
            with self.subTest(a=bad):
                with self.assertRaises(ValueError):
                    monotone_affine_calibration(self.scores, bad, 0.0)
        for bad in (float("nan"), float("inf"), (1.0, float("nan"))):
            with self.subTest(b=bad):
                with self.assertRaises(ValueError):
                    monotone_affine_calibration(self.scores, 1.0, bad)
        with self.assertRaises(ValueError):
            monotone_affine_calibration(self.scores[0], 1.0, 0.0)
        with self.assertRaises(ValueError):
            monotone_affine_calibration(self.scores, True, 0.0)

    def test_coefficients_are_validated_after_the_dtype_cast(self):
        half = self.scores.to(torch.float16)
        threshold = calibration_logit_threshold(2.0, 0.5, dtype=torch.float16)
        calibrated = monotone_affine_calibration(half, 2.0, 0.5)
        self.assertTrue(torch.isfinite(calibrated).all().item())
        self.assertTrue(torch.equal(calibrated[:, 0] > 0, half[:, 0] > threshold[0]))
        for coerced in (1e-9, 1e30):  # 0.0 and inf once cast to float16
            with self.subTest(coerced=coerced):
                with self.assertRaises(ValueError):
                    monotone_affine_calibration(half, coerced, 0.0)
                with self.assertRaises(ValueError):
                    calibration_logit_threshold(coerced, 0.0, dtype=torch.float16)
        with self.assertRaises(ValueError):
            calibration_logit_threshold(2.0, 0.0, dtype="float16")

    def test_overflowing_calibration_raises_instead_of_returning_inf(self):
        for scores, slope in ((torch.full((1, 2, 1, 1), 6.0e4, dtype=torch.float16), 2.0),
                              (torch.full((1, 2, 1, 1), 3.0e38), 2.0)):
            with self.subTest(dtype=scores.dtype):
                with self.assertRaises(FloatingPointError):
                    monotone_affine_calibration(scores, slope, 0.0)

    def test_non_floating_logits_rejected(self):
        with self.assertRaises(ValueError):
            monotone_affine_calibration(torch.zeros(1, 2, 2, 2, dtype=torch.int64), 1.0, 0.0)

    def test_curve_counts_match_a_brute_force_scan(self):
        thresholds = [-1.0, 0.0, 1.0]
        curve = br_ha_curve(self.scores, self.beneficial, self.harmful, thresholds)
        self.assertEqual(len(curve["points"]), len(thresholds))
        self.assertEqual(curve["direction_order"], ["cut", "fill"])
        for point in curve["points"]:
            for index, direction in enumerate(CONTEXT_DIRECTIONS):
                accepted = self.scores[:, index:index + 1] > point["threshold"]
                entry = point["directions"][direction]
                good = self.beneficial[:, index:index + 1]
                bad = self.harmful[:, index:index + 1]
                self.assertEqual(entry["accepted_beneficial"],
                                 int((accepted & good).sum().item()))
                self.assertEqual(entry["accepted_harmful"],
                                 int((accepted & bad).sum().item()))
                total_good = int(good.sum().item())
                total_bad = int(bad.sum().item())
                expected_br = ((total_good - int((accepted & good).sum().item())) / total_good
                               if total_good else None)
                expected_ha = (int((accepted & bad).sum().item()) / total_bad
                               if total_bad else None)
                self.assertEqual(entry["beneficial_rejection_rate"], expected_br)
                self.assertEqual(entry["harmful_acceptance_rate"], expected_ha)

    def test_monotone_calibration_only_reparameterises_the_curve(self):
        a, b = (2.0, 0.5), (1.0, -3.0)
        calibrated = monotone_affine_calibration(self.scores, a, b)
        calibrated_curve = br_ha_curve(calibrated, self.beneficial, self.harmful, [0.0])
        for index, direction in enumerate(CONTEXT_DIRECTIONS):
            with self.subTest(direction=direction):
                raw_curve = br_ha_curve(self.scores, self.beneficial, self.harmful,
                                        [(-b[index]) / a[index]])
                self.assertEqual(
                    calibrated_curve["points"][0]["directions"][direction]["rejected_beneficial"],
                    raw_curve["points"][0]["directions"][direction]["rejected_beneficial"])
                self.assertEqual(
                    calibrated_curve["points"][0]["directions"][direction]["accepted_harmful"],
                    raw_curve["points"][0]["directions"][direction]["accepted_harmful"])

    def test_zero_denominators_are_null(self):
        scores = torch.zeros(1, 2, 1, 1)
        good = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        bad = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        point = br_ha_curve(scores, good, bad, [0.0])["points"][0]
        for direction in CONTEXT_DIRECTIONS:
            entry = point["directions"][direction]
            self.assertIsNone(entry["beneficial_rejection_rate"])
            self.assertIsNone(entry["harmful_acceptance_rate"])
            self.assertIsNone(entry["acceptance_precision"])
            self.assertEqual(entry["beneficial_total"], 0)

    def test_curve_guards(self):
        with self.assertRaises(ValueError):
            br_ha_curve(self.scores[:, :1], self.beneficial, self.harmful, [0.0])
        with self.assertRaises(ValueError):
            br_ha_curve(self.scores, self.beneficial[0], self.harmful, [0.0])
        with self.assertRaises(ValueError):
            br_ha_curve(self.scores, self.beneficial, self.harmful, [True])
        with self.assertRaises(ValueError):
            br_ha_curve(self.scores, self.beneficial, self.harmful, ["zero"])

    def test_curve_rejects_overlap_and_non_finite_scores(self):
        overlap = self.beneficial | self.harmful
        with self.assertRaises(ValueError):
            br_ha_curve(self.scores, self.beneficial, overlap, [0.0])
        with self.assertRaises(ValueError):
            br_ha_curve(self.scores, self.beneficial.long(), self.harmful, [0.0])
        with self.assertRaises(ValueError):
            br_ha_curve(self.scores.long(), self.beneficial, self.harmful, [0.0])
        for bad in (float("nan"), float("inf")):
            broken = self.scores.clone()
            broken[0, 0, 0, 0] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(FloatingPointError):
                    br_ha_curve(broken, self.beneficial, self.harmful, [0.0])


class RouteInvariantTests(unittest.TestCase):
    """Live v9 verifier on real candidate layouts: no new edits, exact fallback."""

    def setUp(self):
        torch.manual_seed(79)
        self.size = 9
        self.features = torch.randn(2, 64, self.size, self.size)
        self.raw = torch.randn(2, 1, self.size, self.size)
        self.sr = self.raw + torch.randn(2, 1, self.size, self.size)
        self.delta = self.sr - self.raw
        self.edge = log_softmax_edges(2, self.size, self.size)
        self.module = module_with_live_context(83)

    def test_no_edit_outside_candidates(self):
        with torch.no_grad():
            logits = self.module(self.features, self.raw, self.sr, self.delta, self.edge)
        keep_cut, keep_fill = hard_keep(self.raw, self.sr, self.delta, logits)
        cut, fill = candidate_masks(self.raw, self.sr, self.delta)
        final = gated_region(self.raw, self.delta, keep_cut, keep_fill)
        self.assertFalse(bool((keep_cut & ~cut).any()))
        self.assertFalse(bool((keep_fill & ~fill).any()))
        outside = ~(cut | fill)
        self.assertTrue(torch.equal(final[outside], self.raw[outside]))
        accepted = keep_cut | keep_fill
        self.assertTrue(torch.equal(final[accepted], self.sr[accepted]))
        rejected = (cut | fill) & ~accepted
        self.assertTrue(torch.equal(final[rejected], self.raw[rejected]))

    def test_full_accept_is_sr_and_full_reject_is_raw(self):
        raw = torch.full((1, 1, 4, 4), 2.0)
        sr = torch.full((1, 1, 4, 4), -2.0)
        delta = sr - raw
        accept = torch.full((1, 2, 1, 1), 5.0).expand(1, 2, 4, 4)
        keep_cut, keep_fill = hard_keep(raw, sr, delta, accept)
        self.assertTrue(torch.equal(gated_region(raw, delta, keep_cut, keep_fill), sr))
        reject = torch.full_like(accept, -5.0)
        keep_cut, keep_fill = hard_keep(raw, sr, delta, reject)
        self.assertTrue(torch.equal(gated_region(raw, delta, keep_cut, keep_fill), raw))

    def test_loss_is_finite_on_a_dense_candidate_layout(self):
        with torch.no_grad():
            logits = self.module(self.features, self.raw, self.sr, self.delta, self.edge)
        truth = (torch.rand(2, 1, self.size, self.size) > 0.5).to(torch.float32)
        loss, stats = verifier_loss(logits.detach().requires_grad_(True),
                                    self.raw, self.sr, self.delta, truth)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(stats["present_directions"], 1)
        self.assertGreater(stats["edit_candidates"], 0)

    def test_empty_candidates_are_a_connected_zero_on_a_forward(self):
        raw = torch.randn(2, 1, 6, 6)
        logits = self.module(self.features[:, :, :6, :6], raw, raw,
                             torch.zeros_like(raw), log_softmax_edges(2, 6, 6))
        truth = torch.zeros_like(raw)
        loss, stats = verifier_loss(logits, raw, raw, torch.zeros_like(raw), truth)
        self.assertEqual(stats["edit_candidates"], 0)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    module = RegionCandidateVerifier()
    control = RegionCandidateVerifier(aggregation="center_only")
    report = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "status": "pass" if result.wasSuccessful() else "fail",
        "context_descriptor_channels": CONTEXT_DESCRIPTOR_CHANNELS,
        "center_channels": CENTER_CHANNELS,
        "support_channels": SUPPORT_CHANNELS,
        "support_group_channels": SUPPORT_GROUP_CHANNELS,
        "support_scales": list(SUPPORT_SCALES),
        "support_masks": list(SUPPORT_MASKS),
        "context_directions": list(CONTEXT_DIRECTIONS),
        "aggregation_modes": list(AGGREGATION_MODES),
        "verifier_parameters": sum(p.numel() for p in module.parameters()),
        "context_parameters": sum(p.numel() for p in module.context_projection.parameters()) +
            sum(p.numel() for p in module.cut_context.parameters()) +
            sum(p.numel() for p in module.fill_context.parameters()),
        "v8_parameters": V8_VERIFIER_PARAMETERS,
        "control_parameters": sum(p.numel() for p in control.parameters()),
        "torch": torch.__version__,
        "dependencies": "torch only (v8+v9 tools; no rasterio/scipy/numpy/yaml/dataset/model)",
    }
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
