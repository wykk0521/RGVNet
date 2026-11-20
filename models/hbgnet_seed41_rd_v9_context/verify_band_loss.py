"""Standalone synthetic CPU unit checks for the SR v2 GT band loss.

Run with a working PyTorch installation, from any current directory::

    python /path/to/staging/models/hbgnet_sr_v2/verify_band_loss.py

``run_checks()`` returns JSON-serializable evidence and raises AssertionError
on failure. No model/package initializers, datasets, CUDA, training, or files
are used by the checks. The slow pixel-loop oracle is TEST-ONLY and runs on CPU.
"""

import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest

import torch


def _load_loss_module():
    path = Path(__file__).resolve().parents[2] / "tools" / "region_boundary_band_loss_v2.py"
    spec = importlib.util.spec_from_file_location("_standalone_region_boundary_band_loss_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load band loss module at {path}")
    module = importlib.util.module_from_spec(spec)
    # Loading source directly avoids importing tools/__init__.py or the models.
    # Avoid bytecode artifacts when these checks are run against a frozen tree.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


_loss = _load_loss_module()
region_boundary_band = _loss.region_boundary_band
boundary_band_bce = _loss.boundary_band_bce


def _mask(rows):
    return torch.tensor(
        [[int(value) for value in row] for row in rows], dtype=torch.bool, device="cpu"
    )[None, None]


def _oracle_band(target, radius):
    """Independent set/coordinate implementation, exclusively for tiny CPU GTs."""
    if target.device.type != "cpu":
        raise AssertionError("The standalone oracle is CPU-only")
    batch, _, height, width = target.shape
    labels = target.tolist()
    expected = torch.zeros((batch, 1, height, width), dtype=torch.bool, device="cpu")
    for image in range(batch):
        seeds = set()
        for row in range(height):
            for col in range(width):
                for other_row, other_col in ((row + 1, col), (row, col + 1)):
                    if other_row < height and other_col < width:
                        if labels[image][0][row][col] != labels[image][0][other_row][other_col]:
                            seeds.update(((row, col), (other_row, other_col)))
        for row, col in seeds:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy, xx = row + dy, col + dx
                    if dy * dy + dx * dx <= radius * radius and 0 <= yy < height and 0 <= xx < width:
                        expected[image, 0, yy, xx] = True
    return expected


def _scalar_bce(logit, target):
    # Independent stable scalar formula, not torch BCE or sigmoid/log.
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def _oracle_loss(logits, target, radius):
    band = _oracle_band(target, radius).tolist()
    zs, ys = logits.tolist(), target.tolist()
    batch, _, height, width = target.shape
    images = []
    for image in range(batch):
        classes = [[], []]
        for row in range(height):
            for col in range(width):
                if band[image][0][row][col]:
                    label = int(ys[image][0][row][col])
                    classes[label].append(_scalar_bce(zs[image][0][row][col], label))
        means = [sum(values) / len(values) for values in classes if values]
        images.append(sum(means) / len(means) if means else 0.0)
    return sum(images) / batch


class BandLossChecks(unittest.TestCase):
    def assert_mask(self, actual, expected):
        self.assertEqual(actual.dtype, torch.bool)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.shape, expected.shape)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(torch.equal(actual, expected), f"Actual mask differs:\n{actual}\n{expected}")

    def assert_scalar_close(self, actual, expected, tolerance=1e-12):
        self.assertEqual(actual.ndim, 0)
        self.assertTrue(torch.isfinite(actual).item())
        self.assertAlmostEqual(actual.item(), expected, delta=tolerance)

    def test_uniform_empty_all_zero_all_one(self):
        for height, width in ((1, 1), (1, 8), (7, 1), (7, 8), (8, 7)):
            target = torch.zeros((2, 1, height, width), device="cpu")
            target[1] = 1
            for radius in (0, 1, 2, 4, 10**30):
                with self.subTest(shape=(height, width), radius=radius):
                    self.assert_mask(region_boundary_band(target, radius), torch.zeros_like(target, dtype=torch.bool))
                    # Large finite values must not overflow an empty-loss fallback.
                    logits = torch.full_like(target, 1e38, requires_grad=True)
                    loss = boundary_band_bce(logits, target, radius)
                    self.assert_scalar_close(loss, 0.0)
                    self.assertTrue(loss.requires_grad)
                    loss.backward()
                    self.assertIsNotNone(logits.grad)
                    self.assertTrue(torch.isfinite(logits.grad).all().item())
                    self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_exact_horizontal_vertical_steps_odd_even(self):
        for height, width in ((9, 12), (10, 13)):
            for axis in (-2, -1):
                with self.subTest(shape=(height, width), axis=axis):
                    target = torch.zeros((1, 1, height, width), device="cpu")
                    boundary = target.shape[axis] // 2
                    region = [slice(None)] * 4
                    region[axis] = slice(boundary, None)
                    target[tuple(region)] = 1
                    for radius in (0, 1, 2):
                        expected = torch.zeros_like(target, dtype=torch.bool)
                        region[axis] = slice(boundary - 1 - radius, boundary + 1 + radius)
                        expected[tuple(region)] = True
                        self.assert_mask(region_boundary_band(target, radius), expected)
                    self.assert_mask(region_boundary_band(target), region_boundary_band(target, 2))

    def test_exact_one_pixel_gap_and_foreground_line(self):
        for height, width in ((11, 12), (12, 11)):
            for axis in (-2, -1):
                for fill in (0, 1):
                    with self.subTest(shape=(height, width), axis=axis, fill=fill):
                        target = torch.full((1, 1, height, width), fill, dtype=torch.int64, device="cpu")
                        center = target.shape[axis] // 2
                        region = [slice(None)] * 4
                        region[axis] = center
                        target[tuple(region)] = 1 - fill
                        expected = torch.zeros_like(target, dtype=torch.bool)
                        region[axis] = slice(center - 3, center + 4)
                        expected[tuple(region)] = True
                        # Three seed columns/rows plus two pixels on each side.
                        self.assert_mask(region_boundary_band(target), expected)

    def test_exact_disk2_around_single_pixel(self):
        target = torch.zeros((1, 1, 9, 9), device="cpu")
        target[0, 0, 4, 4] = 1
        expected = _mask([
            "000000000",
            "000010000",
            "000111000",
            "001111100",
            "011111110",
            "001111100",
            "000111000",
            "000010000",
            "000000000",
        ])
        self.assert_mask(region_boundary_band(target), expected)
        # The corner offset (2, 2) is NOT in the Euclidean disk2 dilation.
        self.assertFalse(expected[0, 0, 2, 2].item())
        self.assert_mask(region_boundary_band(1 - target), expected)

    def test_exact_edge_and_corner_clipping(self):
        corner = torch.zeros((1, 1, 7, 8), device="cpu")
        corner[0, 0, 0, 0] = 1
        self.assert_mask(region_boundary_band(corner), _mask([
            "11110000", "11100000", "11000000", "10000000",
            "00000000", "00000000", "00000000",
        ]))
        # A foreground column touching the entire left border does not wrap.
        edge = torch.zeros_like(corner)
        edge[..., 0] = 1
        self.assert_mask(region_boundary_band(edge), _mask(["11110000"] * 7))
        self.assert_mask(region_boundary_band(1 - edge), _mask(["11110000"] * 7))

    def test_one_dimensional_images_and_large_radius(self):
        target = _mask(["000100000"])
        for image in (target, target.transpose(-2, -1)):
            self.assert_mask(region_boundary_band(image, 0), _oracle_band(image, 0))
            self.assert_mask(region_boundary_band(image, 2), _oracle_band(image, 2))
            self.assert_mask(region_boundary_band(image, 10**30), torch.ones_like(image))
        self.assert_mask(region_boundary_band(target), _mask(["111111100"]))
        pair = _mask(["01"])
        self.assert_mask(region_boundary_band(pair, 0), torch.ones_like(pair))

    def test_independent_mask_oracle_diagonals_and_mixed_batches(self):
        for height, width in ((5, 6), (6, 5), (8, 10), (9, 7)):
            yy = torch.arange(height, device="cpu")[:, None]
            xx = torch.arange(width, device="cpu")[None, :]
            target = torch.stack([
                yy == xx,
                (3 * yy + xx) % 7 == 0,
                (yy < height // 2) & (xx < width // 2),
                torch.zeros((height, width), dtype=torch.bool, device="cpu"),
                torch.ones((height, width), dtype=torch.bool, device="cpu"),
            ])[:, None]
            for radius in (0, 1, 2, 4):
                with self.subTest(shape=(height, width), radius=radius):
                    self.assert_mask(region_boundary_band(target, radius), _oracle_band(target, radius))

    def test_flip_rotation_transpose_equivariance(self):
        for height, width in ((7, 10), (8, 9)):
            yy = torch.arange(height, device="cpu")[:, None]
            xx = torch.arange(width, device="cpu")[None, :]
            target = torch.stack([(yy + xx) % 5 == 0, (yy < 3) & (xx > 4)])[:, None]
            logits = torch.linspace(-3, 4, target.numel(), dtype=torch.float64, device="cpu").reshape(target.shape)
            transforms = [
                lambda x: x.flip(-1),
                lambda x: x.flip(-2),
                lambda x: x.transpose(-2, -1),
            ] + [lambda x, k=k: torch.rot90(x, k, (-2, -1)) for k in (1, 2, 3)]
            for radius in (0, 2, 4):
                band = region_boundary_band(target, radius)
                loss = boundary_band_bce(logits, target, radius)
                for index, transform in enumerate(transforms):
                    with self.subTest(shape=(height, width), radius=radius, transform=index):
                        self.assert_mask(region_boundary_band(transform(target), radius), transform(band))
                        self.assert_scalar_close(boundary_band_bce(transform(logits), transform(target), radius), loss.item())

    def test_manual_class_balance_selected_pixel_normalization(self):
        target = torch.ones((1, 1, 3, 9), dtype=torch.float64, device="cpu")
        target[..., 3] = 0  # A one-pixel background gap.
        logits = torch.zeros_like(target)
        logits[..., 3] = math.log(3)
        # Outside-band errors must have neither loss nor gradient contribution.
        logits[..., 7:] = -2500
        logits.requires_grad_()
        expected_band = _mask(["111111100"] * 3)
        self.assert_mask(region_boundary_band(target), expected_band)
        # BG: 3 pixels, BCE=log(4); FG: 18 pixels, BCE=log(2).
        expected = (math.log(4) + math.log(2)) / 2
        loss = boundary_band_bce(logits, target)
        self.assert_scalar_close(loss, expected)
        pixel_weighted = (3 * math.log(4) + 18 * math.log(2)) / 21
        self.assertGreater(abs(loss.item() - pixel_weighted), 0.1)
        loss.backward()
        expected_grad = torch.zeros_like(logits)
        expected_grad[..., 3] = 0.75 / (2 * 3)
        expected_grad[..., :3] = -0.5 / (2 * 18)
        expected_grad[..., 4:7] = -0.5 / (2 * 18)
        torch.testing.assert_close(logits.grad, expected_grad, rtol=1e-12, atol=1e-12)

    def test_empty_images_remain_in_batch_denominator(self):
        active = _mask(["000111111"] * 3).to(dtype=torch.float64)
        active_logits = torch.linspace(-2, 3, active.numel(), dtype=torch.float64, device="cpu").reshape(active.shape).requires_grad_()
        single = boundary_band_bce(active_logits, active)
        single_grad, = torch.autograd.grad(single, active_logits)
        target = torch.cat((active, torch.zeros_like(active), torch.ones_like(active)))
        logits = torch.cat((active_logits.detach(), torch.full_like(active, 8), torch.full_like(active, -8))).requires_grad_()
        loss = boundary_band_bce(logits, target)
        self.assert_scalar_close(loss, single.item() / 3)
        loss.backward()
        torch.testing.assert_close(logits.grad[:1], single_grad / 3, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(logits.grad[1:], torch.zeros_like(logits.grad[1:])))
        # Reordering the batch must not change the normalization.
        self.assert_scalar_close(boundary_band_bce(logits.flip(0), target.flip(0)), loss.item())

    def test_independent_scalar_loss_oracle(self):
        target = torch.cat((
            _mask(["00000001", "00000001", "00000011", "00111111", "00000000"]),
            _mask(["11111111"] * 5),
            _mask(["10000000", "01000000", "00100000", "00010000", "00001000"]),
        )).to(dtype=torch.float64)
        logits = torch.linspace(-9, 7, target.numel(), dtype=torch.float64, device="cpu").reshape(target.shape)
        for radius in (0, 1, 2, 4):
            self.assert_scalar_close(boundary_band_bce(logits, target, radius), _oracle_loss(logits, target, radius))

    def test_confident_wrong_prediction_gradient_signs(self):
        target = _mask(["000111111"] * 3)
        logits = torch.where(target, -8.0, 8.0).to(dtype=torch.float64).requires_grad_()
        loss = boundary_band_bce(logits, target)
        self.assert_scalar_close(loss, math.log1p(math.exp(8)))
        loss.backward()
        band = region_boundary_band(target)
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue((logits.grad[band & ~target] > 0).all().item())
        self.assertTrue((logits.grad[band & target] < 0).all().item())
        self.assertTrue(torch.equal(logits.grad[~band], torch.zeros_like(logits.grad[~band])))
        # One ordinary gradient-descent step moves each selected logit correctly.
        updated = logits.detach() - logits.grad
        self.assertTrue((updated[band & ~target] < 8).all().item())
        self.assertTrue((updated[band & target] > -8).all().item())

    def test_extreme_finite_logits_and_gradient(self):
        target = _mask(["01", "10"])
        for magnitude in (1000.0, 1e30):
            logits = torch.where(target, -magnitude, magnitude).requires_grad_()
            loss = boundary_band_bce(logits, target)
            self.assertTrue(torch.isfinite(loss).item())
            self.assertGreater(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all().item())
            self.assertTrue((logits.grad[target] < 0).all().item())
            self.assertTrue((logits.grad[~target] > 0).all().item())

    def test_dtypes_noncontiguous_inputs_and_immutable_gt(self):
        base = _mask(["00011111", "00001111", "00000111", "00000011", "00000001"])
        dtypes = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                  torch.float16, torch.bfloat16, torch.float32, torch.float64)
        for dtype in dtypes:
            target = base.to(dtype=dtype).transpose(-2, -1)
            original = target.clone()
            self.assertFalse(target.is_contiguous())
            self.assert_mask(region_boundary_band(target), _oracle_band(target, 2))
            logits = torch.zeros(base.shape, dtype=torch.float64, device="cpu").transpose(-2, -1).requires_grad_()
            self.assert_scalar_close(boundary_band_bce(logits, target), math.log(2))
            self.assertTrue(torch.equal(target, original))
        target = base.to(dtype=torch.float64).requires_grad_()
        logits = torch.zeros_like(target, requires_grad=True)
        boundary_band_bce(logits, target).backward()
        self.assertIsNone(target.grad)  # GT is not an optimization variable.
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(target.detach(), base.to(dtype=torch.float64)))

    def test_half_precision_safe_large_band_reduction(self):
        # Both class counts exceed float16's finite range (65,536 and 65,792).
        target = (torch.arange(513, device="cpu") % 2 == 0)[None, None, None].expand(1, 1, 256, 513)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            logits = torch.zeros(target.shape, dtype=dtype, device="cpu", requires_grad=True)
            loss = boundary_band_bce(logits, target)
            expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            self.assertEqual(loss.dtype, expected_dtype)
            self.assert_scalar_close(loss, math.log(2), tolerance=2e-6)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all().item())
            self.assertTrue((logits.grad[target] < 0).all().item())
            self.assertTrue((logits.grad[~target] > 0).all().item())

    def test_autograd_gradcheck(self):
        target = _mask(["001", "011"])
        logits = torch.linspace(-1.5, 1.5, 6, dtype=torch.float64, device="cpu").reshape(1, 1, 2, 3).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda z: boundary_band_bce(z, target), (logits,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_reject_invalid_shapes_types_layouts_and_devices(self):
        valid = torch.zeros((1, 1, 3, 4), device="cpu")
        for target in (None, [[0, 1]], 1, "binary"):
            with self.assertRaises(TypeError):
                region_boundary_band(target)
            with self.assertRaises(TypeError):
                boundary_band_bce(valid, target)
        for shape in ((), (3, 4), (1, 3, 4), (1, 2, 3, 4), (1, 1, 1, 3, 4),
                      (0, 1, 3, 4), (1, 1, 0, 4), (1, 1, 3, 0)):
            invalid = torch.zeros(shape, device="cpu")
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    region_boundary_band(invalid)
                with self.assertRaises(ValueError):
                    boundary_band_bce(valid, invalid)
                with self.assertRaises(ValueError):
                    boundary_band_bce(invalid, valid)
        for shape in ((2, 1, 3, 4), (1, 1, 4, 3), (1, 1, 1, 4)):
            with self.assertRaises(ValueError):
                boundary_band_bce(torch.zeros(shape, device="cpu"), valid)
        for dtype in (torch.int64, torch.uint8, torch.bool, torch.complex64):
            with self.assertRaises(TypeError):
                boundary_band_bce(valid.to(dtype=dtype), valid)
        for logits in (None, [[0.0]], 1.0):
            with self.assertRaises(TypeError):
                boundary_band_bce(logits, valid)
        with self.assertRaises(TypeError):
            region_boundary_band(valid.to(dtype=torch.complex64))
        sparse = valid.to_sparse()
        with self.assertRaises(TypeError):
            region_boundary_band(sparse)
        with self.assertRaises(TypeError):
            boundary_band_bce(sparse, valid)
        meta = torch.empty(valid.shape, device="meta")
        with self.assertRaises(ValueError):
            region_boundary_band(meta)
        with self.assertRaises(ValueError):
            boundary_band_bce(meta, valid)
        with self.assertRaises(ValueError):
            boundary_band_bce(valid, meta)

    def test_reject_invalid_targets_and_all_nonfinite_logits(self):
        valid = torch.zeros((1, 1, 3, 4), dtype=torch.float64, device="cpu")
        for value in (-1.0, 2.0, 0.5, 1e-12, float("nan"), float("inf"), -float("inf")):
            invalid = valid.clone()
            invalid[0, 0, 1, 1] = value
            with self.subTest(target_value=value):
                with self.assertRaises(ValueError):
                    region_boundary_band(invalid)
                with self.assertRaises(ValueError):
                    boundary_band_bce(valid, invalid)
        active = valid.clone()
        active[..., 2:] = 1
        for target in (valid, torch.ones_like(valid), active):
            for value in (float("nan"), float("inf"), -float("inf")):
                invalid = valid.clone()
                # Invalid values must be rejected even with an empty band.
                invalid[0, 0, 0, 0] = value
                with self.assertRaises(ValueError):
                    boundary_band_bce(invalid, target)
        # Also reject a nonfinite value outside a nonempty band.
        wide = _mask(["000111111111"])
        logits = torch.zeros(wide.shape, device="cpu")
        logits[..., -1] = float("nan")
        with self.assertRaises(ValueError):
            boundary_band_bce(logits, wide)

    def test_reject_invalid_radius_in_both_apis(self):
        target = _mask(["01"])
        logits = torch.zeros(target.shape, device="cpu")
        for radius in (True, False, 2.0, 2.5, "2", None, torch.tensor(2), torch.tensor(True)):
            with self.subTest(radius=repr(radius)):
                with self.assertRaises(TypeError):
                    region_boundary_band(target, radius)
                with self.assertRaises(TypeError):
                    boundary_band_bce(logits, target, radius)
        for radius in (-1, -2):
            with self.assertRaises(ValueError):
                region_boundary_band(target, radius)
            with self.assertRaises(ValueError):
                boundary_band_bce(logits, target, radius)


def run_checks():
    """Run synthetic CPU checks; return evidence or raise with failure details."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BandLossChecks)
    checks = [test._testMethodName for test in suite]
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    evidence = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "checks": checks,
        "datasets_used": False,
        "gpu_executed": False,
    }
    if not result.wasSuccessful():
        raise AssertionError(json.dumps(evidence, indent=2) + "\n" + output.getvalue())
    return evidence


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
