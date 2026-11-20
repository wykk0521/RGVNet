"""Unit tests for targets.py (FHAPD three-channel target builder).

Run from anywhere (dependencies permitting):

    py -3 -B -m unittest discover -s <this directory> -t <this directory> -p "test_targets.py"

The FHAPD builder needs numpy and OpenCV.  When either is missing the whole
module is reported as skipped with the exact missing dependency instead of
erroring, so the remaining tests in this directory can still run.

The exact reference operations under test are the BsiNet preprocess steps
copied into targets.py:

    region   = source uint8 0/1 mask as float32
    boundary = cv2.Canny(mask * 255, 100, 200) > 0
    distance = per-tile min-max of cv2.distanceTransform(mask * 255, L2, 3),
               truncating uint8 quantization, /255; constant maps -> zeros
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
    import numpy as np
    import targets
except ImportError as exc:
    cv2 = None
    np = None
    targets = None
    DEPENDENCY_ERROR = "%s: %s" % (type(exc).__name__, exc)
else:
    DEPENDENCY_ERROR = ""

requires_deps = unittest.skipUnless(
    DEPENDENCY_ERROR == "", "missing dependency: " + DEPENDENCY_ERROR
)


def rectangle_mask(height=32, width=32, r0=8, r1=24, c0=8, c1=24):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    return mask


def reference_targets(mask):
    """Independent reimplementation of the documented BsiNet preprocessing."""
    binary = mask.astype(np.uint8) * 255
    contour = cv2.Canny(binary, 100, 200)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    low = float(distance.min())
    high = float(distance.max())
    if high > low:
        quantized = ((distance - low) / (high - low) * 255.0).astype(np.uint8)
    else:
        quantized = np.zeros(mask.shape, dtype=np.uint8)
    return np.stack(
        (
            mask.astype(np.float32),
            (contour > 0).astype(np.float32),
            quantized.astype(np.float32) / 255.0,
        )
    )


@requires_deps
class TestReferenceOperations(unittest.TestCase):
    def setUp(self):
        self.mask = rectangle_mask()
        self.output = targets.build_targets(self.mask)
        self.reference = reference_targets(self.mask)

    def test_output_contract(self):
        self.assertEqual(self.output.shape, (3,) + self.mask.shape)
        self.assertEqual(self.output.dtype, np.float32)
        self.assertTrue(np.isfinite(self.output).all())

    def test_region_channel_is_source_mask(self):
        np.testing.assert_array_equal(self.output[0], self.mask.astype(np.float32))

    def test_boundary_matches_reference_canny(self):
        np.testing.assert_array_equal(self.output[1], self.reference[1])

    def test_boundary_is_binary(self):
        self.assertTrue(np.isin(self.output[1], (0.0, 1.0)).all())

    def test_boundary_present_for_0_1_source(self):
        # A tile with an interior parcel must produce a non-empty boundary.
        self.assertGreater(int(self.output[1].sum()), 0)
        # Documented bug this catches: Canny run directly on a 0/1 image never
        # reaches the low threshold 100 (a 0/1 step has Sobel response <= 4),
        # so a 0/1-source boundary would silently be empty.
        naive = cv2.Canny(self.mask, 100, 200)
        self.assertEqual(int(naive.sum()), 0)

    def test_distance_matches_reference(self):
        np.testing.assert_array_equal(self.output[2], self.reference[2])

    def test_distance_range_and_quantization_grid(self):
        self.assertGreaterEqual(float(self.output[2].min()), 0.0)
        self.assertLessEqual(float(self.output[2].max()), 1.0)
        scaled = self.output[2].astype(np.float64) * 255.0
        np.testing.assert_allclose(scaled, np.round(scaled), atol=1e-4)

    def test_distance_minmax_span(self):
        # A non-constant distance map spans the full 0..255 code range.
        self.assertEqual(float(self.output[2].min()), 0.0)
        self.assertEqual(float(self.output[2].max()), 1.0)

    def test_distance_quantization_truncates(self):
        # 5x5 tile with a 3x3 block: L2 distances are exactly 1 on the ring
        # and 2 in the centre, so min-max scaling gives 127.5 -> truncated
        # uint8 127 (not rounded 128) and 255 for the centre.
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        output = targets.build_targets(mask)
        self.assertLess(abs(float(output[2, 2, 2]) - 1.0), 1e-6)
        self.assertLess(abs(float(output[2, 1, 1]) * 255.0 - 127.0), 1e-3)
        self.assertLess(abs(float(output[2, 0, 0])), 1e-7)
        np.testing.assert_array_equal(output, reference_targets(mask))

    def test_input_not_modified(self):
        before = self.mask.copy()
        targets.build_targets(self.mask)
        np.testing.assert_array_equal(self.mask, before)


@requires_deps
class TestDegenerateMasks(unittest.TestCase):
    def test_empty_mask_is_finite_with_zero_distance_guard(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        output = targets.build_targets(mask)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(output[0], np.zeros((16, 16), dtype=np.float32))
        np.testing.assert_array_equal(output[1], np.zeros((16, 16), dtype=np.float32))
        np.testing.assert_array_equal(output[2], np.zeros((16, 16), dtype=np.float32))
        np.testing.assert_array_equal(output, reference_targets(mask))

    def test_full_mask_stays_finite(self):
        mask = np.ones((16, 16), dtype=np.uint8)
        output = targets.build_targets(mask)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue((output[0] == 1.0).all())
        self.assertTrue(((output[2] >= 0.0) & (output[2] <= 1.0)).all())
        np.testing.assert_array_equal(output, reference_targets(mask))


@requires_deps
class TestIllegalMasks(unittest.TestCase):
    def test_float_mask_rejected(self):
        with self.assertRaises(ValueError):
            targets.build_targets(np.zeros((8, 8), dtype=np.float32))
        with self.assertRaises(ValueError):
            targets.build_targets(np.array([[0.0, 1.0]], dtype=np.float64))

    def test_non_uint8_integer_rejected(self):
        for dtype in (np.int8, np.int32, np.int64, np.uint16):
            with self.assertRaises(ValueError):
                targets.build_targets(np.zeros((4, 4), dtype=dtype))

    def test_bool_mask_rejected(self):
        with self.assertRaises(ValueError):
            targets.build_targets(np.zeros((4, 4), dtype=bool))

    def test_255_encoding_rejected(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = 255
        with self.assertRaises(ValueError):
            targets.build_targets(mask)

    def test_values_above_one_rejected(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 2
        with self.assertRaises(ValueError):
            targets.build_targets(mask)

    def test_wrong_ndim_rejected(self):
        with self.assertRaises(ValueError):
            targets.build_targets(np.zeros((4, 4, 1), dtype=np.uint8))
        with self.assertRaises(ValueError):
            targets.build_targets(np.zeros((4,), dtype=np.uint8))

    def test_list_input_rejected(self):
        with self.assertRaises(ValueError):
            targets.build_targets([[0, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
