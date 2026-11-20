"""Unit tests for evaluation_adapter.py (streaming FHAPD split evaluator).

Run from anywhere (dependencies permitting):

    py -3 -B -m unittest discover -s <this directory> -t <this directory> -p "test_evaluation_adapter.py"

The adapter itself needs numpy; the shared metric functions in metrics.py
additionally need scikit-image.  Tests that only exercise settings/count/id/
shape/encoding validation run with numpy alone; aggregation tests are skipped
with the exact missing dependency when scikit-image is unavailable.  No model,
checkpoint, file or CLI is involved: every fixture is a synthetic in-memory
array and the configured ``dataset.input_size`` is deliberately small (8x8 or
10x10) so the adapter must read the size from the config.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    np = None
    NUMPY_ERROR = "%s: %s" % (type(exc).__name__, exc)
else:
    NUMPY_ERROR = ""

try:
    import evaluation_adapter as ea
except ImportError as exc:  # pragma: no cover - dependency guard
    ea = None
    ADAPTER_ERROR = "%s: %s" % (type(exc).__name__, exc)
else:
    ADAPTER_ERROR = ""

try:
    import skimage  # noqa: F401
except ImportError as exc:  # pragma: no cover - dependency guard
    SKIMAGE_ERROR = "%s: %s" % (type(exc).__name__, exc)
else:
    SKIMAGE_ERROR = ""

ADAPTER_READY = ea is not None
METRICS_READY = ADAPTER_READY and SKIMAGE_ERROR == ""

requires_adapter = unittest.skipUnless(
    ADAPTER_READY, "evaluation_adapter unavailable: " + (ADAPTER_ERROR or NUMPY_ERROR)
)
requires_metrics = unittest.skipUnless(
    METRICS_READY, "shared metrics unavailable: " + (SKIMAGE_ERROR or ADAPTER_ERROR)
)

PROTOCOL = "HBGNet-metrics-FHAPD-256-3px-spatial-v1"
AGGREGATION = "region_micro_boundary_micro_object_global_area_weighted"
SPLIT_COUNTS = {"train": 2, "val": 3, "test": 3014}


def make_config(size=(8, 8), tolerance=3, counts=None, **evaluation_overrides):
    evaluation = {
        "protocol": PROTOCOL,
        "region_threshold": 0.5,
        "threshold_comparison": ">",
        "tta": False,
        "small_object_removal": False,
        "aggregation": AGGREGATION,
        "spatial_resolution_m": None,
        "boundary_tolerance_m": None,
        "requested_boundary_tolerance_m": 3.0,
        "boundary_tolerance_pixels": tolerance,
        "physical_tolerance_verified": False,
    }
    evaluation.update(evaluation_overrides)
    return {
        "dataset": {
            "input_size": list(size),
            "expected_counts": dict(counts or SPLIT_COUNTS),
        },
        "evaluation": evaluation,
    }


def one_sample_config(**kwargs):
    return make_config(counts={"train": 1, "val": 1, "test": 3014}, **kwargs)


def square(size=8, r0=2, r1=6, c0=2, c1=6):
    mask = np.zeros((size, size), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def as_probability(mask, fg=0.9, bg=0.1):
    probability = np.full(mask.shape, bg, dtype=np.float64)
    probability[mask] = fg
    return probability


def sample(ident, gt=None, probability=None):
    gt = square() if gt is None else gt
    if probability is None:
        probability = as_probability(gt)
    return (ident, probability, gt)


def load_metrics():
    import metrics

    return metrics


@requires_adapter
class TestValidateEvaluationSettings(unittest.TestCase):
    def test_default_settings_resolved(self):
        settings = ea.validate_evaluation_settings(make_config())
        self.assertEqual(settings["protocol"], PROTOCOL)
        self.assertEqual(settings["region_threshold"], 0.5)
        self.assertEqual(settings["threshold_comparison"], ">")
        self.assertIs(settings["tta"], False)
        self.assertIs(settings["small_object_removal"], False)
        self.assertEqual(settings["aggregation"], AGGREGATION)
        self.assertIsNone(settings["spatial_resolution_m"])
        self.assertIsNone(settings["boundary_tolerance_m"])
        self.assertEqual(settings["requested_boundary_tolerance_m"], 3.0)
        self.assertEqual(settings["boundary_tolerance_pixels"], 3)
        self.assertIs(settings["physical_tolerance_verified"], False)
        self.assertEqual(settings["input_size"], (8, 8))

    def test_configured_input_size_is_read(self):
        settings = ea.validate_evaluation_settings(make_config(size=(256, 256)))
        self.assertEqual(settings["input_size"], (256, 256))

    def test_threshold_must_be_exactly_half(self):
        for value in (0.6, 0.4999, 0.5000001, 0.0, 1.0):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.validate_evaluation_settings(make_config(region_threshold=value))
        with self.assertRaises(ea.EvaluationConfigError):
            ea.validate_evaluation_settings(make_config(region_threshold=True))

    def test_threshold_comparison_must_be_strict_greater(self):
        for value in (">=", "<", "=>", "greater"):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.validate_evaluation_settings(make_config(threshold_comparison=value))

    def test_tta_must_be_false(self):
        for value in (True, 1, "false"):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.validate_evaluation_settings(make_config(tta=value))

    def test_small_object_removal_must_be_false(self):
        for value in (True, 1):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.validate_evaluation_settings(make_config(small_object_removal=value))

    def test_aggregation_string_is_fixed(self):
        for value in ("region_micro", AGGREGATION + "_x", None):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.validate_evaluation_settings(make_config(aggregation=value))

    def test_protocol_string_is_fixed(self):
        for value in ("HBGNet-metrics-other", None):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.validate_evaluation_settings(make_config(protocol=value))

    def test_missing_evaluation_mapping_rejected(self):
        with self.assertRaises(ea.EvaluationConfigError):
            ea.validate_evaluation_settings({})
        with self.assertRaises(ea.EvaluationConfigError):
            ea.validate_evaluation_settings({"evaluation": None})
        with self.assertRaises(ea.EvaluationConfigError):
            ea.validate_evaluation_settings({"evaluation": []})
        with self.assertRaises(ea.EvaluationConfigError):
            ea.validate_evaluation_settings(None)

    def test_input_size_from_variant_resolver(self):
        config = make_config()
        del config["dataset"]["input_size"]
        config["dataset"]["variant"] = "fhapd_xj"
        config["dataset"]["variants"] = {"fhapd_xj": {"input_size": [12, 10]}}
        settings = ea.validate_evaluation_settings(config)
        self.assertEqual(settings["input_size"], (12, 10))
        self.assertEqual(ea.resolve_input_size(config), (12, 10))

    def test_unknown_or_missing_variant_rejected(self):
        config = make_config()
        del config["dataset"]["input_size"]
        config["dataset"]["variant"] = "nope"
        config["dataset"]["variants"] = {"fhapd_xj": {"input_size": [12, 10]}}
        with self.assertRaises(ea.EvaluationConfigError):
            ea.resolve_input_size(config)

        config = make_config()
        del config["dataset"]["input_size"]
        config["dataset"]["variants"] = {"fhapd_xj": {"input_size": [12, 10]}}
        with self.assertRaises(ea.EvaluationConfigError):
            ea.resolve_input_size(config)

    def test_invalid_input_size_rejected(self):
        for value in ([0, 8], [8], [8, 8, 1], "8x8", [True, 8], [8.5, 8], [8, -8]):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.validate_evaluation_settings(make_config(size=value))


@requires_adapter
class TestResolveBoundaryTolerance(unittest.TestCase):
    def test_default_is_three_pixels(self):
        tolerance = ea.resolve_boundary_tolerance(make_config())
        self.assertIsInstance(tolerance, int)
        self.assertEqual(tolerance, 3)

    def test_zero_pixels_allowed(self):
        self.assertEqual(ea.resolve_boundary_tolerance(make_config(tolerance=0)), 0)

    def test_integral_float_allowed(self):
        self.assertEqual(ea.resolve_boundary_tolerance(make_config(tolerance=3.0)), 3)

    def test_boolean_rejected(self):
        for value in (True, False):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.resolve_boundary_tolerance(make_config(tolerance=value))

    def test_negative_rejected(self):
        for value in (-1, -0.5):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.resolve_boundary_tolerance(make_config(tolerance=value))

    def test_non_integral_rejected(self):
        for value in (2.5, 0.25):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.resolve_boundary_tolerance(make_config(tolerance=value))

    def test_non_number_rejected(self):
        for value in ("3", None):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.resolve_boundary_tolerance(make_config(tolerance=value))

    def test_unverified_tolerance_forbids_physical_fields(self):
        with self.assertRaises(ea.EvaluationConfigError):
            ea.resolve_boundary_tolerance(make_config(spatial_resolution_m=0.5))
        with self.assertRaises(ea.EvaluationConfigError):
            ea.resolve_boundary_tolerance(make_config(boundary_tolerance_m=1.5))

    def test_three_px_is_not_three_m_while_unverified(self):
        with self.assertRaises(ea.EvaluationConfigError) as ctx:
            ea.resolve_boundary_tolerance(make_config(boundary_tolerance_m=3.0))
        self.assertIn("3 px", str(ctx.exception))

    def test_verified_conversion_valid(self):
        self.assertEqual(
            ea.resolve_boundary_tolerance(
                make_config(
                    tolerance=3,
                    spatial_resolution_m=0.5,
                    boundary_tolerance_m=1.5,
                    physical_tolerance_verified=True,
                )
            ),
            3,
        )
        self.assertEqual(
            ea.resolve_boundary_tolerance(
                make_config(
                    tolerance=3,
                    spatial_resolution_m=1.0,
                    boundary_tolerance_m=3.0,
                    physical_tolerance_verified=True,
                )
            ),
            3,
        )
        self.assertEqual(
            ea.resolve_boundary_tolerance(
                make_config(
                    tolerance=2,
                    spatial_resolution_m=0.5,
                    boundary_tolerance_m=1.0,
                    physical_tolerance_verified=True,
                )
            ),
            2,
        )

    def test_verified_conversion_mismatch_rejected(self):
        with self.assertRaises(ea.EvaluationConfigError):
            ea.resolve_boundary_tolerance(
                make_config(
                    tolerance=3,
                    spatial_resolution_m=0.5,
                    boundary_tolerance_m=3.0,
                    physical_tolerance_verified=True,
                )
            )

    def test_verified_non_integral_pixel_conversion_rejected(self):
        with self.assertRaises(ea.EvaluationConfigError):
            ea.resolve_boundary_tolerance(
                make_config(
                    tolerance=3,
                    spatial_resolution_m=0.3,
                    boundary_tolerance_m=1.0,
                    physical_tolerance_verified=True,
                )
            )

    def test_verified_requires_both_physical_fields(self):
        with self.assertRaises(ea.EvaluationConfigError):
            ea.resolve_boundary_tolerance(
                make_config(
                    tolerance=3,
                    spatial_resolution_m=None,
                    boundary_tolerance_m=1.5,
                    physical_tolerance_verified=True,
                )
            )
        with self.assertRaises(ea.EvaluationConfigError):
            ea.resolve_boundary_tolerance(
                make_config(
                    tolerance=3,
                    spatial_resolution_m=0.5,
                    boundary_tolerance_m=None,
                    physical_tolerance_verified=True,
                )
            )

    def test_verified_flag_must_be_boolean(self):
        for value in ("false", 1, 0):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.resolve_boundary_tolerance(make_config(physical_tolerance_verified=value))

    def test_requested_tolerance_must_be_positive(self):
        for value in (-3.0, 0.0, None, float("nan")):
            with self.assertRaises(ea.EvaluationConfigError):
                ea.resolve_boundary_tolerance(
                    make_config(requested_boundary_tolerance_m=value)
                )


@requires_adapter
class TestAccumulatorContract(unittest.TestCase):
    """Failures that must be raised before any shared metric is computed."""

    def test_count_mismatch_against_3014(self):
        config = make_config(counts={"train": 1, "val": 1, "test": 3014})
        with self.assertRaises(ea.EvaluationAdapterError) as ctx:
            ea.accumulate_prediction_masks(config, ["a", "b"], [], "test")
        self.assertIn("3014", str(ctx.exception))

    def test_ids_length_must_match_expected_count(self):
        config = make_config(counts={"train": 1, "val": 2, "test": 3014})
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(config, ["a"], [], "val")

    def test_missing_expected_count_for_split_rejected(self):
        config = make_config(counts={"train": 1, "test": 3014})
        with self.assertRaises(ea.EvaluationConfigError):
            ea.accumulate_prediction_masks(config, [], [], "val")

    def test_unknown_split_rejected(self):
        with self.assertRaises(ea.EvaluationConfigError):
            ea.accumulate_prediction_masks(make_config(), [], [], "validation")

    def test_duplicate_ids_rejected(self):
        config = make_config(counts={"train": 1, "val": 2, "test": 3014})
        with self.assertRaises(ea.EvaluationAdapterError) as ctx:
            ea.accumulate_prediction_masks(config, ["a", "a"], [], "val")
        self.assertIn("duplicate", str(ctx.exception))

    def test_ids_must_be_a_list_not_string(self):
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(one_sample_config(), "a", [], "val")
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(one_sample_config(), {"a": 1}, [], "val")

    def test_non_clean_ids_rejected(self):
        bad_ids = (123, "", " a", "a ", "a/b", "a\\b", ".", "..", "a\x00b")
        for bad in bad_ids:
            with self.assertRaises(ea.EvaluationAdapterError):
                ea.accumulate_prediction_masks(one_sample_config(), [bad], [], "val")

    def test_stream_item_must_be_triple(self):
        config = one_sample_config()
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(config, ["a"], [("a", np.zeros((8, 8)), True, "x")], "val")
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(config, ["a"], ["not-a-tuple"], "val")

    def test_stream_id_order_mismatch_rejected(self):
        config = one_sample_config()
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(config, ["a"], [sample("b")], "val")

    def test_stream_ending_early_rejected(self):
        config = one_sample_config()
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(config, ["a"], [], "val")

    def test_extra_stream_item_rejected_for_empty_split(self):
        config = make_config(counts={"train": 1, "val": 0, "test": 3014})
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(
                config, [], [("a", np.zeros((8, 8)), np.zeros((8, 8), dtype=bool))], "val"
            )

    def test_shape_mismatch_rejected_before_metrics(self):
        config = one_sample_config()
        with self.assertRaises(ea.EvaluationAdapterError) as ctx:
            ea.accumulate_prediction_masks(
                config, ["a"], [("a", np.zeros((7, 8)), np.zeros((8, 8), dtype=bool))], "val"
            )
        self.assertIn("input_size", str(ctx.exception))

        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(
                config, ["a"], [("a", np.zeros((8, 8, 1)), np.zeros((8, 8), dtype=bool))], "val"
            )
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(
                config, ["a"], [("a", np.zeros((8, 8)), np.zeros((7, 8), dtype=bool))], "val"
            )

    def test_probability_non_finite_rejected(self):
        config = one_sample_config()
        for bad in (float("nan"), float("inf"), float("-inf")):
            probability = np.full((8, 8), 0.5, dtype=np.float64)
            probability[0, 0] = bad
            with self.assertRaises(ea.EvaluationAdapterError):
                ea.accumulate_prediction_masks(
                    config, ["a"], [("a", probability, square())], "val"
                )

    def test_probability_out_of_range_and_logits_rejected(self):
        config = one_sample_config()
        for value in (1.0001, -0.001, 2.0):
            probability = np.zeros((8, 8), dtype=np.float64)
            probability[0, 0] = value
            with self.assertRaises(ea.EvaluationAdapterError):
                ea.accumulate_prediction_masks(
                    config, ["a"], [("a", probability, square())], "val"
                )
        logits = np.zeros((8, 8), dtype=np.float32)
        logits[0, 0] = -2.5
        logits[0, 1] = 3.0
        with self.assertRaises(ea.EvaluationAdapterError) as ctx:
            ea.accumulate_prediction_masks(config, ["a"], [("a", logits, square())], "val")
        self.assertIn("logits", str(ctx.exception))

    def test_probability_integer_dtype_rejected(self):
        config = one_sample_config()
        for array in (np.zeros((8, 8), dtype=np.uint8), np.ones((8, 8), dtype=np.int64)):
            with self.assertRaises(ea.EvaluationAdapterError):
                ea.accumulate_prediction_masks(config, ["a"], [("a", array, square())], "val")

    def test_ground_truth_255_encoding_rejected(self):
        config = one_sample_config()
        truth = np.zeros((8, 8), dtype=np.uint8)
        truth[2:6, 2:6] = 255
        with self.assertRaises(ea.EvaluationAdapterError) as ctx:
            ea.accumulate_prediction_masks(
                config, ["a"], [("a", as_probability(square()), truth)], "val"
            )
        self.assertIn("255", str(ctx.exception))

    def test_ground_truth_non_binary_rejected(self):
        config = one_sample_config()
        for truth in (
            np.full((8, 8), 0.5, dtype=np.float64),
            np.full((8, 8), 2, dtype=np.int32),
            np.full((8, 8), float("nan"), dtype=np.float64),
            np.full((8, 8), "0", dtype="<U1"),
        ):
            with self.assertRaises(ea.EvaluationAdapterError):
                ea.accumulate_prediction_masks(config, ["a"], [("a", square(), truth)], "val")

    def test_invalid_settings_fail_before_stream(self):
        config = one_sample_config(tta=True)
        with self.assertRaises(ea.EvaluationConfigError):
            ea.accumulate_prediction_masks(config, ["a"], ["garbage"], "val")

    def test_non_iterable_predictions_rejected(self):
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(one_sample_config(), ["a"], 42, "val")


@requires_metrics
class TestAccumulatorMetrics(unittest.TestCase):
    def test_perfect_sample_report(self):
        config = one_sample_config()
        gt = square()
        report = ea.accumulate_prediction_masks(
            config, ["tile_0"], [sample("tile_0", gt=gt)], "val"
        )
        self.assertTrue(report["complete_split"])
        self.assertEqual(report["counts"], {"expected": 1, "evaluated": 1})
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["protocol"], PROTOCOL)
        self.assertEqual(report["aggregation"], AGGREGATION)
        self.assertEqual(report["boundary_tolerance_pixels"], 3)
        metrics = report["metrics"]
        self.assertAlmostEqual(metrics["f1"], 1.0, places=9)
        self.assertAlmostEqual(metrics["iou"], 1.0, places=9)
        self.assertAlmostEqual(metrics["f_boundary"], 1.0, places=9)
        self.assertAlmostEqual(metrics["oa"], 1.0, places=9)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=9)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=9)
        self.assertAlmostEqual(metrics["boundary_completeness"], 1.0, places=9)
        self.assertAlmostEqual(metrics["boundary_correctness"], 1.0, places=9)
        self.assertEqual(metrics["goc"], 0.0)
        self.assertEqual(metrics["guc"], 0.0)
        self.assertEqual(metrics["gtc"], 0.0)
        self.assertEqual(report["pixel_counts"], [int(gt.sum()), 0, 0, int((~gt).sum())])
        self.assertEqual(report["predicted_object_area"], int(gt.sum()))
        self.assertEqual(len(report["per_image"]), 1)
        self.assertEqual(report["per_image"][0]["id"], "tile_0")
        json.dumps(report, allow_nan=False)

    def test_strict_greater_than_half(self):
        gt = square()
        half = np.full((8, 8), 0.5, dtype=np.float64)
        report = ea.accumulate_prediction_masks(
            one_sample_config(), ["a"], [("a", half, gt)], "val"
        )
        self.assertEqual(report["pixel_counts"], [0, 0, int(gt.sum()), int((~gt).sum())])
        self.assertAlmostEqual(report["metrics"]["oa"], (64 - int(gt.sum())) / 64.0, places=12)

        above = np.full((8, 8), 0.5 + 1e-9, dtype=np.float64)
        report = ea.accumulate_prediction_masks(
            one_sample_config(), ["a"], [("a", above, gt)], "val"
        )
        self.assertEqual(report["pixel_counts"], [int(gt.sum()), 64 - int(gt.sum()), 0, 0])

        half32 = np.full((8, 8), 0.5, dtype=np.float32)
        report = ea.accumulate_prediction_masks(
            one_sample_config(), ["a"], [("a", half32, gt)], "val"
        )
        self.assertEqual(report["pixel_counts"][0], 0)
        self.assertEqual(report["pixel_counts"][2], int(gt.sum()))

    def test_boolean_mask_is_used_directly(self):
        gt = square()
        decision = gt.copy()
        report = ea.accumulate_prediction_masks(
            one_sample_config(), ["a"], [("a", decision, gt)], "val"
        )
        self.assertEqual(report["pixel_counts"], [int(gt.sum()), 0, 0, int((~gt).sum())])
        self.assertAlmostEqual(report["metrics"]["f1"], 1.0, places=9)

    def test_empty_prediction_and_empty_truth(self):
        gt = np.zeros((8, 8), dtype=bool)
        probability = np.zeros((8, 8), dtype=np.float64)
        report = ea.accumulate_prediction_masks(
            one_sample_config(), ["a"], [("a", probability, gt)], "val"
        )
        self.assertEqual(report["pixel_counts"], [0, 0, 0, 64])
        metrics = report["metrics"]
        self.assertEqual(metrics["oa"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["boundary_completeness"], 1.0)
        self.assertEqual(metrics["boundary_correctness"], 1.0)
        self.assertEqual(metrics["f_boundary"], 1.0)
        self.assertIsNone(metrics["goc"])
        self.assertIsNone(metrics["guc"])
        self.assertIsNone(metrics["gtc"])
        self.assertEqual(report["predicted_object_area"], 0)
        self.assertEqual(report["boundary_counts"], [0, 0, 0, 0])
        self.assertEqual(report["per_image"][0]["object_numerators"], [0.0, 0.0, 0.0])
        json.dumps(report, allow_nan=False)

    def test_false_negatives_with_empty_prediction(self):
        gt = square()
        prediction = np.zeros((8, 8), dtype=bool)
        report = ea.accumulate_prediction_masks(
            one_sample_config(), ["a"], [("a", prediction, gt)], "val"
        )
        self.assertEqual(report["pixel_counts"], [0, 0, int(gt.sum()), int((~gt).sum())])
        self.assertIsNone(report["metrics"]["goc"])
        self.assertIsNone(report["metrics"]["guc"])
        self.assertIsNone(report["metrics"]["gtc"])
        self.assertEqual(report["metrics"]["boundary_completeness"], 0.0)
        self.assertEqual(report["metrics"]["boundary_correctness"], 0.0)

    def test_false_positives_with_empty_truth(self):
        gt = np.zeros((8, 8), dtype=bool)
        prediction = square()
        report = ea.accumulate_prediction_masks(
            one_sample_config(), ["a"], [("a", prediction, gt)], "val"
        )
        self.assertEqual(report["pixel_counts"], [0, int(prediction.sum()), 0, 64 - int(prediction.sum())])
        self.assertEqual(report["predicted_object_area"], int(prediction.sum()))
        self.assertAlmostEqual(report["metrics"]["goc"], 1.0, places=9)
        self.assertAlmostEqual(report["metrics"]["guc"], 1.0, places=9)
        self.assertAlmostEqual(report["metrics"]["gtc"], 1.0, places=9)
        self.assertEqual(report["metrics"]["boundary_completeness"], 0.0)
        self.assertEqual(report["metrics"]["boundary_correctness"], 0.0)

    def test_object_error_is_global_area_weighted_not_per_image_mean(self):
        config = make_config(counts={"train": 1, "val": 2, "test": 3014})
        gt_a = square(8, 2, 6, 2, 6)
        gt_b = np.zeros((8, 8), dtype=bool)
        gt_b[0, 0] = True
        prediction_b = np.zeros((8, 8), dtype=bool)
        prediction_b[4:6, 4:6] = True
        report = ea.accumulate_prediction_masks(
            config,
            ["a", "b"],
            [("a", as_probability(gt_a), gt_a), ("b", prediction_b, gt_b)],
            "val",
        )
        # a: perfect parcel (16 px, error 0); b: one disjunct 4 px parcel with
        # error 1 per numerator.  Area weighted = 4 / 20, not the mean 0.5.
        self.assertAlmostEqual(report["metrics"]["goc"], 0.2, places=9)
        self.assertAlmostEqual(report["metrics"]["guc"], 0.2, places=9)
        self.assertAlmostEqual(report["metrics"]["gtc"], 0.2, places=9)
        self.assertGreater(abs(report["metrics"]["goc"] - 0.5), 0.2)
        entries = {entry["id"]: entry for entry in report["per_image"]}
        self.assertEqual(entries["a"]["object_numerators"], [0.0, 0.0, 0.0])
        self.assertEqual(entries["b"]["object_numerators"], [4.0, 4.0, 4.0])
        self.assertEqual(entries["a"]["predicted_object_area"], 16)
        self.assertEqual(entries["b"]["predicted_object_area"], 4)
        self.assertEqual(report["predicted_object_area"], 20)

    def test_boundary_tolerance_is_taken_from_config(self):
        size = 10
        gt = square(size, 2, 8, 2, 8)
        prediction = np.zeros((size, size), dtype=bool)
        prediction[2:8, 4:10] = True
        strict = ea.accumulate_prediction_masks(
            make_config(size=(size, size), counts={"train": 1, "val": 1, "test": 3014}, tolerance=0),
            ["a"],
            [("a", prediction, gt)],
            "val",
        )
        tolerant = ea.accumulate_prediction_masks(
            make_config(size=(size, size), counts={"train": 1, "val": 1, "test": 3014}, tolerance=3),
            ["a"],
            [("a", prediction, gt)],
            "val",
        )
        self.assertEqual(strict["boundary_tolerance_pixels"], 0)
        self.assertEqual(tolerant["boundary_tolerance_pixels"], 3)
        self.assertLess(strict["metrics"]["boundary_completeness"], 1.0)
        self.assertAlmostEqual(tolerant["metrics"]["boundary_completeness"], 1.0, places=9)
        self.assertAlmostEqual(tolerant["metrics"]["boundary_correctness"], 1.0, places=9)
        metrics = load_metrics()
        self.assertEqual(
            strict["boundary_counts"], list(metrics.boundary_statistics(prediction, gt, 0))
        )
        self.assertEqual(
            tolerant["boundary_counts"], list(metrics.boundary_statistics(prediction, gt, 3))
        )

    def test_per_image_counts_recompute_split_metrics(self):
        metrics = load_metrics()
        config = make_config(counts={"train": 1, "val": 3, "test": 3014})
        gt_a = square(8, 2, 6, 2, 6)
        gt_b = np.zeros((8, 8), dtype=bool)
        prediction_b = np.zeros((8, 8), dtype=bool)
        prediction_b[0:3, 0:3] = True
        gt_c = square(8, 4, 7, 4, 7)
        prediction_c = np.zeros((8, 8), dtype=bool)
        report = ea.accumulate_prediction_masks(
            config,
            ["a", "b", "c"],
            [
                ("a", as_probability(gt_a), gt_a),
                ("b", prediction_b, gt_b),
                ("c", prediction_c, gt_c),
            ],
            "val",
        )
        entries = report["per_image"]
        self.assertEqual(len(entries), 3)
        for key in ("pixel_counts", "boundary_counts", "predicted_object_area", "object_numerators"):
            self.assertIn(key, entries[0])

        pixels = [sum(entry["pixel_counts"][i] for entry in entries) for i in range(4)]
        boundary = [sum(entry["boundary_counts"][i] for entry in entries) for i in range(4)]
        area = sum(entry["predicted_object_area"] for entry in entries)
        self.assertEqual(report["pixel_counts"], pixels)
        self.assertEqual(report["boundary_counts"], boundary)
        self.assertEqual(report["predicted_object_area"], area)

        oa, precision, recall, f1, iou = metrics.metrics_from_confusion(*pixels)
        self.assertAlmostEqual(report["metrics"]["oa"], oa, places=12)
        self.assertAlmostEqual(report["metrics"]["precision"], precision, places=12)
        self.assertAlmostEqual(report["metrics"]["recall"], recall, places=12)
        self.assertAlmostEqual(report["metrics"]["f1"], f1, places=12)
        self.assertAlmostEqual(report["metrics"]["iou"], iou, places=12)
        completeness, correctness, f_boundary = metrics.boundary_metrics_from_statistics(*boundary)
        self.assertAlmostEqual(report["metrics"]["boundary_completeness"], completeness, places=12)
        self.assertAlmostEqual(report["metrics"]["boundary_correctness"], correctness, places=12)
        self.assertAlmostEqual(report["metrics"]["f_boundary"], f_boundary, places=12)

        numerators = [
            sum(entry["object_numerators"][i] for entry in entries) for i in range(3)
        ]
        self.assertAlmostEqual(report["metrics"]["goc"], numerators[0] / area, places=12)
        self.assertAlmostEqual(report["metrics"]["guc"], numerators[1] / area, places=12)
        self.assertAlmostEqual(report["metrics"]["gtc"], numerators[2] / area, places=12)
        self.assertAlmostEqual(report["metrics"]["goc"], 9.0 / 25.0, places=12)
        json.dumps(report, allow_nan=False)

    def test_integer_and_float_binary_encodings_accepted(self):
        for truth in (square().astype(np.int8), square().astype(np.uint8), square().astype(np.float64)):
            report = ea.accumulate_prediction_masks(
                one_sample_config(), ["a"], [("a", as_probability(square()), truth)], "val"
            )
            self.assertEqual(report["pixel_counts"], [int(square().sum()), 0, 0, 64 - int(square().sum())])

    def test_extra_valid_stream_item_rejected(self):
        config = make_config(counts={"train": 1, "val": 2, "test": 3014})
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(
                config, ["a", "b"], [sample("a"), sample("b"), sample("c")], "val"
            )

    def test_missing_stream_item_rejected(self):
        config = make_config(counts={"train": 1, "val": 2, "test": 3014})
        with self.assertRaises(ea.EvaluationAdapterError):
            ea.accumulate_prediction_masks(config, ["a", "b"], [sample("a")], "val")

    def test_include_per_image_false_omits_list(self):
        report = ea.accumulate_prediction_masks(
            one_sample_config(), ["a"], [sample("a")], "val", include_per_image=False
        )
        self.assertNotIn("per_image", report)
        self.assertTrue(report["complete_split"])


if __name__ == "__main__":
    unittest.main()
