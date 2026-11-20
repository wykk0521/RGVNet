"""Unit tests for validation.py (independent FHAPD validation plumbing).

Run from anywhere (dependencies permitting):

    py -3 -B -m unittest discover -s <this directory> -t <this directory> -p "test_validation.py"

Only the new plumbing is exercised: strict ``sigmoid > 0.5`` decisions, the
boundary tolerance resolved from the config, flat pixel/boundary counts and
``None`` object errors, micro (not per-image mean) aggregation, frozen
manifest order/uniqueness/completeness for full and ``max_batches`` smoke
validation, both runtime identities, the combined candidate table and
diagnostics, and the image-only forward contract.

Fixtures are synthetic tiny tensors and fake inference-only models; the
validation manifest lives in a temporary directory under the project test work
root next to this package (never the system temp directory).  No dataset, no
loader, no checkpoint, no training and no test split is touched; nothing is
downloaded and no external process runs.  torch/numpy/scikit-image/scipy come
from the remote training environment, so the whole module skips cleanly when a
dependency is missing locally.
"""
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Test work root next to the package (the template's sibling), never the system temp.
WORK_ROOT = ROOT.parent / 'validation_test_work'
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    np = None
    NUMPY_ERROR = f'{type(exc).__name__}: {exc}'
else:
    NUMPY_ERROR = ''

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency guard
    torch = None
    TORCH_ERROR = f'{type(exc).__name__}: {exc}'
else:
    TORCH_ERROR = ''

try:
    import validation
except ImportError as exc:  # pragma: no cover - dependency guard
    validation = None
    VALIDATION_ERROR = f'{type(exc).__name__}: {exc}'
else:
    VALIDATION_ERROR = ''

try:
    import skimage  # noqa: F401
except ImportError as exc:  # pragma: no cover - dependency guard
    SKIMAGE_ERROR = f'{type(exc).__name__}: {exc}'
else:
    SKIMAGE_ERROR = ''

try:
    import scipy  # noqa: F401
except ImportError as exc:  # pragma: no cover - dependency guard
    SCIPY_ERROR = f'{type(exc).__name__}: {exc}'
else:
    SCIPY_ERROR = ''

VALIDATION_READY = validation is not None and torch is not None and np is not None
METRICS_READY = VALIDATION_READY and SKIMAGE_ERROR == ''
COMBINED_READY = METRICS_READY and SCIPY_ERROR == ''

requires_validation = unittest.skipUnless(
    VALIDATION_READY, 'validation unavailable: ' + (VALIDATION_ERROR or TORCH_ERROR or NUMPY_ERROR))
requires_metrics = unittest.skipUnless(
    METRICS_READY, 'shared metrics unavailable: ' + (SKIMAGE_ERROR or VALIDATION_ERROR or TORCH_ERROR))
requires_combined = unittest.skipUnless(
    COMBINED_READY, 'combined diagnostics unavailable: ' + (SCIPY_ERROR or SKIMAGE_ERROR))

SIZE = (8, 8)
FINAL_ROUTE_KEYS = (
    'iou', 'f1', 'oa', 'precision', 'recall', 'bf1',
    'boundary_completeness', 'boundary_correctness', 'goc', 'guc', 'gtc',
    'predicted_object_area', 'tp', 'fp', 'fn', 'tn',
    'matched_gt_boundary', 'gt_boundary', 'matched_pred_boundary', 'pred_boundary',
)


def uniform_logits(batch, size=SIZE, value=0.0):
    return torch.full((batch, 1) + tuple(size), float(value), dtype=torch.float32)


def mask_logits(mask, positive=10.0, negative=-10.0):
    return torch.where(mask, torch.full_like(mask, positive, dtype=torch.float32),
                       torch.full_like(mask, negative, dtype=torch.float32))


class FakeModel:
    """Inference-only stand-in that refuses any argument but the image tensor."""

    def __init__(self, region_logits, auxiliary=None):
        self.region_logits = region_logits
        self.auxiliary = auxiliary
        self.calls = []
        self.last_auxiliary = {}

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise AssertionError(f'model received keyword arguments: {sorted(kwargs)}')
        if len(args) != 1:
            raise AssertionError(f'model received {len(args)} positional arguments')
        image = args[0]
        if not isinstance(image, torch.Tensor):
            raise AssertionError('model input is not a tensor')
        if image.ndim != 4 or image.shape[1] != 3:
            raise AssertionError('model input is not a 3-channel image batch')
        if image.shape[0] != self.region_logits.shape[0]:
            raise AssertionError('model input batch size differs from the fixture')
        self.calls.append(image)
        if self.auxiliary is not None:
            self.last_auxiliary = dict(self.auxiliary)
        edge = torch.zeros(image.shape[0], 2, image.shape[2], image.shape[3], dtype=torch.float32)
        return [self.region_logits, edge, torch.zeros_like(self.region_logits)]


class FixedOutputModel:
    """Returns one fixed output regardless of the input batch size."""

    def __init__(self, output):
        self.output = output
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return [self.output]


class ValidationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        cls.addClassCleanup(cls._remove_empty_work_root)

    @classmethod
    def _remove_empty_work_root(cls):
        try:
            if WORK_ROOT.exists() and not any(WORK_ROOT.iterdir()):
                WORK_ROOT.rmdir()
        except OSError:  # pragma: no cover - best effort cleanup
            pass

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix='validation_', dir=str(WORK_ROOT)))
        self.addCleanup(shutil.rmtree, str(self.work), True)

    def manifest(self, ids, name='val.txt'):
        path = self.work / name
        text = '\n'.join(ids)
        path.write_text(text + ('\n' if text else ''), encoding='utf-8')
        return path

    def config(self, ids, *, kind='baseline', tolerance=3, size=SIZE, expected=None,
               manifest_path=None):
        path = Path(manifest_path) if manifest_path is not None else self.manifest(ids)
        config = {
            'runtime_identity': {'MODEL_KIND': kind},
            'dataset': {
                'input_size': list(size),
                'manifests': {'val': str(path)},
                'expected_counts': {'val': len(ids) if expected is None else expected},
            },
            'evaluation': {
                'region_threshold': 0.5,
                'threshold_comparison': '>',
                'boundary_tolerance_pixels': tolerance,
                'physical_tolerance_verified': False,
                'requested_boundary_tolerance_m': 3.0,
                'spatial_resolution_m': None,
                'boundary_tolerance_m': None,
            },
        }
        if kind == 'combined':
            config['model'] = {'refinement': {'max_delta': 2.0}}
        return config

    def batch(self, ids, logits, truth, size=SIZE):
        count = len(ids)
        return {
            'image_id': list(ids),
            'image': torch.zeros(count, 3, *size, dtype=torch.float32),
            'region': truth,
            'boundary': torch.zeros(count, 1, *size, dtype=torch.float32),
            'distance': torch.zeros(count, 1, *size, dtype=torch.float32),
        }

    def id_batches(self, ids):
        return [self.batch([ident], uniform_logits(1), torch.zeros(1, 1, *SIZE)) for ident in ids]

    def combined_auxiliary(self, raw, sr, delta, truth, keep_cut, keep_fill, *,
                           valid_count=None, verifier_logits=None):
        from protocol.candidate_diagnostics import strict_labels
        labels = strict_labels(raw, sr, delta, truth)
        keep_cut = keep_cut & labels['cut_edit']
        keep_fill = keep_fill & labels['fill_edit']
        return {
            'raw_region_logits': raw,
            'sr_region_logits': sr,
            'final_region_logits': sr.clone(),
            'delta': delta,
            'edit': labels['edit'],
            'cut_edit': labels['cut_edit'],
            'fill_edit': labels['fill_edit'],
            'keep': keep_cut | keep_fill,
            'cut_keep': keep_cut,
            'fill_keep': keep_fill,
            'valid_count': (torch.ones_like(raw, dtype=torch.int64)
                            if valid_count is None else valid_count),
            'verifier_logits': (torch.zeros(raw.shape[0], 2, raw.shape[2], raw.shape[3])
                                if verifier_logits is None else verifier_logits),
        }

    def main_candidate_fixture(self):
        """32x32 fixture: one beneficial cut accepted, one harmful fill rejected."""
        raw = torch.full((1, 1, 32, 32), -4.0)
        sr = torch.full((1, 1, 32, 32), -4.0)
        truth = torch.zeros(1, 1, 32, 32)
        truth[0, 0, 4:9, 4:9] = 1.0
        raw[0, 0, 0, 0] = 4.0
        sr[0, 0, 0, 1] = 4.0
        delta = sr - raw
        keep_cut = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
        keep_cut[0, 0, 0, 0] = True
        keep_fill = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
        return raw, sr, delta, truth, keep_cut, keep_fill

    def run_combined(self, *, tolerance=3, valid_count=None):
        raw, sr, delta, truth, keep_cut, keep_fill = self.main_candidate_fixture()
        auxiliary = self.combined_auxiliary(raw, sr, delta, truth, keep_cut, keep_fill,
                                            valid_count=valid_count)
        model = FakeModel(auxiliary['final_region_logits'], auxiliary)
        # The combined fixture is 32x32 (candidate geometry and the expected
        # valid_pixels/delta denominators below); the config must resolve the
        # same input size or validation rightly refuses the output shape.
        cfg = self.config(['a'], kind='combined', tolerance=tolerance, size=(32, 32))
        batches = [self.batch(['a'], auxiliary['final_region_logits'], truth, size=(32, 32))]
        result = validation.evaluate_validation(
            model, batches, torch.device('cpu'), cfg=cfg)
        return result, model, (raw, sr, delta, truth, keep_cut, keep_fill)


class StrictThresholdTests(ValidationTestCase):
    @requires_metrics
    def test_zero_logit_is_background_and_positive_logit_is_foreground(self):
        truth = torch.ones(1, 1, *SIZE)
        cfg = self.config(['a'])
        exact_half = FakeModel(uniform_logits(1, value=0.0))
        result = validation.evaluate_validation(
            exact_half, [self.batch(['a'], uniform_logits(1), truth)],
            torch.device('cpu'), cfg=cfg)
        # sigmoid(0) == 0.5 exactly; the FHAPD decision is strict > 0.5.
        self.assertEqual(result['tp'], 0)
        self.assertEqual(result['fp'], 0)
        self.assertEqual(result['fn'], 64)
        self.assertEqual(result['tn'], 0)
        self.assertEqual(result['iou'], 0.0)
        self.assertEqual(result['oa'], 0.0)
        # Truth is all foreground, so the truth boundary is empty as well; the
        # frozen shared metrics define BF1=1.0 for two empty boundaries, hence
        # the score is exactly 0.7*0 + 0.3*1.0. (A real split always has
        # background, so an empty prediction scores 0.0 there.)
        self.assertAlmostEqual(result['checkpoint_score'], 0.3, places=9)
        positive = FakeModel(uniform_logits(1, value=1.0))
        better = validation.evaluate_validation(
            positive, [self.batch(['a'], uniform_logits(1), truth)],
            torch.device('cpu'), cfg=cfg)
        self.assertAlmostEqual(better['iou'], 1.0, places=9)
        self.assertAlmostEqual(better['checkpoint_score'], 1.0, places=9)

    @requires_metrics
    def test_boundary_tolerance_comes_from_config(self):
        from protocol.metrics import boundary_statistics
        truth = torch.zeros(1, 1, *SIZE)
        truth[..., :, 0:4] = 1.0
        prediction = torch.zeros(1, 1, *SIZE)
        prediction[..., :, 5:8] = 1.0
        logits = mask_logits(prediction.bool())
        target = truth[0, 0].numpy().astype(bool)
        predicted = prediction[0, 0].numpy().astype(bool)
        observed = {}
        for tolerance in (0, 3):
            cfg = self.config(['a'], tolerance=tolerance)
            result = validation.evaluate_validation(
                FakeModel(logits), [self.batch(['a'], logits, truth)],
                torch.device('cpu'), cfg=cfg)
            expected = boundary_statistics(predicted, target, tolerance)
            self.assertEqual(result['matched_gt_boundary'], expected[0])
            self.assertEqual(result['gt_boundary'], expected[1])
            self.assertEqual(result['matched_pred_boundary'], expected[2])
            self.assertEqual(result['pred_boundary'], expected[3])
            # Only the boundary tolerance changed: pixel counts stay identical.
            self.assertEqual(tuple(result[key] for key in ('tp', 'fp', 'fn', 'tn')),
                             (0, 24, 32, 8))
            observed[tolerance] = (expected, result)
        loose, strict = observed[0][0], observed[3][0]
        self.assertNotEqual(loose, strict)
        self.assertGreater(_boundary_bf1(strict), _boundary_bf1(loose))
        self.assertGreater(observed[3][1]['bf1'], observed[0][1]['bf1'])


def _boundary_bf1(statistics):
    from protocol.metrics import boundary_metrics_from_statistics
    return boundary_metrics_from_statistics(*statistics)[2]


class AggregationTests(ValidationTestCase):
    @requires_metrics
    def test_micro_aggregation_is_not_a_per_image_mean(self):
        truth = torch.zeros(2, 1, *SIZE)
        truth[0, 0, 0, :] = 1.0
        truth[1, 0, 0, :] = 1.0
        logits = uniform_logits(2, value=-10.0)
        logits[0, 0, 0, :] = 10.0
        logits[1, 0, 1:, :] = 10.0
        cfg = self.config(['a', 'b'])
        result = validation.evaluate_validation(
            FakeModel(logits), [self.batch(['a', 'b'], logits, truth)],
            torch.device('cpu'), cfg=cfg)
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['tn']),
                         (8, 56, 8, 56))
        self.assertAlmostEqual(result['iou'], 8.0 / 72.0, places=9)
        self.assertAlmostEqual(result['precision'], 0.125, places=9)
        self.assertAlmostEqual(result['recall'], 0.5, places=9)
        self.assertAlmostEqual(result['oa'], 0.5, places=9)
        # The per-image mean IoU here is 0.5; micro must not match it.
        self.assertNotAlmostEqual(result['iou'], 0.5, places=3)

    @requires_metrics
    def test_no_predicted_object_reports_none_and_zero_area(self):
        truth = torch.zeros(1, 1, *SIZE)
        truth[0, 0, 2:5, 2:5] = 1.0
        logits = uniform_logits(1, value=-20.0)
        cfg = self.config(['a'])
        result = validation.evaluate_validation(
            FakeModel(logits), [self.batch(['a'], logits, truth)],
            torch.device('cpu'), cfg=cfg)
        self.assertIsNone(result['goc'])
        self.assertIsNone(result['guc'])
        self.assertIsNone(result['gtc'])
        self.assertEqual(result['predicted_object_area'], 0)
        self.assertIsInstance(result['predicted_object_area'], int)
        self.assertEqual(result['tp'], 0)
        self.assertGreater(result['fn'], 0)
        json.dumps(result, allow_nan=False)


class SplitIdentityTests(ValidationTestCase):
    @requires_metrics
    def test_full_validation_accepts_the_frozen_order(self):
        ids = ['a', 'b', 'c']
        cfg = self.config(ids)
        result = validation.evaluate_validation(
            FakeModel(uniform_logits(1)), self.id_batches(ids),
            torch.device('cpu'), cfg=cfg)
        self.assertEqual(result['samples'], 3)
        self.assertEqual(result['expected_samples'], 3)
        self.assertIs(result['complete_split'], True)

    @requires_metrics
    def test_out_of_order_ids_rejected(self):
        cfg = self.config(['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), self.id_batches(['b', 'a', 'c']),
                torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_missing_sample_rejected(self):
        cfg = self.config(['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), self.id_batches(['a', 'b']),
                torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_duplicate_sample_rejected(self):
        cfg = self.config(['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), self.id_batches(['a', 'a', 'c']),
                torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_extra_sample_rejected(self):
        cfg = self.config(['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), self.id_batches(['a', 'b', 'c', 'd']),
                torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_duplicate_manifest_ids_rejected(self):
        cfg = self.config(['a', 'a', 'b'])
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), [], torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_manifest_count_must_match_expected_counts(self):
        cfg = self.config(['a', 'b', 'c'], expected=4)
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), [], torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_empty_manifest_rejected(self):
        cfg = self.config([], expected=1)
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), [], torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_empty_batches_rejected(self):
        cfg = self.config(['a'])
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), [], torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_batch_without_image_id_rejected(self):
        cfg = self.config(['a'])
        batch = self.batch(['a'], uniform_logits(1), torch.zeros(1, 1, *SIZE))
        del batch['image_id']
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), [batch], torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_batch_ids_must_match_the_batch_size(self):
        cfg = self.config(['a', 'b'])
        batch = self.batch(['a', 'b'], uniform_logits(1), torch.zeros(1, 1, *SIZE))
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FixedOutputModel(uniform_logits(1)), [batch],
                torch.device('cpu'), cfg=cfg)

    @requires_metrics
    def test_max_batches_smoke_uses_a_prefix_subset(self):
        ids = [f'val_{index:03d}' for index in range(83)]
        cfg = self.config(ids)
        logits = uniform_logits(8)
        batches = [self.batch(ids[start:start + 8], logits, torch.zeros(len(ids[start:start + 8]), 1, *SIZE))
                   for start in range(0, len(ids), 8)]
        pulls = {'count': 0}

        def counting(items):
            for item in items:
                pulls['count'] += 1
                yield item

        model = FakeModel(logits)
        result = validation.evaluate_validation(
            model, counting(batches), torch.device('cpu'), max_batches=8, cfg=cfg)
        self.assertEqual(result['samples'], 64)
        self.assertEqual(result['expected_samples'], 83)
        self.assertIs(result['complete_split'], False)
        self.assertEqual(result['batches_consumed'], 8)
        self.assertEqual(len(model.calls), 8)
        self.assertLessEqual(pulls['count'], 9)

    @requires_metrics
    def test_smoke_that_covers_the_whole_split_is_complete(self):
        ids = ['a', 'b', 'c']
        cfg = self.config(ids)
        result = validation.evaluate_validation(
            FakeModel(uniform_logits(1)), self.id_batches(ids),
            torch.device('cpu'), max_batches=3, cfg=cfg)
        self.assertEqual(result['samples'], 3)
        self.assertIs(result['complete_split'], True)

    @requires_metrics
    def test_smoke_prefix_mismatch_rejected(self):
        cfg = self.config(['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), self.id_batches(['a', 'x']),
                torch.device('cpu'), max_batches=2, cfg=cfg)

    @requires_metrics
    def test_max_batches_must_be_a_positive_integer(self):
        cfg = self.config(['a'])
        batches = self.id_batches(['a'])
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                validation.evaluate_validation(
                    FakeModel(uniform_logits(1)), batches, torch.device('cpu'),
                    max_batches=value, cfg=cfg)

    @requires_validation
    def test_invalid_model_kind_rejected(self):
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(uniform_logits(1)), [],
                torch.device('cpu'),
                cfg={'runtime_identity': {'MODEL_KIND': 'other'}})


class CombinedDiagnosticsTests(ValidationTestCase):
    @requires_combined
    def test_candidate_table_and_rates_match_shared_functions(self):
        from protocol.candidate_diagnostics import (COUNT_KEYS, LEGACY_ALIASES, RATE_KEYS,
                                                    contingency, net_pixels, rates, strict_labels)
        result, _, (raw, sr, delta, truth, keep_cut, keep_fill) = self.run_combined()
        labels = strict_labels(raw, sr, delta, truth)
        expected = contingency(labels, keep_cut, keep_fill)
        for key in COUNT_KEYS:
            self.assertEqual(result[key], expected[key], key)
        for legacy, canonical in LEGACY_ALIASES.items():
            self.assertEqual(result[legacy], expected[canonical], legacy)
        for key, value in rates(expected, undefined=None).items():
            self.assertEqual(result[key], value, key)
        for key, value in net_pixels(expected).items():
            self.assertEqual(result[key], value, key)
        self.assertEqual(expected['edit_pixels'], 2)
        self.assertEqual(expected['accepted_beneficial_pixels'], 1)
        self.assertEqual(expected['rejected_harmful_pixels'], 1)
        self.assertEqual(result['acceptance_precision'], 1.0)
        self.assertEqual(result['harmful_rejection_rate'], 1.0)
        self.assertEqual(result['rejected_beneficial_rate'], 0.0)
        # The rejected fill exists, but no fill candidate was accepted at all.
        self.assertIsNone(result['fill_acceptance_precision'])
        self.assertEqual(result['net_pixels_vs_raw'], 1)
        for key in validation.CANDIDATE_SMOKE_KEYS:
            self.assertIsInstance(result[key], int)
            self.assertNotIsInstance(result[key], bool)

    @requires_combined
    def test_delta_valid_and_goc_plumbing(self):
        result, _, _ = self.run_combined()
        self.assertEqual(result['valid_pixels'], 32 * 32)
        self.assertAlmostEqual(result['delta_abs_mean'], 16.0 / (32 * 32), places=9)
        self.assertAlmostEqual(result['delta_saturation_fraction'], 2.0 / (32 * 32), places=9)
        # The single predicted pixel overlaps no truth object: every error is 1.
        self.assertEqual(result['predicted_object_area'], 1)
        self.assertEqual(result['goc'], 1.0)
        self.assertEqual(result['guc'], 1.0)
        self.assertEqual(result['gtc'], 1.0)

    @requires_combined
    def test_far_diagnostics_use_the_shared_distance_threshold(self):
        from protocol.metrics import mask_boundary
        from scipy.ndimage import distance_transform_edt
        result, _, (raw, sr, delta, truth, _, _) = self.run_combined()
        target = truth[0, 0].numpy().astype(bool)
        gt_boundary = mask_boundary(target)
        far = (distance_transform_edt(~gt_boundary) > 10 if gt_boundary.any()
               else np.ones_like(target, dtype=bool))
        self.assertEqual(result['far_pixels'], int(far.sum()))
        predictions = {
            'raw': raw[0, 0].sigmoid().numpy() > 0.5,
            'sr': sr[0, 0].sigmoid().numpy() > 0.5,
            'final': sr[0, 0].sigmoid().numpy() > 0.5,
        }
        for name, prediction in predictions.items():
            self.assertEqual(result[name + '_far_fp'],
                             int((far & prediction & ~target).sum()))
            self.assertEqual(result[name + '_far_fn'],
                             int((far & ~prediction & target).sum()))

    @requires_combined
    def test_thin_proxy_semantics_on_a_single_pixel_truth(self):
        raw = uniform_logits(1, value=-4.0)
        sr = uniform_logits(1, value=-4.0)
        delta = torch.zeros_like(raw)
        truth = torch.zeros(1, 1, *SIZE)
        truth[0, 0, 3, 3] = 1.0
        keep = torch.zeros(1, 1, *SIZE, dtype=torch.bool)
        auxiliary = self.combined_auxiliary(raw, sr, delta, truth, keep, keep,
                                            valid_count=torch.zeros_like(raw, dtype=torch.int64))
        cfg = self.config(['a'], kind='combined')
        model = FakeModel(auxiliary['final_region_logits'], auxiliary)
        result = validation.evaluate_validation(
            model, [self.batch(['a'], auxiliary['final_region_logits'], truth)],
            torch.device('cpu'), cfg=cfg)
        # The two-sided thin proxy keeps only the truth pixel for the foreground
        # gap and nothing for the background candidate on an isolated pixel.
        self.assertEqual(result['thin_bg_candidates'], 0)
        self.assertEqual(result['thin_fg_candidates'], 1)
        self.assertEqual(result['edit_pixels'], 0)
        self.assertEqual(result['valid_pixels'], 0)
        self.assertEqual(result['delta_abs_mean'], 0.0)
        self.assertEqual(result['delta_saturation_fraction'], 0.0)

    @requires_combined
    def test_baseline_and_combined_final_route_agree(self):
        raw, sr, delta, truth, keep_cut, keep_fill = self.main_candidate_fixture()
        auxiliary = self.combined_auxiliary(raw, sr, delta, truth, keep_cut, keep_fill)
        # Both configs must resolve the 32x32 fixture size used by every batch here.
        combined_cfg = self.config(['a'], kind='combined', size=(32, 32))
        combined = validation.evaluate_validation(
            FakeModel(auxiliary['final_region_logits'], auxiliary),
            [self.batch(['a'], auxiliary['final_region_logits'], truth, size=(32, 32))],
            torch.device('cpu'), cfg=combined_cfg)
        baseline_cfg = self.config(['a'], kind='baseline', size=(32, 32))
        baseline_model = FakeModel(sr.clone())
        baseline = validation.evaluate_validation(
            baseline_model, [self.batch(['a'], sr.clone(), truth, size=(32, 32))],
            torch.device('cpu'), cfg=baseline_cfg)
        for key in FINAL_ROUTE_KEYS:
            self.assertEqual(baseline[key], combined[key], key)
        self.assertEqual(baseline['checkpoint_score'], combined['checkpoint_score'])
        self.assertEqual(baseline['samples'], combined['samples'])
        self.assertIn('raw_iou', combined)
        self.assertIn('sr_iou', combined)
        self.assertNotIn('raw_iou', baseline)
        self.assertNotIn('edit_pixels', baseline)
        self.assertIn('edit_pixels', combined)

    @requires_combined
    def test_combined_rejects_routing_mask_disagreement(self):
        raw, sr, delta, truth, keep_cut, keep_fill = self.main_candidate_fixture()
        auxiliary = self.combined_auxiliary(raw, sr, delta, truth, keep_cut, keep_fill)
        empty = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
        auxiliary.update({'edit': empty, 'cut_edit': empty, 'fill_edit': empty,
                          'keep': empty, 'cut_keep': empty, 'fill_keep': empty})
        # 32x32 config so the refusal comes from the routing masks, not the size.
        cfg = self.config(['a'], kind='combined', size=(32, 32))
        with self.assertRaises(ValueError):
            validation.evaluate_validation(
                FakeModel(auxiliary['final_region_logits'], auxiliary),
                [self.batch(['a'], auxiliary['final_region_logits'], truth, size=(32, 32))],
                torch.device('cpu'), cfg=cfg)

    @requires_combined
    def test_combined_rejects_non_finite_auxiliary(self):
        raw, sr, delta, truth, keep_cut, keep_fill = self.main_candidate_fixture()
        for key in ('delta', 'verifier_logits'):
            auxiliary = self.combined_auxiliary(raw, sr, delta, truth, keep_cut, keep_fill)
            broken = auxiliary[key].clone()
            broken.view(-1)[0] = float('nan')
            auxiliary[key] = broken
            cfg = self.config(['a'], kind='combined', size=(32, 32))
            with self.assertRaises(FloatingPointError):
                validation.evaluate_validation(
                    FakeModel(auxiliary['final_region_logits'], auxiliary),
                    [self.batch(['a'], auxiliary['final_region_logits'], truth, size=(32, 32))],
                    torch.device('cpu'), cfg=cfg)

    @requires_combined
    def test_combined_auxiliary_with_mixed_finite_field_dtypes(self):
        raw, sr, delta, truth, keep_cut, keep_fill = self.main_candidate_fixture()
        auxiliary = self.combined_auxiliary(raw, sr, delta, truth, keep_cut, keep_fill)
        auxiliary['raw_region_logits'] = raw.to(torch.float64)
        auxiliary['sr_region_logits'] = sr.to(torch.float64)
        auxiliary['delta'] = delta.to(torch.float64)
        auxiliary['verifier_logits'] = auxiliary['verifier_logits'].to(torch.float16)
        auxiliary['valid_count'] = torch.ones_like(raw, dtype=torch.int64)
        cfg = self.config(['a'], kind='combined', size=(32, 32))
        result = validation.evaluate_validation(
            FakeModel(auxiliary['final_region_logits'], auxiliary),
            [self.batch(['a'], auxiliary['final_region_logits'], truth, size=(32, 32))],
            torch.device('cpu'), cfg=cfg)
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['edit_pixels'], 2)


class ForwardContractTests(ValidationTestCase):
    @requires_metrics
    def test_model_receives_only_the_image_batch(self):
        truth = torch.zeros(1, 1, *SIZE)
        truth[0, 0, 1:4, 1:4] = 1.0
        logits = mask_logits(truth.bool())
        cfg = self.config(['a'])
        model = FakeModel(logits)
        validation.evaluate_validation(
            model, [self.batch(['a'], logits, truth)], torch.device('cpu'), cfg=cfg)
        self.assertEqual(len(model.calls), 1)
        image = model.calls[0]
        self.assertEqual(tuple(image.shape), (1, 3) + SIZE)
        self.assertFalse(torch.equal(image, truth))

    @requires_combined
    def test_combined_model_receives_only_the_image_batch(self):
        result, model, _ = self.run_combined()
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(tuple(model.calls[0].shape), (1, 3, 32, 32))
        self.assertEqual(result['samples'], 1)


class ResultContractTests(ValidationTestCase):
    @requires_combined
    def test_result_is_flat_and_json_safe(self):
        result, _, _ = self.run_combined()
        json.dumps(result, allow_nan=False)
        for name, value in result.items():
            self.assertNotIsInstance(value, (list, dict, tuple), name)
            if value is None:
                continue
            self.assertIsInstance(value, (int, float, bool), name)
            self.assertNotIsInstance(value, np.ndarray, name)
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), name)
        self.assertIsInstance(result['complete_split'], bool)


class RuntimeIdentityTests(ValidationTestCase):
    def _run_with_modules(self, stub_modules, config, *, load_through_common=False):
        """Evaluate one valid baseline batch with the given module overrides.

        With ``load_through_common`` the configuration is omitted so
        ``evaluate_validation`` must resolve it through ``common.config``.
        """
        saved = {name: sys.modules.get(name) for name in ('common', 'runtime_identity')}
        try:
            for name, module in stub_modules.items():
                sys.modules[name] = module
            ids = ['a']
            logits = uniform_logits(1)
            kwargs = {} if load_through_common else {'cfg': config}
            return validation.evaluate_validation(
                FakeModel(logits), [self.batch(ids, logits, torch.zeros(1, 1, *SIZE))],
                torch.device('cpu'), **kwargs)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    @requires_metrics
    def test_config_is_loaded_through_common_config(self):
        config = self.config(['a'])
        config.pop('runtime_identity')
        calls = []
        stub = types.ModuleType('common')
        stub.MODEL_KIND = 'baseline'

        def stub_config(path):
            calls.append(Path(path))
            return config

        stub.config = stub_config
        result = self._run_with_modules({'common': stub, 'runtime_identity': None}, config,
                                        load_through_common=True)
        self.assertEqual(calls, [ROOT / 'config_protocol_b.yaml'])
        self.assertEqual(result['samples'], 1)

    @requires_metrics
    def test_runtime_identity_module_is_the_primary_kind_source(self):
        config = self.config(['a'])
        config.pop('runtime_identity')
        stub = types.ModuleType('runtime_identity')
        stub.MODEL_KIND = 'baseline'
        result = self._run_with_modules({'runtime_identity': stub}, config)
        self.assertEqual(result['samples'], 1)


if __name__ == '__main__':
    unittest.main()
