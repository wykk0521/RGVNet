"""Focused negative guards and selection/loss checks for FHAPD training."""
import copy
import unittest
from unittest.mock import patch

import common
import train
import training_state


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = common.config(common.ROOT / 'config_protocol_b.yaml')

    def test_frozen_recipe_and_no_cross_dataset_floor(self):
        train._check_recipe(self.cfg)
        self.assertIsNone(self.cfg['training']['iou_floor'])
        self.assertEqual(self.cfg['training']['seed'], 73)
        self.assertIsNone(self.cfg['evaluation']['spatial_resolution_m'])
        self.assertFalse(self.cfg['evaluation']['physical_tolerance_verified'])
        self.assertEqual(self.cfg['evaluation']['boundary_tolerance_pixels'], 3)

    def test_config_rejects_identity_data_loss_and_recipe_drift(self):
        cases = [('experiment', 'run_name', 'other'), ('training', 'epochs', 39),
                 ('training', 'seed', 17), ('training', 'iou_floor', .8205741376917083),
                 ('dataset', 'expected_counts', {'train': 13890, 'val': 3067, 'test': 3014}),
                 ('loss', 'weights', [2, 1, 1]), ('evaluation', 'region_threshold', .6),
                 ('evaluation', 'boundary_tolerance_pixels', 4),
                 ('model', 'drop_rate', .3)]
        for section, key, value in cases:
            changed = copy.deepcopy(self.cfg)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                common._validate_config(changed)

    def test_disabled_floor_does_not_create_constrained_selection(self):
        best, constrained, new, constrained_new = training_state.selection_update(
            {'iou': .3, 'bf1': .7, 'checkpoint_score': .7 * .3 + .3 * .7}, 1, None, None, None)
        self.assertTrue(new)
        self.assertEqual(best['epoch'], 1)
        self.assertIsNone(constrained)
        self.assertFalse(constrained_new)
        later = training_state.selection_update({'iou': .8, 'bf1': .2}, 2, best, None, None)
        self.assertIsNone(later[1])
        self.assertFalse(later[3])
        self.assertEqual(later[0]['epoch'], 2)

    def test_no_loss_coefficient_drift(self):
        self.assertEqual(self.cfg['loss'], common._expected_loss(common.MAIN_RUN))
        if common.MODEL_KIND == 'combined':
            self.assertEqual(common.LOSS_COEFFICIENTS[common.MAIN_RUN], (.5, .5, .125, .125))
            self.assertFalse(self.cfg['loss']['final_region_loss'])
            self.assertEqual(self.cfg['loss']['verifier_loss_mode'], 'ordinary_bce_present_direction_average')
        else:
            torch = common.torch
            outputs = [torch.randn(1, 1, 8, 8, requires_grad=True),
                       torch.log_softmax(torch.randn(1, 2, 8, 8, requires_grad=True), dim=1),
                       torch.randn(1, 1, 8, 8, requires_grad=True)]
            batch = {'region': torch.zeros(1, 1, 8, 8), 'boundary': torch.zeros(1, 1, 8, 8),
                     'distance': torch.zeros(1, 1, 8, 8)}
            from implementation.losses import LossF
            expected = LossF([1, 1, 1])(*outputs, batch['region'], batch['boundary'].long(), batch['distance'])
            actual, components = common.loss_fn(outputs, batch, self.cfg, return_components=True)
            self.assertTrue(torch.equal(actual, expected))
            actual.backward()
            self.assertTrue(torch.isfinite(outputs[0].grad).all())
            self.assertEqual(components['total'], float(expected.detach()))

    def test_missing_pretraining_refuses_fallback(self):
        with patch('common.Path.is_file', return_value=False), self.assertRaises(FileNotFoundError):
            common.build(self.cfg, pretrained=True)

    def test_model_loss_and_backbone_sources_stay_frozen(self):
        from verify_fhapd_training import check_sources
        self.assertGreater(check_sources()['unchanged_source_files'], 0)

    def test_formal_test_entrypoint_disabled(self):
        import evaluate
        with self.assertRaisesRegex(RuntimeError, 'not enabled'):
            evaluate.run(self.cfg, 'not-read.pt', 'test', common.MAIN_RUN + '_test', 'cpu')


if __name__ == '__main__':
    unittest.main()
