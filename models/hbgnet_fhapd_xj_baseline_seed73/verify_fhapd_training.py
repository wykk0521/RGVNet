"""CPU contract and real train/val GPU acceptance for a frozen FHAPD run."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import time
from pathlib import Path

import common
import train

ROOT = Path(__file__).resolve().parent


def assert_true(value, message):
    if not value:
        raise AssertionError(message)


def parameter_counts(model):
    return {'field': sum(p.numel() for p in model.parameters()),
            'refiner': sum(p.numel() for p in model.region_refiner.parameters()) if getattr(model, 'region_refiner', None) is not None else 0,
            'verifier': sum(p.numel() for p in model.edit_verifier.parameters()) if getattr(model, 'edit_verifier', None) is not None else 0}


def check_sources():
    provenance = json.loads((ROOT / 'clone_provenance.json').read_text(encoding='utf-8'))
    for name, expected in provenance['unchanged_source_files_sha256'].items():
        assert_true(common.sha256(ROOT / name) == expected, f'Copied source changed: {name}')
    if common.MODEL_KIND == 'baseline':
        def forwards(path):
            return {node.name: ast.dump(item, include_attributes=False)
                    for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node, ast.ClassDef)
                    for item in node.body if isinstance(item, ast.FunctionDef) and item.name == 'forward'}
        assert_true(forwards(ROOT / 'reference/baseline_model.py') == forwards(ROOT / 'implementation/model.py'),
                    'Baseline official forward bodies changed')
    return {'unchanged_source_files': len(provenance['unchanged_source_files_sha256']),
            'model_kind': common.MODEL_KIND}


def cpu_acceptance(cfg, fingerprint):
    import copy
    import unittest
    import io

    sources = check_sources()
    contract = common.verify_data_contract(cfg, verify_bytes=True)
    train._check_recipe(cfg)
    stream = io.StringIO()
    modules = ['test_training_contract', 'test_validation']
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    assert_true(result.wasSuccessful() and not result.skipped, 'FHAPD training contract tests failed/skipped')
    from verify_training_state import verify
    recovery = verify()
    if common.MODEL_KIND == 'combined':
        from verify_rd_v9_context import (check_architecture, check_v8_parity, check_masks_and_gate,
                                        check_module_detach_and_modes, check_gradient_staging,
                                        check_calibration, check_loss, check_field)
        modules = {'architecture': check_architecture(), 'v8_parity': check_v8_parity(),
                   'routing': check_masks_and_gate(), 'detach': check_module_detach_and_modes(),
                   'gradient_staging': check_gradient_staging(), 'calibration': check_calibration(),
                   'loss': check_loss(), 'field': check_field()}
    else:
        common.seed_all(73)
        model = common.build(cfg, pretrained=False).eval()
        counts = parameter_counts(model)
        assert_true(counts == {'field': 30453790, 'refiner': 0, 'verifier': 0}, 'Baseline architecture changed')
        batch = next(iter(common.loader(cfg, 'train', batch=1, workers=0)))
        with common.torch.no_grad():
            outputs = model(batch['image'])
            expected_shapes = [(1, 1, 256, 256), (1, 2, 256, 256), (1, 1, 256, 256)]
            assert_true([tuple(v.shape) for v in outputs] == expected_shapes, 'Baseline output shapes')
            from implementation.losses import LossF
            expected = LossF([1, 1, 1])(*outputs, batch['region'], batch['boundary'].long(), batch['distance'])
            actual = common.loss_fn(outputs, batch, cfg)
            assert_true(common.torch.equal(actual, expected), 'Official LossF parity')
            assert_true(common.torch.allclose(outputs[1].exp().sum(1), common.torch.ones((1, 256, 256)), atol=1e-6), 'Boundary log probabilities')
        modules = {'parameter_counts': counts, 'official_loss_parity': True,
                   'baseline_model_source_exact': True, 'real_train_forward': True}
        del model, batch, outputs, actual, expected
        gc.collect()
    common.atomic_json(ROOT / 'engineering_cpu.json', {
        'status': 'pass', 'kind': 'CPU engineering; no formal training or model test',
        'config': cfg, 'source_fingerprint': fingerprint, 'sources': sources,
        'data_contract': contract, 'tests': result.testsRun, 'failures': 0, 'errors': 0, 'skipped': 0,
        'training_state': recovery, 'model_checks': modules, 'test_pixels_read': False,
        'formal_training_started': False, 'checkpoint_written': False,
    })


def baseline_gpu_smoke(cfg, fingerprint):
    torch = common.torch
    torch.cuda.set_device(0)
    common.seed_all(cfg['training']['seed'])
    model = common.build(cfg, pretrained=True).to('cuda:0')
    assert_true(parameter_counts(model) == {'field': 30453790, 'refiner': 0, 'verifier': 0}, 'Baseline parameter counts')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    iterator = iter(common.loader(cfg, 'train', batch=8, workers=0))
    before = model.out_feature.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    updates = 0
    for index in range(1, 5):
        batch = common.device_batch(next(iterator), torch.device('cuda:0'))
        loss = common.loss_fn(model(batch['image']), batch, cfg)
        losses.append(float(loss.detach()))
        (loss / 2).backward()
        if index % 2 == 0:
            train._finite_step(model, optimizer, 0, index)
            updates += 1
        del loss, batch
    assert_true(updates == 2 and not torch.equal(before, model.out_feature.weight.detach()), 'Baseline optimizer updates must change Region head')
    validation = common.evaluate(model, common.loader(cfg, 'val', batch=8, workers=0), torch.device('cuda:0'), max_batches=8)
    assert_true(validation['samples'] == 64, 'Baseline GPU validation must evaluate 64 real val samples')
    report = {'status': 'pass', 'configs': {common.MAIN_RUN: cfg}, 'source_fingerprint': fingerprint,
              'parameter_counts': parameter_counts(model), 'real_smoke': {
                  'device': 'cuda:0', 'train_batches': 4, 'optimizer_updates': 2,
                  'validation_batches': 8, 'validation_samples': 64, 'losses': losses,
                  'validation': validation, 'official_pretraining_only': True, 'test_started': False,
                  'checkpoint_written': False, 'region_head_moved': True,
              }}
    common.atomic_json(ROOT / 'verification.json', report)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    if args.cpu == args.gpu:
        raise ValueError('Choose exactly one engineering phase')
    common.torch.set_num_threads(1)
    cfg = common.config(ROOT / 'config_protocol_b.yaml')
    with train.training_lock(cfg, purpose='engineering'):
        train._validate_outputs(*train._output_paths(cfg), None)
        fingerprint = common.source_fingerprint()
        common.verify_data_contract(cfg)
        if args.cpu:
            if (ROOT / 'engineering_cpu.json').exists():
                raise FileExistsError('CPU engineering receipt already exists')
            cpu_acceptance(cfg, fingerprint)
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != str(common.PHYSICAL_GPU):
                raise ValueError('Physical GPU binding differs from the frozen runtime identity')
            if not common.torch.cuda.is_available() or common.torch.cuda.device_count() != 1:
                raise RuntimeError('GPU engineering requires one visible CUDA device')
            cpu = json.loads((ROOT / 'engineering_cpu.json').read_text(encoding='utf-8'))
            assert_true(cpu['status'] == 'pass' and cpu['source_fingerprint'] == fingerprint, 'Fresh CPU acceptance required')
            if (ROOT / 'verification.json').exists():
                raise FileExistsError('GPU verification receipt already exists')
            if common.MODEL_KIND == 'combined':
                from verify_rd_v9_context import check_real_smoke
                check_real_smoke('cuda:0')
            else:
                baseline_gpu_smoke(cfg, fingerprint)
            common.require_engineering(cfg, fingerprint)
        assert_true(common.source_fingerprint() == fingerprint, 'Source changed during engineering')
        print(json.dumps({'status': 'pass', 'phase': 'cpu' if args.cpu else 'gpu',
                          'run_name': common.MAIN_RUN, 'physical_gpu': common.PHYSICAL_GPU,
                          'formal_training_started': False, 'formal_model_test_started': False,
                          'checkpoint_written': False}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
