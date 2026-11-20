"""Small CPU acceptance checks; no train/model/dataset imports or formal outputs."""
import copy
import io
import json
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import training_state as state


ROOT = Path(__file__).resolve().parent
FLOOR = 0.8205741376917083


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def _roundtrip(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return torch.load(stream, map_location='cpu', weights_only=True)


def _must_fail(function, exception):
    try:
        function()
    except exception:
        return
    raise AssertionError(f'Expected {exception.__name__}')


def _val(iou, bf1):
    return {'iou': iou, 'bf1': bf1, 'checkpoint_score': .7 * iou + .3 * bf1}


def _selection_test():
    best, constrained, legacy_new, constrained_new = state.selection_update(
        _val(FLOOR - 1e-10, .95), 1, None, None, FLOOR)
    assert legacy_new and not constrained_new and constrained is None
    best, constrained, _, changed = state.selection_update(
        _val(FLOOR, .75), 2, best, constrained, FLOOR)
    assert changed and constrained == {'epoch': 2, 'bf1': .75, 'iou': FLOOR}
    _, tied, _, changed = state.selection_update(_val(FLOOR, .75), 3, best, constrained, FLOOR)
    assert tied == constrained and not changed
    _, stronger, _, changed = state.selection_update(_val(.83, .75), 4, best, constrained, FLOOR)
    assert changed and stronger['epoch'] == 4
    _, preferred, _, changed = state.selection_update(_val(FLOOR, .76), 5, best, stronger, FLOOR)
    assert changed and preferred['epoch'] == 5  # BF1 first, not IoU.
    exact = _val(.85, .8)
    first, selected, _, _ = state.selection_update(exact, 1, None, None, FLOOR)
    second, same, new, constrained_new = state.selection_update(exact, 2, first, selected, FLOOR)
    assert second == first and same == selected and not new and not constrained_new
    earlier, _, new, _ = state.selection_update(exact, 1, {'epoch': 2, 'score': first['score']},
                                               None, FLOOR)
    assert new and earlier['epoch'] == 1
    assert state.selection_update(None, 3, first, selected, FLOOR) == (first, selected, False, False)
    _must_fail(lambda: state.selection_update(_val(float('nan'), .8), 1, None, None, FLOOR), ValueError)
    _must_fail(lambda: state.selection_update({'iou': .9, 'bf1': .9, 'checkpoint_score': 0},
                                              1, None, None, FLOOR), ValueError)
    return {'floor_inclusive': True, 'score_ties_keep_earlier': True,
            'constrained_bf1_then_iou_then_earlier': True, 'nonfinite_rejected': True}


def _rng_test():
    generators = {'train': torch.Generator().manual_seed(17),
                  'val': torch.Generator().manual_seed(42)}
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    # Populate NumPy's cached Gaussian to test all five MT19937 fields.
    np.random.normal()
    captured = state.capture_rng(generators)
    assert isinstance(captured['numpy'][1], list)
    assert all(type(value) is int for value in captured['numpy'][1])
    loaded = _roundtrip(captured)
    _assert_tree_equal(captured, loaded)

    def sample():
        return {'python': [random.random() for _ in range(4)],
                'numpy': np.random.normal(size=4).tolist(),
                'torch': torch.rand(4),
                'train': torch.randperm(19, generator=generators['train']),
                'val': torch.randperm(11, generator=generators['val'])}

    expected = sample()
    sample()
    state.restore_rng(loaded, generators)
    _assert_tree_equal(expected, sample())
    single = torch.Generator().manual_seed(71)
    saved_single = _roundtrip(state.capture_rng(single))
    expected_single = torch.rand(4, generator=single)
    state.restore_rng(saved_single, single)
    assert torch.equal(expected_single, torch.rand(4, generator=single))
    return {'weights_only_roundtrip': True, 'numpy_list_encoding': True,
            'python_numpy_torch_exact': True, 'both_loader_streams_exact': True,
            'single_generator_exact': True, 'cuda_states_captured': len(captured['cuda'])}


def _tiny_components():
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10_000_000_000, eta_min=1e-5)
    return model, optimizer, scheduler


def _tiny_step(model, optimizer, scheduler, inputs):
    optimizer.zero_grad(set_to_none=True)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def _optimizer_test():
    model, optimizer, scheduler = _tiny_components()
    _tiny_step(model, optimizer, scheduler, torch.rand(2, 3))
    saved = _roundtrip({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'rng': state.capture_rng(None)})
    # Verify actual serialized Adam moments and scheduler epoch, not just config.
    assert saved['optimizer']['state'] and saved['scheduler']['last_epoch'] == 1
    inputs = torch.rand(2, 3)
    expected_loss = _tiny_step(model, optimizer, scheduler, inputs)
    expected = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    restored, restored_optimizer, restored_scheduler = _tiny_components()
    restored.load_state_dict(saved['model'])
    restored_optimizer.load_state_dict(saved['optimizer'])
    restored_scheduler.load_state_dict(saved['scheduler'])
    state.restore_rng(saved['rng'], None)
    resumed_inputs = torch.rand(2, 3)
    assert torch.equal(inputs, resumed_inputs)
    assert _tiny_step(restored, restored_optimizer, restored_scheduler, resumed_inputs) == expected_loss
    _assert_tree_equal(expected, (restored.state_dict(), restored_optimizer.state_dict(),
                                 restored_scheduler.state_dict()))
    return {'adam_moments_roundtrip': True, 'scheduler_roundtrip': True,
            'next_step_bitwise_equal_cpu': True}


def _transaction_test(root):
    log, weights = root / 'log', root / 'weights'
    log.mkdir()
    weights.mkdir()
    model, optimizer, scheduler = _tiny_components()
    generator = torch.Generator().manual_seed(17)
    fingerprint = {'fixture.py': 'fixture-hash'}
    best = constrained = None
    check_calls = []

    def guard():
        check_calls.append(True)

    def checkpoint(epoch, validation):
        nonlocal best, constrained
        _tiny_step(model, optimizer, scheduler, torch.rand(2, 3))
        best, constrained, legacy_new, constrained_new = state.selection_update(
            validation, epoch, best, constrained, FLOOR)
        record = {'epoch': epoch, 'validation': validation, 'best': best,
                  'constrained': constrained, 'train_loss': .1}
        return {'epoch': epoch, 'run_name': 'fixture', 'config': {'training': {'iou_floor': FLOOR}},
                'model': copy.deepcopy(model.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'scheduler': copy.deepcopy(scheduler.state_dict()),
                'rng': state.capture_rng(generator), 'record': record,
                'validation': validation, 'best': best, 'constrained': constrained,
                'legacy_new': legacy_new, 'constrained_new': constrained_new,
                'source_fingerprint': fingerprint, 'pretrained_provenance': {'sha256': 'fixture'},
                'split_manifest_sha256': {'train': 'a', 'val': 'b', 'test': 'c'},
                'torch_version': str(torch.__version__)}

    first = checkpoint(1, _val(.83, .7))
    real_text = state._atomic_text

    def fail_record(path, text):
        if Path(path).parent.name == 'epoch_records':
            raise OSError('Injected post-last/pre-record crash')
        return real_text(path, text)

    with patch.object(state, '_atomic_text', side_effect=fail_record):
        _must_fail(lambda: state.commit_epoch(log, weights, first, check_source=guard), OSError)
    assert (weights / 'last.pt').is_file() and (log / 'pending_epoch.json').is_file()
    assert not (weights / 'best.pt').exists()
    current = torch.load(weights / 'last.pt', map_location='cpu', weights_only=True)
    state.recover_epoch(log, weights, current, check_source=guard)
    assert not (log / 'pending_epoch.json').exists()
    assert (log / 'epoch_records' / 'epoch_001.json').is_file()
    assert set(path.name for path in weights.iterdir()) == {
        'last.pt', 'best.pt', 'best_iou_constrained.pt'}

    second = checkpoint(2, _val(.84, .8))
    real_save = state._atomic_checkpoint

    def fail_best(path, cp, check_source):
        if Path(path).name == 'best.pt':
            raise OSError('Injected post-last/pre-best crash')
        return real_save(path, cp, check_source)

    with patch.object(state, '_atomic_checkpoint', side_effect=fail_best):
        _must_fail(lambda: state.commit_epoch(log, weights, second, check_source=guard), OSError)
    current = torch.load(weights / 'last.pt', map_location='cpu', weights_only=True)
    assert current['epoch'] == 2
    (log / 'epoch_records' / 'epoch_002.json').unlink()
    (log / 'metrics.jsonl').unlink()
    (weights / 'best.pt.tmp').write_bytes(b'incomplete selected checkpoint')
    state.recover_epoch(log, weights, current, check_source=guard)
    before = {str(path.relative_to(root)): path.read_bytes()
              for path in root.rglob('*') if path.is_file()}
    state.recover_epoch(log, weights, current, check_source=guard)
    after = {str(path.relative_to(root)): path.read_bytes()
             for path in root.rglob('*') if path.is_file()}
    assert before == after  # Idempotent, including checkpoint bytes.
    for name in ('best.pt', 'best_iou_constrained.pt'):
        compact = torch.load(weights / name, map_location='cpu', weights_only=True)
        assert compact['epoch'] == 2 and compact['source_fingerprint'] == fingerprint
        assert compact['pretrained_provenance'] == second['pretrained_provenance']
        assert compact['split_manifest_sha256'] == second['split_manifest_sha256']
        assert not {'optimizer', 'scheduler', 'rng'} & compact.keys()

    third = checkpoint(3, None)
    state.commit_epoch(log, weights, third, check_source=guard)
    current = torch.load(weights / 'last.pt', map_location='cpu', weights_only=True)
    assert current['epoch'] == 3
    assert len(list(weights.iterdir())) == 3
    old_best = (weights / 'best.pt').read_bytes()
    (weights / 'best.pt').unlink()
    _must_fail(lambda: state.recover_epoch(log, weights, current), FileNotFoundError)
    (weights / 'best.pt').write_bytes(old_best)
    extra = log / 'epoch_records' / 'epoch_004.json'
    extra.write_text('{}', encoding='utf-8')
    _must_fail(lambda: state.recover_epoch(log, weights, current), FileExistsError)
    extra.unlink()
    unexpected = weights / 'epoch_001.pt'
    unexpected.write_bytes(b'non-best historical weights forbidden')
    _must_fail(lambda: state.recover_epoch(log, weights, current), FileExistsError)
    unexpected.unlink()

    fourth = checkpoint(4, _val(.85, .85))
    last_bytes = (weights / 'last.pt').read_bytes()

    def fail_last(path, cp, check_source):
        if Path(path).name == 'last.pt':
            raise OSError('Injected pre-last crash')
        return real_save(path, cp, check_source)

    with patch.object(state, '_atomic_checkpoint', side_effect=fail_last):
        _must_fail(lambda: state.commit_epoch(log, weights, fourth), OSError)
    assert (weights / 'last.pt').read_bytes() == last_bytes
    pending_bytes = (log / 'pending_epoch.json').read_bytes()
    _must_fail(lambda: state.recover_epoch(log, weights, current), FileExistsError)
    _must_fail(lambda: state.commit_epoch(log, weights, fourth), FileExistsError)
    assert (log / 'pending_epoch.json').read_bytes() == pending_bytes
    assert (weights / 'last.pt').read_bytes() == last_bytes
    assert check_calls
    return {'post_last_missing_record_and_best_recovered': True,
            'post_last_matching_temporary_recovered': True, 'recovery_idempotent': True,
            'pre_last_orphan_refused_without_overwrite': True,
            'future_record_and_historical_weight_orphans_refused': True,
            'missing_historical_best_refused': True, 'only_three_checkpoints_retained': True,
            'compact_provenance_present': True, 'source_guard_calls': len(check_calls)}


def verify():
    """Return JSON-safe results; preserve the caller's RNG and remove all fixtures."""
    initial_rng = state.capture_rng(None)
    try:
        results = {'selection': _selection_test(), 'rng': _rng_test(),
                   'optimizer': _optimizer_test()}
        with tempfile.TemporaryDirectory(prefix='.verify-training-state-', dir=ROOT) as temporary:
            results['transactions'] = _transaction_test(Path(temporary))
        results['status'] = 'pass'
        json.dumps(results, allow_nan=False)
        return results
    finally:
        state.restore_rng(initial_rng, None)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, allow_nan=False))
