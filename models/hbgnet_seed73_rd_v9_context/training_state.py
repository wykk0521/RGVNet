"""Independent selection, RNG and crash-consistent epoch state helpers.

No model, dataset or common imports: selection tests need only the standard
library. RNG/checkpoint operations import NumPy/PyTorch lazily. ``last.pt`` is
the sole authority; JSON records and selected weights are materialized views.
A pending marker contains metadata only, never historical model weights.
"""
import json
import math
import os
import random
import re
import uuid
from pathlib import Path


FORMAT_VERSION = 1
BEST_FILES = {'best': 'best.pt', 'constrained': 'best_iou_constrained.pt'}
RUN_KEYS = ('run_name', 'config', 'source_fingerprint', 'pretrained_provenance',
            'split_manifest_sha256')
STATE_KEYS = {'model', 'optimizer', 'scheduler', 'rng'}


def selection_update(val, epoch, best, constrained, floor):
    """Return selections and write flags; exact ties retain the earlier epoch."""
    if type(epoch) is not int or epoch < 1:
        raise ValueError('Epoch must be a positive integer')
    if not math.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError('IoU floor must be finite and in [0, 1]')
    best = None if best is None else dict(best)
    constrained = None if constrained is None else dict(constrained)
    if val is None:
        return best, constrained, False, False
    iou, bf1 = float(val['iou']), float(val['bf1'])
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (iou, bf1)):
        raise ValueError('Validation IoU/BF1 must be finite and in [0, 1]')
    score = .7 * iou + .3 * bf1
    if 'checkpoint_score' in val and float(val['checkpoint_score']) != score:
        raise ValueError('checkpoint_score must equal 0.7 * IoU + 0.3 * BF1')
    legacy_new = (best is None or score > best['score'] or
                  (score == best['score'] and epoch < best['epoch']))
    constrained_new = iou >= floor and (
        constrained is None or (bf1, iou) > (constrained['bf1'], constrained['iou']) or
        ((bf1, iou) == (constrained['bf1'], constrained['iou']) and
         epoch < constrained['epoch']))
    if legacy_new:
        best = {'epoch': epoch, 'score': score}
    if constrained_new:
        constrained = {'epoch': epoch, 'bf1': bf1, 'iou': iou}
    return best, constrained, legacy_new, constrained_new


def _generator_state(generator):
    if isinstance(generator, dict):
        return {name: item.get_state().cpu().clone() for name, item in generator.items()}
    return None if generator is None else generator.get_state().cpu().clone()


def capture_rng(generator):
    """Capture global RNGs and one Generator or a named map of loader generators.

    NumPy's MT19937 ndarray is encoded as a list of Python ints. All values can
    therefore round-trip through torch.load(..., weights_only=True).
    """
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        'torch': torch.get_rng_state().cpu().clone(),
        'cuda': [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [],
        'python': random.getstate(),
        'numpy': (str(numpy_state[0]), numpy_state[1].tolist(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])),
        'loader': _generator_state(generator),
    }


def restore_rng(state, generator):
    """Restore after building/loading everything and before creating iterators."""
    import numpy as np
    import torch

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state['cuda']) != cuda_count:
        raise ValueError('Resume requires the same number of visible CUDA devices')
    loader_state = state['loader']
    if isinstance(generator, dict):
        if not isinstance(loader_state, dict) or set(generator) != set(loader_state):
            raise ValueError('Loader RNG names differ from checkpoint')
    elif isinstance(loader_state, dict) or (generator is None) != (loader_state is None):
        raise ValueError('Loader RNG configuration differs from checkpoint')
    numpy_state = state['numpy']
    if not isinstance(numpy_state[1], list):
        raise ValueError('NumPy RNG must use the safe list encoding')
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])))
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])
    if isinstance(generator, dict):
        for name, item in generator.items():
            item.set_state(loader_state[name].cpu())
    elif generator is not None:
        generator.set_state(loader_state.cpu())


def _sync_directory(path):
    # Directory fsync makes rename/unlink durable on the Linux training host.
    if os.name == 'posix':
        descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    # Never silently truncate evidence from a failed earlier transaction.
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _atomic_checkpoint(path, cp, check_source):
    import torch

    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if check_source is not None:
        check_source()
    with temporary.open('xb') as stream:
        torch.save(cp, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if check_source is not None:
        check_source()
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _load_checkpoint(path):
    import torch

    return torch.load(path, map_location='cpu', weights_only=True)


def _summary(cp):
    return {key: value for key, value in cp.items() if key not in STATE_KEYS}


def _validate_checkpoint(cp):
    required = set(RUN_KEYS) | STATE_KEYS | {
        'epoch', 'record', 'best', 'constrained', 'validation', 'legacy_new',
        'constrained_new', 'format_version', 'checkpoint_kind', 'transaction_id',
    }
    if not isinstance(cp, dict) or not required <= cp.keys():
        raise ValueError('Incomplete transactional last checkpoint')
    if cp['format_version'] != FORMAT_VERSION or cp['checkpoint_kind'] != 'last':
        raise ValueError('Only this format of last.pt can resume')
    if type(cp['epoch']) is not int or cp['epoch'] < 1:
        raise ValueError('Invalid checkpoint epoch')
    if not isinstance(cp['transaction_id'], str) or not cp['transaction_id']:
        raise ValueError('Missing transaction identity')
    record = cp['record']
    if record['epoch'] != cp['epoch']:
        raise ValueError('Checkpoint and embedded record epochs differ')
    for key in ('best', 'constrained', 'validation'):
        if record[key] != cp[key]:
            raise ValueError(f'Checkpoint and record differ: {key}')
    for flag, key in (('legacy_new', 'best'), ('constrained_new', 'constrained')):
        if type(cp[flag]) is not bool:
            raise ValueError(f'Invalid checkpoint flag: {flag}')
        current = cp[key] is not None and cp[key]['epoch'] == cp['epoch']
        if cp[flag] != current:
            raise ValueError(f'Checkpoint selection/write flag mismatch: {flag}')
    # Reject NumPy scalars, tensors or NaNs accidentally inserted in metadata.
    _json_text(_summary(cp))


def _require_same_run(left, right):
    for key in RUN_KEYS:
        if left[key] != right[key]:
            raise ValueError(f'Checkpoint provenance/config differs: {key}')


def _inventory(log, weights, epoch, pending):
    allowed_weights = {'last.pt', *BEST_FILES.values()}
    recoverable = {log / 'metrics.jsonl.tmp',
                   log / 'epoch_records' / f'epoch_{epoch:03d}.json.tmp'}
    if pending:
        for key, flag in (('best', 'legacy_new'), ('constrained', 'constrained_new')):
            if pending[flag]:
                recoverable.add(weights / (BEST_FILES[key] + '.tmp'))
    for path in weights.iterdir():
        if path.name not in allowed_weights and (not pending or path not in recoverable):
            raise FileExistsError(f'Unexpected weight/orphan evidence: {path}')
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Weights must be regular local files: {path}')
    if (log / 'pending_epoch.json.tmp').exists():
        raise FileExistsError('Pre-last pending marker temporary exists; inspect manually')
    records = log / 'epoch_records'
    if records.exists():
        for path in records.iterdir():
            match = re.fullmatch(r'epoch_(\d{3})\.json', path.name)
            if not match:
                if pending and path in recoverable and path.is_file():
                    continue
                raise FileExistsError(f'Unexpected record evidence: {path}')
            if not path.is_file() or path.is_symlink() or not 1 <= int(match[1]) <= epoch:
                raise FileExistsError(f'Record is not committed by last.pt: {path}')
    if (log / 'metrics.jsonl.tmp').exists() and not pending:
        raise FileExistsError('Unexplained metrics temporary; inspect manually')
    return recoverable


def _history(log, cp):
    records = []
    best = constrained = None
    floor = cp['config']['training']['iou_floor']
    for epoch in range(1, cp['epoch'] + 1):
        path = log / 'epoch_records' / f'epoch_{epoch:03d}.json'
        if epoch == cp['epoch']:
            record = cp['record']
            if path.exists() and _read_json(path) != record:
                raise ValueError(f'Committed record differs from last.pt: {path}')
        else:
            if not path.is_file():
                raise FileNotFoundError(f'Historical record cannot be recovered: {path}')
            record = _read_json(path)
        if record['epoch'] != epoch:
            raise ValueError(f'Noncontiguous epoch history: {path}')
        best, constrained, legacy_new, constrained_new = selection_update(
            record['validation'], epoch, best, constrained, floor)
        if record['best'] != best or record['constrained'] != constrained:
            raise ValueError(f'Selection history mismatch: {path}')
        if epoch == cp['epoch'] and (legacy_new != cp['legacy_new'] or
                                    constrained_new != cp['constrained_new']):
            raise ValueError('Checkpoint write flags differ from selection history')
        _json_text(record)
        records.append(record)
    metrics = log / 'metrics.jsonl'
    if metrics.exists():
        with metrics.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) > len(records) or rows != records[:len(rows)]:
            raise ValueError('metrics.jsonl is not a prefix of committed epoch records')
    return records


def _best_needs_write(weights, key, cp, records):
    path = weights / BEST_FILES[key]
    selection = cp[key]
    if selection is None:
        if path.exists():
            raise FileExistsError(f'Unexpected unselected checkpoint: {path}')
        return False
    selected_epoch = selection['epoch']
    reconstructible = selected_epoch == cp['epoch']
    if not path.exists():
        if not reconstructible:
            raise FileNotFoundError(f'Historical selected weights cannot be recovered: {path}')
        return True
    saved = _load_checkpoint(path)
    _require_same_run(saved, cp)
    if saved.get('checkpoint_kind') != path.stem or saved.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'Unexpected checkpoint format: {path}')
    saved_epoch = saved.get('epoch')
    if type(saved_epoch) is not int or not 1 <= saved_epoch <= selected_epoch:
        raise ValueError(f'Selected checkpoint is ahead of last.pt or invalid: {path}')
    expected = records[saved_epoch - 1]
    if expected[key] is None or expected[key]['epoch'] != saved_epoch:
        raise ValueError(f'Checkpoint epoch was never selected: {path}')
    for field in ('best', 'constrained', 'validation'):
        if saved.get(field) != expected[field]:
            raise ValueError(f'Selected checkpoint metadata differs: {path} ({field})')
    if saved.get('record') != expected:
        raise ValueError(f'Selected checkpoint record differs: {path}')
    if saved_epoch == cp['epoch'] and saved.get('transaction_id') != cp['transaction_id']:
        raise ValueError(f'Selected checkpoint belongs to another transaction: {path}')
    if saved_epoch == selected_epoch:
        return False
    if not reconstructible:
        raise ValueError(f'Stale selected weights cannot be reconstructed: {path}')
    return True


def recover_epoch(log, weights, cp, *, check_source=None):
    """Materialize one authoritative last checkpoint, idempotently.

    ``cp`` MUST be the weights_only=True CPU load of ``weights/last.pt`` (or the
    object just atomically written there by commit_epoch). Future/unexplained
    evidence is refused, not discarded. Missing older records/selected weights
    cannot be reconstructed because non-selected historical weights never exist.
    """
    log, weights = Path(log), Path(weights)
    if not log.is_dir() or not weights.is_dir() or not (weights / 'last.pt').is_file():
        raise FileNotFoundError('Resume requires existing log/weights and last.pt')
    _validate_checkpoint(cp)
    marker = log / 'pending_epoch.json'
    pending = _read_json(marker) if marker.exists() else None
    if pending is not None and pending != _summary(cp):
        raise FileExistsError('Pending epoch is not last.pt; pre-last crash evidence requires inspection')
    recoverable = _inventory(log, weights, cp['epoch'], pending)
    records = _history(log, cp)
    writes = {key: _best_needs_write(weights, key, cp, records) for key in BEST_FILES}
    if check_source is not None:
        check_source()
    metrics_text = ''.join(_json_text(record) + '\n' for record in records)
    metrics = log / 'metrics.jsonl'
    record_path = log / 'epoch_records' / f"epoch_{cp['epoch']:03d}.json"
    needs_repair = (not record_path.exists() or any(writes.values()) or
                    not metrics.exists() or metrics.read_text(encoding='utf-8') != metrics_text)
    # Recovery itself is transactional, including repair after a removed marker.
    if pending is None and needs_repair:
        pending = _summary(cp)
        _atomic_text(marker, _json_text(pending) + '\n')
        recoverable = _inventory(log, weights, cp['epoch'], pending)
    # Only temporaries whose intent is vouched for by this last.pt may be removed.
    if pending:
        for path in recoverable:
            if path.exists():
                path.unlink()
                _sync_directory(path.parent)
    record_root = log / 'epoch_records'
    record_root.mkdir(exist_ok=True)
    record_path = record_root / f"epoch_{cp['epoch']:03d}.json"
    if not record_path.exists():
        _atomic_text(record_path, _json_text(cp['record']) + '\n')
    for key, needed in writes.items():
        if needed:
            compact = {name: value for name, value in cp.items()
                       if name not in {'optimizer', 'scheduler', 'rng'}}
            compact['checkpoint_kind'] = Path(BEST_FILES[key]).stem
            _atomic_checkpoint(weights / BEST_FILES[key], compact, check_source)
    if not metrics.exists() or metrics.read_text(encoding='utf-8') != metrics_text:
        _atomic_text(metrics, metrics_text)
    if pending:
        marker.unlink()
        _sync_directory(log)
    return cp['record']


def commit_epoch(log, weights, cp, *, check_source=None):
    """Commit pending metadata -> last.pt -> record/best/metrics -> clear pending.

    The caller supplies all training states plus record and selection flags.
    This function adds format_version, checkpoint_kind and transaction_id to cp.
    ``check_source`` is called immediately before every checkpoint save/rename.
    Exceptions deliberately retain transaction evidence for safe recovery.
    """
    log, weights = Path(log), Path(weights)
    if not log.is_dir() or not weights.is_dir():
        raise FileNotFoundError('Create this run log and weights directories first')
    cp.setdefault('format_version', FORMAT_VERSION)
    cp.setdefault('checkpoint_kind', 'last')
    cp.setdefault('transaction_id', uuid.uuid4().hex)
    _validate_checkpoint(cp)
    last = weights / 'last.pt'
    if last.exists():
        previous = _load_checkpoint(last)
        _require_same_run(previous, cp)
        if previous['epoch'] + 1 != cp['epoch']:
            raise ValueError('Commit must advance last.pt by exactly one epoch')
        recover_epoch(log, weights, previous, check_source=check_source)
        del previous
    else:
        if cp['epoch'] != 1:
            raise ValueError('First commit must be epoch 1')
        if (log / 'pending_epoch.json').exists() or (log / 'metrics.jsonl').exists():
            raise FileExistsError('Pre-last orphan evidence; refusing a new first commit')
        _inventory(log, weights, 0, None)
        if any(weights.iterdir()):
            raise FileExistsError('Weights without last.pt are orphan evidence')
    # Validate candidate history before publishing last.pt, not afterwards.
    _inventory(log, weights, cp['epoch'] - 1, None)
    _history(log, cp)
    if check_source is not None:
        check_source()
    _atomic_text(log / 'pending_epoch.json', _json_text(_summary(cp)) + '\n')
    _atomic_checkpoint(last, cp, check_source)
    return recover_epoch(log, weights, cp, check_source=check_source)
