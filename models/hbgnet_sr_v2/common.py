"""Frozen HBGNet Protocol B helpers for joint local-separation v2 supervision."""
from pathlib import Path, PurePosixPath
import copy
import hashlib
import json
import random
import re
import sys

ROOT = Path(__file__).resolve().parent
# The source staging layout is package/ + tools/; deployment is models/<package>/.
PROJECT_ROOT = ROOT.parent if ROOT.name == 'package' else ROOT.parents[1]
SHARED_DEPS = ROOT.parent / 'hbgnet' / '_deps'
for directory in (PROJECT_ROOT, PROJECT_ROOT / '.python_packages', SHARED_DEPS, ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from protocol.dataset import AI4BoundariesSEANetDataset, load_config, resolve_dataset_variant
from implementation.model import Field
from implementation.losses import BCEDiceLoss, LossMulti
from tools.region_boundary_band_loss_v2 import boundary_band_bce, region_boundary_band

RUN_CONFIGS = {
    'hbgnet_protocol_b_sr_v2': 'config_protocol_b.yaml',
    'hbgnet_protocol_b_sr_v2_disabled': 'config_disabled.yaml',
    'hbgnet_protocol_b_sr_v2_raw': 'config_raw.yaml',
    'hbgnet_protocol_b_sr_v2_band': 'config_band.yaml',
    'hbgnet_protocol_b_sr_v2_nosr_band': 'config_nosr_band.yaml',
}
RESERVED_RUNS = {
    'hbgnet_protocol_b_sr_v2_' + suffix
    for suffix in ('stopraw', 'stopall', 'linear', 'finalband')
}
STAGE = 'HBGNet_local_separation_refinement_v2'
REFINEMENT = {
    'enabled': True, 'max_delta': 2.0, 'checkpoint_groups': True,
    'mode': 'local_separation_v2', 'hidden_channels': 16,
    'path_channels': 4, 'radii': [2, 4],
}
# Ordered raw region, final region, raw band, final band. No schedules or overrides.
LOSS_COEFFICIENTS = {
    'hbgnet_protocol_b_sr_v2': (0.5, 0.5, 0.125, 0.125),
    'hbgnet_protocol_b_sr_v2_disabled': (0.0, 1.0, 0.0, 0.0),
    'hbgnet_protocol_b_sr_v2_raw': (0.5, 0.5, 0.0, 0.0),
    'hbgnet_protocol_b_sr_v2_band': (0.0, 1.0, 0.0, 0.25),
    'hbgnet_protocol_b_sr_v2_nosr_band': (1.0, 0.0, 0.25, 0.0),
}
PROTOCOL_SECTIONS = (
    'dataset', 'augmentation', 'normalization', 'loss', 'training', 'evaluation', 'paths',
)
EXTERNAL_FILES = {
    'tools/__init__.py', 'tools/local_separation_refinement_v2.py',
    'tools/region_boundary_band_loss_v2.py',
}
REQUIRED_PACKAGE_FILES = {
    'common.py', 'train.py', 'evaluate.py', 'training_state.py',
    'config_protocol_b.yaml', 'config_disabled.yaml', 'config_raw.yaml',
    'config_band.yaml', 'config_nosr_band.yaml', 'run_train.sh', 'run_evaluate.sh',
    'implementation/__init__.py', 'implementation/model.py', 'implementation/losses.py',
    'implementation/lib/__init__.py', 'implementation/lib/pvtv2.py',
    'protocol/__init__.py', 'protocol/dataset.py', 'protocol/metrics.py',
    'protocol/hbgnet_baseline.yaml', 'verify_sr.py', 'verify_band_loss.py',
    'verify_local_separation.py', 'verify_training_state.py',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _same(left, right):
    """Type-aware equality: True is not 1, and unknown nested keys are refused."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f'Configuration keys must be unique strings: {key!r}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _raw_config(path):
    with Path(path).open(encoding='utf-8') as stream:
        result = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(result, dict):
        raise ValueError('Configuration must be a mapping')
    return result


def _expected_loss(run):
    if run in RESERVED_RUNS:
        raise ValueError(f'Reserved v2 experiment is not implemented in this delivery: {run}')
    if run not in RUN_CONFIGS:
        raise ValueError('Unexpected local-separation v2 run name')
    raw, final, raw_band, final_band = LOSS_COEFFICIENTS[run]
    return {
        'name': 'dual_region_band_v2',
        # Integer multipliers for the combined Region objective, Boundary, Distance.
        'weights': [1, 1, 1],
        'raw_region_weight': raw, 'final_region_weight': final,
        'raw_band_weight': raw_band, 'final_band_weight': final_band,
        'band_radius_pixels': 2,
    }


def _expected_config(run, *, resolved):
    loss = _expected_loss(run)
    baseline_path = ROOT / 'protocol' / 'hbgnet_baseline.yaml'
    expected = _raw_config(baseline_path)
    if resolved:
        expected = resolve_dataset_variant(expected)
    expected['experiment'] = {'run_name': run, 'stage': STAGE}
    expected['model']['refinement'] = copy.deepcopy(REFINEMENT)
    expected['model']['refinement']['enabled'] = run not in {
        'hbgnet_protocol_b_sr_v2_disabled', 'hbgnet_protocol_b_sr_v2_nosr_band',
    }
    expected['loss'] = loss
    return expected


def _validate_config(cfg, *, resolved=True):
    sections = {'experiment', 'model', *PROTOCOL_SECTIONS}
    if not isinstance(cfg, dict) or set(cfg) != sections:
        raise ValueError('Unexpected configuration sections')
    experiment = cfg['experiment']
    if not isinstance(experiment, dict) or not isinstance(experiment.get('run_name'), str):
        raise ValueError('Missing experiment run name')
    expected = _expected_config(experiment['run_name'], resolved=resolved)
    for key in ('experiment', 'model', *PROTOCOL_SECTIONS):
        if not _same(cfg[key], expected[key]):
            raise ValueError(f'Frozen HBGNet/local-separation v2 configuration mismatch: {key}')
    return cfg


def config(path):
    # Validate unoverlaid YAML too, so forbidden dataset overrides cannot disappear.
    _validate_config(_raw_config(path), resolved=False)
    cfg = _validate_config(load_config(path))
    # All five canonical basenames have fixed identities, including supplied copies.
    named_run = {filename: run for run, filename in RUN_CONFIGS.items()}.get(Path(path).name)
    if named_run is not None and cfg['experiment']['run_name'] != named_run:
        raise ValueError('Canonical configuration filename/run/refinement/loss mismatch')
    config_source_path(cfg)
    return cfg


def config_source_path(cfg):
    """Return the selected canonical YAML without adding source metadata to cfg."""
    _validate_config(cfg)
    path = ROOT / RUN_CONFIGS[cfg['experiment']['run_name']]
    _validate_config(_raw_config(path), resolved=False)
    if not _same(load_config(path), cfg):
        raise ValueError('Selected canonical YAML differs from the run configuration')
    return path


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate manifest/report key: {key}')
        result[key] = value
    return result


def _manifest_file(base, name):
    """Only canonical relative POSIX file keys, with no escape or symlink hops."""
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
            '\x00' in name or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Unsafe manifest path: {name!r}')
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name:
        raise ValueError(f'Non-relative manifest path: {name!r}')
    base = Path(base).resolve()
    path = base
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Manifest file must not traverse symlinks: {name}')
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise ValueError(f'Manifest file escapes its root or is not a file: {name}')
    return resolved


def source_fingerprint():
    manifest = json.loads((ROOT / 'package_manifest.json').read_text(encoding='utf-8'),
                          object_pairs_hook=_unique_json)
    if not isinstance(manifest, dict):
        raise ValueError('Invalid package manifest')
    package_files = manifest.get('files_sha256')
    external_files = manifest.get('external_files_sha256')
    if not isinstance(package_files, dict) or not REQUIRED_PACKAGE_FILES <= package_files.keys():
        raise ValueError('Package manifest omits required source/config files')
    if not isinstance(external_files, dict) or set(external_files) != EXTERNAL_FILES:
        raise ValueError('External manifest must cover exactly the three local-separation v2 tools files')
    actual = {}
    for base, entries, prefix in ((ROOT, package_files, ''),
                                   (PROJECT_ROOT, external_files, '@project/')):
        for name, expected in entries.items():
            if not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
                raise ValueError(f'Invalid SHA256 in package manifest: {name!r}')
            path = _manifest_file(base, name)
            digest = sha256(path)
            if digest != expected:
                raise ValueError(f'Source/config changed after packaging: {prefix}{name}')
            actual[prefix + name] = digest
    return actual


def require_engineering(cfg, fingerprint):
    _validate_config(cfg)
    report = json.loads((ROOT / 'verification.json').read_text(encoding='utf-8'),
                        object_pairs_hook=_unique_json)
    if (not isinstance(report, dict) or report.get('status') != 'pass' or
            not _same(report.get('source_fingerprint'), fingerprint)):
        raise ValueError('Missing or stale local-separation v2 engineering verification')
    configs = report.get('configs')
    if not isinstance(configs, dict) or set(configs) != set(RUN_CONFIGS):
        raise ValueError('Verification must cover all five v2 configurations')
    for run in RUN_CONFIGS:
        if not _same(configs[run], _expected_config(run, resolved=True)):
            raise ValueError(f'Verification configs/run mismatch: {run}; v1 verification cannot be reused')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def loader(cfg, split, batch=None, workers=None):
    data = AI4BoundariesSEANetDataset(cfg, split, augment=(split == 'train'))
    generator = torch.Generator().manual_seed(cfg['training']['seed'])
    return DataLoader(data, batch_size=batch or cfg['dataset']['batch_size'], shuffle=(split == 'train'),
        num_workers=cfg['dataset']['num_workers'] if workers is None else workers,
        drop_last=(split == 'train'), pin_memory=True, worker_init_fn=seed_worker,
        generator=generator, persistent_workers=False)


def build(cfg, pretrained=True):
    _validate_config(cfg)
    options = cfg['model']
    path = Path(options['pretrained_path'])
    if pretrained:
        if not path.is_file():
            raise FileNotFoundError(f'Official ImageNet PVT-v2-B2 pretraining required: {path}; random fallback forbidden')
        receipt = path.with_suffix('.json')
        if not receipt.is_file():
            raise ValueError('Missing pretraining provenance receipt')
        evidence = json.loads(receipt.read_text(encoding='utf-8'))
        if evidence.get('sha256') != sha256(path) or not evidence.get('source_url') or not evidence.get('source_evidence'):
            raise ValueError('Mismatched or incomplete pretraining provenance receipt')
    return Field(channel=options['channel'], num_classes=options['boundary_classes'],
                 drop_rate=options['drop_rate'], pretrained_path=str(path) if pretrained else None,
                 refinement=copy.deepcopy(options['refinement']))


def loss_fn(outputs, batch, cfg, training_aux, return_components=False):
    """Joint v2 objective; auxiliary raw logits must come from the training return.

    Components are flat, detached Python floats (inactive terms are zero), with
    unweighted names and weighted_<name> counterparts, plus total. The integer
    weights [1, 1, 1] multiply combined Region, Boundary NLL and Distance MSE.
    """
    if not isinstance(cfg, dict) or not isinstance(cfg.get('experiment'), dict):
        raise ValueError('Loss requires an explicit v2 run configuration')
    run = cfg['experiment'].get('run_name')
    if not isinstance(run, str) or not _same(cfg.get('loss'), _expected_loss(run)):
        raise ValueError('Frozen v2 loss configuration mismatch')
    options = cfg['loss']
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise ValueError('HBGNet must return exactly Region, Boundary, Distance outputs')
    final, edge, distance = outputs
    for name, tensor in zip(('region', 'boundary', 'distance'), outputs):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or not tensor.is_floating_point():
            raise ValueError(f'HBGNet output must be a floating [B,C,H,W] tensor: {name}')
        if not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f'Non-finite HBGNet output: {name}')
    if final.shape[1] != 1 or min(final.shape) < 1:
        raise ValueError('Region logits must have nonempty shape [B,1,H,W]')
    shape = final.shape
    if edge.shape != (shape[0], 2, shape[2], shape[3]) or distance.shape != shape:
        raise ValueError('Boundary/Distance output shape mismatch')
    for name in ('region', 'boundary', 'distance'):
        target = batch[name]
        if not isinstance(target, torch.Tensor) or target.shape != shape or target.device != final.device:
            raise ValueError(f'Target must match Region shape/device: {name}')
        if not torch.isfinite(target).all().item():
            raise FloatingPointError(f'Non-finite HBGNet target: {name}')
    if edge.device != final.device or distance.device != final.device:
        raise ValueError('All HBGNet outputs must be on the same device')
    target = batch['region']
    if not target.is_floating_point() or not ((target == 0) | (target == 1)).all().item():
        raise ValueError('Region GT must be a floating binary mask after augmentation')
    if not ((batch['boundary'] == 0) | (batch['boundary'] == 1)).all().item():
        raise ValueError('Boundary GT must be binary before conversion to long')
    raw = None
    if options['raw_region_weight'] > 0 or options['raw_band_weight'] > 0:
        if not isinstance(training_aux, dict) or 'raw_region_logits' not in training_aux:
            raise ValueError('Enabled raw supervision requires return_training_aux=True')
        raw = training_aux['raw_region_logits']
        if (not isinstance(raw, torch.Tensor) or raw.shape != shape or
                raw.dtype != final.dtype or raw.device != final.device):
            raise ValueError('Training raw logits must match final [B,1,H,W], dtype and device')
        if not torch.isfinite(raw).all().item():
            raise FloatingPointError('Non-finite training raw logits')
        if torch.is_grad_enabled() and (not raw.requires_grad or raw.grad_fn is None):
            raise ValueError('Raw loss requires connected model logits, not detached diagnostic auxiliary')
    zero = final.new_zeros(())
    region_criterion = BCEDiceLoss()
    radius = options['band_radius_pixels']
    components = {
        'raw_region': region_criterion(raw, target) if options['raw_region_weight'] > 0 else zero,
        'final_region': region_criterion(final, target) if options['final_region_weight'] > 0 else zero,
        'raw_band': boundary_band_bce(raw, target, radius=radius) if options['raw_band_weight'] > 0 else zero,
        'final_band': boundary_band_bce(final, target, radius=radius) if options['final_band_weight'] > 0 else zero,
        'boundary_nll': LossMulti(num_classes=2)(edge, batch['boundary'].long()),
        'distance_mse': torch.nn.MSELoss()(distance, batch['distance']),
    }
    weighted = {
        name: options[name + '_weight'] * value
        for name, value in components.items() if name not in ('boundary_nll', 'distance_mse')
    }
    # Avoid extra additions/multiplications in the disabled objective: this is
    # the identical criterion order and arithmetic as original LossF([1,1,1]).
    if run == 'hbgnet_protocol_b_sr_v2_disabled':
        region_loss = components['final_region']
    elif run == 'hbgnet_protocol_b_sr_v2_nosr_band':
        region_loss = weighted['raw_region'] + weighted['raw_band']
    else:
        region_loss = (weighted['raw_region'] + weighted['final_region'] +
                       weighted['raw_band'] + weighted['final_band'])
    task_weights = options['weights']
    total = (task_weights[0] * region_loss + task_weights[1] * components['boundary_nll'] +
             task_weights[2] * components['distance_mse'])
    if not torch.isfinite(total).all().item():
        raise FloatingPointError('Non-finite v2 total loss')
    if not return_components:
        return total
    weighted = {name: task_weights[0] * value for name, value in weighted.items()}
    weighted['boundary_nll'] = task_weights[1] * components['boundary_nll']
    weighted['distance_mse'] = task_weights[2] * components['distance_mse']
    values = {name: float(value.detach().item()) for name, value in components.items()}
    values.update({'weighted_' + name: float(value.detach().item()) for name, value in weighted.items()})
    values['total'] = float(total.detach().item())
    return total, values


def device_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)


def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def _thin_background_proxy(truth):
    """Zero-filled two-sided foreground geometry; not instance labels or a loss."""
    height, width = truth.shape[-2:]

    def sample(dy, dx):
        result = np.zeros_like(truth, dtype=bool)
        if abs(dy) >= height or abs(dx) >= width:
            return result
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        result[..., y0:y1, x0:x1] = truth[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        return result

    candidate = np.zeros_like(truth, dtype=bool)
    for radius in (2, 4):
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            candidate |= sample(radius * dy, radius * dx) & sample(-radius * dy, -radius * dx)
    return candidate & ~truth


@torch.no_grad()
def evaluate(model, batches, device, max_batches=None):
    """Unchanged micro validation metrics plus detached, flat scalar diagnostics.

    Band counters use augmented binary Region GT and the training radius 2.
    Far counters use distance >10px from the metric's inner GT boundary. If no
    GT boundary exists, every pixel is far (no fictitious outside-image seed).
    Delta statistics use valid_count >0; saturation is abs(delta) >=0.95*2.
    """
    from scipy.ndimage import distance_transform_edt
    from protocol.metrics import (binary_confusion, boundary_statistics, mask_boundary,
                                  metrics_from_confusion, boundary_metrics_from_statistics)
    model.eval()
    pixel = np.zeros(4, dtype=np.int64)
    boundary = np.zeros(4, dtype=np.int64)
    raw_pixel = np.zeros(4, dtype=np.int64)
    raw_boundary = np.zeros(4, dtype=np.int64)
    count = corrected = regressed = changed = valid_pixels = total_pixels = saturated = 0
    delta_abs_sum = 0.0
    diagnostics = dict.fromkeys((
        'band_pixels', 'band_fg_pixels', 'band_bg_pixels', 'band_empty_images',
        'band_valid_pixels', 'valid_empty_images', 'far_pixels', 'far_fg_pixels', 'far_bg_pixels',
        'raw_band_fg_errors', 'raw_band_bg_errors', 'final_band_fg_errors', 'final_band_bg_errors',
        'raw_far_fp', 'raw_far_fn', 'final_far_fp', 'final_far_fn',
        'valid_corrected_pixels', 'valid_regressed_pixels',
        'invalid_corrected_pixels', 'invalid_regressed_pixels',
        'thin_bg_candidates', 'thin_bg_raw_fp', 'thin_bg_final_fp',
        'thin_bg_corrected', 'thin_bg_regressed',
        'thin_bg_residual_fp_raw_logit_gt2', 'thin_bg_residual_fp_delta_positive',
        'thin_bg_raw_fp_gt2_total', 'thin_bg_raw_fp_gt2_corrected',
        'thin_bg_raw_fp_le2_total', 'thin_bg_raw_fp_le2_corrected',
    ), 0)
    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        out = model(batch['image'].to(device))[0]
        if not torch.isfinite(out).all():
            raise FloatingPointError('Non-finite validation output')
        auxiliary = getattr(model, 'last_auxiliary', None)
        keys = ('raw_region_logits', 'refined_region_logits', 'delta', 'valid_count')
        if not isinstance(auxiliary, dict) or any(key not in auxiliary for key in keys):
            raise ValueError('Model must expose raw/refined/delta/valid_count validation diagnostics')
        tensors = [auxiliary[key] for key in keys]
        for key, tensor in zip(keys, tensors):
            if not isinstance(tensor, torch.Tensor) or tensor.shape != out.shape:
                raise ValueError(f'Auxiliary must match region logits [B,1,H,W]: {key}')
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f'Non-finite validation auxiliary: {key}')
        raw, refined, delta, valid_count = [tensor.detach() for tensor in tensors]
        if not torch.equal(out.detach(), refined):
            raise ValueError('Refined auxiliary does not match the primary region output')
        pred = refined.sigmoid().cpu().numpy() >= .5
        raw_pred = raw.sigmoid().cpu().numpy() >= .5
        truth = batch['region'].detach().cpu().numpy() >= .5
        if pred.shape != truth.shape or pred.ndim != 4 or pred.shape[1] != 1:
            raise ValueError('Region prediction/target shape mismatch')
        corrected_mask = (raw_pred != truth) & (pred == truth)
        regressed_mask = (raw_pred == truth) & (pred != truth)
        changed += int(np.count_nonzero(pred != raw_pred))
        corrected += int(corrected_mask.sum())
        regressed += int(regressed_mask.sum())
        valid = valid_count > 0
        n_valid = int(valid.sum().item())
        valid_pixels += n_valid
        total_pixels += pred.size
        if n_valid:
            absolute = delta[valid].abs()
            delta_abs_sum += float(absolute.double().sum().item())
            saturated += int((absolute >= .95 * REFINEMENT['max_delta']).sum().item())
            del absolute
        valid_cpu = valid.cpu().numpy()
        band = region_boundary_band(batch['region'].detach().cpu(), radius=2).bool().numpy()
        if band.shape != truth.shape:
            raise ValueError('GT boundary band must match Region [B,1,H,W]')
        band_fg, band_bg = band & truth, band & ~truth
        diagnostics['band_pixels'] += int(band.sum())
        diagnostics['band_fg_pixels'] += int(band_fg.sum())
        diagnostics['band_bg_pixels'] += int(band_bg.sum())
        diagnostics['band_empty_images'] += int((~band.reshape(len(band), -1).any(axis=1)).sum())
        diagnostics['band_valid_pixels'] += int((band & valid_cpu).sum())
        diagnostics['valid_empty_images'] += int((~valid_cpu.reshape(len(band), -1).any(axis=1)).sum())
        for name, prediction in (('raw', raw_pred), ('final', pred)):
            diagnostics[name + '_band_fg_errors'] += int((band_fg & ~prediction).sum())
            diagnostics[name + '_band_bg_errors'] += int((band_bg & prediction).sum())
        for name, mask in (('corrected', corrected_mask), ('regressed', regressed_mask)):
            diagnostics['valid_' + name + '_pixels'] += int((mask & valid_cpu).sum())
            diagnostics['invalid_' + name + '_pixels'] += int((mask & ~valid_cpu).sum())
        candidate = _thin_background_proxy(truth)
        raw_fp, final_fp = candidate & raw_pred, candidate & pred
        raw_gt2 = raw.cpu().numpy() > 2
        diagnostics['thin_bg_candidates'] += int(candidate.sum())
        diagnostics['thin_bg_raw_fp'] += int(raw_fp.sum())
        diagnostics['thin_bg_final_fp'] += int(final_fp.sum())
        diagnostics['thin_bg_corrected'] += int((raw_fp & ~pred).sum())
        diagnostics['thin_bg_regressed'] += int((final_fp & ~raw_pred).sum())
        diagnostics['thin_bg_residual_fp_raw_logit_gt2'] += int((final_fp & raw_gt2).sum())
        diagnostics['thin_bg_residual_fp_delta_positive'] += int((final_fp & (delta.cpu().numpy() > 0)).sum())
        for name, mask in (('gt2', raw_gt2), ('le2', ~raw_gt2)):
            diagnostics['thin_bg_raw_fp_' + name + '_total'] += int((raw_fp & mask).sum())
            diagnostics['thin_bg_raw_fp_' + name + '_corrected'] += int((raw_fp & mask & ~pred).sum())
        for p, rp, t in zip(pred[:, 0], raw_pred[:, 0], truth[:, 0]):
            pixel += binary_confusion(p, t)
            boundary += boundary_statistics(p, t, 3)
            raw_pixel += binary_confusion(rp, t)
            raw_boundary += boundary_statistics(rp, t, 3)
            gt_boundary = mask_boundary(t)
            far = (distance_transform_edt(~gt_boundary) > 10 if gt_boundary.any()
                   else np.ones_like(t, dtype=bool))
            diagnostics['far_pixels'] += int(far.sum())
            diagnostics['far_fg_pixels'] += int((far & t).sum())
            diagnostics['far_bg_pixels'] += int((far & ~t).sum())
            for name, prediction in (('raw', rp), ('final', p)):
                diagnostics[name + '_far_fp'] += int((far & prediction & ~t).sum())
                diagnostics[name + '_far_fn'] += int((far & ~prediction & t).sum())
            count += 1
        # No graph/tensor history is kept across batches or returned to TensorBoard.
        del out, auxiliary, tensors, tensor, raw, refined, delta, valid_count, valid
    if count == 0:
        raise ValueError('Validation must contain at least one sample')
    _, p, r, f1, iou = metrics_from_confusion(*map(int, pixel))
    com, corr, bf1 = boundary_metrics_from_statistics(*map(int, boundary))
    raw_iou = metrics_from_confusion(*map(int, raw_pixel))[4]
    raw_com, raw_corr, raw_bf1 = boundary_metrics_from_statistics(*map(int, raw_boundary))
    return dict(samples=count, iou=iou, f1=f1, precision=p, recall=r,
                boundary_completeness=com, boundary_correctness=corr, bf1=bf1,
                checkpoint_score=.7 * iou + .3 * bf1,
                raw_iou=raw_iou, raw_bf1=raw_bf1,
                raw_boundary_completeness=raw_com, raw_boundary_correctness=raw_corr,
                raw_fp=int(raw_pixel[1]), raw_fn=int(raw_pixel[2]),
                final_fp=int(pixel[1]), final_fn=int(pixel[2]),
                corrected_pixels=corrected, regressed_pixels=regressed,
                net_corrected_pixels=corrected - regressed, changed_pixels=changed,
                delta_abs_mean=delta_abs_sum / valid_pixels if valid_pixels else 0.0,
                delta_saturation_fraction=saturated / valid_pixels if valid_pixels else 0.0,
                valid_pixels=valid_pixels, invalid_pixels=total_pixels - valid_pixels,
                total_pixels=total_pixels,
                valid_fraction=valid_pixels / total_pixels if total_pixels else 0.0,
                band_fraction=diagnostics['band_pixels'] / total_pixels if total_pixels else 0.0,
                **diagnostics)
