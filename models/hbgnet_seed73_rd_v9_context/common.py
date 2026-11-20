"""HBGNet Protocol B helpers for context-supported rd_v9_context candidate verification."""
from pathlib import Path, PurePosixPath
import copy
import hashlib
import json
import random
import re
import sys

ROOT = Path(__file__).resolve().parent
# Staging is package/ + tools/; deployment is SEANet/models/<package> with
# shared tools in SEANet/tools. Resolve both layouts explicitly.
if ROOT.name == 'package':
    PROJECT_ROOT = ROOT.parent
elif ROOT.parent.name == 'models':
    PROJECT_ROOT = ROOT.parents[1]
else:
    PROJECT_ROOT = ROOT.parents[1]
SHARED_DEPS = ROOT.parent / 'hbgnet' / '_deps'
for directory in (PROJECT_ROOT, PROJECT_ROOT / '.python_packages', SHARED_DEPS, ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from protocol.dataset import AI4BoundariesSEANetDataset, load_config, resolve_dataset_variant
from protocol.candidate_diagnostics import (accumulate, contingency, legacy_aliases,
                                            net_pixels, new_counts, rates, strict_labels)
from implementation.model import Field
from implementation.losses import BCEDiceLoss, LossMulti
from tools.region_boundary_band_loss_v2 import boundary_band_bce, region_boundary_band
from tools.region_discrimination_v9 import verifier_loss

MAIN_RUN = 'hbgnet_protocol_b_seed73_rd_v9_context'
RUN_CONFIGS = {MAIN_RUN: 'config_protocol_b.yaml'}
# The historical four-group comparison is retained as explicit provenance. A post-SR
# candidate verifier has no meaningful RD-only route when SR is disabled, so those
# identities are refused rather than silently mapped to another objective. The
# center-only control and the historical v8 identities are reserved as well: this
# package is authorized for one identity only.
RESERVED_RUNS = {
    'hbgnet_protocol_b_baseline',
    'hbgnet_protocol_b_baseline_rd',
    'hbgnet_protocol_b_sr_v2',
    'hbgnet_protocol_b_rd_v9_context_sr_v2_verifier_disabled',
    'hbgnet_protocol_b_rd_v9_context_sr_v2_always_reject',
    'hbgnet_protocol_b_rd_v9_context_sr_v2_loss_matched_no_verifier',
    'hbgnet_protocol_b_rd_v9_context_sr_v2_symmetric_bce',
    'hbgnet_protocol_b_rd_v9_context_sr_v2_center_only_control',
    'hbgnet_protocol_b_seed17_rd_v8_grok_fullloss',
    'hbgnet_protocol_b_seed73_rd_v8_grok_fullloss',
}
ABLATION_GROUPS = {
    'baseline': {
        'run': 'hbgnet_protocol_b_baseline',
        'status': 'historical_reference_only',
        'sr_enabled': False, 'verifier_enabled': False,
        'reason': 'use the frozen baseline package for this reference',
    },
    'baseline_plus_rd': {
        'run': 'hbgnet_protocol_b_baseline_rd',
        'status': 'reserved_not_representable',
        'sr_enabled': False, 'verifier_enabled': False,
        'reason': 'the post-SR verifier has no candidate edits when SR is disabled',
    },
    'baseline_plus_sr_v2': {
        'run': 'hbgnet_protocol_b_sr_v2',
        'status': 'historical_reference_only',
        'sr_enabled': True, 'verifier_enabled': False,
        'reason': 'use the independently frozen SR v2 package for this reference',
    },
    'baseline_plus_rd_plus_sr_v2': {
        'run': MAIN_RUN,
        'status': 'main_group_only',
        'sr_enabled': True, 'verifier_enabled': True,
        'reason': 'the only group authorized for this training transaction',
    },
}
STAGE = 'HBGNet_seed73_rd_v9_context'
VERIFIER_LOSS_MODE = 'ordinary_bce_present_direction_average'
REFINEMENT = {
    'enabled': True, 'max_delta': 2.0, 'checkpoint_groups': True,
    'mode': 'local_separation_v2', 'hidden_channels': 16,
    'path_channels': 4, 'radii': [2, 4],
}
# Ordered raw region, SR region, raw band, SR band. The gated final is not supervised.
LOSS_COEFFICIENTS = {
    MAIN_RUN: (0.5, 0.5, 0.125, 0.125),
}
VERIFIER_SPECS = {
    MAIN_RUN: {'enabled': True, 'mode': 'enabled', 'aggregation': 'multi_scale_support'},
}
PROTOCOL_SECTIONS = (
    'dataset', 'augmentation', 'normalization', 'loss', 'training', 'evaluation', 'paths',
)
EXTERNAL_FILES = {
    'tools/__init__.py', 'tools/local_separation_refinement_v2.py',
    'tools/region_boundary_band_loss_v2.py',
    'tools/region_discrimination_v9.py',
    # The v9 module subclasses the v8 verifier, so the v8 source is a hashed
    # dependency of this package as well: the preserved scoring path cannot
    # change without failing the source fingerprint.
    'tools/region_discrimination_v8_grok_fullloss.py',
}
REQUIRED_PACKAGE_FILES = {
    'common.py', 'train.py', 'evaluate.py', 'training_state.py',
    'config_protocol_b.yaml', 'run_train.sh', 'run_evaluate.sh',
    'implementation/__init__.py', 'implementation/model.py', 'implementation/losses.py',
    'implementation/lib/__init__.py', 'implementation/lib/pvtv2.py',
    'protocol/__init__.py', 'protocol/dataset.py', 'protocol/metrics.py',
    'protocol/candidate_diagnostics.py', 'protocol/hbgnet_baseline.yaml',
    'verify_rd_v9_context.py', 'verify_v9_diagnostics.py', 'verify_band_loss.py',
    'verify_local_separation.py', 'verify_training_state.py', 'launch_rd_v9_context_audited.py',
    'README.md',
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
        raise ValueError(f'Reserved experiment is not an entry point: {run}')
    if run not in RUN_CONFIGS:
        raise ValueError('Unexpected rd_v9_context run name')
    raw, sr, raw_band, sr_band = LOSS_COEFFICIENTS[run]
    return {
        'name': 'rd_v9_context_fullloss_sr_band',
        # Integer multipliers for the combined Region objective, Boundary, Distance.
        'weights': [1, 1, 1],
        'raw_region_weight': raw, 'sr_region_weight': sr,
        'raw_band_weight': raw_band, 'sr_band_weight': sr_band,
        'band_radius_pixels': 2,
        'verifier_weight': 1.0,
        'verifier_loss_mode': 'ordinary_bce_present_direction_average',
        'final_region_loss': False,
    }


def _expected_config(run, *, resolved):
    loss = _expected_loss(run)
    baseline_path = ROOT / 'protocol' / 'hbgnet_baseline.yaml'
    expected = _raw_config(baseline_path)
    if resolved:
        expected = resolve_dataset_variant(expected)
    expected['experiment'] = {'run_name': run, 'stage': STAGE}
    expected['model']['refinement'] = copy.deepcopy(REFINEMENT)
    expected['model']['refinement']['enabled'] = True
    expected['model']['refinement']['verifier'] = copy.deepcopy(VERIFIER_SPECS[run])
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
            raise ValueError(f'rd_v9_context configuration mismatch: {key}')
    return cfg


def config(path):
    # Validate unoverlaid YAML too, so forbidden dataset overrides cannot disappear.
    _validate_config(_raw_config(path), resolved=False)
    cfg = _validate_config(load_config(path))
    # The sole canonical basename has a fixed identity, including supplied copies.
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
        raise ValueError('External manifest does not match the rd_v8_grok external source set')
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
        raise ValueError('Missing or stale rd_v9_context engineering verification')
    configs = report.get('configs')
    if not isinstance(configs, dict) or set(configs) != set(RUN_CONFIGS):
        raise ValueError('Verification must cover the canonical rd_v9_context configuration')
    for run in RUN_CONFIGS:
        if not _same(configs[run], _expected_config(run, resolved=True)):
            raise ValueError(f'Verification configs/run mismatch: {run}; another package verification cannot be reused')


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
    """Supervise raw and SR. The gated final output is not a Region target.

    Fullloss adds ordinary verifier BCE over candidate pixels and averages only
    directions present in the batch. An empty direction contributes a connected
    zero. Components are detached floats.
    """
    if not isinstance(cfg, dict) or not isinstance(cfg.get('experiment'), dict):
        raise ValueError('Loss requires an explicit rd_v8_grok run configuration')
    run = cfg['experiment'].get('run_name')
    if not isinstance(run, str) or not _same(cfg.get('loss'), _expected_loss(run)):
        raise ValueError('rd_v9_context loss configuration mismatch')
    options = cfg['loss']
    if options.get('final_region_loss') is not False:
        raise ValueError('Gated final Region loss is forbidden')
    if options.get('verifier_loss_mode') != VERIFIER_LOSS_MODE:
        raise ValueError('Verifier loss mode must be ordinary BCE with present-direction averaging')
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
    if not isinstance(training_aux, dict):
        raise ValueError('rd_v9_context requires connected raw and SR training logits')
    def connected(name):
        tensor = training_aux.get(name)
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape or
                tensor.dtype != final.dtype or tensor.device != final.device):
            raise ValueError(f'Training {name} must match final [B,1,H,W], dtype and device')
        if not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f'Non-finite training {name}')
        if torch.is_grad_enabled() and (not tensor.requires_grad or tensor.grad_fn is None):
            raise ValueError(f'{name} requires connected model logits, not detached diagnostics')
        return tensor
    raw = connected('raw_region_logits')
    sr = connected('sr_region_logits')
    zero = final.new_zeros(())
    region_criterion = BCEDiceLoss()
    radius = options['band_radius_pixels']
    components = {
        'raw_region': region_criterion(raw, target) if options['raw_region_weight'] > 0 else zero,
        'sr_region': region_criterion(sr, target) if options['sr_region_weight'] > 0 else zero,
        'raw_band': boundary_band_bce(raw, target, radius=radius) if options['raw_band_weight'] > 0 else zero,
        'sr_band': boundary_band_bce(sr, target, radius=radius) if options['sr_band_weight'] > 0 else zero,
        'boundary_nll': LossMulti(num_classes=2)(edge, batch['boundary'].long()),
        'distance_mse': torch.nn.MSELoss()(distance, batch['distance']),
    }
    verifier_term = zero
    verifier_stats = {}
    if options['verifier_weight'] > 0:
        logits = training_aux.get('verifier_logits')
        delta = training_aux.get('delta')
        expected_logits = (shape[0], 2, shape[2], shape[3])
        if (not isinstance(logits, torch.Tensor) or logits.shape != expected_logits or
                logits.dtype != final.dtype or logits.device != final.device or
                not isinstance(delta, torch.Tensor) or delta.shape != shape or
                delta.dtype != final.dtype or delta.device != final.device):
            raise ValueError('Verifier loss requires [B,2,H,W] connected logits and SR delta')
        if not torch.isfinite(logits).all().item() or not torch.isfinite(delta).all().item():
            raise FloatingPointError('Non-finite verifier loss input')
        if torch.is_grad_enabled() and (not logits.requires_grad or logits.grad_fn is None):
            raise ValueError('Verifier loss requires connected verifier logits')
        verifier_term, verifier_stats = verifier_loss(logits, raw, sr, delta, target)
    components['verifier_bce'] = verifier_term
    weighted = {
        'raw_region': options['raw_region_weight'] * components['raw_region'],
        'sr_region': options['sr_region_weight'] * components['sr_region'],
        'raw_band': options['raw_band_weight'] * components['raw_band'],
        'sr_band': options['sr_band_weight'] * components['sr_band'],
        'verifier_bce': options['verifier_weight'] * components['verifier_bce'],
    }
    region_loss = (weighted['raw_region'] + weighted['sr_region'] +
                   weighted['raw_band'] + weighted['sr_band'])
    task_weights = options['weights']
    total = (task_weights[0] * region_loss + task_weights[1] * components['boundary_nll'] +
             task_weights[2] * components['distance_mse'] + weighted['verifier_bce'])
    if not torch.isfinite(total).all().item():
        raise FloatingPointError('Non-finite rd_v9_context total loss')
    if not return_components:
        return total
    reported = {name: task_weights[0] * value for name, value in weighted.items() if name != 'verifier_bce'}
    reported['verifier_bce'] = weighted['verifier_bce']
    reported['boundary_nll'] = task_weights[1] * components['boundary_nll']
    reported['distance_mse'] = task_weights[2] * components['distance_mse']
    values = {name: float(value.detach().item()) for name, value in components.items()}
    values.update({'weighted_' + name: float(value.detach().item()) for name, value in reported.items()})
    values.update({'verifier_' + name: int(value) for name, value in verifier_stats.items()})
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
    """Validation on the gated final output, with raw and SR diagnostics.

    Best selection uses only final IoU and BF1. Object errors are diagnostic
    and never enter the loss or the checkpoint score.

    Direction-level candidate labels use the STRICT routing convention
    (``raw_logit > 0``, ``sr_logit > 0``, ``delta != 0``, ``truth > 0.5``) from
    ``protocol.candidate_diagnostics``; the frozen historical ``sigmoid >= 0.5``
    booleans are kept only for the route IoU/BF1 metrics. Every returned value
    is a finite int/float because ``train.py`` forwards each one to
    ``SummaryWriter.add_scalar``. An undefined rate is therefore reported as
    ``0.0`` together with its always-present denominator count; see the
    ``protocol.candidate_diagnostics`` docstring for that policy.
    """
    from scipy.ndimage import distance_transform_edt
    from protocol.metrics import (binary_confusion, boundary_statistics, mask_boundary,
                                  metrics_from_confusion, boundary_metrics_from_statistics,
                                  object_error_sums)
    model.eval()
    routes = ('raw', 'sr', 'final')
    pixel = {name: np.zeros(4, dtype=np.int64) for name in routes}
    boundary = {name: np.zeros(4, dtype=np.int64) for name in routes}
    objects = {name: np.zeros(4, dtype=np.float64) for name in routes}
    count = total_pixels = valid_pixels = saturated = 0
    delta_abs_sum = 0.0
    candidate_counts = new_counts()
    diagnostics = dict.fromkeys((
        'thin_bg_candidates', 'thin_bg_raw_fp', 'thin_bg_sr_fp', 'thin_bg_final_fp',
        'thin_fg_candidates', 'thin_fg_raw_fn', 'thin_fg_sr_fn', 'thin_fg_final_fn',
        'far_pixels', 'raw_far_fp', 'sr_far_fp', 'final_far_fp',
        'raw_far_fn', 'sr_far_fn', 'final_far_fn',
    ), 0)
    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        out = model(batch['image'].to(device))[0]
        if not torch.isfinite(out).all():
            raise FloatingPointError('Non-finite validation output')
        auxiliary = getattr(model, 'last_auxiliary', None)
        scalar_keys = ('raw_region_logits', 'sr_region_logits', 'final_region_logits',
                       'delta', 'edit', 'cut_edit', 'fill_edit', 'keep',
                       'cut_keep', 'fill_keep', 'valid_count')
        if not isinstance(auxiliary, dict) or any(key not in auxiliary for key in scalar_keys + ('verifier_logits',)):
            raise ValueError('Model must expose raw/SR/final and directional verifier diagnostics')
        tensors = {key: auxiliary[key] for key in scalar_keys}
        for key, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor) or tensor.shape != out.shape:
                raise ValueError(f'Auxiliary must match region logits [B,1,H,W]: {key}')
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise FloatingPointError(f'Non-finite validation auxiliary: {key}')
        verifier_logits = auxiliary['verifier_logits']
        expected_verifier_shape = (out.shape[0], 2, out.shape[2], out.shape[3])
        if (not isinstance(verifier_logits, torch.Tensor) or
                verifier_logits.shape != expected_verifier_shape):
            raise ValueError('Verifier diagnostics must have shape [B,2,H,W]')
        if not torch.isfinite(verifier_logits).all():
            raise FloatingPointError('Non-finite validation verifier logits')
        if not torch.equal((tensors['cut_edit'].bool() | tensors['fill_edit'].bool()), tensors['edit'].bool()):
            raise ValueError('Directional edit diagnostics do not match edit union')
        if not torch.equal((tensors['cut_keep'].bool() | tensors['fill_keep'].bool()), tensors['keep'].bool()):
            raise ValueError('Directional keep diagnostics do not match keep union')
        if not torch.equal(out.detach(), tensors['final_region_logits'].detach()):
            raise ValueError('Final auxiliary does not match the primary region output')
        raw = tensors['raw_region_logits'].detach()
        sr = tensors['sr_region_logits'].detach()
        final = tensors['final_region_logits'].detach()
        delta = tensors['delta'].detach()
        valid = tensors['valid_count'].detach()
        # Validation preserves the frozen historical >=0.5 threshold for the
        # route IoU/BF1 metrics only. Direction labels use strict logit > 0.
        predictions = {
            'raw': raw.sigmoid().cpu().numpy() >= .5,
            'sr': sr.sigmoid().cpu().numpy() >= .5,
            'final': final.sigmoid().cpu().numpy() >= .5,
        }
        truth = batch['region'].detach().cpu().numpy() >= .5
        if predictions['final'].shape != truth.shape or truth.ndim != 4 or truth.shape[1] != 1:
            raise ValueError('Region prediction/target shape mismatch')
        # Recompute the direction labels from the strict routing convention and
        # refuse any disagreement with the model's own candidate masks.
        cpu = lambda tensor: tensor.detach().cpu()
        labels = strict_labels(cpu(raw), cpu(sr), cpu(delta),
                               torch.from_numpy(truth.astype(np.float32)))
        for key, name in (('edit', 'edit'), ('cut_edit', 'cut_edit'), ('fill_edit', 'fill_edit')):
            if not torch.equal(tensors[key].detach().cpu().bool(), labels[name]):
                raise ValueError(f'Model routing mask disagrees with strict direction labels: {key}')
        accumulate(candidate_counts, contingency(
            labels, cpu(tensors['cut_keep']).bool(), cpu(tensors['fill_keep']).bool()))
        n_valid = int((valid > 0).sum().item())
        valid_pixels += n_valid
        total_pixels += predictions['final'].size
        if n_valid:
            absolute = delta[valid > 0].abs()
            delta_abs_sum += float(absolute.double().sum().item())
            saturated += int((absolute >= .95 * REFINEMENT['max_delta']).sum().item())
        candidate = _thin_background_proxy(truth)
        foreground_gap = _thin_background_proxy(~truth)
        diagnostics['thin_bg_candidates'] += int(candidate.sum())
        diagnostics['thin_fg_candidates'] += int(foreground_gap.sum())
        for name, prediction in predictions.items():
            diagnostics['thin_bg_' + name + '_fp'] += int((candidate & prediction).sum())
            diagnostics['thin_fg_' + name + '_fn'] += int((foreground_gap & ~prediction).sum())
        for p, s, f, t in zip(predictions['raw'][:, 0], predictions['sr'][:, 0],
                              predictions['final'][:, 0], truth[:, 0]):
            for name, prediction in (('raw', p), ('sr', s), ('final', f)):
                pixel[name] += binary_confusion(prediction, t)
                boundary[name] += boundary_statistics(prediction, t, 3)
                goc, guc, gtc, area = object_error_sums(prediction, t)
                objects[name] += (goc, guc, gtc, area)
            gt_boundary = mask_boundary(t)
            far = (distance_transform_edt(~gt_boundary) > 10 if gt_boundary.any()
                   else np.ones_like(t, dtype=bool))
            diagnostics['far_pixels'] += int(far.sum())
            for name, prediction in (('raw', p), ('sr', s), ('final', f)):
                diagnostics[name + '_far_fp'] += int((far & prediction & ~t).sum())
                diagnostics[name + '_far_fn'] += int((far & ~prediction & t).sum())
            count += 1
        del out, auxiliary, tensors, raw, sr, final, delta, valid
    if count == 0:
        raise ValueError('Validation must contain at least one sample')
    result = {'samples': count, 'total_pixels': total_pixels,
              'valid_pixels': valid_pixels,
              'delta_abs_mean': delta_abs_sum / valid_pixels if valid_pixels else 0.0,
              'delta_saturation_fraction': saturated / valid_pixels if valid_pixels else 0.0}
    for name in routes:
        _, precision, recall, f1, iou = metrics_from_confusion(*map(int, pixel[name]))
        completeness, correctness, bf1 = boundary_metrics_from_statistics(*map(int, boundary[name]))
        area = objects[name][3]
        prefix = '' if name == 'final' else name + '_'
        result[prefix + 'iou'] = iou
        result[prefix + 'f1'] = f1
        result[prefix + 'precision'] = precision
        result[prefix + 'recall'] = recall
        result[prefix + 'bf1'] = bf1
        result[prefix + 'boundary_completeness'] = completeness
        result[prefix + 'boundary_correctness'] = correctness
        result[prefix + 'fp'] = int(pixel[name][1])
        result[prefix + 'fn'] = int(pixel[name][2])
        result[prefix + 'goc'] = float(objects[name][0] / area) if area else 0.0
        result[prefix + 'guc'] = float(objects[name][1] / area) if area else 0.0
        result[prefix + 'gtc'] = float(objects[name][2] / area) if area else 0.0
    result['checkpoint_score'] = .7 * result['iou'] + .3 * result['bf1']
    # Canonical directional contingency table, its retained v6 aliases, the
    # finite-rate view (zero denominator -> 0.0 plus the count that explains it)
    # and the accepted-benefit minus accepted-harm totals.
    result.update(candidate_counts)
    result.update(legacy_aliases(candidate_counts))
    result.update(rates(candidate_counts, undefined=0.0))
    result.update(net_pixels(candidate_counts))
    result.update(diagnostics)
    if not all(isinstance(value, (int, float)) and value == value for value in result.values()):
        raise FloatingPointError('Non-finite validation diagnostic')
    return result


