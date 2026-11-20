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
LOCAL_DEPS = ROOT / '_deps'
for directory in (PROJECT_ROOT, PROJECT_ROOT / '.python_packages', LOCAL_DEPS, ROOT):
    if str(directory) in sys.path:
        sys.path.remove(str(directory))
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

from runtime_identity import MAIN_RUN, STAGE, MODEL_KIND, PHYSICAL_GPU
RUN_CONFIGS = {MAIN_RUN: 'config_protocol_b.yaml'}
RESERVED_RUNS = {'hbgnet_protocol_b_rd_v9_context_sr_v2_center_only_control'}
ABLATION_GROUPS = {
    'baseline': {'run': 'hbgnet_fhapd_xj_baseline_seed73', 'status': 'separate_package',
                 'sr_enabled': False, 'verifier_enabled': False},
    'baseline_plus_sr_v2': {'status': 'not_requested', 'sr_enabled': True, 'verifier_enabled': False},
    'baseline_plus_rd': {'status': 'reserved_not_representable', 'sr_enabled': False, 'verifier_enabled': False},
    'baseline_plus_rd_plus_sr_v2': {'run': 'hbgnet_fhapd_xj_sr_v2_rd_v9_seed73',
                 'status': 'separate_package', 'sr_enabled': True, 'verifier_enabled': True},
}
VERIFIER_LOSS_MODE = 'ordinary_bce_present_direction_average'
REFINEMENT = {
    'enabled': True, 'max_delta': 2.0, 'checkpoint_groups': True,
    'mode': 'local_separation_v2', 'hidden_channels': 16,
    'path_channels': 4, 'radii': [2, 4],
}
LOSS_COEFFICIENTS = {MAIN_RUN: (0.5, 0.5, 0.125, 0.125)}
VERIFIER_SPECS = {MAIN_RUN: {'enabled': True, 'mode': 'enabled', 'aggregation': 'multi_scale_support'}}
PROTOCOL_SECTIONS = ('dataset', 'augmentation', 'normalization', 'loss', 'training', 'evaluation', 'paths')
EXTERNAL_FILES = set()
REQUIRED_PACKAGE_FILES = {
    'common.py', 'train.py', 'evaluate.py', 'validation.py', 'training_state.py',
    'runtime_identity.py', 'data_contract.json', 'config_protocol_b.yaml',
    'run_train.sh', 'run_evaluate.sh', 'launch_training_audited.py',
    'implementation/__init__.py', 'implementation/model.py', 'implementation/losses.py',
    'implementation/lib/__init__.py', 'implementation/lib/pvtv2.py',
    'protocol/__init__.py', 'protocol/dataset.py', 'protocol/metrics.py',
    'protocol/evaluation_adapter.py', 'protocol/candidate_diagnostics.py',
    'protocol/hbgnet_baseline.yaml', 'verify_fhapd_training.py', 'test_training_contract.py', 'test_validation.py',
    'README.md', 'VERSION_RECORD.md',
}
if MODEL_KIND == 'combined':
    REQUIRED_PACKAGE_FILES.update({
        'tools/__init__.py', 'tools/local_separation_refinement_v2.py',
        'tools/region_boundary_band_loss_v2.py', 'tools/region_discrimination_v9.py',
        'tools/region_discrimination_v8_grok_fullloss.py', 'verify_rd_v9_context.py',
        'verify_v9_diagnostics.py', 'launch_rd_v9_context_audited.py',
    })


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
    if MODEL_KIND == 'baseline':
        return {'name': 'official_LossF', 'weights': [1, 1, 1]}
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
    if MODEL_KIND == 'combined':
        expected['model']['refinement'] = copy.deepcopy(REFINEMENT)
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
    cpu = json.loads((ROOT / 'engineering_cpu.json').read_text(encoding='utf-8'), object_pairs_hook=_unique_json)
    if (cpu.get('status') != 'pass' or cpu.get('source_fingerprint') != fingerprint
            or not _same(cpu.get('config'), cfg)):
        raise ValueError('Missing or stale FHAPD CPU engineering acceptance')
    real = report.get('real_smoke', {})
    if (real.get('train_batches') != 4 or real.get('optimizer_updates') != 2
            or real.get('validation_samples') != 64 or real.get('test_started') is not False):
        raise ValueError('Engineering receipt lacks the required real train/val GPU smoke')


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
    arguments = dict(channel=options['channel'], num_classes=options['boundary_classes'],
                     drop_rate=options['drop_rate'], pretrained_path=str(path) if pretrained else None)
    if MODEL_KIND == 'combined':
        arguments['refinement'] = copy.deepcopy(options['refinement'])
    return Field(**arguments)


def loss_fn(outputs, batch, cfg, training_aux=None, return_components=False):
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
    if MODEL_KIND == 'baseline':
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise ValueError('Baseline requires Region/Boundary/Distance outputs')
        region, boundary, distance = outputs
        if any(not torch.isfinite(value).all().item() for value in outputs):
            raise FloatingPointError('Non-finite baseline output')
        if region.shape != batch['region'].shape or distance.shape != batch['distance'].shape:
            raise ValueError('Baseline target/output shape mismatch')
        if boundary.shape != (region.shape[0], 2, region.shape[2], region.shape[3]):
            raise ValueError('Baseline requires two-class log probabilities')
        from implementation.losses import LossF
        total = LossF([1, 1, 1])(*outputs, batch['region'], batch['boundary'].long(), batch['distance'])
        if not torch.isfinite(total).all().item():
            raise FloatingPointError('Non-finite official LossF')
        if not return_components:
            return total
        components = {
            'raw_region': BCEDiceLoss()(region, batch['region']),
            'boundary_nll': LossMulti(num_classes=2)(boundary, batch['boundary'].long()),
            'distance_mse': torch.nn.MSELoss()(distance, batch['distance']),
        }
        values = {key: float(value.detach().item()) for key, value in components.items()}
        values['total'] = float(total.detach().item())
        return total, values
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


@torch.no_grad()
def evaluate(model, batches, device, max_batches=None):
    from validation import evaluate_validation
    return evaluate_validation(model, batches, device, max_batches=max_batches)


def verify_data_contract(cfg, *, verify_bytes=False):
    contract = json.loads((ROOT / 'data_contract.json').read_text(encoding='utf-8'),
                          object_pairs_hook=_unique_json)
    if contract['dataset'] != cfg['dataset']['name']:
        raise ValueError('Data contract has another regional identity')
    data_root = Path(cfg['dataset']['root'])
    if sha256(data_root / 'DATA_MANIFEST.json') != contract['data_manifest_sha256']:
        raise ValueError('Frozen DATA_MANIFEST changed')
    splits = {}
    for split, path in cfg['dataset']['manifests'].items():
        if sha256(path) != contract['split_manifest_sha256'][split]:
            raise ValueError(f'Frozen split changed: {split}')
        ids = Path(path).read_text(encoding='utf-8').splitlines()
        if (len(ids) != cfg['dataset']['expected_counts'][split] or len(ids) != len(set(ids))
                or any(not ident or '/' in ident or chr(92) in ident or ident in ('.', '..') for ident in ids)):
            raise ValueError(f'Invalid split identity/count: {split}')
        splits[split] = set(ids)
    if any(splits[a] & splits[b] for a, b in (('train', 'val'), ('val', 'test'), ('train', 'test'))):
        raise ValueError('Split IDs overlap')
    if verify_bytes:
        manifest = json.loads((data_root / 'DATA_MANIFEST.json').read_text(encoding='utf-8'),
                              object_pairs_hook=_unique_json)
        # Only train/val pixel data are accessed during training engineering checks.
        for name, expected in manifest['files'].items():
            if not name.startswith(('images/train/', 'masks/train/', 'images/val/', 'masks/val/')):
                continue
            path = _manifest_file(data_root, name)
            if sha256(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
                raise ValueError(f'Train/val bytes changed: {name}')
    return contract
