"""Build isolated FHAPD data/configs; never launch training or model evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import warnings

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
import yaml

from spatial_split import build_split, spatial_audit, parse_tile_id
from targets import build_targets, TARGET_VERSION, REFERENCE_URL

REGIONS = ('XJ', 'JS')
SPLITS = ('train', 'val', 'test')
PACKAGE = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path('/root/wyk5/dataset/FHAPD_256')
DEFAULT_OUTPUT = Path('/root/wyk5/dataset')
PROJECT = Path('/root/wyk5/RGVNet')
SCHEMA = 'fhapd-protocol-b-compatible-data-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def inventory(root):
    return {p.relative_to(root).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()}


def source_files(root):
    files = {}
    for kind in ('img', 'mask'):
        directory = root / kind
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f'Expected real source directory: {directory}')
        found = sorted(directory.iterdir())
        if not found or any(not p.is_file() or p.is_symlink() or p.suffix != '.png' for p in found):
            raise ValueError(f'Expected only regular .png files in {directory}')
        files[kind] = {p.stem: p for p in found}
        if len(files[kind]) != len(found):
            raise ValueError('Duplicate source IDs')
    if set(files['img']) != set(files['mask']):
        raise ValueError('Source image/mask pairs are incomplete')
    return files


def read_source(image_path, mask_path):
    with Image.open(image_path) as image:
        if image.mode != 'RGB' or image.size != (256, 256) or image.format != 'PNG':
            raise ValueError(f'Expected native RGB uint8 256x256 PNG: {image_path}')
        rgb = np.asarray(image)
    with Image.open(mask_path) as image:
        if image.mode != 'L' or image.size != (256, 256) or image.format != 'PNG':
            raise ValueError(f'Expected native one-channel uint8 256x256 PNG: {mask_path}')
        mask = np.asarray(image)
    if rgb.dtype != np.uint8 or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError(f'Source mask must be uint8 0/1: {mask_path}')
    return rgb, mask


def make_config(region, root, split, data_manifest_hash):
    dataset_name = f'FHAPD_{region}'
    training_reference = {
        'epochs': 40, 'gradient_accumulation_steps': 2, 'amp': False,
        'optimizer': 'Adam', 'learning_rate': 1e-4, 'weight_decay': 0.0,
        'scheduler': 'CosineAnnealingLR', 'scheduler_t_max': 10_000_000_000,
        'min_lr': 1e-5, 'validation_interval': 2, 'final_validation_epochs': 5,
        'log_interval': 20, 'selection': '0.7_iou_0.3_bf1',
    }
    return {
        'adaptation': {
            'schema': SCHEMA, 'dataset': dataset_name, 'status': 'data_adapter_only',
            'run_config_ready': False, 'auto_train': False, 'auto_test': False,
            'target_version': TARGET_VERSION, 'data_manifest_sha256': data_manifest_hash,
            'split_manifest_sha256': {s: sha256(root / 'manifests' / f'{s}.txt') for s in SPLITS},
            'training_package_note': 'A new model/seed run must freeze its own identity, fingerprint and engineering receipt.',
        },
        'dataset': {
            'name': dataset_name, 'variant': 'spatial_nonoverlap_v1',
            'variants': {'spatial_nonoverlap_v1': {
                'protocol': 'FHAPD-ProtocolB-compatible-spatial-v1',
                'purpose': 'within_region_spatial_holdout', 'root': str(root),
                'image_dir': str(root / 'images'), 'mask_dir': str(root / 'masks'),
                'manifests': {s: str(root / 'manifests' / f'{s}.txt') for s in SPLITS},
                'image_suffix': '.png', 'mask_suffix': '.tif', 'input_size': [256, 256],
            }},
            'image_bands': [1, 2, 3], 'region_band': 1, 'boundary_band': 2, 'distance_band': 3,
            'expected_counts': {s: split['summary']['counts'][s] for s in SPLITS},
            'batch_size': 8, 'val_batch_size': 8, 'num_workers': 8, 'persistent_workers': False,
        },
        'augmentation': {
            'enabled': True, 'horizontal_flip_probability': 0.5,
            'vertical_flip_probability': 0.5, 'rotate_90_probability': 0.5,
        },
        'normalization': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'training_recipe_reference': training_reference,
        'initialization_reference': 'Official ImageNet PVT-v2-B2 only; no resume from AI4Boundaries trained checkpoints.',
        'iou_floor_reference': {
            'ai4boundaries_value': 0.8205741376917083,
            'apply_to_fhapd': False,
            'reason': 'This value belongs to the AI4Boundaries control; it is not an established FHAPD floor.',
        },
        'loss_reference': {
            'baseline': 'BCEDice region + two-class Boundary NLL + Distance MSE; weights [1,1,1]',
            'sr_rd': 'Copy the authorized module-specific loss and supervision sites into each new model run.',
            'target_changes': 'FHAPD boundary/distance generated from binary mask; not native AI4Boundaries labels.',
        },
        'evaluation': {
            'protocol': 'HBGNet-metrics-FHAPD-256-3px-spatial-v1',
            'region_threshold': 0.5, 'threshold_comparison': '>', 'primary_output': 'region',
            'tta': False, 'small_object_removal': False,
            'spatial_resolution_m': None, 'boundary_tolerance_m': None,
            'requested_boundary_tolerance_m': 3.0, 'boundary_tolerance_pixels': 3,
            'physical_tolerance_verified': False,
            'physical_resolution_evidence': 'Source PNG files have no CRS/geotransform; identity pixel transform is not GSD.',
            'aggregation': 'region_micro_boundary_micro_object_global_area_weighted',
            'primary_metrics': ['f1', 'iou', 'f_boundary', 'goc', 'guc', 'gtc'],
            'diagnostic_metrics': ['oa', 'precision', 'recall', 'boundary_completeness', 'boundary_correctness'],
            'narrow_structure_metrics': {'status': 'not_defined_in_frozen_evaluator', 'value': None},
        },
        'paths': {
            'train_log_root': str(PROJECT / 'z_1_trainning_log' / dataset_name),
            'test_log_root': str(PROJECT / 'z_2_test' / dataset_name),
            'weights_root': str(PROJECT / 'z_3_weights' / dataset_name),
            'results_root': str(PROJECT / 'z_4_results' / dataset_name),
            'local_backup_root': f'E:/A-94服务器备份-地块分割/{dataset_name}',
        },
    }


def prepare(region, source_parent, output_parent):
    source = source_parent / region
    root = output_parent / f'FHAPD_{region}_protocol_b_v1'
    config_path = PACKAGE / 'configs' / f'config_fhapd_{region.lower()}.yaml'
    if root.exists() or root.is_symlink() or config_path.exists():
        raise FileExistsError(f'Refuse to replace data or config: {root}, {config_path}')
    files = source_files(source)
    ids = sorted(files['img'])
    split = build_split(ids)
    audit = spatial_audit(ids, split['assignments'])
    if not audit['ok']:
        raise ValueError('Spatial audit failed')
    for parent, value in split['summary']['parents'].items():
        for name in ('train_val', 'val_test'):
            if value['cuts'][name]['clear_gap_px'] < 256:
                raise ValueError(f'Missing spatial gap: {parent}/{name}')
    root.mkdir(parents=True, exist_ok=False)
    json_write(root / 'preparation_status.json', {'status': 'building', 'schema': SCHEMA, 'region': region})
    for name in SPLITS:
        (root / 'images' / name).mkdir(parents=True, exist_ok=False)
        (root / 'masks' / name).mkdir(parents=True, exist_ok=False)
    (root / 'manifests').mkdir()
    source_records = {}
    stats = {s: {'images': 0, 'foreground_pixels': 0, 'empty_masks': 0, 'full_masks': 0,
                 'boundary_pixels': 0, 'constant_distance_maps': 0} for s in (*SPLITS, 'excluded')}
    target_records = []
    for index, ident in enumerate(ids):
        geometry = parse_tile_id(ident)
        if not geometry['parent'].startswith(region):
            raise ValueError('Source region and ID disagree')
        image_path, mask_path = files['img'][ident], files['mask'][ident]
        rgb, mask = read_source(image_path, mask_path)
        label = split['assignments'][ident]
        record = {'id': ident, **geometry, 'assignment': label,
                  'excluded_reason': split['excluded_reasons'].get(ident),
                  'image': {'path': str(image_path), 'sha256': sha256(image_path), 'bytes': image_path.stat().st_size},
                  'mask': {'path': str(mask_path), 'sha256': sha256(mask_path), 'bytes': mask_path.stat().st_size},
                  'foreground_pixels': int(mask.sum())}
        source_records[ident] = record
        count = stats[label]
        count['images'] += 1
        count['foreground_pixels'] += int(mask.sum())
        count['empty_masks'] += int(not mask.any())
        count['full_masks'] += int(mask.all())
        if label != 'excluded':
            image_out = root / 'images' / label / f'{ident}.png'
            mask_out = root / 'masks' / label / f'{ident}.tif'
            shutil.copyfile(image_path, image_out)
            target = build_targets(mask)
            with rasterio.open(mask_out, 'w', driver='GTiff', height=256, width=256,
                               count=3, dtype='float32', compress='deflate', predictor=3) as dst:
                dst.write(target)
                dst.set_band_description(1, 'region_binary_0_1')
                dst.set_band_description(2, 'canny_boundary_binary_0_1')
                dst.set_band_description(3, 'per_tile_distance_u8_div_255')
                dst.update_tags(target_version=TARGET_VERSION, source_mask_sha256=record['mask']['sha256'],
                                spatial_resolution_m='unknown', georeference_status='not_available')
            count['boundary_pixels'] += int(target[1].sum())
            count['constant_distance_maps'] += int(not target[2].any())
            target_records.append({'id': ident, 'split': label, 'image_sha256': sha256(image_out),
                                   'target_sha256': sha256(mask_out), 'target_pixel_sha256':
                                   hashlib.sha256(target.tobytes(order='C')).hexdigest()})
        if (index + 1) % 500 == 0:
            print(f'{region}: validated/converted {index + 1}/{len(ids)}', flush=True)
    for name in SPLITS:
        selected = [ident for ident in ids if split['assignments'][ident] == name]
        (root / 'manifests' / f'{name}.txt').write_text(''.join(ident + '\n' for ident in selected), encoding='utf-8')
    for name in ('all', 'excluded'):
        selected = ids if name == 'all' else [ident for ident in ids if split['assignments'][ident] == 'excluded']
        (root / 'manifests' / f'{name}.txt').write_text(''.join(ident + '\n' for ident in selected), encoding='utf-8')
    json_write(root / 'provenance' / 'source_inventory.json', {
        'schema': SCHEMA, 'source_root': str(source), 'count': len(ids), 'records': source_records})
    json_write(root / 'provenance' / 'split_assignment.json', split)
    json_write(root / 'provenance' / 'spatial_audit.json', audit)
    json_write(root / 'provenance' / 'target_inventory.json', {
        'target_version': TARGET_VERSION, 'reference_url': REFERENCE_URL,
        'region_encoding': 'Source uint8 0/1 preserved; image uint8 RGB divided by255 in shared reader.',
        'boundary': 'cv2.Canny(source_mask*255,100,200)>0, no dilation',
        'distance': 'cv2.distanceTransform(mask*255,DIST_L2,3); per-tile minmax*255 cast uint8 then /255',
        'constant_map_policy': 'distance min==max -> all zeros, finite; no source/sample dropped',
        'generation_scope': 'Independently within each retained 256x256 tile, not parent-wide distances.',
        'stats': stats, 'records': target_records,
        'versions': {'cv2': cv2.__version__, 'rasterio': rasterio.__version__, 'numpy': np.__version__, 'torch': torch.__version__},
    })
    for record in source_records.values():
        for kind in ('image', 'mask'):
            if sha256(record[kind]['path']) != record[kind]['sha256']:
                raise RuntimeError('Source bytes changed during preparation')
    (root / 'preparation_status.json').unlink()
    json_write(root / 'preparation_status.json', {
        'status': 'prepared', 'schema': SCHEMA, 'region': region,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'counts': split['summary']['counts'],
        'auto_train': False, 'auto_test': False,
    })
    data_inventory = inventory(root)
    json_write(root / 'DATA_MANIFEST.json', {
        'schema': SCHEMA, 'region': region, 'root': str(root),
        'files': data_inventory, 'excluded_from_inventory': ['DATA_MANIFEST.json'],
    })
    config = make_config(region, root, split, sha256(root / 'DATA_MANIFEST.json'))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open('x', encoding='utf-8', newline='\n') as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    print(json.dumps({'region': region, 'source_count': len(ids), 'counts': split['summary']['counts'],
                      'retained': split['summary']['retained'], 'root': str(root),
                      'config': str(config_path), 'data_manifest_sha256': sha256(root / 'DATA_MANIFEST.json')},
                     ensure_ascii=False), flush=True)
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-parent', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--region', choices=REGIONS, action='append')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
    for region in args.region or REGIONS:
        prepare(region, args.source_root, args.output_parent)


if __name__ == '__main__':
    main()
