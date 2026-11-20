"""Verify prepared FHAPD data and loader contracts without a model or optimizer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
import torch.nn.functional as F

from dataset import AI4BoundariesSEANetDataset, load_config
from evaluation_adapter import validate_evaluation_settings
from prepare_fhapd import sha256, inventory, json_write, read_source
from spatial_split import build_split, spatial_audit
from targets import build_targets

PACKAGE = Path(__file__).resolve().parent
SPLITS = ('train', 'val', 'test')


def check_tensor_batch(dataset):
    sample = dataset[0]
    batch = {name: sample[name].unsqueeze(0) for name in ('image', 'region', 'boundary', 'distance')}
    expected = {'image': (1, 3, 256, 256), 'region': (1, 1, 256, 256),
                'boundary': (1, 1, 256, 256), 'distance': (1, 1, 256, 256)}
    for name, tensor in batch.items():
        if tuple(tensor.shape) != expected[name] or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f'Bad target/image contract: {name}')
    for name in ('region', 'boundary'):
        if not ((batch[name] == 0) | (batch[name] == 1)).all():
            raise ValueError('Nonbinary target tensor')
    if not ((batch['distance'] >= 0) & (batch['distance'] <= 1)).all():
        raise ValueError('Distance not normalized')
    logits = torch.zeros_like(batch['region'], requires_grad=True)
    edge_logits = torch.zeros((1, 2, 256, 256), requires_grad=True)
    distance_prediction = torch.zeros_like(batch['distance'], requires_grad=True)
    boundary_nll = F.nll_loss(F.log_softmax(edge_logits, dim=1), batch['boundary'][:, 0].long())
    region_bce = F.binary_cross_entropy_with_logits(logits, batch['region'])
    distance_mse = F.mse_loss(distance_prediction, batch['distance'])
    total = region_bce + boundary_nll + distance_mse
    total.backward()
    for tensor in (logits, edge_logits, distance_prediction):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise ValueError('Nonfinite loss gradient')
    return {'id': sample['image_id'], 'shapes': {k: list(v.shape) for k, v in batch.items()},
            'dtype': 'float32', 'boundary_target_for_nll': 'long, 0/1',
            'synthetic_target_loss_backward': 'pass', 'model_forward': 'not_run',
            'optimizer': 'not_created', 'checkpoint': 'not_written'}


def verify(config_path):
    cfg = load_config(config_path)
    settings = validate_evaluation_settings(cfg)
    root = Path(cfg['dataset']['root'])
    data_manifest_path = root / 'DATA_MANIFEST.json'
    if sha256(data_manifest_path) != cfg['adaptation']['data_manifest_sha256']:
        raise ValueError('Data manifest hash does not match config')
    manifest = json.loads(data_manifest_path.read_text(encoding='utf-8'))
    found = inventory(root)
    found.pop('DATA_MANIFEST.json')
    if found != manifest['files']:
        raise ValueError('Prepared data inventory differs from DATA_MANIFEST')
    provenance = json.loads((root / 'provenance/source_inventory.json').read_text(encoding='utf-8'))
    sources = provenance['records']
    source_ids = sorted(sources)
    assignments = json.loads((root / 'provenance/split_assignment.json').read_text(encoding='utf-8'))
    if assignments != build_split(source_ids):
        raise ValueError('Spatial split is not reproducible from source IDs')
    audit = spatial_audit(source_ids, assignments['assignments'])
    if audit != json.loads((root / 'provenance/spatial_audit.json').read_text(encoding='utf-8')) or not audit['ok']:
        raise ValueError('Spatial audit mismatch')
    target_info = json.loads((root / 'provenance/target_inventory.json').read_text(encoding='utf-8'))
    target_records = {item['id']: item for item in target_info['records']}
    for ident, item in sources.items():
        for name in ('image', 'mask'):
            path = Path(item[name]['path'])
            if path.stat().st_size != item[name]['bytes'] or sha256(path) != item[name]['sha256']:
                raise ValueError(f'Raw source changed: {ident}/{name}')
    results = {}
    seen = set()
    for split in SPLITS:
        split_path = Path(cfg['dataset']['manifests'][split])
        if sha256(split_path) != cfg['adaptation']['split_manifest_sha256'][split]:
            raise ValueError('Split manifest hash mismatch')
        data = AI4BoundariesSEANetDataset(cfg, split, augment=False)
        ids = data.image_ids
        expected = [ident for ident in source_ids if assignments['assignments'][ident] == split]
        if ids != expected or len(ids) != cfg['dataset']['expected_counts'][split] or seen.intersection(ids):
            raise ValueError('Split count/identity/order mismatch')
        seen.update(ids)
        for subdir, suffix in (('images', '.png'), ('masks', '.tif')):
            paths = list((root / subdir / split).iterdir())
            if {p.name for p in paths} != {ident + suffix for ident in ids}:
                raise ValueError('Incomplete/extra prepared files')
        foreground = boundary = empty = 0
        for index, ident in enumerate(ids):
            sample = data[index]
            rgb, mask = read_source(Path(sources[ident]['image']['path']), Path(sources[ident]['mask']['path']))
            expected_target = build_targets(mask)
            image_path = root / 'images' / split / (ident + '.png')
            target_path = root / 'masks' / split / (ident + '.tif')
            if sha256(image_path) != sources[ident]['image']['sha256']:
                raise ValueError('RGB image bytes were altered')
            with rasterio.open(target_path) as src:
                array = src.read()
                if src.count != 3 or src.dtypes != ('float32',) * 3 or array.shape != (3, 256, 256):
                    raise ValueError('Target band shape/dtype mismatch')
            if not np.array_equal(array, expected_target):
                raise ValueError('Generated targets differ from reference operations')
            if hashlib.sha256(array.tobytes(order='C')).hexdigest() != target_records[ident]['target_pixel_sha256']:
                raise ValueError('Target pixel digest mismatch')
            expected_image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
            mean = torch.tensor(cfg['normalization']['mean']).view(3, 1, 1)
            std = torch.tensor(cfg['normalization']['std']).view(3, 1, 1)
            if not torch.equal(sample['image'], (expected_image - mean) / std):
                raise ValueError('Shared reader RGB normalization changed')
            for band, name in enumerate(('region', 'boundary', 'distance')):
                value = sample[name]
                if value.shape != (1, 256, 256) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError(f'Reader tensor contract mismatch: {name}')
                if not np.array_equal(value.numpy()[0], expected_target[band]):
                    raise ValueError(f'Reader changed target: {name}')
            foreground += int(mask.sum())
            boundary += int(array[1].sum())
            empty += int(not mask.any())
        results[split] = {'samples': len(ids), 'foreground_pixels': foreground,
                          'boundary_pixels': boundary, 'empty_masks': empty,
                          'per_file_readback_and_target_rebuild': 'pass',
                          'tensor_contract': check_tensor_batch(data)}
    augmented = AI4BoundariesSEANetDataset(cfg, 'train', augment=True)
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(917)
        augmented_contract = check_tensor_batch(augmented)
    finally:
        torch.set_rng_state(rng_state)
    return {'status': 'pass', 'scope': 'data-only verification, not model training or formal test',
            'dataset': cfg['dataset']['name'], 'config_sha256': sha256(config_path),
            'data_manifest_sha256': sha256(data_manifest_path), 'data_files_hashed': len(found),
            'source_pairs_hashed': len(sources), 'split_results': results,
            'augmented_training_tensor_contract': augmented_contract, 'spatial_audit': audit,
            'evaluation_settings': settings, 'physical_tolerance_verified': False,
            'training_started': False, 'formal_model_test_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    result = verify(args.config)
    json_write(args.receipt, result)
    print(json.dumps({k: result[k] for k in ('status', 'dataset', 'data_files_hashed', 'source_pairs_hashed',
                                           'physical_tolerance_verified', 'training_started', 'formal_model_test_started')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
