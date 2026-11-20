from pathlib import Path
import copy
import hashlib
import json
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '_deps'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from protocol.dataset import AI4BoundariesSEANetDataset, load_config
from implementation.model import Field
from implementation.losses import LossF

RUN = 'hbgnet_protocol_b_v1'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def config(path):
    cfg = load_config(path)
    expected_keys = {'experiment','dataset','augmentation','normalization','model','loss','training','evaluation','paths'}
    if set(cfg) != expected_keys: raise ValueError('Unexpected configuration sections')
    if cfg['experiment'] != {'run_name':RUN,'stage':'official_HBGNet_ProtocolB_adaptation'}:
        raise ValueError('Unexpected experiment configuration')
    ref = load_config(ROOT/'protocol/seanet_reference.yaml')
    for key in ('dataset', 'augmentation', 'normalization', 'evaluation', 'paths'):
        if cfg[key] != ref[key]: raise ValueError(f'Protocol mismatch: {key}')
    if cfg['experiment']['run_name'] != RUN: raise ValueError('Unexpected run name')
    if cfg['model'] != {'name':'HBGNet', 'backbone':'pvt_v2_b2', 'channel':32, 'boundary_classes':2, 'drop_rate':0.4,
                         'pretrained_path':str(ROOT/'pretrained/pvt_v2_b2.pth')}:
        raise ValueError('HBGNet architecture or pretraining path changed')
    expected = {'epochs':40,'seed':41,'gradient_accumulation_steps':2,'amp':False,
                'optimizer':'Adam','learning_rate':0.0001,'weight_decay':0.0,
                'scheduler':'CosineAnnealingLR','scheduler_t_max':10000000000,'min_lr':0.00001,
                'validation_interval':2,'final_validation_epochs':5,'log_interval':20,
                'selection':'0.7_iou_0.3_bf1','iou_floor':0.8205741376917083}
    if cfg['training'] != expected: raise ValueError('HBGNet training recipe mismatch')
    if cfg['loss'] != {'name':'official_LossF','weights':[1,1,1]}: raise ValueError('Official loss changed')
    return cfg


def source_fingerprint():
    manifest = json.loads((ROOT/'package_manifest.json').read_text(encoding='utf-8'))
    actual = {name: sha256(ROOT/name) for name in manifest['files_sha256']}
    if actual != manifest['files_sha256']:
        raise ValueError('Source/config changed after packaging')
    return actual


def require_engineering(cfg, fingerprint):
    report = json.loads((ROOT/'verification.json').read_text(encoding='utf-8'))
    if report.get('status') != 'pass' or report.get('source_fingerprint') != fingerprint:
        raise ValueError('Missing or stale engineering verification')
    if report.get('config') != cfg:
        raise ValueError('Verification configuration mismatch')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed); random.seed(seed)


def loader(cfg, split, batch=None, workers=None):
    data = AI4BoundariesSEANetDataset(cfg, split, augment=(split=='train'))
    generator = torch.Generator().manual_seed(cfg['training']['seed'])
    return DataLoader(data, batch_size=batch or cfg['dataset']['batch_size'], shuffle=(split=='train'),
        num_workers=cfg['dataset']['num_workers'] if workers is None else workers,
        drop_last=(split=='train'), pin_memory=True, worker_init_fn=seed_worker,
        generator=generator, persistent_workers=False)


def build(cfg, pretrained=True):
    path = Path(cfg['model']['pretrained_path'])
    if pretrained:
        if not path.is_file():
            raise FileNotFoundError(f'Official ImageNet PVT-v2-B2 pretraining required: {path}; random fallback forbidden')
        receipt = path.with_suffix('.json')
        if not receipt.is_file(): raise ValueError('Missing pretraining provenance receipt')
        evidence = json.loads(receipt.read_text())
        if evidence.get('sha256') != sha256(path) or not evidence.get('source_url') or not evidence.get('source_evidence'):
            raise ValueError('Mismatched or incomplete pretraining provenance receipt')
    return Field(channel=32, num_classes=2, drop_rate=.4, pretrained_path=str(path) if pretrained else None)


def loss_fn(outputs, batch):
    for tensor in outputs:
        if not torch.isfinite(tensor).all(): raise FloatingPointError('Non-finite HBGNet output')
    # The published edge branch emits two-class log probabilities, not one-channel logits.
    return LossF([1,1,1])(*outputs, batch['region'], batch['boundary'].long(), batch['distance'])


def device_batch(batch, device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def atomic_json(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_save(path, value):
    path = Path(path); tmp = path.with_suffix('.pt.tmp')
    torch.save(value,tmp); tmp.replace(path)


@torch.no_grad()
def evaluate(model, batches, device, max_batches=None):
    from protocol.metrics import binary_confusion, boundary_statistics, metrics_from_confusion, boundary_metrics_from_statistics
    model.eval(); pixel=np.zeros(4,dtype=np.int64); boundary=np.zeros(4,dtype=np.int64); count=0
    for i,batch in enumerate(batches):
        if max_batches is not None and i>=max_batches: break
        out=model(batch['image'].to(device))[0]
        if not torch.isfinite(out).all(): raise FloatingPointError('Non-finite validation output')
        pred=(out.sigmoid().cpu().numpy()>=.5)
        for p,t in zip(pred[:,0],batch['region'].numpy()[:,0]):
            pixel+=binary_confusion(p,t);boundary+=boundary_statistics(p,t,3);count+=1
    oa,p,r,f1,iou=metrics_from_confusion(*map(int,pixel));com,corr,bf1=boundary_metrics_from_statistics(*map(int,boundary))
    return dict(samples=count,iou=iou,f1=f1,precision=p,recall=r,boundary_completeness=com,
                boundary_correctness=corr,bf1=bf1,checkpoint_score=.7*iou+.3*bf1)
