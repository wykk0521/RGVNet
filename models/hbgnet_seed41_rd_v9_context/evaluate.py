"""Full-split evaluation with the identical parcel metrics used by SEANet.

One forward pass per batch yields the raw, SR and gated-final Region logits from
``model.last_auxiliary``. Route metrics keep the frozen historical formal test
threshold ``sigmoid > 0.5``; direction-level candidate labels use the strict
routing convention (``raw_logit > 0``, ``sr_logit > 0``, ``delta != 0``,
``truth > 0.5``) from ``protocol.candidate_diagnostics``. The ground-truth
Region mask is only ever compared, never fed forward. Undefined rates are
reported as JSON ``null`` next to the always-present denominator count. No
rerun, threshold tuning or checkpoint reselection is performed here.
"""
import argparse
import json
import re
from pathlib import Path
import cv2
from common import (ROOT, _same, np, torch, config, source_fingerprint, require_engineering,
                    sha256, loader, build, atomic_json)
from protocol.metrics import (binary_confusion, boundary_statistics, object_error_sums,
                              metrics_from_confusion, boundary_metrics_from_statistics)
from protocol.candidate_diagnostics import (accumulate, contingency, legacy_aliases,
                                            net_pixels, new_counts, rates, strict_labels)

ROUTES = ('raw', 'sr', 'final')
AUXILIARY_KEYS = ('raw_region_logits', 'sr_region_logits', 'final_region_logits', 'delta',
                  'edit', 'cut_edit', 'fill_edit', 'keep', 'cut_keep', 'fill_keep',
                  'valid_count', 'verifier_logits')
PER_IMAGE_DIAGNOSTICS = 'per_image_diagnostics.jsonl'


def verify_predictions(directory, ids, size):
    expected = {f'{ident}.png' for ident in ids}
    if len(expected) != len(ids) or {p.name for p in directory.iterdir()} != expected:
        raise ValueError('Prediction ID set mismatch')
    for name in expected:
        image = cv2.imread(str(directory/name), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != tuple(size) or image.dtype != np.uint8:
            raise ValueError(f'Invalid prediction PNG: {name}')
        if not np.isin(image, [0,255]).all(): raise ValueError(f'Non-binary PNG: {name}')


@torch.no_grad()
def run(cfg, checkpoint, split, name, device):
    if split not in ('val', 'test'):
        raise ValueError('Formal evaluation requires the full val or test split')
    run_name = cfg['experiment']['run_name']
    fingerprint = source_fingerprint()
    require_engineering(cfg, fingerprint)
    if not re.fullmatch(re.escape(run_name) + r'_[A-Za-z0-9_-]+', name):
        raise ValueError('Use an isolated HBGNet evaluation name')
    checkpoint = Path(checkpoint).absolute()
    weights = Path(cfg['paths']['weights_root'])/run_name
    selected = (weights/'best.pt').absolute()
    if checkpoint != selected or checkpoint.is_symlink() or weights.is_symlink():
        raise ValueError('Formal evaluation requires this run best.pt; last/constrained checkpoints are forbidden')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing selected best checkpoint: {checkpoint}')
    status = json.loads((Path(cfg['paths']['train_log_root'])/run_name/'training_status.json').read_text())
    if (status.get('status') != 'completed' or not _same(status.get('epoch'), 40) or
            status.get('run_name') != run_name):
        raise ValueError('Full training must complete before formal evaluation')
    before = sha256(checkpoint)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if cp.get('run_name') != run_name or not _same(cp.get('config'), cfg) or cp.get('source_fingerprint') != fingerprint:
        raise ValueError('Checkpoint/run/config/source mismatch')
    if cp.get('checkpoint_kind') != 'best' or not _same(cp.get('format_version'), 1):
        raise ValueError('Checkpoint selection kind/format mismatch')
    if (type(cp.get('epoch')) is not int or not 1 <= cp['epoch'] <= 40 or
            not isinstance(cp.get('best'), dict) or cp['best'].get('epoch') != cp['epoch']):
        raise ValueError('Checkpoint does not identify a selected best epoch')
    if status.get('source_fingerprint') != fingerprint:
        raise ValueError('Training completion belongs to different source files')
    split_hashes = {k:sha256(v) for k,v in cfg['dataset']['manifests'].items()}
    if (cp.get('split_manifest_sha256') != split_hashes or
            status.get('split_manifest_sha256') != split_hashes):
        raise ValueError('Split manifests changed since training')
    for key, value in cp['model'].items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'Checkpoint model entry is not a tensor: {key}')
        # The verifier carries an integer aggregation sentinel; only floating
        # tensors can be non-finite, so it is scanned too but never rejected.
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f'Non-finite checkpoint tensor: {key}')
    metric_root = Path(cfg['paths']['test_log_root'])/name
    output = Path(cfg['paths']['results_root'])/name
    if metric_root.exists() or output.exists() or metric_root.is_symlink() or output.is_symlink():
        raise FileExistsError('Refuse to overwrite evaluation output')
    model = build(cfg, pretrained=False)
    model.load_state_dict(cp['model'], strict=True)
    model.to(device).eval()
    batches = loader(cfg, split)
    ids = batches.dataset.image_ids
    if len(ids) != {'val':3067,'test':3014}[split] or len(set(ids)) != len(ids):
        raise ValueError('Full-split sample count mismatch')
    if any('/' in ident or '\\' in ident or ident in ('.','..') for ident in ids):
        raise ValueError('Unsafe sample ID')
    metric_root.mkdir(parents=True, exist_ok=False)
    (output/'predictions').mkdir(parents=True, exist_ok=False)
    atomic_json(metric_root/'status.json', {'status':'running','complete_split':False})
    try:
        px = {route: np.zeros(4,dtype=np.int64) for route in ROUTES}
        bd = {route: np.zeros(4,dtype=np.int64) for route in ROUTES}
        ob = {route: np.zeros(3,dtype=np.float64) for route in ROUTES}
        area = dict.fromkeys(ROUTES, 0)
        counts = new_counts()
        count = 0
        seen = []
        with (metric_root/'accuracy.txt').open('x') as text, \
                (metric_root/PER_IMAGE_DIAGNOSTICS).open('x') as stream:
            for batch in batches:
                # Ground truth is never a forward input; only the image is.
                logits = model(batch['image'].to(device))[0]
                if not torch.isfinite(logits).all(): raise FloatingPointError('Non-finite prediction')
                auxiliary = getattr(model, 'last_auxiliary', None)
                if not isinstance(auxiliary, dict) or any(key not in auxiliary for key in AUXILIARY_KEYS):
                    raise ValueError('Model must expose raw/SR/final and directional verifier diagnostics')
                for key in AUXILIARY_KEYS:
                    if not isinstance(auxiliary[key], torch.Tensor):
                        raise ValueError(f'Auxiliary diagnostics must be tensors: {key}')
                for key in AUXILIARY_KEYS:
                    if key != 'verifier_logits' and auxiliary[key].shape != logits.shape:
                        raise ValueError(f'Auxiliary must match region logits [B,1,H,W]: {key}')
                if auxiliary['verifier_logits'].shape != (logits.shape[0], 2, logits.shape[2], logits.shape[3]):
                    raise ValueError('Verifier diagnostics must have shape [B,2,H,W]')
                if not torch.equal(logits.detach(), auxiliary['final_region_logits'].detach()):
                    raise ValueError('Final auxiliary does not match the primary region output')
                if not torch.equal((auxiliary['cut_edit'].bool() | auxiliary['fill_edit'].bool()),
                                   auxiliary['edit'].bool()):
                    raise ValueError('Directional edit diagnostics do not match edit union')
                if not torch.equal((auxiliary['cut_keep'].bool() | auxiliary['fill_keep'].bool()),
                                   auxiliary['keep'].bool()):
                    raise ValueError('Directional keep diagnostics do not match keep union')
                for key in AUXILIARY_KEYS:
                    if auxiliary[key].is_floating_point() and not torch.isfinite(auxiliary[key]).all():
                        raise FloatingPointError(f'Non-finite evaluation auxiliary: {key}')
                raw = auxiliary['raw_region_logits'].detach().cpu()
                sr = auxiliary['sr_region_logits'].detach().cpu()
                final = auxiliary['final_region_logits'].detach().cpu()
                delta = auxiliary['delta'].detach().cpu()
                truth = batch['region'].numpy() >= .5
                if final.shape != torch.from_numpy(truth).shape:
                    raise ValueError('Region prediction/target shape mismatch')
                # Same forward, same logits: strict routing labels for the table.
                labels = strict_labels(raw, sr, delta, torch.from_numpy(truth.astype(np.float32)))
                for key, label in (('edit', 'edit'), ('cut_edit', 'cut_edit'), ('fill_edit', 'fill_edit')):
                    if not torch.equal(auxiliary[key].detach().cpu().bool(), labels[label]):
                        raise ValueError(f'Model routing mask disagrees with strict direction labels: {key}')
                keep_cut = auxiliary['cut_keep'].detach().cpu().bool()
                keep_fill = auxiliary['fill_keep'].detach().cpu().bool()
                # Formal test threshold: strict sigmoid > .5 for every route.
                route_pred = {route: value.sigmoid().numpy() > .5 for route, value in
                              (('raw', raw), ('sr', sr), ('final', final))}
                for index, ident in enumerate(batch['image_id']):
                    target = truth[index, 0]
                    per_route = {}
                    for route in ROUTES:
                        prediction = route_pred[route][index, 0]
                        x = binary_confusion(prediction, target)
                        b = boundary_statistics(prediction, target, 3)
                        o = object_error_sums(prediction, target)
                        px[route] += x; bd[route] += b; ob[route] += o[:3]; area[route] += o[3]
                        per_route[route] = {
                            'pixel_counts': [int(value) for value in x],
                            'boundary_counts': [int(value) for value in b],
                            'iou': metrics_from_confusion(*map(int, x))[4],
                            'bf1': boundary_metrics_from_statistics(*map(int, b))[2],
                        }
                    image_counts = contingency(
                        {key: value[index:index + 1] for key, value in labels.items()},
                        keep_cut[index:index + 1], keep_fill[index:index + 1])
                    accumulate(counts, image_counts)
                    count += 1
                    seen.append(ident)
                    if not cv2.imwrite(str(output/'predictions'/f'{ident}.png'),
                                       route_pred['final'][index, 0].astype(np.uint8)*255):
                        raise OSError('PNG write failed')
                    stream.write(json.dumps({'id': ident, 'routes': per_route,
                                             'candidate_counts': image_counts},
                                            ensure_ascii=False, allow_nan=False) + '\n')
                    text.write(f'Eval {count}/{len(ids)} ID={ident} '
                               f'IoU={per_route["final"]["iou"]:.9f} '
                               f'BF1={per_route["final"]["bf1"]:.9f}\n')
        if seen != ids or count != len(ids): raise ValueError('Evaluation omitted or duplicated samples')
        with (metric_root/PER_IMAGE_DIAGNOSTICS).open(encoding='utf-8') as stream:
            if sum(1 for line in stream if line.strip()) != count:
                raise ValueError('Per-image diagnostic JSONL is incomplete')
        verify_predictions(output/'predictions',ids,cfg['dataset']['input_size'])
        if int(px['final'].sum()) != count*256*256: raise ValueError('Pixel count mismatch')
        if sha256(checkpoint) != before or source_fingerprint() != fingerprint:
            raise ValueError('Checkpoint/source changed during evaluation')
        if {k:sha256(v) for k,v in cfg['dataset']['manifests'].items()} != split_hashes:
            raise ValueError('Split manifests changed during evaluation')
        oa,p,r,f1,iou = metrics_from_confusion(*map(int,px['final']))
        com,corr,bf1 = boundary_metrics_from_statistics(*map(int,bd['final']))
        objects = ob['final']/area['final'] if area['final'] else np.full(3,0. if px['final'][2]==0 else 1.)
        route_metrics = {}
        for route in ROUTES:
            route_oa, route_p, route_r, route_f1, route_iou = metrics_from_confusion(*map(int, px[route]))
            route_com, route_corr, route_bf1 = boundary_metrics_from_statistics(*map(int, bd[route]))
            route_objects = (ob[route]/area[route] if area[route]
                             else np.full(3, 0. if px[route][2] == 0 else 1.))
            route_metrics[route] = {
                'region_threshold': .5, 'threshold_comparison': '>', 'tta': False,
                'f1': route_f1, 'iou': route_iou, 'f_boundary': route_bf1,
                'goc': float(route_objects[0]), 'guc': float(route_objects[1]),
                'gtc': float(route_objects[2]),
                'oa': route_oa, 'precision': route_p, 'recall': route_r,
                'boundary_completeness': route_com, 'boundary_correctness': route_corr,
                'pixel_counts': px[route].tolist(), 'boundary_counts': bd[route].tolist(),
                'predicted_object_area_pixels': int(area[route]),
            }
        candidate_diagnostics = {
            'label_convention': 'strict_logit_positive',
            'raw_positive': 'raw_region_logit > 0',
            'sr_positive': 'sr_region_logit > 0',
            'changed': 'delta != 0',
            'truth_positive': 'region_truth > 0.5',
            'validation_route_threshold': 'sigmoid >= 0.5',
            'formal_route_threshold': 'sigmoid > 0.5',
            'undefined_rate_policy': 'null plus the always-present denominator count',
            'counts': dict(counts),
            'legacy_alias_counts': legacy_aliases(counts),
            'rates': rates(counts, undefined=None),
            'route_error_reduction_pixels': int(px['raw'][1] + px['raw'][2]
                                                - px['final'][1] - px['final'][2]),
            **net_pixels(counts),
        }
        report = {'complete_split':True,'samples':count,'split':split,'checkpoint':str(checkpoint),
            'checkpoint_sha256':before,'checkpoint_epoch':cp['epoch'],'protocol':cfg['evaluation']['protocol'],
            'region_threshold':.5,'threshold_comparison':'>','boundary_tolerance_pixels':3,'tta':False,'postprocessing':False,
            'primary_route':'final',
            'primary_metrics':{'f1':f1,'iou':iou,'f_boundary':bf1,'goc':float(objects[0]),'guc':float(objects[1]),'gtc':float(objects[2])},
            'diagnostic_metrics':{'oa':oa,'precision':p,'recall':r,'boundary_completeness':com,'boundary_correctness':corr},
            'pixel_counts':px['final'].tolist(),'boundary_counts':bd['final'].tolist(),'predicted_object_area_pixels':area['final'],
            'route_metrics':route_metrics,'candidate_diagnostics':candidate_diagnostics,
            'per_image_diagnostics':PER_IMAGE_DIAGNOSTICS,'per_image_diagnostics_rows':count,
            'config':cfg,'source_fingerprint':fingerprint,'split_manifest_sha256':split_hashes,
            'pretrained_provenance':cp['pretrained_provenance'],'png_verified':count}
        atomic_json(metric_root/'metrics.json',report)
        atomic_json(metric_root/'status.json',{'status':'completed','complete_split':True,'exit_code':0})
        print(json.dumps(report,indent=2))
    except BaseException as exc:
        atomic_json(metric_root/'status.json',{'status':'failed','complete_split':False,'error':str(exc),'exit_code':1})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'config_protocol_b.yaml'))
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--test-name')
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    cfg = config(args.config)
    run(cfg,args.checkpoint,args.split,args.test_name or cfg['experiment']['run_name']+'_best',torch.device(args.device))
