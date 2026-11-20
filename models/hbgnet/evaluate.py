"""Full-split evaluation with the identical parcel metrics used by SEANet."""
import argparse
import json
import re
from pathlib import Path
import cv2
from common import (ROOT, RUN, np, torch, config, source_fingerprint, require_engineering,
                    sha256, loader, build, atomic_json)
from protocol.metrics import (binary_confusion, boundary_statistics, object_error_sums,
                              metrics_from_confusion, boundary_metrics_from_statistics)


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
    fingerprint = source_fingerprint()
    require_engineering(cfg, fingerprint)
    if not re.fullmatch(r'hbgnet_protocol_b_v1_[A-Za-z0-9_-]+', name):
        raise ValueError('Use an isolated HBGNet evaluation name')
    checkpoint = Path(checkpoint).resolve()
    weights = Path(cfg['paths']['weights_root'])/RUN
    if checkpoint not in {(weights/n).resolve() for n in ('best.pt','best_iou_constrained.pt')}:
        raise ValueError('Evaluate only this run selected best checkpoint; never last.pt')
    status = json.loads((Path(cfg['paths']['train_log_root'])/RUN/'training_status.json').read_text())
    if status.get('status') != 'completed' or status.get('epoch') != 40:
        raise ValueError('Full training must complete before formal evaluation')
    before = sha256(checkpoint)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if cp.get('run_name') != RUN or cp.get('config') != cfg or cp.get('source_fingerprint') != fingerprint:
        raise ValueError('Checkpoint/run/config/source mismatch')
    if cp.get('checkpoint_kind') != checkpoint.stem or cp.get('format_version') != 1:
        raise ValueError('Checkpoint selection kind/format mismatch')
    if status.get('source_fingerprint') != fingerprint:
        raise ValueError('Training completion belongs to different source files')
    split_hashes = {k:sha256(v) for k,v in cfg['dataset']['manifests'].items()}
    if cp.get('split_manifest_sha256') != split_hashes:
        raise ValueError('Split manifests changed since training')
    if any(not torch.isfinite(t).all() for t in cp['model'].values()):
        raise FloatingPointError('Non-finite checkpoint')
    metric_root = Path(cfg['paths']['test_log_root'])/name
    output = Path(cfg['paths']['results_root'])/name
    if metric_root.exists() or output.exists(): raise FileExistsError('Refuse to overwrite evaluation output')
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
        px = np.zeros(4,dtype=np.int64)
        bd = np.zeros(4,dtype=np.int64)
        ob = np.zeros(3,dtype=np.float64)
        area = count = 0
        seen = []
        with (metric_root/'accuracy.txt').open('x') as text:
            for batch in batches:
                logits = model(batch['image'].to(device))[0]
                if not torch.isfinite(logits).all(): raise FloatingPointError('Non-finite prediction')
                pred = logits.sigmoid().cpu().numpy() > .5
                for ident, p, t in zip(batch['image_id'],pred[:,0],batch['region'].numpy()[:,0]):
                    x = binary_confusion(p,t)
                    b = boundary_statistics(p,t,3)
                    o = object_error_sums(p,t)
                    px += x; bd += b; ob += o[:3]; area += o[3]; count += 1
                    seen.append(ident)
                    if not cv2.imwrite(str(output/'predictions'/f'{ident}.png'),p.astype(np.uint8)*255):
                        raise OSError('PNG write failed')
                    text.write(f'Eval {count}/{len(ids)} ID={ident} IoU={metrics_from_confusion(*x)[4]:.9f} BF1={boundary_metrics_from_statistics(*b)[2]:.9f}\n')
        if seen != ids or count != len(ids): raise ValueError('Evaluation omitted or duplicated samples')
        verify_predictions(output/'predictions',ids,cfg['dataset']['input_size'])
        if int(px.sum()) != count*256*256: raise ValueError('Pixel count mismatch')
        if sha256(checkpoint) != before or source_fingerprint() != fingerprint:
            raise ValueError('Checkpoint/source changed during evaluation')
        if {k:sha256(v) for k,v in cfg['dataset']['manifests'].items()} != split_hashes:
            raise ValueError('Split manifests changed during evaluation')
        oa,p,r,f1,iou = metrics_from_confusion(*map(int,px))
        com,corr,bf1 = boundary_metrics_from_statistics(*map(int,bd))
        objects = ob/area if area else np.full(3,0. if px[2]==0 else 1.)
        report = {'complete_split':True,'samples':count,'split':split,'checkpoint':str(checkpoint),
            'checkpoint_sha256':before,'checkpoint_epoch':cp['epoch'],'protocol':cfg['evaluation']['protocol'],
            'region_threshold':.5,'threshold_comparison':'>','boundary_tolerance_pixels':3,'tta':False,'postprocessing':False,
            'primary_metrics':{'f1':f1,'iou':iou,'f_boundary':bf1,'goc':float(objects[0]),'guc':float(objects[1]),'gtc':float(objects[2])},
            'diagnostic_metrics':{'oa':oa,'precision':p,'recall':r,'boundary_completeness':com,'boundary_correctness':corr},
            'pixel_counts':px.tolist(),'boundary_counts':bd.tolist(),'predicted_object_area_pixels':area,
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
    parser.add_argument('--test-name',default=RUN+'_best')
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    run(config(args.config),args.checkpoint,args.split,args.test_name,torch.device(args.device))
