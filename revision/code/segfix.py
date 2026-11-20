"""AI4 parcel adaptation of the authors' HRNet-W48 SegFix. GPU model never sees test labels for offsets."""
import argparse,os,sys,json,time,math,hashlib,random
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from scipy import ndimage as ndi
from eval_revision import BASE,SRC,atomic,sha,total,score,add_score,stats
THIRD=BASE/'third_party'; UP=THIRD/'openseg'; WEIGHT=THIRD/'hrnetv2_w48_imagenet_pretrained.pth'
sys.path[:0]=[str(UP),str(SRC/'models/hbgnet_seed73_baseline')]
os.environ.update(dt_num_classes='8',dt_min_distance='0',dt_max_distance='5',dt_scale='1')
import common as href
from lib.models.nets.segfix import SegFix_HRNet,BackboneSelector
from lib.models.backbones.hrnet.hrnet_backbone import HighResolutionNet
from lib.models.backbones.hrnet.hrnet_config import MODEL_CONFIGS
from lib.utils.helpers.offset_helper import DTOffsetHelper as DT,label_to_vector_mapping
from protocol.metrics import binary_confusion,boundary_statistics,metrics_from_confusion,boundary_metrics_from_statistics
TRAIN=BASE/'jobs/segfix_73'
RECIPE={'source':'https://github.com/openseg-group/openseg.pytorch','commit':'aefc75517b09068d7131a69420bc5f66cb41f0ee',
        'backbone':'hrnet48','initialization':'authors ImageNet backbone, unfrozen','heads':'official SegFix_HRNet mask + eight directions',
        'target':'binary region mask, Euclidean transform per class, class area cutoff 100, authors 5x5 gradient, quantize angle to two degrees',
        'pixel_shift':2,'edge_probability_threshold':.5,'edge_distance_inclusive':[0,5],'class_weights':'1 - class frequency; empty directional support gives zero',
        'seed':73,'epochs':40,'batch':8,'accumulation':2,'dtype':'FP32','optimizer':'SGD','momentum':.9,'decay':.0005,'initial_lr':.04,
        'lr_policy':'poly power 0.9 per optimizer update','single_gpu_normalization':'PyTorch BatchNorm2d instead of legacy in-place synchronized BN',
        'selection':'corrected HBGNet validation, 0.7 IoU + 0.3 BF1; earlier tie','test_augmentation':False}
def basecfg():
    return href.config(href.ROOT/'config_protocol_b.yaml')
class Cfg:
    def get(self,*keys):
        d={'network':{'backbone':'hrnet48','bn_type':'torchbn','pretrained':None}}
        for k in keys:d=d[k]
        return d
def model(pretrained=True):
    # The upstream 2019 factory hardcodes a CUDA extension for synchronized BN.
    # Use precisely its HRNet48 constructor, with PyTorch BN for this single-GPU run.
    with patch.object(BackboneSelector,'get_backbone',lambda self: HighResolutionNet(MODEL_CONFIGS['hrnet48'],bn_type='torchbn',bn_momentum=.1)):
        m=SegFix_HRNet(Cfg())
    if pretrained:
        rec=json.loads((THIRD/'provenance.json').read_text())
        if WEIGHT.stat().st_size!=rec['bytes'] or sha(WEIGHT)!=rec['sha256']:raise ValueError('Official ImageNet weight digest mismatch')
        w=torch.load(WEIGHT,map_location='cpu',weights_only=True)
        if isinstance(w,dict) and 'state_dict' in w:w=w['state_dict']
        w={k.removeprefix('module.'):v for k,v in w.items() if torch.is_tensor(v)}
        b=m.backbone.state_dict();good={k:v for k,v in w.items() if k in b and v.shape==b[k].shape}
        missing=[k for k in b if k not in good and not k.endswith('num_batches_tracked')]
        if missing:raise ValueError(f'Uninitialized parameters/buffers in official backbone: {missing[:12]}')
        b.update(good);m.backbone.load_state_dict(b,strict=True)
        print('Loaded official ImageNet backbone tensors',len(good),flush=True)
    return m
Y,X=np.mgrid[-2:3,-2:3].astype(np.float32);DEN=Y*Y+X*X;DEN[2,2]=1.;KY=Y/DEN;KX=X/DEN
def targets(mask):
    h,w=mask.shape;depth=np.zeros((h,w),np.float32);dy=np.zeros_like(depth);dx=np.zeros_like(depth)
    for c in (0,1):
        region=mask==c
        if int(region.sum())<100:continue
        d=ndi.distance_transform_edt(region).astype(np.float32)
        depth+=d;dy+=ndi.correlate(d,KY,mode='constant',cval=0)*region;dx+=ndi.correlate(d,KX,mode='constant',cval=0)*region
    depth=np.minimum(depth,250).astype(np.uint8)
    angle=(((np.degrees(np.arctan2(dy,dx))+180)/2).astype(np.uint8).astype(np.float32)*2-180)
    return torch.from_numpy(depth),torch.from_numpy(angle)
class WithTargets(Dataset):
    def __init__(self,cfg):self.base=href.AI4BoundariesSEANetDataset(cfg,'train',augment=True)
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        b=self.base[i];seg=b['region'][0].byte();dep,ang=targets(seg.numpy())
        return b['image'],seg,dep,ang
def balance(logits,labels,n):
    chosen=labels>=0
    if not chosen.any():return logits.sum()*0
    hist=torch.bincount(labels[chosen].reshape(-1),minlength=n).to(torch.float)
    weight=1-hist/hist.sum()
    # On degenerate single-class batches the authors' denominator is zero. Use
    # unweighted CE only in that case; preserve their weighting otherwise.
    if (weight*hist).sum()==0:weight=None
    return F.cross_entropy(logits,labels,weight=weight,ignore_index=-1)
def loss_fn(maps,seg,dep,angle):
    shape=seg.shape[-2:]
    mask,ds=[F.interpolate(x,shape,mode='bilinear',align_corners=True) for x in maps]
    gt=DT.distance_to_mask_label(dep,seg,return_tensor=True)
    isedge=mask.softmax(1)[:,1]>.5
    gt_dir=DT.angle_to_direction_label(angle,seg_label_map=seg,extra_ignore_mask=~isedge,return_tensor=True)
    return balance(mask,gt,2)+balance(ds,gt_dir,8)
def correct(maps,inputs):
    # Author's single-scale probability/offset decoder and integer label shift, with border replication.
    shape=inputs.shape[-2:];mask,ds=maps
    prob=F.interpolate(mask.softmax(1),shape,mode='bilinear',align_corners=True)[:,1]
    ds=F.interpolate(ds,shape,mode='bilinear',align_corners=True).argmax(1)
    return apply_offsets(inputs,ds,prob>.5)
def apply_offsets(inputs,labels,on):
    h,w=inputs.shape[-2:];vectors=torch.as_tensor(label_to_vector_mapping[8],device=labels.device)[labels]*2
    iy=(torch.arange(h,device=labels.device)[None,:,None]+vectors[...,0]*on).clamp(0,h-1)
    ix=(torch.arange(w,device=labels.device)[None,None,:]+vectors[...,1]*on).clamp(0,w-1)
    return torch.gather(inputs.flatten(1),1,(iy*w+ix).flatten(1)).reshape_as(inputs)
def check_core():
    # Inside the two vertical edges of a field, offsets point toward its interior.
    m=np.zeros((64,64),np.uint8);m[:,16:48]=1;d,a=targets(m)
    s=torch.from_numpy(m)[None];direction=DT.angle_to_direction_label(a[None],seg_label_map=s,return_tensor=True)
    assert direction[0,32,17]==4 and direction[0,32,46]==0
    inp=torch.zeros(1,9,9,dtype=torch.bool);inp[0,4,6]=True
    lab=torch.full_like(inp,4,dtype=torch.long);active=torch.zeros_like(inp);active[0,4,4]=True
    out=apply_offsets(inp,lab,active);assert out[0,4,4] and out[0,4,6] and out.sum()==2
    p=torch.randn(2,8,4,4,requires_grad=True);q=torch.full((2,4,4),-1,dtype=torch.long);empty=balance(p,q,8);empty.backward();assert p.grad.abs().sum()==0
def cached(ids,run,split,device,final=False):
    root=BASE/'cache'/f'{run}_{split}'
    ary=[]
    for i in ids:
        p=root/(i+('.final.bin' if final else '.bin'));b=np.fromfile(p,np.uint8);assert b.size==8192
        ary.append(np.unpackbits(b).reshape(256,256).astype(bool))
    return torch.from_numpy(np.stack(ary)).to(device)
def validate_cache(run,split,ds):
    meta=json.loads((BASE/'cache'/f'{run}_{split}'/'manifest.json').read_text())
    assert meta['ids']==ds.image_ids and meta['run']==run and meta['epoch']>0
@torch.no_grad()
def validate(m,dl,device):
    m.eval();px=np.zeros(4,np.int64);bd=np.zeros(4,np.int64);n=0
    for b in dl:
        pred=correct(m(b['image'].to(device)),cached(b['image_id'],'hbg_73','val',device)).cpu().numpy()
        for p,t in zip(pred,b['region'][:,0].numpy().astype(bool)):
            px+=binary_confusion(p,t);bd+=boundary_statistics(p,t,3);n+=1
    assert n==3067
    i=metrics_from_confusion(*map(int,px))[4];f=boundary_metrics_from_statistics(*map(int,bd))[2]
    return dict(n=n,iou=float(i),bf1=float(f),score=float(.7*i+.3*f))
def safe_save(x,p):
    tmp=p.with_suffix('.pt.tmp');torch.save(x,tmp);os.replace(tmp,p)
def fit():
    import fcntl
    lock=(TRAIN/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (TRAIN/'complete.json').exists():return
    dev=torch.device('cuda:0');cfg=basecfg();check_core();href.seed_all(73);m=model().to(dev);opts=torch.optim.SGD(m.parameters(),lr=.04,momentum=.9,weight_decay=.0005)
    g=torch.Generator().manual_seed(73);vgen=torch.Generator().manual_seed(73)
    train=DataLoader(WithTargets(cfg),batch_size=8,shuffle=True,drop_last=True,num_workers=4,prefetch_factor=1,worker_init_fn=href.seed_worker,generator=g,pin_memory=True)
    valds=href.AI4BoundariesSEANetDataset(cfg,'val',augment=False);validate_cache('hbg_73','val',valds)
    val=DataLoader(valds,batch_size=8,num_workers=2,prefetch_factor=1,shuffle=False,generator=vgen)
    from training_state import capture_rng,restore_rng
    gen={'train':g,'val':vgen};steps=0;start=1;best=None;totalsteps=40*math.ceil(len(train)/2)
    recipe=dict(RECIPE,sha256=sha(WEIGHT),adapter_sha256=sha(__file__),split_sha256={k:sha(v) for k,v in cfg['dataset']['manifests'].items()},train_samples=len(train.dataset),validation_samples=len(valds))
    last=TRAIN/'last.pt'
    if last.is_file():
        ck=torch.load(last,map_location='cpu',weights_only=True);assert ck['recipe']==recipe
        m.load_state_dict(ck['model']);opts.load_state_dict(ck['opt']);start=ck['epoch']+1;steps=ck['steps'];best=ck['best'];restore_rng(ck['rng'],gen);del ck
    atomic(TRAIN/'recipe.json',recipe)
    for ep in range(start,41):
        m.train();opts.zero_grad(set_to_none=True);ls=0.;now=time.monotonic()
        for ii,(img,seg,dep,angle) in enumerate(train,1):
            maps=m(img.to(dev));lv=loss_fn(maps,seg.to(dev),dep.to(dev),angle.to(dev));assert torch.isfinite(lv)
            (lv/min(2,len(train)-((ii-1)//2)*2)).backward();ls+=lv.item()
            if ii%2==0 or ii==len(train):
                for param in opts.param_groups:param['lr']=.04*(1-steps/totalsteps)**.9
                opts.step();opts.zero_grad(set_to_none=True);steps+=1
            if ii%100==0:print('epoch',ep,'batch',ii,'loss',round(lv.item(),5),flush=True)
        v=validate(m,val,dev) if ep==1 or ep%2==0 or ep>35 else None
        if v is not None and (best is None or v['score']>best['score']):
            best={'epoch':ep,**v};safe_save({'recipe':recipe,'model':m.state_dict(),'best':best},TRAIN/'best.pt')
        record=dict(epoch=ep,loss=ls/len(train),validation=v,best=best,steps=steps,seconds=time.monotonic()-now)
        safe_save({'recipe':recipe,'epoch':ep,'steps':steps,'best':best,'model':m.state_dict(),'opt':opts.state_dict(),'rng':capture_rng(gen)},last)
        atomic(TRAIN/f'epoch_{ep:02}.json',record);print(record,flush=True)
    atomic(TRAIN/'complete.json',{'epochs':40,'best':best,'sha256':sha(TRAIN/'best.pt'),'recipe':recipe})
def trained(device):
    receipt=json.loads((TRAIN/'complete.json').read_text());assert sha(TRAIN/'best.pt')==receipt['sha256']
    m=model(False);s=torch.load(TRAIN/'best.pt',map_location='cpu',weights_only=True);m.load_state_dict(s['model'],strict=True)
    return m.to(device).eval()
@torch.no_grad()
def test():
    dev=torch.device('cuda:0');m=trained(dev);cfg=basecfg();ds=href.AI4BoundariesSEANetDataset(cfg,'test',augment=False)
    for r in ('hbg_73','rgv_73'):validate_cache(r,'test',ds)
    dl=DataLoader(ds,batch_size=8,num_workers=2,prefetch_factor=1);sums={};seen=[]
    with Path('per_image.jsonl').open('w') as f:
        for b in dl:
            ids=b['image_id'];seen+=ids;maps=m(b['image'].to(dev));routes={}
            for r in ('hbg_73','rgv_73'):
                src=cached(ids,r,'test',dev);routes[r+'_input']=src.cpu().numpy();routes[r+'_segfix']=correct(maps,src).cpu().numpy()
            routes['rgv_73_final']=cached(ids,'rgv_73','test',dev,True).cpu().numpy()
            for i,(pid,t) in enumerate(zip(ids,b['region'][:,0].numpy().astype(bool))):
                row={'id':pid}
                for k,v in routes.items():
                    px,bd,ob=score(v[i],t);add_score(sums.setdefault(k,total()),(px,bd,ob));row[k]={'px':[int(x) for x in px],'bd':[int(x) for x in bd]}
                f.write(json.dumps(row,separators=(',',':'))+'\n')
    assert seen==ds.image_ids and len(seen)==3014
    atomic(Path('complete.json'),{'samples':len(seen),'seed':73,'routes':{k:stats(v) for k,v in sums.items()},'best_sha256':sha(TRAIN/'best.pt')})
def main():
    op=sys.argv[1]
    if op=='train':fit()
    elif op=='test':test()
    elif op=='check':
        torch.set_num_threads(1);check_core();m=model('--weights' in sys.argv);m.eval()
        ds=href.AI4BoundariesSEANetDataset(basecfg(),'train',augment=False);b=ds[0]
        x=b['image'][None,:,:64,:64];seg=b['region'][:, :64,:64].byte();dep,ang=targets(seg[0].numpy())
        out=m(x);lv=loss_fn(out,seg,dep[None],ang[None]);lv.backward();assert torch.isfinite(lv)
        print('Official SegFix CPU forward/backward and offset geometry verified; params',sum(p.numel() for p in m.parameters()),'loss',lv.item(),flush=True)
    else:raise ValueError(op)
if __name__=='__main__':main()

