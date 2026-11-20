"""Evaluate one validation-selected checkpoint. Each process imports exactly one trained architecture."""
import os,sys,json,hashlib,argparse
from pathlib import Path
import numpy as np,torch
BASE=Path(__file__).resolve().parents[1];SRC=Path('/root/wyk5/RGVNet');ART=BASE/'artifacts'
def atomic(path,obj):
    path=Path(path);tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def load_one(run,device):
    kind,seed=run.split('_');seed=int(seed);job=BASE/'jobs'/run
    if (job/'contract.json').is_file():
        os.environ['REV_JOB']=run;import runtime_adapter as r
        cfg=r.config(job/'config.yaml');info=json.loads((ART/'logs'/run/'training_status.json').read_text())
        assert info['status']=='completed' and info['epoch']==40
        path=ART/'weights'/run/'best.pt';ref_name=run
    else:
        assert kind in ('rgv','hbg') and (seed in (17,41,73) if kind=='rgv' else seed==73)
        folder='rd_v9_context' if kind=='rgv' else 'baseline'
        pkg=SRC/'models'/f'hbgnet_seed{seed}_{folder}';sys.path.insert(0,str(pkg))
        import common as r
        cfg=r.config(pkg/'config_protocol_b.yaml');ref_name=cfg['experiment']['run_name']
        path=SRC/'z_3_weights/AI4Boundaries_1m'/ref_name/'best.pt'
    assert cfg['training']['seed']==seed and path.name=='best.pt'
    data=torch.load(path,map_location='cpu',weights_only=True)
    assert data['run_name']==ref_name and data['checkpoint_kind']=='best' and data['best']['epoch']==data['epoch']
    assert data['split_manifest_sha256']=={k:sha(p) for k,p in cfg['dataset']['manifests'].items()}
    if (job/'contract.json').is_file():
        assert data['config']==cfg and data['source_fingerprint']==r.source_fingerprint()
    m=r.build(cfg,pretrained=False);m.load_state_dict(data['model'],strict=True);m.to(device).eval()
    meta={'run':run,'seed':seed,'source_checkpoint':str(path),'sha256':sha(path),'epoch':data['epoch'],'selection':'best validation 0.7 IoU + 0.3 BF1; earlier tie','device':str(device)}
    del data
    return r,cfg,m,meta
def total():
    return {'px':np.zeros(4,np.int64),'bd':np.zeros(4,np.int64),'ob':np.zeros(3,np.float64),'area':0}
def score(p,t):
    from protocol.metrics import binary_confusion,boundary_statistics,object_error_sums
    px=binary_confusion(p,t);bd=boundary_statistics(p,t,3);ob=object_error_sums(p,t)
    return px,bd,ob
def add_score(a,vals):
    px,bd,ob=vals;a['px']+=px;a['bd']+=bd;a['ob']+=ob[:3];a['area']+=ob[3]
def stats(a):
    from protocol.metrics import metrics_from_confusion,boundary_metrics_from_statistics
    oa,p,r,f,i=metrics_from_confusion(*map(int,a['px']));c,k,b=boundary_metrics_from_statistics(*map(int,a['bd']))
    return {'iou':float(i),'bf1':float(b),'f1':float(f),'goc':float(a['ob'][0]/a['area']) if a['area'] else None,'guc':float(a['ob'][1]/a['area']) if a['area'] else None,'gtc':float(a['ob'][2]/a['area']) if a['area'] else None,'pixel_counts':a['px'].tolist(),'boundary_counts':a['bd'].tolist(),'pred_area':int(a['area'])}
def margin_masks(raw,sr,aux,i,truth):
    result=raw.copy();out={}
    for branch,key in (('cut','cut'),('fill','fill')):
        cand=aux[key+'_edit'][i,0];keep=aux[key+'_keep'][i,0];ix=np.flatnonzero(cand);k=int(keep.sum())
        assert not (keep&~cand).any()
        # Stable raster order breaks ties. Ranking and k do not use the reference.
        top=ix[np.argsort(-np.abs(aux['sr'][i,0]).ravel()[ix],kind='stable')[:k]]
        result.ravel()[top]=sr.ravel()[top]
        good=~truth if branch=='cut' else truth
        out[branch]=dict(proposed=len(ix),beneficial_proposed=int((good&cand).sum()),accepted=k,
                         beneficial_cev=int((good&keep).sum()),beneficial_margin=int(good.ravel()[top].sum()))
    return result,out
def write_cache(path,mask):
    with path.open('wb') as f: f.write(np.packbits(mask,axis=None).tobytes())
def run_tiles(a,r,cfg,model,meta,device,work):
    from torch.utils.data import DataLoader
    ds=r.loader(cfg,a.split,batch=8,workers=0).dataset
    loader=DataLoader(ds,batch_size=4,num_workers=2,prefetch_factor=1,shuffle=False,drop_last=False)
    rgv=a.run.startswith(('rgv_','win7_'))
    cache=BASE/'cache'/f'{a.run}_{a.split}';dump=a.cache
    if dump:cache.mkdir(parents=True,exist_ok=True)
    sums={};candidate={d:dict.fromkeys(['proposed','beneficial_proposed','accepted','beneficial_cev','beneficial_margin'],0) for d in ('cut','fill')};seen=[]
    if a.cache_only and not dump:raise ValueError('cache_only requires cache')
    with (work/'per_image.jsonl').open('w') as log,torch.no_grad():
        for batch in loader:
            y=batch['region'][:,0].numpy().astype(bool)
            z=model(batch['image'].to(device))[0]
            if not torch.isfinite(z).all():raise ValueError('Nonfinite prediction')
            final=((z.sigmoid()>=0.5) if a.split=='val' else (z.sigmoid()>0.5))[:,0].cpu().numpy()
            if rgv:
                b=model.last_auxiliary
                aux={k:b[k].bool().cpu().numpy() for k in ('cut_edit','cut_keep','fill_edit','fill_keep')}
                aux['sr']=b['sr_region_logits'].cpu().numpy()
                raw=(b['raw_region_logits'].sigmoid()>.5)[:,0].cpu().numpy();sr=(b['sr_region_logits'].sigmoid()>.5)[:,0].cpu().numpy()
                assert torch.equal(z,b['final_region_logits'])
            for i,pid in enumerate(batch['image_id']):
                seen.append(pid);routes={'final':final[i]};cnt=None
                if rgv:
                    routes.update(raw=raw[i],drr=sr[i])
                    if not a.cache_only:
                        margin,cnt=margin_masks(raw[i],sr[i],aux,i,y[i]);routes['margin_equal_k']=margin
                        for d,yc in cnt.items():
                            for key,val in yc.items():candidate[d][key]+=val
                if dump:
                    write_cache(cache/(pid+'.bin'),raw[i] if rgv else final[i])
                    if rgv:write_cache(cache/(pid+'.final.bin'),final[i])
                row={'id':pid}
                if not a.cache_only:
                    for name,pred in routes.items():
                        vals=score(pred,y[i]);add_score(sums.setdefault(name,total()),vals)
                        row[name]={'px':[int(v) for v in vals[0]],'bd':[int(v) for v in vals[1]],'ob':[float(v) for v in vals[2][:3]]+[int(vals[2][3])]}
                    if cnt:row['edit_counts']=cnt
                log.write(json.dumps(row,separators=(',',':'))+'\n')
            if len(seen)%400==0:print(a.run,a.split,len(seen),flush=True)
    assert seen==ds.image_ids and len(set(seen))==len(seen)
    report=dict(meta,split=a.split,samples=len(seen),routes={k:stats(v) for k,v in sums.items()},candidates=candidate if rgv and not a.cache_only else None)
    if dump:atomic(cache/'manifest.json',dict(meta,split=a.split,samples=len(seen),shape=[256,256],packbits='big',primary='raw' if rgv else 'final',ids=seen))
    atomic(work/'complete.json',report)
def region_stats(pred,truth,band):
    from protocol.metrics import mask_boundary
    from skimage.morphology import disk,binary_dilation
    # Extract and match boundaries on the complete parent; never introduce an artificial boundary at the band edge.
    pb=mask_boundary(pred);tb=mask_boundary(truth);ft=disk(3);db=binary_dilation(pb,footprint=ft);dt=binary_dilation(tb,footprint=ft)
    return (
        (int((pred&truth&band).sum()),int((pred&~truth&band).sum()),int((~pred&truth&band).sum()),int((~pred&~truth&band).sum())),
        (int((tb&db&band).sum()),int((tb&band).sum()),int((pb&dt&band).sum()),int((pb&band).sum())),(0.,0.,0.,0))
def seam(a,r,cfg,model,meta,device,work):
    import rasterio
    from PIL import Image
    ids=sorted({s.rsplit('_',1)[0] for s in Path(cfg['dataset']['manifests']['test']).read_text().splitlines() if s})
    assert len(ids)==862
    orig=Path('/root/wyk5/dataset/AI4Boundaries_1m');mean=torch.tensor(cfg['normalization']['mean']).view(3,1,1);std=torch.tensor(cfg['normalization']['std']).view(3,1,1)
    y,x=np.ogrid[:512,:512];b=((y>=248)&(y<264))|((x>=248)&(x<264));regions={'whole':np.ones((512,512),bool),'seam8':b,'interior':~b}
    sums={}
    out_cache=work/'masks';out_cache.mkdir(exist_ok=True)
    with torch.no_grad(),(work/'per_parent.jsonl').open('w') as f:
        for idx,pid in enumerate(ids):
            with rasterio.open(orig/'images/test'/(pid+'_ortho_1m_512.tif')) as src: rgb=src.read((1,2,3))
            with rasterio.open(orig/'masks/test'/(pid+'_ortholabel_1m_512.tif')) as src:t=src.read(1)>.5
            assert rgb.shape==(3,512,512) and t.shape==(512,512)
            if idx in (0,100,400,861):
                # Confirm full parents and retained prepared 256-pixel quadrants match.
                for index,(yy,xx) in enumerate(((0,0),(0,256),(256,0),(256,256))):
                    tile=pid+'_'+str(index)
                    mp=Path(cfg['dataset']['mask_dir'])/'test'/(tile+cfg['dataset']['mask_suffix'])
                    if mp.exists():
                        with rasterio.open(mp) as src:assert np.array_equal(src.read(1)>.5,t[yy:yy+256,xx:xx+256])
                        ip=Path(cfg['dataset']['image_dir'])/'test'/(tile+cfg['dataset']['image_suffix'])
                        with rasterio.open(ip) as src:assert np.array_equal(src.read((1,2,3)),rgb[:,yy:yy+256,xx:xx+256])
            image=(torch.from_numpy(rgb.astype(np.float32))/255-mean)/std
            coords=[(y,x) for y in (0,128,256) for x in (0,128,256)]
            tiles=torch.stack([image[:,y:y+256,x:x+256] for y,x in coords])
            zs=torch.cat([model(ch.to(device))[0].detach().cpu() for ch in tiles.split(3)])[:,0]
            assert torch.isfinite(zs).all()
            row={'id':pid};preds={}
            for mode in ('direct','overlap'):
                acc=torch.zeros(512,512);cnt=torch.zeros(512,512)
                for z,(y0,x0) in zip(zs,coords):
                    if mode=='direct' and (y0==128 or x0==128):continue
                    acc[y0:y0+256,x0:x0+256]+=z;cnt[y0:y0+256,x0:x0+256]+=1
                assert cnt.min()>=1
                p=((acc/cnt).sigmoid()>.5).numpy();preds[mode]=p
                write_cache(out_cache/(pid+'_'+mode+'.bin'),p)
                for rr,band in regions.items():
                    key=mode+'_'+rr;v=region_stats(p,t,band);add_score(sums.setdefault(key,total()),v);row[key]=[list(v[0]),list(v[1])]
            row['disagreement_on_seam']=int(((preds['direct']!=preds['overlap'])&b).sum())
            f.write(json.dumps(row,separators=(',',':'))+'\n')
            # Fixed first parent example, supplementary individual counts permit independent selection.
            if idx==0:
                for key,img in dict(rgb=rgb.transpose(1,2,0),reference=t.astype(np.uint8)*255,**{k:v.astype(np.uint8)*255 for k,v in preds.items()}).items():
                    Image.fromarray(img.astype(np.uint8)).save(work/(pid+'_'+key+'.png'))
            if (idx+1)%75==0:print(a.run,'parents',idx+1,flush=True)
    atomic(work/'complete.json',dict(meta,parents=len(ids),width_each_side_px=8,strides=[256,128],results={k:stats(v) for k,v in sums.items()}))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',choices=['test','val'],default='test')
    p.add_argument('--cache',action='store_true');p.add_argument('--cache-only',action='store_true');p.add_argument('--seam',action='store_true')
    a=p.parse_args();dev=torch.device('cuda:0');r,cfg,m,meta=load_one(a.run,dev)
    if a.seam:seam(a,r,cfg,m,meta,dev,Path.cwd())
    else:run_tiles(a,r,cfg,m,meta,dev,Path.cwd())
if __name__=='__main__':main()

