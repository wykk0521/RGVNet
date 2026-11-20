"""Measure paired, end-to-end mask generation on an otherwise idle A40 after training."""
from pathlib import Path
import os,sys,json,subprocess,importlib.util,time,random,types,statistics,gc
import numpy as np,torch
from eval_revision import BASE,SRC,atomic,load_one,sha
ROUTES=['hbg','hbg_segfix','raw','raw_segfix','drr','final','window7']
ARCHIVE=SRC/'paper_route_latency_20261019_C76OOn/benchmark_routes_v3.py'
def child(route,repeat):
    dev=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ident='hbg_73' if route.startswith('hbg') else 'win7_73' if route=='window7' else 'rgv_73'
    api,cfg,m,meta=load_one(ident,dev)
    ds=api.loader(cfg,'test',batch=1,workers=0).dataset
    ids=sorted(random.Random(73).sample(range(len(ds)),32));imgs=[ds[i]['image'][None].to(dev) for i in ids]
    if not route.startswith('hbg'):
        import implementation.model as impl
        spec=importlib.util.spec_from_file_location('archived_route_timing',ARCHIVE);arch=importlib.util.module_from_spec(spec);spec.loader.exec_module(arch)
        fd,sig=arch.make_forward(impl,Path.cwd())
        r='raw' if route in ('raw','raw_segfix') else 'drr' if route=='drr' else 'final'
        with torch.inference_mode():
            m(imgs[0]);target=m.last_auxiliary[{'raw':'raw_region_logits','drr':'sr_region_logits','final':'final_region_logits'}[r]].detach().clone()
            m.last_auxiliary={};m.benchmark_route=r;m.forward=types.MethodType(fd,m)
            if r=='raw':m.region_refiner=None;m.edit_verifier=None
            if r=='drr':m.edit_verifier=None
            torch.testing.assert_close(m(imgs[0])[0],target,rtol=1e-5,atol=1e-6)
    extra=None
    if route.endswith('segfix'):
        from segfix import trained,correct
        extra=trained(dev)
    def predict(x):
        mask=(m(x)[0].sigmoid()>.5)[:,0]
        return correct(extra(x),mask) if extra is not None else mask
    timers=[torch.cuda.Event(enable_timing=True) for _ in range(2)];vals=[];events=[]
    with torch.inference_mode():
        for i in range(100):predict(imgs[i%32])
        torch.cuda.synchronize()
        for i in range(5):timers[0].record();predict(imgs[i]);timers[1].record();timers[1].synchronize();timers[0].elapsed_time(timers[1])
        gc.collect()
        for i in range(300):
            torch.cuda.synchronize();t=time.perf_counter_ns();timers[0].record();out=predict(imgs[i%32]);timers[1].record();timers[1].synchronize()
            vals.append((time.perf_counter_ns()-t)/1e6);events.append(timers[0].elapsed_time(timers[1]));del out
    atomic(Path(f'round{repeat}_{route}.json'),dict(source=meta,full_mask_ms=vals,event_ms=events,active_parameters=sum(p.numel() for p in m.parameters())+(sum(p.numel() for p in extra.parameters()) if extra is not None else 0)))
def main():
    if len(sys.argv)>1:return child(sys.argv[1],int(sys.argv[2]))
    from collections import defaultdict
    reports=defaultdict(list)
    for i in range(3):
        # No rival job may be active on the measured GPU.
        g=os.environ['CUDA_VISIBLE_DEVICES'];used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if any(s.split(',')[0].strip()==g for s in used.splitlines()):raise RuntimeError('Other process on benchmark GPU')
        for r in ROUTES[i*2:]+ROUTES[:i*2]:
            subprocess.run([sys.executable,'-B','-u',__file__,r,str(i+1)],check=True)
            reports[r].append(json.loads(Path(f'round{i+1}_{r}.json').read_text()))
    summary={}
    for r,rr in reports.items():
        vv=[x for rec in rr for x in rec['full_mask_ms']]
        summary[r]=dict(median_ms=float(np.median(vv)),p95_ms=float(np.percentile(vv,95)),n=len(vv),params=rr[0]['active_parameters'])
    atomic(Path('complete.json'),{'device':'A40','batch':1,'fp32':True,'input_on_gpu':True,'warmup_per_round':100,'timed_per_round':300,
        'rounds':3,'includes':'backbone, selected refinement, binary mask decoding; excludes I/O and computing metrics','results':summary})
if __name__=='__main__':main()

