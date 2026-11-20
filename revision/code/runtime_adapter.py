"""Revision-specific job identity, provenance and genuine preflight. Scientific operations are imported from the selected original package."""
from pathlib import Path
import copy,gc,hashlib,importlib.util,json,os,sys
BASE=Path(__file__).resolve().parents[1]; RUN=os.environ['REV_JOB'];ROOT=BASE/'jobs'/RUN
C=json.loads((ROOT/'contract.json').read_text());assert C['id']==RUN and ROOT.parent==BASE/'jobs'
KIND=C['family'];SEED=C['seed'];SRC=Path('/root/wyk5/RGVNet')
TEMPLATE=Path(C['template']);PKG=TEMPLATE.parent
sys.path.insert(0,str(PKG))
if KIND=='win7':
    import tools as ztools
    n='tools.region_discrimination_v9';sp=importlib.util.spec_from_file_location(n,ROOT/'region_discrimination_win7.py')
    mod=importlib.util.module_from_spec(sp);sys.modules[n]=mod;sp.loader.exec_module(mod);setattr(ztools,'region_discrimination_v9',mod)
import common as ref
import yaml,torch
from protocol.dataset import resolve_dataset_variant
refcfg=ref.config(TEMPLATE)
RUN_CONFIGS={RUN:'config.yaml'}
sha256=ref.sha256; seed_all=ref.seed_all; device_batch=ref.device_batch; evaluate=ref.evaluate;atomic_json=ref.atomic_json
_same=getattr(ref,'_same',lambda a,b:a==b)
def config(path):
    if Path(path).resolve()!=ROOT/'config.yaml':raise ValueError('Unexpected run configuration path')
    cfg=resolve_dataset_variant(yaml.safe_load(Path(path).read_text()))
    expected=copy.deepcopy(refcfg);expected['training']['seed']=SEED
    expected['experiment']={'run_name':RUN,'stage':'ai4_reviewer_revision'}
    for key,folder in [('train_log_root','logs'),('weights_root','weights'),('test_log_root','test'),('results_root','results')]:
        expected['paths'][key]=str(BASE/'artifacts'/folder)
    expected['revision']={'family':KIND,'template':str(TEMPLATE),'seed':SEED,
        'masked_context_windows':[7] if KIND=='win7' else [3,7,15] if KIND=='rgv' else [],'loader_prefetch_factor':1,'archived_num_workers':8}
    if not _same(cfg,expected):raise ValueError('Revision recipe differs outside authorized experiment factors')
    return cfg
def config_source_path(cfg):
    assert _same(cfg,config(ROOT/'config.yaml'));return ROOT/'config.yaml'
def source_fingerprint():
    actual={p:sha256(p) for p in C['sources']}
    if actual!=C['sources']:raise RuntimeError('Revision source/config changed after sealing')
    return actual
def build(cfg,pretrained=True):
    return ref.build(refcfg,pretrained=pretrained)
def loss_fn(outputs,batch,cfg=None,training_aux=None,return_components=False):
    if KIND=='hbg':return ref.loss_fn(outputs,batch)
    return ref.loss_fn(outputs,batch,refcfg,training_aux,return_components=return_components)
def loader(cfg,split,batch=None,workers=None):
    # Same dataset, generator, sampler, 8 logical workers, initialization and batches. Only limit read-ahead to fit the rented host.
    from torch.utils.data import DataLoader
    data=ref.AI4BoundariesSEANetDataset(cfg,split,augment=split=='train')
    g=torch.Generator().manual_seed(SEED);n=cfg['dataset']['num_workers'] if workers is None else workers
    return DataLoader(data,batch_size=batch or cfg['dataset']['batch_size'],shuffle=split=='train',num_workers=n,
                      drop_last=split=='train',pin_memory=True,generator=g,worker_init_fn=ref.seed_worker,persistent_workers=False,
                      **({'prefetch_factor':1} if n else {}))
def require_engineering(cfg,fingerprint):
    assert fingerprint==C['sources'] and _same(cfg,config(ROOT/'config.yaml'))
    assert {k:sha256(v) for k,v in cfg['dataset']['manifests'].items()}==C['dataset_sha256']
    assert sha256(cfg['model']['pretrained_path'])==C['pretrained_sha256']
    seed_all(SEED);dev=torch.device('cuda:0');model=build(cfg,True).to(dev).train()
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    has_drr=any(k.startswith('region_refiner.') for k,_ in model.named_parameters())
    has_cev=any(k.startswith('edit_verifier.') for k,_ in model.named_parameters())
    assert has_drr==(KIND in ('drr','rgv','win7')) and has_cev==(KIND in ('rgv','win7')), (KIND,has_drr,has_cev)
    if has_cev:
        from tools.region_discrimination_v9 import SUPPORT_SCALES,CONTEXT_DESCRIPTOR_CHANNELS
        assert list(SUPPORT_SCALES)==cfg['revision']['masked_context_windows']
        assert CONTEXT_DESCRIPTOR_CHANNELS==(53 if KIND=='win7' else 131)
    batch=next(iter(loader(cfg,'train',batch=2,workers=0)));ids=batch['image_id'];batch=device_batch(batch,dev)
    if KIND=='hbg':
        loss=loss_fn(model(batch['image']),batch)
    else:
        out,aux=model(batch['image'],return_training_aux=True)
        loss,components=loss_fn(out,batch,cfg,aux,True)
    assert torch.isfinite(loss);loss.backward()
    ng=sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all().item() and torch.count_nonzero(p.grad).item()>0)
    assert ng>10
    if KIND=='matched':assert cfg['loss']['raw_region_weight']==1 and cfg['loss']['raw_band_weight']==.25 and not has_drr
    atomic_json(ROOT/'preflight.json',{'actual_forward_backward':'pass','run':RUN,'seed':SEED,'device':torch.cuda.get_device_name(0),
        'train_samples':ids,'input_size':list(batch['image'].shape),'loss':float(loss.detach()),'gradient_tensors_nonzero':ng,
        'params':trainable,'context_windows':cfg['revision']['masked_context_windows'],'split_sha256':C['dataset_sha256'],
        'torch':torch.__version__,'source_hashes':fingerprint})
    print(f'PREFLIGHT PASSED {RUN}: {trainable} params, loss={loss.item():.5f}, {ng} nonzero finite gradients',flush=True)
    del model,loss,batch
    if KIND!='hbg':del out,aux
    gc.collect();torch.cuda.empty_cache()

