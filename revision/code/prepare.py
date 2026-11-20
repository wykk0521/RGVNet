"""Create new revision contracts without changing any archived job."""
from pathlib import Path
import json,hashlib,os
import yaml
BASE=Path(__file__).resolve().parents[1]
SRC=Path('/root/wyk5/RGVNet')
JOB_ORDER=['rgv_101','rgv_137','hbg_101','drr_101','hbg_137','drr_137',
           'win7_17','win7_41','win7_73','win7_101','win7_137','matched_17','matched_41','matched_73']
FAMILY={'rgv':'rd_v9_context','win7':'rd_v9_context','hbg':'baseline','drr':'sr_v2','matched':'sr_v2'}
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact(p,data):
    p=Path(p)
    if p.exists():
        if p.read_bytes()!=data: raise RuntimeError(f'Different existing task artifact: {p}')
    else:
        with p.open('xb') as f:f.write(data)
def emit(p,obj): exact(p,(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode())
def main():
    gpu_mem=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    (BASE/'jobs').mkdir(exist_ok=True);(BASE/'control').mkdir(exist_ok=True)
    for sub in ('logs','weights','test','results'):(BASE/'artifacts'/sub).mkdir(parents=True,exist_ok=True)
    for run in JOB_ORDER:
        kind,seed=run.split('_');seed=int(seed);job=BASE/'jobs'/run;job.mkdir(exist_ok=True)
        pkg=SRC/'models'/('hbgnet_seed73_'+FAMILY[kind]);source_cfg='config_nosr_band.yaml' if kind=='matched' else 'config_protocol_b.yaml'
        y=yaml.safe_load((pkg/source_cfg).read_text())
        y['training']['seed']=seed;y['experiment']={'run_name':run,'stage':'ai4_reviewer_revision'}
        for key,area in [('train_log_root','logs'),('weights_root','weights'),('test_log_root','test'),('results_root','results')]:
            y['paths'][key]=str(BASE/'artifacts'/area)
        y['revision']={'family':kind,'template':str(pkg/source_cfg),'seed':seed,
            'masked_context_windows':[7] if kind=='win7' else [3,7,15] if kind=='rgv' else [],
            'loader_prefetch_factor':1,'archived_num_workers':8}
        exact(job/'config.yaml',yaml.safe_dump(y,sort_keys=False).encode())
        trainer=(pkg/'train.py').read_text()
        assert trainer.count('from common import (')==1 and trainer.count("'seed': 73")==1
        trainer=trainer.replace('from common import (','from runtime_adapter import (').replace("'seed': 73",f"'seed': {seed}")
        if kind=='hbg':
            # The baseline driver used the historical hard-coded YAML filename for the log copy.
            trainer=trainer.replace("ROOT / 'config_protocol_b.yaml'","ROOT / 'config.yaml'").replace("ROOT/'config_protocol_b.yaml'","ROOT/'config.yaml'")
        exact(job/'trainer.py',trainer.encode())
        files=[pkg/source_cfg,pkg/'train.py',pkg/'common.py',pkg/'training_state.py',
               *(pkg/'protocol').rglob('*.py'),*(pkg/'protocol').glob('*.yaml'),*(pkg/'implementation').rglob('*.py')]
        if kind not in ('hbg',):
            files.extend(SRC/'tools'/s for s in ('__init__.py','local_separation_refinement_v2.py','region_boundary_band_loss_v2.py'))
        if kind in ('rgv','win7'):
            files.extend(SRC/'tools'/s for s in ('region_discrimination_v9.py','region_discrimination_v8_grok_fullloss.py'))
        ref=json.loads((pkg/'package_manifest.json').read_text())
        for p in files:
            if p.is_relative_to(pkg): prev=ref.get('files_sha256',{}).get(p.relative_to(pkg).as_posix())
            else:prev=ref.get('external_files_sha256',{}).get(p.relative_to(SRC).as_posix())
            if prev and p.suffix=='.py' and digest(p)!=prev:
                # The migration rewrote absolute path literals. Require exact original after restoring the known root substitution.
                hermetic=Path('/root/wyk5/.rgvnet-relocation/original/SEANet')/p.relative_to(SRC)
                if not hermetic.is_file() or digest(hermetic)!=prev or hermetic.read_bytes().replace(b'/root/wyk3',b'/root/wyk5').replace(b'/root/wyk2',b'/root/wyk5')!=p.read_bytes():
                    raise ValueError(f'Original scientific source checksum mismatch: {p}')
        files.extend([job/'config.yaml',job/'trainer.py',BASE/'code/runtime_adapter.py',BASE/'code/prepare.py'])
        if kind=='win7':
            p=SRC/'tools/region_discrimination_v9.py';t=p.read_text();needle='\nSUPPORT_SCALES = (3, 7, 15)\n';assert t.count(needle)==1
            exact(job/'region_discrimination_win7.py',t.replace(needle,'\nSUPPORT_SCALES = (7,)\n').encode())
            files.append(job/'region_discrimination_win7.py')
        stages={'split_manifest_sha256':json.loads((SRC/'splits/manifest.json').read_text())['ai4boundaries']}
        sp=y['dataset']['variants']['parcel_focused']['manifests']
        split={s:digest(p) for s,p in sp.items()};assert split==stages['split_manifest_sha256']
        counts={s:len([x for x in Path(p).read_text().splitlines() if x.strip()]) for s,p in sp.items()}
        assert counts=={'train':13890,'val':3067,'test':3014}
        emit(job/'contract.json',{'id':run,'family':kind,'seed':seed,'template':str(pkg/source_cfg),'dataset_sha256':split,
             'samples':counts,'pretrained_sha256':'80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d',
             'sources':{str(p):digest(p) for p in files},'runtime_cgroup_memory_max':gpu_mem})
    emit(BASE/'control/training_plan.json',JOB_ORDER)
    print('Prepared',len(JOB_ORDER),'isolated training contracts with archived scientific source and split checks.')
if __name__=='__main__':main()

