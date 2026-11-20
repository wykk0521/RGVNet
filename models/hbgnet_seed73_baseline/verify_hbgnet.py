"""CPU engineering checks only; no formal training, held-out labels, or smoke weights."""
import ast
import copy
import gc
import io
import json
import platform
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch
from common import (ROOT, np, torch, config, build, loader, loss_fn, seed_all,
                    sha256, source_fingerprint, atomic_json, evaluate)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    seed_all(41)
    cfg = config(ROOT/'config_protocol_b.yaml')
    fingerprint = source_fingerprint()
    report = {'status':'running','kind':'CPU engineering, not a trained-model result',
              'device':'cpu','config':cfg,'source_fingerprint':fingerprint,'checks':[],
              'test_labels_read':False,'formal_training_started':False,'torch':str(torch.__version__),
              'python':platform.python_version()}
    started = time.time()
    def passed(name, **details):
        report['checks'].append({'name':name,'status':'pass',**details})
        print('PASS',name,details,flush=True)
    try:
        upstream = json.loads((ROOT/'source_manifest.json').read_text())
        for name,digest in upstream['files_sha256'].items():
            assert sha256(ROOT/'upstream'/name) == digest, name
        assert sha256(ROOT/'upstream/losses.py') == sha256(ROOT/'implementation/losses.py')
        assert sha256(ROOT/'upstream/lib/pvtv2.py') == sha256(ROOT/'implementation/lib/pvtv2.py')
        passed('official source integrity and unchanged loss/backbone', files=len(upstream['files_sha256']))
        reference = ROOT / 'protocol/seanet_control/ai4boundaries'
        assert sha256(ROOT/'protocol/dataset.py') == sha256(reference/'dataset.py')
        assert (ROOT/'protocol/seanet_reference.yaml').read_bytes() == (reference/'references/config_v5.yaml').read_bytes().replace(b'/root/wyk5/', b'/root/wyk5/')
        passed('live SEANet dataset and frozen protocol reference parity')
        # Only import and pretraining I/O are adapted; compare every forward's AST.
        def forwards(text):
            return {n.name:ast.dump(f,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.ClassDef)
                    for f in n.body if isinstance(f,ast.FunctionDef) and f.name=='forward'}
        original = (ROOT/'upstream/models.py').read_text()
        assert forwards(original) == forwards((ROOT/'implementation/model.py').read_text())
        passed('all official forward bodies unchanged')
        for key, change in [('training',{'epochs':39}),('evaluation',{'region_threshold':.6}),('model',{'drop_rate':.3})]:
            import yaml
            changed = copy.deepcopy(cfg); changed[key].update(change)
            with patch('common.load_config', side_effect=[changed,config(ROOT/'config_protocol_b.yaml')]):
                try: config(ROOT/'config_protocol_b.yaml')
                except ValueError: pass
                else: raise AssertionError(f'Guard accepted {key}')
        passed('configuration negative guards')
        with patch('common.Path.is_file',return_value=False):
            try: build(cfg,pretrained=True)
            except FileNotFoundError as exc: assert 'random fallback forbidden' in str(exc)
            else: raise AssertionError('Missing pretraining accepted')
        passed('missing pretraining refuses formal build')
        train_loader = loader(cfg,'train',batch=1,workers=0)
        val_loader = loader(cfg,'val',batch=1,workers=0)
        assert len(train_loader.dataset)==13890 and len(val_loader.dataset)==3067
        ids = {}
        for split,path in cfg['dataset']['manifests'].items():
            values = Path(path).read_text().splitlines()
            assert len(values)=={'train':13890,'val':3067,'test':3014}[split]
            assert len(values)==len(set(values))
            ids[split] = set(values)
        assert not (ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
        batch = next(iter(train_loader))
        for name in ('image','region','boundary','distance'): assert torch.isfinite(batch[name]).all()
        assert batch['image'].shape == (1,3,256,256)
        for name in ('region','boundary'): assert ((batch[name]==0)|(batch[name]==1)).all()
        assert batch['distance'].min()>=0 and batch['distance'].max()<=1
        passed('fixed split and real training data',train=13890,val=3067,test_manifest_only=3014)
        seed_all(41)
        model = build(cfg,pretrained=False).eval()
        original = original.replace('from lib.pvtv2 import pvt_v2_b2','from implementation.lib.pvtv2 import pvt_v2_b2')
        module = types.ModuleType('hbgnet_official_control')
        exec(compile(original,str(ROOT/'upstream/models.py'),'exec'),module.__dict__)
        seed_all(41)
        with patch('torch.load',return_value=model.backbone.state_dict()):
            control = module.Field().eval()
        assert model.state_dict().keys()==control.state_dict().keys()
        assert all(torch.equal(v,control.state_dict()[k]) for k,v in model.state_dict().items())
        assert torch.isfinite(next(model.parameters())).all()
        with torch.no_grad():
            outputs = model(batch['image'])
            expected = control(batch['image'])
        assert [list(v.shape) for v in outputs] == [[1,1,256,256],[1,2,256,256],[1,1,256,256]]
        assert all(torch.equal(a,b) for a,b in zip(outputs,expected))
        assert torch.allclose(outputs[1].exp().sum(1),torch.ones(1,256,256),atol=1e-6)
        passed('official initialized state and 256px forward parity',parameters=sum(p.numel() for p in model.parameters()),
               outputs=[list(v.shape) for v in outputs])
        del control, expected; gc.collect()
        from implementation.model import Field
        backbone_buffer = io.BytesIO()
        torch.save(model.backbone.state_dict(),backbone_buffer)
        safe_load = torch.load
        def controlled_load(path, **kwargs):
            assert kwargs == {'map_location':'cpu','weights_only':True}
            backbone_buffer.seek(0)
            return safe_load(backbone_buffer,**kwargs)
        with patch('torch.load',side_effect=controlled_load):
            initialized = Field(pretrained_path='synthetic-backbone-for-check-only')
        assert all(torch.equal(v,initialized.backbone.state_dict()[k]) for k,v in model.backbone.state_dict().items())
        del initialized, backbone_buffer; gc.collect()
        with patch('torch.load',return_value={}):
            try: Field(pretrained_path='incomplete-synthetic-check')
            except ValueError as exc: assert 'Incomplete PVT pretraining' in str(exc)
            else: raise AssertionError('Incomplete backbone accepted')
        passed('strict pretrained loader positive and incomplete-key negative',weights='synthetic engineering state, not official pretraining')
        from implementation.losses import BCEDiceLoss
        official = BCEDiceLoss()(outputs[0],batch['region']) + torch.nn.functional.nll_loss(outputs[1],batch['boundary'].long().squeeze(1)) + torch.nn.functional.mse_loss(outputs[2],batch['distance'])
        assert torch.equal(loss_fn(outputs,batch),official)
        passed('official three-task loss parity',loss=float(official))
        laplace = [p for n,p in model.edge_lap.named_parameters() if p.ndim==4 and tuple(p.shape[-2:])==(3,3)]
        kernel = torch.tensor([[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]])
        assert len(laplace)==1 and laplace[0].requires_grad and torch.equal(laplace[0],kernel.repeat(1,3,1,1))
        passed('Laplace initializer remains trainable')
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        registered = [id(p) for g in optimizer.param_groups for p in g['params']]
        assert len(registered)==len(set(registered))==len(list(model.parameters()))
        before = model.out_feature.weight.detach().clone()
        loss = loss_fn(model(batch['image']),batch)
        loss.backward()
        gradients = {n:p.grad for n,p in model.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        assert laplace[0].grad is not None and laplace[0].grad.abs().sum()>0
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        assert not torch.equal(before,model.out_feature.weight)
        unused = [n for n,p in model.named_parameters() if n not in gradients]
        passed('real sample backward and Adam update',loss=float(loss.detach()),gradient_tensors=len(gradients),unused_official_parameters=unused)
        del gradients, optimizer, loss; gc.collect()
        model.eval()
        buffer = io.BytesIO(); torch.save({'model':model.state_dict()},buffer);buffer.seek(0)
        restored = build(cfg,pretrained=False).eval()
        restored.load_state_dict(torch.load(buffer,map_location='cpu',weights_only=True)['model'],strict=True)
        with torch.no_grad():
            a=model(batch['image']); b=restored(batch['image'])
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
        passed('fresh instance checkpoint strict roundtrip')
        del restored, buffer, a, b; gc.collect()
        validation = evaluate(model,val_loader,torch.device('cpu'),max_batches=1)
        assert validation['samples']==1 and all(np.isfinite(v) for v in validation.values())
        passed('one real validation sample',metrics=validation)
        del model; gc.collect()
        # Compile the same seven existing SEANet functions without importing its model.
        from protocol import metrics
        src = (ROOT/'protocol/seanet_control/test_ablation.py').read_text()
        nodes = [n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and hasattr(metrics,n.name)]
        namespace = dict(metrics.__dict__)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<existing-seanet-metrics>','exec'),namespace)
        assert len(nodes)==7
        patterns = []
        z = np.zeros((32,32),bool); q=z.copy();q[6:19,8:21]=True
        patterns.extend([(z,z),(q,q),(q,z),(z,q),(q,np.roll(q,4,axis=0))])
        for p,t in patterns:
            for name,args in [('binary_confusion',(p,t)),('boundary_statistics',(p,t,3)),('object_error_sums',(p,t))]:
                assert getattr(metrics,name)(*args)==namespace[name](*args)
        passed('shared SEANet pixel boundary and object metrics parity',fixtures=len(patterns))
        from verify_training_state import verify as verify_state
        passed('training state recovery',details=verify_state())
        import tempfile
        import cv2
        from evaluate import verify_predictions
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='png_check_') as tmp:
            directory = Path(tmp)
            image = np.zeros((32,32),dtype=np.uint8); image[5:17,6:18]=255
            assert cv2.imwrite(str(directory/'sample.png'),image)
            verify_predictions(directory,['sample'],[32,32])
            image[0,0]=127
            assert cv2.imwrite(str(directory/'sample.png'),image)
            try: verify_predictions(directory,['sample'],[32,32])
            except ValueError: pass
            else: raise AssertionError('Non-binary PNG accepted')
        passed('prediction PNG contract and negative guard')
        report.update(status='pass',seconds=time.time()-started,
                      pretraining_file_present=Path(cfg['model']['pretrained_path']).is_file(),
                      scope_limitations=['No formal batch-8 GPU capacity test','No full training or held-out evaluation','Model checks use random initialization; not an accuracy reproduction'])
        atomic_json(ROOT/'verification.json',report)
        print('ALL ENGINEERING CHECKS PASSED',flush=True)
    except BaseException as exc:
        report.update(status='fail',error=repr(exc),seconds=time.time()-started)
        atomic_json(ROOT/'verification.json',report)
        raise


if __name__=='__main__': main()
