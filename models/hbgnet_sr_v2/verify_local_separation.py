"""CPU checks for the refinement module, without formal outputs."""
import copy
import io
import json

import torch
from tools.local_separation_refinement_v2 import LocalSeparationRefinement


def inputs(height=13, width=17, batch=2, gradients=False):
    f = torch.randn(batch,64,height,width,requires_grad=gradients)
    z = torch.randn(batch,1,height,width,requires_grad=gradients)
    e = torch.randn(batch,2,height,width).log_softmax(1).detach().requires_grad_(gradients)
    return f,z,e


def nonzero(model):
    with torch.no_grad():
        model.head[-1].weight.normal_(0,.1)
        model.head[-1].bias.fill_(.02)


def verify():
    initial = torch.get_rng_state().clone()
    torch.manual_seed(517)
    result = {}
    try:
        model = LocalSeparationRefinement(checkpoint_groups=False)
        assert sum(p.numel() for p in model.parameters()) == 3941
        for h,w in [(1,1),(2,17),(17,2),(8,8),(13,17),(16,20)]:
            x=inputs(h,w,batch=1)
            delta,aux=model(*x,return_aux=True)
            assert delta.shape==(1,1,h,w) and aux['valid_count'].shape==delta.shape
            assert torch.isfinite(delta).all() and torch.count_nonzero(delta)==0
            if min(h,w)<3:assert torch.count_nonzero(aux['valid_count'])==0
        result['parameters_and_zero_init_sizes']=True
        grid=torch.arange(20.).reshape(1,1,4,5)
        shifted=model._sample(grid,1,-2)
        for y in range(4):
            for x in range(5):
                expected=grid[0,0,y+1,x-2] if y+1<4 and x>=2 else 0
                assert shifted[0,0,y,x]==expected
        assert torch.count_nonzero(model._sample(grid,8,0))==0
        for r in (2,4):
            for d in model.DIRECTIONS:
                mask=model._valid_mask(11,13,r,d)[0,0]
                for y in range(11):
                    for x in range(13):
                        expected=all(1<=y+sign*r*d[0]<10 and 1<=x+sign*r*d[1]<12 for sign in (-1,1))
                        assert bool(mask[y,x])==expected
        result['zero_fill_sampling_and_exact_valid_windows']=True
        s=torch.randn(2,16,13,17);c=torch.randn_like(s);u=torch.randn(2,4,13,17)
        p=torch.rand(2,1,13,17);b=torch.rand_like(p)
        for r in (2,4):
            for d in model.DIRECTIONS:
                a=model._descriptor(s,c,u,p,b,r,d)
                rev=model._descriptor(s,c,u,p,b,r,(-d[0],-d[1]))
                assert a.shape==(2,94,13,17) and torch.equal(a,rev)
        result['descriptor_94_and_endpoint_exchange_symmetry']=True
        nonzero(model)
        f,z,e=inputs(gradients=True)
        delta=model(f,z,e)
        delta.square().mean().backward()
        assert f.grad is not None and f.grad.abs().sum()>0
        assert z.grad is None and e.grad is None
        for name,param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
        for part in (model.projection,model.path_projection,model.encoder,model.head):
            assert sum(float(p.grad.abs().sum()) for p in part.parameters())>0
        assert delta.abs().max()<=2 and torch.count_nonzero(delta)>0
        result['semantic_gradient_and_prediction_hint_detach']=True
        with torch.no_grad():
            original=model._group_features(s,c,u,p,b,radius=2,direction=(0,1))
            uu=u.clone();uu[:,:,6,8]+=2
            changed=model._group_features(s,c,uu,p,b,radius=2,direction=(0,1))
            assert (original[:,:,:,3:-3]-changed[:,:,:,3:-3]).abs().max()>0
            modified_s=s.clone();modified_s[:,:,5:8,9:12]+=2
            changed_side=model._group_features(modified_s,c,u,p,b,radius=2,direction=(0,1))
            assert (original-changed_side).abs().max()>0
            base=model(f,z,e)
            zz=z.detach().clone();zz[:,:,4:9,5:12]+=3
            altered=model(f,zz,e)
            assert (base-altered).abs().max()>0
        result['path_endpoints_and_region_hints_affect_forward']=True
        for h,w in [(1,1),(2,17),(17,2)]:
            assert torch.count_nonzero(model(*inputs(h,w,batch=1)))==0
        with torch.no_grad():model.head[-1].bias.fill_(100)
        delta,aux=model(*inputs(),return_aux=True)
        assert delta.abs().max()<=2 and torch.all(delta[aux['valid_count']>0]==2)
        result['bound_and_no_valid_group_mask']=True
        nonzero(model)
        normal=copy.deepcopy(model).train();checked=copy.deepcopy(model).train()
        checked.checkpoint_groups=True
        a=inputs(gradients=True)
        b_in=tuple(v.detach().clone().requires_grad_() for v in a)
        ya=normal(*a);yb=checked(*b_in)
        assert torch.equal(ya,yb)
        ya.square().mean().backward();yb.square().mean().backward()
        assert torch.allclose(a[0].grad,b_in[0].grad,atol=1e-7,rtol=1e-5)
        for (name,pa),(_,pb) in zip(normal.named_parameters(),checked.named_parameters()):
            assert torch.allclose(pa.grad,pb.grad,atol=1e-7,rtol=1e-5),name
        result['checkpoint_output_and_gradient_equivalence']=True
        optimizer=torch.optim.Adam(checked.parameters(),lr=1e-4)
        ids=[id(p) for group in optimizer.param_groups for p in group['params']]
        assert len(ids)==len(set(ids))==len(list(checked.parameters()))
        before={n:p.detach().clone() for n,p in checked.named_parameters()}
        optimizer.step()
        changed=[n for n,p in checked.named_parameters() if not torch.equal(before[n],p)]
        assert len(changed)==len(before),(changed,list(before))
        result['adam_updates_all_refiner_tensors']=len(changed)
        checked.eval();x=inputs()
        with torch.no_grad():expected=checked(*x)
        stream=io.BytesIO();torch.save(checked.state_dict(),stream);stream.seek(0)
        restored=LocalSeparationRefinement().eval()
        restored.load_state_dict(torch.load(stream,map_location='cpu',weights_only=True),strict=True)
        with torch.no_grad():actual=restored(*x)
        assert torch.equal(expected,actual)
        result['safe_strict_roundtrip']=True
        result['status']='pass'
        return result
    finally:
        torch.set_rng_state(initial)


if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(verify(),indent=2))
