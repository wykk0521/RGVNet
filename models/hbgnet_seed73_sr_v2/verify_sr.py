"""Engineering acceptance only; no formal training outputs or test labels."""
import argparse
import ast
import copy
import gc
import io
import json
import tempfile
import time
import types
from pathlib import Path

from common import (ROOT, RUN_CONFIGS, atomic_json, build, config, device_batch, evaluate, loader,
                    loss_fn, seed_all, sha256, source_fingerprint, torch)


def must_fail(function, exception=Exception):
    try:
        function()
    except exception:
        return
    raise AssertionError('Expected rejection')


def control_class():
    text = (ROOT / 'protocol/baseline_model.py').read_text()
    text = text.replace('from .lib.pvtv2 import pvt_v2_b2',
                        'from implementation.lib.pvtv2 import pvt_v2_b2')
    module = types.ModuleType('frozen_hbgnet_control')
    exec(compile(text, str(ROOT / 'protocol/baseline_model.py'), 'exec'), module.__dict__)
    return module.Field


def loss_checks(configs):
    from implementation.losses import BCEDiceLoss, LossF
    from tools.region_boundary_band_loss_v2 import boundary_band_bce
    main = configs['hbgnet_protocol_b_seed73_sr_v2']
    target = torch.zeros(2,1,9,11)
    target[:,:,:,5:] = 1
    raw = torch.randn_like(target, requires_grad=True) + 0.0
    residual = torch.randn_like(target, requires_grad=True)
    final = raw + residual
    edge = torch.randn(2,2,9,11, requires_grad=True).log_softmax(1)
    distance = torch.randn_like(raw, requires_grad=True)
    batch = {'region': target, 'boundary': target, 'distance': target}
    outputs = [final,edge,distance]
    aux = {'raw_region_logits': raw}
    loss, parts = loss_fn(outputs,batch,main,aux,return_components=True)
    region = BCEDiceLoss()
    base = LossF([1,1,1])(final,edge,distance,target,target.long(),target)
    expected = base-region(final,target)+.5*region(raw,target)+.5*region(final,target)
    expected += .125*boundary_band_bce(raw,target)+.125*boundary_band_bce(final,target)
    assert torch.allclose(loss,expected,atol=1e-6,rtol=1e-6)
    assert all(isinstance(v,float) for v in parts.values())
    assert abs(parts['total']-float(loss.detach()))<1e-7
    must_fail(lambda: loss_fn(outputs,batch,main,{'raw_region_logits':raw.detach()}), ValueError)
    collapsed=[raw,edge,distance]
    a=loss_fn(collapsed,batch,main,aux)
    b=loss_fn(collapsed,batch,configs['hbgnet_protocol_b_seed73_sr_v2_nosr_band'],aux)
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
    off=loss_fn(outputs,batch,configs['hbgnet_protocol_b_seed73_sr_v2_disabled'],aux)
    assert torch.equal(off,base)
    for c in configs.values():
        value=loss_fn(outputs,batch,c,aux)
        assert torch.isfinite(value)
    grad_raw,grad_residual=torch.autograd.grad(loss,(raw,residual))
    assert grad_raw.abs().sum()>0 and grad_residual.abs().sum()>0
    z = torch.randn_like(target,requires_grad=True)
    d = torch.randn_like(target,requires_grad=True)
    zz=z+d
    raw_objective=.5*region(z,target)+.5*region(zz,target)
    branch_v2=torch.autograd.grad(raw_objective,d,retain_graph=True)[0]
    branch_v1=torch.autograd.grad(region(zz,target),d)[0]
    assert torch.allclose(branch_v2,.5*branch_v1,atol=1e-7,rtol=1e-6)
    return {'main_formula':True,'collapsed_no_sr_parity':True,'disabled_official_loss_parity':True,
            'raw_loss_graph_required':True,'branch_explicit_half_gradient_verified':True,
            'five_losses_finite':True}


def verify(device):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    configs = {run: config(ROOT / name) for run, name in RUN_CONFIGS.items()}
    cfg = config(ROOT / 'config_protocol_b.yaml')
    disabled = config(ROOT / 'config_disabled.yaml')
    fingerprint = source_fingerprint()
    started = time.monotonic()
    report = {'status': 'running', 'kind': 'engineering_not_accuracy_experiment',
              'configs': configs,
              'source_fingerprint': fingerprint, 'device': str(device),
              'test_labels_read': False, 'formal_training_started': False,
              'formal_evaluation_started': False, 'checks': []}

    def passed(name, **details):
        report['checks'].append({'name': name, 'status': 'pass', **details})
        print('PASS', name, json.dumps(details, ensure_ascii=False), flush=True)

    try:
        from verify_band_loss import run_checks
        passed('GT band mask normalization gradients and rejection checks', details=run_checks())
        from verify_local_separation import verify as verify_module
        passed('local separation module acceptance', details=verify_module())
        baseline_root = Path('/root/wyk5/RGVNet/models/hbgnet_seed73_baseline')
        for name in ('implementation/losses.py', 'implementation/lib/pvtv2.py',
                     'protocol/dataset.py', 'protocol/metrics.py', 'training_state.py'):
            assert sha256(ROOT / name) == sha256(baseline_root / name), name
        assert sha256(ROOT / 'protocol/baseline_model.py') == sha256(baseline_root / 'implementation/model.py')
        assert sha256(ROOT / 'protocol/hbgnet_baseline.yaml') == sha256(baseline_root / 'config_protocol_b.yaml')
        passed('frozen backbone loss dataset metrics and training-state bytes unchanged')

        import yaml
        raw = yaml.safe_load((ROOT / 'config_protocol_b.yaml').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.config-check-') as folder:
            path = Path(folder) / 'bad.yaml'
            for key, field, value in [('training', 'epochs', 39), ('model', 'drop_rate', .3),
                                      ('evaluation', 'region_threshold', .6),
                                      ('experiment', 'run_name', 'hbgnet_protocol_b_v1')]:
                changed = copy.deepcopy(raw)
                changed[key][field] = value
                path.write_text(yaml.safe_dump(changed))
                must_fail(lambda: config(path), ValueError)
            changed = copy.deepcopy(raw)
            changed['model']['refinement']['enabled'] = False
            path.write_text(yaml.safe_dump(changed))
            must_fail(lambda: config(path), ValueError)
        from common import _validate_config
        for section, key, value in [('loss', 'raw_region_weight', True),
                                    ('loss', 'raw_band_weight', .25),
                                    ('loss', 'band_radius_pixels', 3),
                                    ('experiment', 'run_name', 'hbgnet_protocol_b_seed73_sr_v2_stopraw')]:
            changed = copy.deepcopy(cfg)
            changed[section][key] = value
            must_fail(lambda: _validate_config(changed), ValueError)
        changed = copy.deepcopy(cfg)
        changed['model']['refinement']['gradient_mode'] = 'stopall'
        must_fail(lambda: _validate_config(changed), ValueError)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.duplicate-check-') as folder:
            p = Path(folder) / 'duplicate.yaml'
            p.write_text((ROOT/'config_protocol_b.yaml').read_text()+'\nloss: {}\n')
            must_fail(lambda: config(p), ValueError)
        passed('configuration drift duplicate keys reserved modes and run collision rejected')
        passed('five run loss coefficients and connected raw loss', details=loss_checks(configs))

        from common import require_engineering
        from unittest.mock import patch
        valid_report = {'status': 'pass', 'source_fingerprint': fingerprint,
                        'configs': report['configs']}
        with patch('common._validate_config', return_value=cfg), patch('common.json.loads', return_value=valid_report), patch.object(Path, 'read_text', return_value='{}'):
            for candidate in configs.values():
                require_engineering(candidate, fingerprint)
            must_fail(lambda: require_engineering(cfg, {'stale': 'fingerprint'}), ValueError)
        with patch('common._validate_config', return_value=cfg), patch('common.json.loads', return_value={'status': 'pass', 'source_fingerprint': fingerprint, 'config': cfg}), patch.object(Path, 'read_text', return_value='{}'):
            must_fail(lambda: require_engineering(cfg, fingerprint), ValueError)
        passed('verification gate requires new source and per-run configs')

        class DiagnosticFixture(torch.nn.Module):
            def forward(self, image):
                raw = torch.tensor([[[[-1., 1.], [-1., 1.]]]])
                refined = torch.tensor([[[[1., -1.], [-1., 1.]]]])
                self.last_auxiliary = {'raw_region_logits': raw, 'refined_region_logits': refined,
                                       'delta': refined-raw, 'valid_count': torch.ones_like(raw)}
                return [refined]
        fixture = {'image': torch.zeros(1,3,2,2), 'region': torch.ones(1,1,2,2)}
        diagnosis = evaluate(DiagnosticFixture(), [fixture], torch.device('cpu'))
        assert diagnosis['corrected_pixels'] == diagnosis['regressed_pixels'] == 1
        assert diagnosis['changed_pixels'] == 2 and diagnosis['net_corrected_pixels'] == 0
        assert diagnosis['delta_abs_mean'] == 1 and diagnosis['delta_saturation_fraction'] == .5
        passed('raw-refined diagnostic counts and saturation fixture')

        import train as trainer
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.entry-check-') as folder:
            temporary = Path(folder)
            with patch.object(trainer, 'ROOT', temporary):
                with trainer.training_lock(cfg):
                    def collide():
                        with trainer.training_lock(cfg):
                            pass
                    must_fail(collide, RuntimeError)
                    with trainer.training_lock(disabled):
                        pass
                with patch.object(trainer, '_train_locked', return_value=40) as entry:
                    assert trainer.main(['--config',str(ROOT/'config_protocol_b.yaml')]) == 0
                    assert entry.call_count == 1
                log, weights = temporary/'log', temporary/'weights'
                trainer._validate_outputs(log,weights,None)
                log.mkdir(); weights.mkdir()
                (log/'config_snapshot.json').write_text('{}')
                (weights/'last.pt').write_bytes(b'fixture')
                trainer._validate_outputs(log,weights,str(weights/'last.pt'))
                must_fail(lambda: trainer._validate_outputs(log,weights,None), FileExistsError)
                must_fail(lambda: trainer._validate_outputs(log,weights,str(weights/'best.pt')), ValueError)
        passed('per-run locking CLI single entry and resume path guards')

        seed_all(41)
        control = control_class()().eval()
        rng_after = torch.get_rng_state().clone()
        seed_all(41)
        off = build(disabled, pretrained=False).eval()
        assert torch.equal(rng_after, torch.get_rng_state())
        assert control.state_dict().keys() == off.state_dict().keys()
        assert all(torch.equal(v, off.state_dict()[k]) for k, v in control.state_dict().items())
        seed_all(41)
        on = build(cfg, pretrained=False).eval()
        assert torch.equal(rng_after, torch.get_rng_state())
        assert all(torch.equal(v, on.state_dict()[k]) for k, v in control.state_dict().items())
        count = sum(p.numel() for p in on.parameters())
        base_count = sum(p.numel() for p in control.parameters())
        assert count - base_count == 3941
        image = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            expected, off_out, on_out = control(image), off(image), on(image)
        assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(expected, off_out, on_out))
        assert torch.count_nonzero(on.last_auxiliary['delta']) == 0
        assert all(not x.requires_grad for x in on.last_auxiliary.values())
        passed('disabled exact baseline and zero-init enabled parity',
               baseline_parameters=base_count, refined_parameters=count, added_parameters=count-base_count)

        control.train(); off.train(); on.train()
        seed_all(903)
        with torch.no_grad(): a = control(image)
        rng_forward = torch.get_rng_state().clone()
        seed_all(903)
        with torch.no_grad(): b = on(image)
        assert torch.equal(rng_forward, torch.get_rng_state())
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        passed('training-mode dropout stream and outputs unchanged at initialization')

        on.eval()
        last_conv = [m for m in on.region_refiner.modules() if isinstance(m, torch.nn.Conv2d)][-1]
        with torch.no_grad(): last_conv.weight.normal_(0, .08)
        with torch.no_grad():
            output = on(image)
            aux = on.last_auxiliary
        assert torch.count_nonzero(aux['delta']) > 0
        assert torch.equal(output[0], aux['raw_region_logits'] + aux['delta'])
        assert aux['delta'].abs().max() <= 2
        captured = {}
        handles = [on.edge_feature.register_forward_hook(lambda m, a, o: captured.update(edge=o.detach())),
                   on.dis_feature.register_forward_hook(lambda m, a, o: captured.update(distance=o.detach()))]
        with torch.no_grad(): outputs = on(image)
        for handle in handles: handle.remove()
        assert torch.equal(outputs[1], torch.nn.functional.log_softmax(captured['edge'], dim=1))
        assert torch.equal(outputs[2], captured['distance'])
        passed('nonzero branch changes Region only and respects logit bound')
        captured_feature = {}
        handle = on.multi_fusion.register_forward_hook(lambda m, args, output: captured_feature.update(value=output))
        train_outputs, train_aux = on(image, return_training_aux=True)
        handle.remove()
        raw_connected = train_aux['raw_region_logits']
        features = captured_feature['value']
        delta_connected = train_outputs[0] - raw_connected
        direct = torch.autograd.grad(train_outputs[0].sum(), raw_connected, retain_graph=True)[0]
        assert torch.equal(direct, torch.ones_like(direct))
        feature_grad, raw_hint_grad, edge_hint_grad = torch.autograd.grad(
            delta_connected.square().mean(), (features, raw_connected, train_outputs[1]),
            allow_unused=True, retain_graph=True)
        assert feature_grad is not None and feature_grad.abs().sum() > 0
        assert raw_hint_grad is None or torch.count_nonzero(raw_hint_grad) == 0
        assert edge_hint_grad is None or torch.count_nonzero(edge_hint_grad) == 0
        raw_head_grad = torch.autograd.grad(raw_connected.square().mean(), on.out_feature.weight)[0]
        assert raw_head_grad.abs().sum() > 0
        assert raw_connected.requires_grad and all(not value.requires_grad for value in on.last_auxiliary.values())
        passed('connected raw training aux and explicit joint gradient routes')
        del train_outputs, train_aux, raw_connected, features, delta_connected, direct
        del captured_feature, feature_grad, raw_hint_grad, edge_hint_grad, raw_head_grad
        for candidate in configs.values():
            candidate_model = build(candidate, pretrained=False).eval()
            with torch.no_grad():
                candidate_outputs, candidate_aux = candidate_model(image, return_training_aux=True)
                target = (torch.rand_like(candidate_outputs[0]) > .5).float()
                batch_fixture = {'region': target, 'boundary': target, 'distance': target}
                candidate_loss = loss_fn(candidate_outputs, batch_fixture, candidate, candidate_aux)
            assert torch.isfinite(candidate_loss)
            enabled = candidate['model']['refinement']['enabled']
            assert (candidate_model.region_refiner is not None) == enabled
            if not enabled:
                assert torch.equal(candidate_outputs[0], candidate_aux['raw_region_logits'])
                assert sum(p.numel() for p in candidate_model.parameters()) == base_count
            del candidate_model, candidate_outputs, candidate_aux, candidate_loss
        passed('all five configuration model and loss forward paths')

        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.roundtrip-') as folder:
            checkpoint_path = Path(folder) / 'fixture.pt'
            torch.save({'model': on.state_dict(), 'config': cfg, 'source_fingerprint': fingerprint}, checkpoint_path)
            buffer = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert buffer['config'] == cfg and buffer['source_fingerprint'] == fingerprint
        restored = build(cfg, pretrained=False).eval()
        restored.load_state_dict(buffer['model'], strict=True)
        with torch.no_grad(): restored_output = restored(image)
        assert all(torch.equal(x, y) for x, y in zip(outputs, restored_output))
        must_fail(lambda: restored.load_state_dict(control.state_dict(), strict=True), RuntimeError)
        off.load_state_dict(control.state_dict(), strict=True)
        passed('fresh model safe checkpoint roundtrip and cross-architecture strict rejection')
        del control, off, on, restored, buffer, image, outputs, output, aux, captured
        gc.collect()

        from verify_training_state import verify as verify_state
        passed('checkpoint RNG Adam scheduler and transaction recovery', details=verify_state())

        train_batches = loader(cfg, 'train', batch=8, workers=0)
        val_batches = loader(cfg, 'val', batch=1, workers=0)
        assert len(train_batches.dataset) == 13890 and len(val_batches.dataset) == 3067
        assert len(train_batches) == 1736
        assert cfg['training']['gradient_accumulation_steps'] == 2
        from train import _split_hashes
        hashes = _split_hashes(cfg)
        assert hashes == {
            'train': '85e9ff248be6aa66fcbdba82d1f61a0d26d9ead9ef449a6708e211abc7181201',
            'val': '4be22a48eead8fe9c91f58c890dc161e70d81ff956b4ba75b2e92198f86dd5cd',
            'test': 'f669288a374c4a3fed1c3a0379e995ebe1db6a35564a10b6a0241613a4093163'}
        passed('frozen split budget and RGB three-task loader', samples=[13890,3067,3014], train_batches=1736)
        assert device.type == 'cuda' and torch.cuda.is_available(), 'Real batch8 GPU acceptance required'
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        seed_all(41)
        model = build(cfg, pretrained=True).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        iterator = iter(train_batches)
        real_batches = [device_batch(next(iterator), device) for _ in range(2)]
        for batch in real_batches:
            assert batch['image'].shape == (8,3,256,256)
            for key in ('image','region','boundary','distance'):
                assert torch.isfinite(batch[key]).all()
        losses = []
        changed_groups = {}
        for step in range(2):
            before = {n: p.detach().clone() for n,p in model.region_refiner.named_parameters()}
            optimizer.zero_grad(set_to_none=True)
            for batch in real_batches:
                outputs, training_aux = model(batch['image'], return_training_aux=True)
                assert [tuple(v.shape) for v in outputs] == [(8,1,256,256),(8,2,256,256),(8,1,256,256)]
                loss, components = loss_fn(outputs, batch, cfg, training_aux, return_components=True)
                assert abs(sum(v for k,v in components.items() if k.startswith('weighted_')) - components['total']) < 1e-5
                assert training_aux['raw_region_logits'].requires_grad
                assert torch.isfinite(loss)
                (loss / 2).backward()
                losses.append({'total': float(loss.detach()), 'components': components})
                assert model.last_auxiliary['delta'].abs().max() <= 2
                assert all(not v.requires_grad for v in model.last_auxiliary.values())
            del training_aux
            grads = {n: p.grad for n,p in model.named_parameters() if p.grad is not None}
            assert grads and all(torch.isfinite(g).all() for g in grads.values())
            if step == 1:
                for prefix in ('region_refiner.', 'backbone.', 'out_feature.', 'edge_feature.', 'dis_feature.'):
                    norm = sum(float(g.abs().sum()) for n,g in grads.items() if n.startswith(prefix))
                    assert norm > 0, prefix
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            changed = [n for n,p in model.region_refiner.named_parameters() if not torch.equal(before[n],p)]
            changed_groups[str(step+1)] = changed
            if step == 1:
                assert len(changed) == len(before), (changed, list(before))
        passed('real pretrained batch8 accumulation2 backward and two Adam updates',
               losses=losses, changed_refiner_parameters=changed_groups,
               peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
               peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
               real_unique_training_samples=16, samples_reused_for_second_engineering_step=True)
        del optimizer, real_batches, iterator, train_batches, outputs, loss, grads, before
        gc.collect(); torch.cuda.empty_cache()
        validation = evaluate(model, val_batches, device, max_batches=1)
        assert validation['samples'] == 1
        import math
        assert all(math.isfinite(v) for v in validation.values())
        assert validation['net_corrected_pixels'] == validation['corrected_pixels'] - validation['regressed_pixels']
        passed('one real validation sample with raw-refined diagnostics', metrics=validation)
        del model, val_batches
        gc.collect(); torch.cuda.empty_cache()
        assert source_fingerprint() == fingerprint
        for c in configs.values():
            run = c['experiment']['run_name']
            for root_key in ('train_log_root','weights_root'):
                assert not (Path(c['paths'][root_key])/run).exists()
        import evaluate as formal_evaluator
        with patch.object(formal_evaluator, 'source_fingerprint', return_value=fingerprint), patch.object(formal_evaluator, 'require_engineering'):
            for forbidden in ('last.pt', 'best_iou_constrained.pt'):
                path = Path(cfg['paths']['weights_root']) / cfg['experiment']['run_name'] / forbidden
                must_fail(lambda: formal_evaluator.run(cfg, path, 'test', cfg['experiment']['run_name']+'_guard', device), ValueError)
        assert all(not (Path(c['paths'][root])/c['experiment']['run_name']).exists()
                   for c in configs.values() for root in ('train_log_root', 'weights_root'))
        passed('best-only formal checkpoint guard source unchanged and no formal outputs')
        report.update(status='pass',seconds=time.monotonic()-started,
                      scope_limitations=['No full training or formal evaluation.',
                                         'Accuracy benefit and inference speed not established.',
                                         'Engineering-only optimizer states discarded; no smoke weights saved.'])
        atomic_json(ROOT/'verification.json',report)
        print('ALL SR ENGINEERING CHECKS PASSED',flush=True)
    except BaseException as exc:
        report.update(status='fail',error=repr(exc),seconds=time.monotonic()-started)
        atomic_json(ROOT/'verification.json',report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    verify(torch.device(parser.parse_args().device))
