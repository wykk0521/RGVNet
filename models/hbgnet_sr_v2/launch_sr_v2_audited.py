"""Launch the approved v2 trainer; observe its first two real updates only."""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path('/root/wyk5/RGVNet/models/hbgnet_sr_v2')
EVIDENCE = ROOT / 'formal_launch_01'
sys.path.insert(0, str(ROOT))
import common
import train


def emit(record):
    record['unix_time'] = time.time()
    text = json.dumps(record, allow_nan=False)
    with (EVIDENCE / 'activity.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(text + '\n')
    print('STARTUP_AUDIT', text, flush=True)


original_build = train.build
original_step = train._finite_step
original_loss = train.loss_fn
step_count = 0
loss_count = 0


def audited_build(*args, **kwargs):
    model = original_build(*args, **kwargs)
    assert model.region_refiner is not None
    assert sum(p.numel() for p in model.region_refiner.parameters()) == 3941
    calls = 0

    def observe_forward(module, inputs, output):
        nonlocal calls
        calls += 1
        if calls in (1, 3, 5):
            delta = (output[0] if isinstance(output, tuple) else output).detach()
            emit({'kind': 'refiner_forward', 'forward': calls,
                  'delta_abs_mean': float(delta.abs().mean()),
                  'delta_abs_max': float(delta.abs().max())})
        if calls == 5:
            handle.remove()

    handle = model.region_refiner.register_forward_hook(observe_forward)
    return model


def audited_step(model, optimizer, epoch, batch_index):
    global step_count
    step_count += 1
    selected = {name: p for name, p in model.named_parameters()
                if name.startswith(('region_refiner.', 'out_feature.',
                                    'edge_feature.', 'dis_feature.'))}
    counts = Counter(id(p) for group in optimizer.param_groups for p in group['params'])
    assert all(counts[id(p)] == 1 and p.requires_grad for p in selected.values())
    before = {name: p.detach().clone() for name, p in selected.items()}
    gradients = {name: None if p.grad is None else float(p.grad.detach().abs().sum())
                 for name, p in selected.items()}
    result = original_step(model, optimizer, epoch, batch_index)
    changes = {name: float((p.detach() - before[name]).abs().max())
               for name, p in selected.items()}
    emit({'kind': 'parameter_update', 'optimizer_update': step_count,
          'epoch': epoch, 'batch': batch_index, 'grad_l1': gradients,
          'max_parameter_change': changes})
    if step_count == 2:
        train._finite_step = original_step
    return result


def audited_loss(*args, **kwargs):
    global loss_count
    result = original_loss(*args, **kwargs)
    loss_count += 1
    if isinstance(result, tuple):
        emit({'kind': 'loss_components', 'batch': loss_count, 'components': result[1]})
    if loss_count == 4:
        train.loss_fn = original_loss
    return result


if __name__ == '__main__':
    cfg = common.config(ROOT / 'config_protocol_b.yaml')
    fingerprint = common.source_fingerprint()
    common.require_engineering(cfg, fingerprint)
    train._validate_outputs(*train._output_paths(cfg), None)
    EVIDENCE.mkdir(exist_ok=False)
    common.atomic_json(EVIDENCE / 'launch.json', {
        'pid': os.getpid(), 'start_unix': time.time(),
        'config': str(ROOT / 'config_protocol_b.yaml'), 'resume': False,
        'auto_test': False, 'instrumentation_only': True,
        'launcher_sha256': common.sha256(__file__),
        'source_fingerprint': fingerprint,
    })
    train.build = audited_build
    train._finite_step = audited_step
    train.loss_fn = audited_loss
    code = 1
    try:
        code = train.main(['--config', str(ROOT / 'config_protocol_b.yaml'),
                           '--device', 'cuda:0'])
    finally:
        common.atomic_json(EVIDENCE / 'exit.json',
                           {'exit_code': code, 'finished_unix': time.time()})
    sys.exit(code)
