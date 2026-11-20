"""Start one frozen FHAPD run and remove instrumentation after its startup audit."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import common
import train

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'formal_launch_01'
original_build = train.build
original_step = train._finite_step
original_loss = train.loss_fn
hooks = []
loss_count = step_count = 0
forward_counts = {'field': 0, 'refiner': 0, 'verifier': 0}
audit_complete = False


def emit(record):
    record = dict(record, unix_time=time.time())
    text = json.dumps(record, allow_nan=False)
    with (EVIDENCE / 'activity.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(text + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print('STARTUP_AUDIT', text, flush=True)


def cleanup():
    for hook in hooks:
        hook.remove()
    hooks.clear()
    train.build = original_build
    train._finite_step = original_step
    train.loss_fn = original_loss


def maybe_finish():
    global audit_complete
    required = ('field', 'refiner', 'verifier') if common.MODEL_KIND == 'combined' else ('field',)
    if not audit_complete and loss_count >= 5 and step_count >= 2 and all(forward_counts[key] >= 5 for key in required):
        audit_complete = True
        cleanup()
        emit({'kind': 'audit_complete', 'loss_batches': loss_count, 'optimizer_updates': step_count,
              'forward_counts': dict(forward_counts), 'instrumentation_removed': True,
              'training_continues': True, 'auto_test': False})


def audited_build(*args, **kwargs):
    model = original_build(*args, **kwargs)
    from verify_fhapd_training import parameter_counts
    expected = {'field': 30468959, 'refiner': 3941, 'verifier': 11228} if common.MODEL_KIND == 'combined' else {
        'field': 30453790, 'refiner': 0, 'verifier': 0}
    if parameter_counts(model) != expected:
        raise AssertionError('Startup model parameter counts differ from the frozen architecture')

    def observe(name):
        def callback(module, inputs, output):
            forward_counts[name] += 1
            tensors = output[0] if isinstance(output, tuple) else output
            if name == 'field':
                tensors = tensors if isinstance(tensors, list) else [tensors]
            else:
                tensors = [tensors]
            if any(not common.torch.isfinite(value).all().item() for value in tensors):
                raise FloatingPointError(f'Non-finite {name} startup output')
            if forward_counts[name] <= 5:
                emit({'kind': name + '_forward', 'count': forward_counts[name]})
            maybe_finish()
        return callback

    hooks.append(model.register_forward_hook(observe('field')))
    if common.MODEL_KIND == 'combined':
        if model.region_refiner is None or model.edit_verifier is None or model.edit_verifier.aggregation != 'multi_scale_support':
            raise AssertionError('Startup must construct the approved SRv2 and RDv9 aggregation')
        hooks.extend((model.region_refiner.register_forward_hook(observe('refiner')),
                      model.edit_verifier.register_forward_hook(observe('verifier'))))
    emit({'kind': 'model_built', 'parameter_counts': expected, 'model_kind': common.MODEL_KIND,
          'official_pretraining_only': True, 'resume': False, 'auto_test': False})
    return model


def audited_step(*args, **kwargs):
    global step_count
    result = original_step(*args, **kwargs)
    step_count += 1
    if step_count <= 2:
        emit({'kind': 'optimizer_update', 'count': step_count})
    maybe_finish()
    return result


def audited_loss(*args, **kwargs):
    global loss_count
    result = original_loss(*args, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise AssertionError('Startup training loss must expose its components')
    loss_count += 1
    if loss_count <= 5:
        emit({'kind': 'loss_components', 'count': loss_count, 'components': result[1]})
    maybe_finish()
    return result


def main():
    global audit_complete
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(common.PHYSICAL_GPU):
        raise ValueError('Physical GPU does not match this frozen run')
    if not common.torch.cuda.is_available() or common.torch.cuda.device_count() != 1:
        raise RuntimeError('Formal training requires one visible CUDA GPU')
    cfg = common.config(ROOT / 'config_protocol_b.yaml')
    with train.training_lock(cfg, purpose='training'):
        fingerprint = common.source_fingerprint()
        common.require_engineering(cfg, fingerprint)
        common.verify_data_contract(cfg)
        log, weights = train._output_paths(cfg)
        train._validate_outputs(log, weights, None)
        if EVIDENCE.exists() or EVIDENCE.is_symlink():
            raise FileExistsError('Startup evidence already exists')
        EVIDENCE.mkdir()
        common.atomic_json(EVIDENCE / 'launch.json', {
            'pid': os.getpid(), 'start_unix': time.time(), 'run_name': common.MAIN_RUN,
            'physical_gpu': common.PHYSICAL_GPU, 'seed': 73, 'model_kind': common.MODEL_KIND,
            'resume': False, 'auto_test': False, 'source_fingerprint': fingerprint,
            'config': str(ROOT / 'config_protocol_b.yaml'),
        })
        train.build, train._finite_step, train.loss_fn = audited_build, audited_step, audited_loss
        code = 1
        try:
            device = common.torch.device('cuda:0')
            common.torch.cuda.set_device(device)
            train._train_locked(cfg, device, resume=None)
            if not audit_complete:
                raise RuntimeError('Training completed without the required startup observations')
            code = 0
        except BaseException as error:
            code = 130 if isinstance(error, KeyboardInterrupt) else 1
            epoch = 0
            try:
                if (weights / 'last.pt').is_file():
                    epoch = int(common.torch.load(weights / 'last.pt', map_location='cpu', weights_only=True)['epoch'])
                train._status(cfg, log, weights, 'interrupted' if code == 130 else 'failed', epoch,
                              error=f'{type(error).__name__}: {error}')
            except Exception as bookkeeping_error:
                print(f'Could not read last.pt/write failure status: {bookkeeping_error}',
                      file=sys.stderr, flush=True)
            finally:
                emit({'kind': 'launch_exit', 'exit_code': code, 'audit_complete': audit_complete})
            raise
        finally:
            cleanup()
            common.atomic_json(EVIDENCE / 'exit.json', {
                'exit_code': code, 'audit_complete': audit_complete,
                'loss_batches': loss_count, 'optimizer_updates': step_count,
                'forward_counts': dict(forward_counts), 'finished_unix': time.time(), 'auto_test': False,
            })
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f'FHAPD launch failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
