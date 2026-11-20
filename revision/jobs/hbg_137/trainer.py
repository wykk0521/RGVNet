"""Isolated HBGNet Protocol B training. Validation only; never auto-test."""
import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

from runtime_adapter import (ROOT, RUN, atomic_json, build, config, device_batch, evaluate,
                    loader, loss_fn, require_engineering, seed_all, sha256,
                    source_fingerprint, torch)
from training_state import (capture_rng, commit_epoch, recover_epoch, restore_rng,
                            selection_update)


@contextlib.contextmanager
def training_lock():
    """Keep one stable inode in ROOT; release the advisory lock, never unlink it."""
    import fcntl

    path = ROOT / 'train.lock'
    with path.open('a+', encoding='utf-8') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another HBGNet trainer holds {path}') from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({'pid': os.getpid(), 'run_name': RUN}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _output_paths(cfg):
    return (Path(cfg['paths']['train_log_root']) / RUN,
            Path(cfg['paths']['weights_root']) / RUN)


def _validate_outputs(log, weights, resume):
    if resume is None:
        if log.exists() or weights.exists() or log.is_symlink() or weights.is_symlink():
            raise FileExistsError('Run output already exists; new runs never overwrite it')
        return
    if log.is_symlink() or weights.is_symlink():
        raise ValueError('Run output directories must not be symlinks')
    if not log.is_dir() or not weights.is_dir():
        raise FileNotFoundError('Resume requires both existing log and weight directories')
    last = weights / 'last.pt'
    requested = Path(resume).absolute()
    if requested != last.absolute() or requested.is_symlink() or not last.is_file():
        raise ValueError(f'Resume is restricted to this run last.pt: {last}')
    if not (log / 'config_snapshot.json').is_file():
        raise FileNotFoundError('Resume requires the existing config snapshot')


def _split_hashes(cfg):
    dataset = cfg['dataset']
    manifests = dataset.get('manifests')
    if manifests is None:
        manifests = dataset['variants'][dataset['variant']]['manifests']
    if set(manifests) != {'train', 'val', 'test'}:
        raise ValueError('Expected exactly the frozen train/val/test manifests')
    # Hash only the manifest text; no test images or labels are opened.
    return {split: sha256(path) for split, path in manifests.items()}


def _pretrained_provenance(cfg):
    path = Path(cfg['model']['pretrained_path'])
    receipt_path = path.with_suffix('.json')
    if not path.is_file():
        raise FileNotFoundError(f'Official PVT-v2-B2 pretraining required; no random fallback: {path}')
    if not receipt_path.is_file():
        raise FileNotFoundError(f'Official pretraining provenance receipt required: {receipt_path}')
    with receipt_path.open(encoding='utf-8') as stream:
        receipt = json.load(stream)
    if not isinstance(receipt, dict) or receipt.get('sha256') != sha256(path):
        raise ValueError('Missing/mismatched official PVT-v2-B2 provenance')
    json.dumps(receipt, allow_nan=False)
    return {'path': str(path), 'receipt_path': str(receipt_path),
            'sha256': receipt['sha256'], 'receipt': receipt}


def _check_recipe(cfg):
    expected = {
        'epochs': 40, 'seed': 137, 'gradient_accumulation_steps': 2, 'amp': False,
        'optimizer': 'Adam', 'learning_rate': 1e-4, 'weight_decay': 0.0,
        'scheduler': 'CosineAnnealingLR', 'scheduler_t_max': 10_000_000_000,
        'min_lr': 1e-5, 'iou_floor': 0.8205741376917083,
        'validation_interval': 2, 'final_validation_epochs': 5,
    }
    for name, value in expected.items():
        if cfg['training'].get(name) != value:
            raise ValueError(f'Frozen HBGNet training recipe differs: {name}')
    if cfg['dataset']['batch_size'] != 8 or cfg['dataset'].get('persistent_workers', False):
        raise ValueError('Training requires batch size 8 and nonpersistent workers')


def _check_optimizer_resume(cp):
    """Reject silent optimizer/scheduler recipe drift in a nominally matching cfg."""
    scheduler = cp['scheduler']
    if (scheduler.get('T_max') != 10_000_000_000 or scheduler.get('eta_min') != 1e-5 or
            scheduler.get('last_epoch') != cp['epoch'] or
            scheduler.get('base_lrs') != [1e-4]):
        raise ValueError('Resume scheduler does not match the official recipe/epoch')
    groups = cp['optimizer']['param_groups']
    if len(groups) != 1:
        raise ValueError('Expected a single official Adam parameter group')
    group = groups[0]
    if (group.get('weight_decay') != 0 or tuple(group.get('betas', ())) != (.9, .999) or
            group.get('eps') != 1e-8 or group.get('amsgrad', False) or
            group.get('maximize', False) or group.get('initial_lr') != 1e-4 or
            not math.isfinite(group.get('lr', float('nan'))) or
            scheduler.get('_last_lr') != [group['lr']]):
        raise ValueError('Resume Adam state differs from the official recipe')


def _finite_step(model, optimizer, epoch, batch_index):
    """Check every accumulated gradient, including the short final group."""
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None:
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            if not torch.isfinite(values).all().item():
                raise FloatingPointError(
                    f'Non-finite gradient before Adam: epoch={epoch} batch={batch_index} {name}')
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f'Non-finite parameter after Adam: {name}')
    optimizer.zero_grad(set_to_none=True)


def _status(log, weights, status, epoch, **extra):
    # If mkdir succeeded only for weights, retain failure evidence there as well.
    destination = log if log.is_dir() else weights if weights.is_dir() else None
    if destination is not None:
        atomic_json(destination / 'training_status.json', {
            'status': status, 'run_name': RUN, 'epoch': epoch,
            'pid': os.getpid(), 'updated_unix': time.time(), **extra,
        })


def train(cfg, device, resume=None):
    """Programmatic entry point obeys the same duplicate-training lock as CLI."""
    with training_lock():
        return _train_locked(cfg, device, resume)


def _train_locked(cfg, device, resume=None):
    """Implementation for callers already holding training_lock()."""
    log, weights = _output_paths(cfg)
    _validate_outputs(log, weights, resume)
    _check_recipe(cfg)
    fingerprint = source_fingerprint()
    require_engineering(cfg, fingerprint)
    split_hashes = _split_hashes(cfg)

    def check_source():
        if source_fingerprint() != fingerprint:
            raise RuntimeError('Immutable HBGNet source changed during this run')
        if _split_hashes(cfg) != split_hashes:
            raise RuntimeError('Frozen split manifests changed during this run')

    cp = None
    start = 1
    best = constrained = None
    if resume is not None:
        cp = torch.load(weights / 'last.pt', map_location='cpu', weights_only=True)
        if cp.get('config') != cfg or cp.get('run_name') != RUN:
            raise ValueError('Resume config/run mismatch')
        if cp.get('source_fingerprint') != fingerprint:
            raise ValueError('Resume source fingerprint mismatch')
        if cp.get('split_manifest_sha256') != split_hashes:
            raise ValueError('Resume split manifest mismatch')
        provenance = cp.get('pretrained_provenance')
        if (not isinstance(provenance, dict) or not provenance.get('sha256') or
                provenance.get('path') != cfg['model']['pretrained_path'] or
                provenance.get('receipt', {}).get('sha256') != provenance['sha256']):
            raise ValueError('Resume checkpoint lacks valid pretraining provenance')
        with (log / 'config_snapshot.json').open(encoding='utf-8') as stream:
            if json.load(stream) != cfg:
                raise ValueError('Existing config snapshot differs from resume config')
        if type(cp.get('epoch')) is not int or not 1 <= cp['epoch'] <= 40:
            raise ValueError('Resume epoch must be in [1, 40]')
        _check_optimizer_resume(cp)
        recover_epoch(log, weights, cp, check_source=check_source)
        start = cp['epoch'] + 1
        best, constrained = cp['best'], cp['constrained']
    else:
        provenance = _pretrained_provenance(cfg)

    if start > cfg['training']['epochs']:
        check_source()
        _status(log, weights, 'completed', 40, source_fingerprint=fingerprint,
                split_manifest_sha256=split_hashes)
        return 40

    seed_all(cfg['training']['seed'])
    # No pretrained file download/read is needed when resuming this run's last.pt.
    model = build(cfg, pretrained=resume is None).to(device)
    if resume is None and _pretrained_provenance(cfg) != provenance:
        raise RuntimeError('Pretraining receipt/weights changed during model construction')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10_000_000_000, eta_min=1e-5)
    train_loader = loader(cfg, 'train')
    val_loader = loader(cfg, 'val')
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError('Training and validation loaders must be nonempty')
    if train_loader.generator is None or val_loader.generator is None:
        raise ValueError('Both loaders require explicit, checkpointable generators')
    generators = {'train': train_loader.generator, 'val': val_loader.generator}
    if cp is not None:
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler'])
    check_source()
    if resume is None:
        log.mkdir(parents=True, exist_ok=False)
        weights.mkdir(parents=True, exist_ok=False)
        (log / 'epoch_records').mkdir()
        atomic_json(log / 'config_snapshot.json', cfg)
        (log / 'config_snapshot.yaml').write_bytes((ROOT / 'config.yaml').read_bytes())
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(str(log / 'tensorboard'), purge_step=start if resume else None)
    try:
        if cp is not None:
            restore_rng(cp['rng'], generators)
            del cp
        _status(log, weights, 'running', start - 1, source_fingerprint=fingerprint,
                split_manifest_sha256=split_hashes)
        accumulation = cfg['training']['gradient_accumulation_steps']
        epochs = cfg['training']['epochs']
        for epoch in range(start, epochs + 1):
            model.train()
            total_loss = 0.0
            batches = len(train_loader)
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for index, batch in enumerate(train_loader, 1):
                batch = device_batch(batch, device)
                # AMP is deliberately disabled; retain official full precision.
                loss = loss_fn(model(batch['image']), batch)
                if not torch.isfinite(loss).all().item():
                    raise FloatingPointError(f'Non-finite training loss at {epoch}/{index}')
                group_start = ((index - 1) // accumulation) * accumulation
                divisor = min(accumulation, batches - group_start)
                (loss / divisor).backward()
                if index % accumulation == 0 or index == batches:
                    _finite_step(model, optimizer, epoch, index)
                loss_value = float(loss.detach().item())
                total_loss += loss_value
                if index % cfg['training']['log_interval'] == 0:
                    print(f'epoch={epoch}/{epochs} batch={index}/{batches} '
                          f'loss={loss_value:.6f}', flush=True)
            validate = (epoch == 1 or epoch % cfg['training']['validation_interval'] == 0 or
                        epoch > epochs - cfg['training']['final_validation_epochs'])
            val = evaluate(model, val_loader, device) if validate else None
            best, constrained, legacy_new, constrained_new = selection_update(
                val, epoch, best, constrained, cfg['training']['iou_floor'])
            scheduler.step()
            record = {
                'epoch': epoch, 'train_loss': total_loss / batches,
                'train_batches': batches, 'seconds': time.monotonic() - started,
                'validation': val, 'best': best, 'constrained': constrained,
                'learning_rate': float(optimizer.param_groups[0]['lr']),
            }
            cp = {
                'epoch': epoch, 'run_name': RUN, 'config': cfg,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'best': best,
                'constrained': constrained, 'validation': val, 'record': record,
                'legacy_new': legacy_new, 'constrained_new': constrained_new,
                'rng': capture_rng(generators), 'source_fingerprint': fingerprint,
                'pretrained_provenance': provenance, 'split_manifest_sha256': split_hashes,
                'torch_version': str(torch.__version__),
            }
            commit_epoch(log, weights, cp, check_source=check_source)
            del cp
            _status(log, weights, 'running', epoch, source_fingerprint=fingerprint,
                    split_manifest_sha256=split_hashes)
            writer.add_scalar('train/loss', record['train_loss'], epoch)
            if val is not None:
                for name, value in val.items():
                    writer.add_scalar('validation/' + name, value, epoch)
            writer.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
        check_source()
        _status(log, weights, 'completed', epochs, source_fingerprint=fingerprint,
                split_manifest_sha256=split_hashes)
        return epochs
    finally:
        writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--resume')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    # Lock acquisition failures and rejected new runs must not overwrite the
    # active/previous trainer's status. All accepted-run failures are recorded.
    try:
        with training_lock():
            cfg = config(args.config)
            log, weights = _output_paths(cfg)
            _validate_outputs(log, weights, args.resume)
            try:
                device = torch.device(args.device)
                if device.type != 'cuda' or not torch.cuda.is_available():
                    raise RuntimeError('Formal HBGNet training requires CUDA')
                torch.cuda.set_device(device)
                _train_locked(cfg, device, args.resume)
            except BaseException as exc:
                epoch = 0
                # Read only last.pt for authoritative failure progress, safely.
                try:
                    if (weights / 'last.pt').is_file():
                        last = torch.load(weights / 'last.pt', map_location='cpu', weights_only=True)
                        epoch = int(last.get('epoch', 0))
                    _status(log, weights, 'interrupted' if isinstance(exc, KeyboardInterrupt)
                            else 'failed', epoch, error=f'{type(exc).__name__}: {exc}')
                except Exception as status_error:
                    print(f'Could not write failure status: {status_error}', file=sys.stderr, flush=True)
                raise
    except KeyboardInterrupt:
        print('Training interrupted; resume only from this run last.pt.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'HBGNet training failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
