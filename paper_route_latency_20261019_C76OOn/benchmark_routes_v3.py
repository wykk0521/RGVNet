"""Isolated seed73 route benchmark. Never edits model sources or checkpoints.

All routes retain the original region, edge and distance heads. Unused DRR/CEV
modules are removed after strict loading. The common forward prefix is copied
from the verified original AST; training/diagnostic tensor caches are omitted.
"""
from pathlib import Path
import argparse, ast, csv, gc, hashlib, importlib, importlib.util, inspect
import itertools, json, os, platform, random, statistics, subprocess, sys
import textwrap, time, traceback, types

ROOT = Path('/root/wyk5/RGVNet')
PACKAGE = ROOT / 'models/hbgnet_seed73_rd_v9_context'
CHECKPOINT = ROOT / 'z_3_weights/AI4Boundaries_1m/hbgnet_protocol_b_seed73_rd_v9_context/best.pt'
RECORD = ROOT / 'z_2_test/AI4Boundaries_1m/hbgnet_protocol_b_seed73_rd_v9_context_best/metrics.json'
EXPECTED_CHECKPOINT = '92d162c7085982422d7c9d11946addb236427b87a75876b90eb680d4d063b466'
EXPECTED_PARAMETERS = {'raw': 30453790, 'drr': 30457731, 'final': 30468959}
ROUTES = ('raw', 'drr', 'final')
sys.path[:0] = [str(ROOT / '.python_packages'), str(ROOT), str(ROOT / 'models/hbgnet/_deps')]
import numpy as np
import torch
import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def emit(event, **values):
    print(json.dumps({'event': event, **values}), flush=True)


def cpu_stats():
    p = Path('/sys/fs/cgroup/cpu.stat')
    return {a[0]: int(a[1]) for line in p.read_text().splitlines() if len(a := line.split()) == 2}


def gpu_snapshot():
    commands = {
        'gpu': ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,driver_version', '--format=csv'],
        'processes': ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'],
    }
    return {k: subprocess.run(v, text=True, capture_output=True, timeout=20).stdout.strip()
            for k, v in commands.items()}


def setup():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available(), 'CUDA unavailable; GPU benchmark cannot run'
    assert torch.cuda.device_count() == 1, 'Expose exactly one GPU for this benchmark'
    record = json.loads(RECORD.read_text())
    assert record['checkpoint_sha256'] == EXPECTED_CHECKPOINT
    assert sha(CHECKPOINT) == EXPECTED_CHECKPOINT
    assert record['checkpoint_epoch'] == 34
    assert record['config']['training']['seed'] == 73
    assert record['complete_split'] and record['samples'] == 3014
    verified = {}
    source_files = {
        'implementation/model.py': PACKAGE / 'implementation/model.py',
        'implementation/lib/pvtv2.py': PACKAGE / 'implementation/lib/pvtv2.py',
        'protocol/dataset.py': PACKAGE / 'protocol/dataset.py',
        '@project/tools/local_separation_refinement_v2.py': ROOT / 'tools/local_separation_refinement_v2.py',
        '@project/tools/region_discrimination_v9.py': ROOT / 'tools/region_discrimination_v9.py',
        '@project/tools/region_discrimination_v8_grok_fullloss.py': ROOT / 'tools/region_discrimination_v8_grok_fullloss.py',
    }
    for name, p in source_files.items():
        digest = sha(p)
        assert record['source_fingerprint'][name] == digest, ('source changed', name)
        verified[name] = digest
    namespace = '_route_benchmark_impl'
    stub = types.ModuleType(namespace)
    stub.__path__ = [str(PACKAGE / 'implementation')]
    sys.modules[namespace] = stub
    impl = importlib.import_module(namespace + '.model')
    ds_spec = importlib.util.spec_from_file_location('_route_benchmark_dataset', PACKAGE / 'protocol/dataset.py')
    ds_module = importlib.util.module_from_spec(ds_spec)
    ds_spec.loader.exec_module(ds_module)
    cfg = ds_module.load_config(PACKAGE / 'config_protocol_b.yaml')
    assert cfg['training']['seed'] == 73
    assert cfg['model'] == record['config']['model']
    assert sha(cfg['dataset']['manifests']['test']) == record['split_manifest_sha256']['test']
    dataset = ds_module.AI4BoundariesSEANetDataset(cfg, 'test', augment=False)
    indices = sorted(random.Random(73).sample(range(len(dataset)), 32))
    inputs, ids = [], []
    for index in indices:
        item = dataset[index]
        x = item['image'].unsqueeze(0).contiguous()
        assert list(x.shape) == [1, 3, 256, 256] and x.dtype == torch.float32
        assert torch.isfinite(x).all()
        inputs.append(x)
        ids.append(item['image_id'])
    return impl, cfg, inputs, {
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': EXPECTED_CHECKPOINT,
        'checkpoint_epoch': 34, 'verified_sources': verified,
        'test_manifest_sha256': record['split_manifest_sha256']['test'],
        'input_ids': ids, 'input_indices': indices,
        'input_tensor_sha256': [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in inputs],
    }


def make_forward(impl, output_dir):
    original = textwrap.dedent(inspect.getsource(impl.Field.forward))
    fn = ast.parse(original).body[0]
    prefix = []
    for statement in fn.body:
        prefix.append(statement)
        if (isinstance(statement, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'raw_region' for t in statement.targets)):
            break
    else:
        raise AssertionError('Original forward prefix marker not found')
    tail = ast.parse(textwrap.dedent('''
        if self.benchmark_route == 'raw':
            return [raw_region, edge_out, dis_out]
        delta, auxiliary = self.region_refiner(out, raw_region, edge_out, return_aux=True)
        sr_region = raw_region + delta
        del auxiliary
        if self.benchmark_route == 'drr':
            return [sr_region, edge_out, dis_out]
        verifier_logits = self.edit_verifier(out, raw_region, sr_region, delta, edge_out)
        keep_cut, keep_fill = hard_keep(raw_region, sr_region, delta, verifier_logits)
        final_region = gated_region(raw_region, delta, keep_cut, keep_fill)
        return [final_region, edge_out, dis_out]
    ''')).body
    fn.body = prefix + tail
    fn.name = 'route_forward'
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    generated = ast.unparse(module) + '\n'
    globals_copy = dict(impl.__dict__)
    exec(compile(module, '<audited_route_forward>', 'exec'), globals_copy)
    path = output_dir / 'route_forward_generated.py'
    if path.exists():
        assert path.read_text() == generated
    else:
        path.write_text(generated)
    return globals_copy['route_forward'], hashlib.sha256(generated.encode()).hexdigest()


def build(impl, cfg, route=None, forward=None):
    model = impl.Field(channel=cfg['model']['channel'], num_classes=cfg['model']['boundary_classes'],
                       drop_rate=cfg['model']['drop_rate'], pretrained_path=None,
                       refinement=cfg['model']['refinement'])
    state = torch.load(CHECKPOINT, map_location='cpu')['model']
    model.load_state_dict(state, strict=True)
    del state
    if route:
        if route == 'raw':
            model.region_refiner = None
            model.edit_verifier = None
        elif route == 'drr':
            model.edit_verifier = None
        model.benchmark_route = route
        model.forward = types.MethodType(forward, model)
        assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS[route]
    model.eval()
    return model.to('cuda', dtype=torch.float32)


def stats(values):
    values = [float(v) for v in values]
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': float(np.percentile(values, 95)), 'std': statistics.stdev(values),
            'min': min(values), 'max': max(values)}


def validate(args):
    impl, cfg, inputs, provenance = setup()
    forward, generated_sha = make_forward(impl, args.output)
    validation_inputs = inputs[:4] + [torch.zeros_like(inputs[0]), torch.randn_like(inputs[0])]
    model = build(impl, cfg)
    expected = {r: [] for r in ROUTES}
    with torch.inference_mode():
        for x in validation_inputs:
            result = model(x.cuda())
            aux = model.last_auxiliary
            for route, key in [('raw', 'raw_region_logits'), ('drr', 'sr_region_logits'), ('final', 'final_region_logits')]:
                expected[route].append([aux[key].cpu(), result[1].cpu(), result[2].cpu()])
            del result, aux
    model.last_auxiliary = {}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    records = []
    for route in ROUTES:
        model = build(impl, cfg, route, forward)
        calls = {'region_refiner': 0, 'edit_verifier': 0}
        handles = []
        for name in calls:
            module = getattr(model, name)
            if module is not None:
                def observe(_module, _inputs, _output, key=name):
                    calls[key] += 1
                handles.append(module.register_forward_hook(observe))
        comparison = []
        with torch.inference_mode():
            for i, x in enumerate(validation_inputs):
                outputs = model(x.cuda())
                diffs = []
                for output, reference in zip(outputs, expected[route][i]):
                    value = output.cpu()
                    diffs.append(float((value - reference).abs().max()))
                    assert torch.equal(value, reference), (route, i, diffs)
                comparison.append({'input_index': i, 'max_abs_differences': diffs,
                                   'binary_mask_differences': int(((outputs[0].cpu() > 0) != (expected[route][i][0] > 0)).sum())})
                del outputs
        expected_calls = {'region_refiner': 0 if route == 'raw' else len(validation_inputs),
                          'edit_verifier': len(validation_inputs) if route == 'final' else 0}
        assert calls == expected_calls
        assert model.last_auxiliary == {}
        for h in handles:
            h.remove()
        records.append({'route': route, 'module_calls': calls, 'checks': comparison,
                        'parameters': sum(p.numel() for p in model.parameters())})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result = {'status': 'PASS', 'provenance': provenance, 'generated_forward_sha256': generated_sha,
              'input_count': len(validation_inputs), 'exact_bitwise_equality': True,
              'routes': records, 'gpu': gpu_snapshot()}
    save(args.output / 'validation.json', result)
    emit('VALIDATION_PASS', inputs=len(validation_inputs), routes=list(ROUTES))


def bench(args):
    before = gpu_snapshot()
    impl, cfg, inputs, provenance = setup()
    forward, generated_sha = make_forward(impl, args.output)
    model = build(impl, cfg, args.route, forward)
    inputs = [x.pin_memory() for x in inputs]
    device_input = torch.empty_like(inputs[0], device='cuda')
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for i in range(args.warmup):
            device_input.copy_(inputs[i % len(inputs)], non_blocking=True)
            output = model(device_input)
            del output
        torch.cuda.synchronize()
        # CUDA Event handles are created lazily. Warm the exact measurement
        # instrumentation too, so event creation cannot enter the first sample.
        for _ in range(5):
            event_start.record()
            output = model(device_input)
            event_end.record()
            event_end.synchronize()
            event_start.elapsed_time(event_end)
            del output
        gc.collect()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        measurements = []
        cpu_before = cpu_stats()
        gc.disable()
        try:
            for i in range(args.iterations):
                input_index = i % len(inputs)
                device_input.copy_(inputs[input_index], non_blocking=True)
                torch.cuda.synchronize()
                wall_start = time.perf_counter_ns()
                event_start.record()
                output = model(device_input)
                event_end.record()
                event_end.synchronize()
                wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
                event_ms = event_start.elapsed_time(event_end)
                measurements.append({'iteration': i, 'input_index': input_index,
                                     'wall_sync_ms': wall_ms, 'cuda_event_ms': event_ms})
                del output
        finally:
            gc.enable()
        torch.cuda.synchronize()
    cpu_after = cpu_stats()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    assert model.last_auxiliary == {}
    result = {
        'status': 'PASS', 'round': args.round, 'route': args.route,
        'parameters': sum(p.numel() for p in model.parameters()),
        'warmup': args.warmup, 'instrumented_warmup': 5, 'iterations': args.iterations,
        'cpu_cgroup_delta': {k: cpu_after[k] - cpu_before[k] for k in cpu_before},
        'wall_sync_ms': stats([m['wall_sync_ms'] for m in measurements]),
        'cuda_event_ms': stats([m['cuda_event_ms'] for m in measurements]),
        'memory_bytes': {'baseline_allocated': baseline_allocated, 'baseline_reserved': baseline_reserved,
                         'peak_allocated': peak_allocated, 'peak_reserved': peak_reserved,
                         'peak_increment_allocated': peak_allocated - baseline_allocated},
        'conditions': {'input_shape': [1, 3, 256, 256], 'dtype': 'float32', 'amp': False,
                       'tf32_matmul': False, 'tf32_cudnn': False,
                       'cudnn_benchmark': False, 'cudnn_deterministic': True,
                       'torch_threads': torch.get_num_threads(), 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'inference_mode': True,
                       'timing_scope': 'forward with CUDA synchronization; includes Python launch and event overhead; excludes input copy, disk I/O, preprocessing, thresholding and metrics',
                       'memory_scope': 'PyTorch live-tensor peak allocated, with model, one input, outputs and temporaries; cached reserved bytes reported separately; excludes CUDA context/other non-PyTorch allocations',
                       'auxiliary_heads_retained': True, 'diagnostic_caches': False,
                       'isolated_process_per_route_round': True, 'cuda_event_handles_warmed': True},
        'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
                        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                        'gpu_properties': str(torch.cuda.get_device_properties(0)),
                        'platform': platform.platform(), 'cpu_threads': os.cpu_count()},
        'provenance': provenance, 'generated_forward_sha256': generated_sha,
        'gpu_before': before, 'gpu_after': gpu_snapshot(), 'measurements': measurements,
    }
    save(args.output / f'round{args.round}_{args.route}.json', result)
    emit('BENCH_DONE', round=args.round, route=args.route,
         median_ms=result['wall_sync_ms']['median'], p95_ms=result['wall_sync_ms']['p95'],
         peak_allocated_mib=peak_allocated / 2**20)


def run_child(args, extra):
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--output', str(args.output)] + extra
    subprocess.run(command, check=True)


def run_all(args):
    snapshot = gpu_snapshot()
    assert snapshot['gpu'].startswith('index,'), snapshot
    process_lines = snapshot['processes'].splitlines()
    assert len(process_lines) == 1, ('GPU already occupied', process_lines)
    save(args.output / 'initial_gpu.json', snapshot)
    run_child(args, ['--mode', 'validate'])
    orders = [('raw', 'drr', 'final'), ('drr', 'final', 'raw'), ('final', 'raw', 'drr')]
    all_runs = []
    for round_index, order in enumerate(orders, 1):
        for route in order:
            current = gpu_snapshot()
            assert len(current['processes'].splitlines()) == 1, ('Concurrent GPU process detected', current)
            emit('BENCH_START', round=round_index, route=route)
            run_child(args, ['--mode', 'bench', '--route', route, '--round', str(round_index),
                             '--warmup', str(args.warmup), '--iterations', str(args.iterations)])
            record = json.loads((args.output / f'round{round_index}_{route}.json').read_text())
            all_runs.append(record)
    summary = {'status': 'PASS', 'orders': orders, 'conditions': all_runs[0]['conditions'],
               'environment': all_runs[0]['environment'], 'routes': {}, 'final_gpu': gpu_snapshot()}
    for route in ROUTES:
        records = [r for r in all_runs if r['route'] == route]
        summary['routes'][route] = {
            'parameters': records[0]['parameters'],
            'wall_sync_ms': stats([m['wall_sync_ms'] for r in records for m in r['measurements']]),
            'cuda_event_ms': stats([m['cuda_event_ms'] for r in records for m in r['measurements']]),
            'round_medians_ms': [r['wall_sync_ms']['median'] for r in records],
            'round_means_ms': [r['wall_sync_ms']['mean'] for r in records],
            'peak_allocated_mib': [r['memory_bytes']['peak_allocated'] / 2**20 for r in records],
            'peak_reserved_mib': [r['memory_bytes']['peak_reserved'] / 2**20 for r in records],
            'baseline_allocated_mib': [r['memory_bytes']['baseline_allocated'] / 2**20 for r in records],
        }
    save(args.output / 'summary.json', summary)
    with (args.output / 'timing_samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['round', 'route', 'iteration', 'input_index', 'wall_sync_ms', 'cuda_event_ms'])
        writer.writeheader()
        for record in all_runs:
            for m in record['measurements']:
                writer.writerow({'round': record['round'], 'route': record['route'], **m})
    emit('ALL_DONE', summary=summary['routes'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'validate', 'bench'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route', choices=ROUTES)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == 'all':
            run_all(args)
        elif args.mode == 'validate':
            validate(args)
        else:
            bench(args)
    except Exception:
        failure = {'status': 'FAILED', 'mode': args.mode, 'route': args.route,
                   'round': args.round, 'traceback': traceback.format_exc()}
        save(args.output / f'failure_{args.mode}_{args.route}_{args.round}.json', failure)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
