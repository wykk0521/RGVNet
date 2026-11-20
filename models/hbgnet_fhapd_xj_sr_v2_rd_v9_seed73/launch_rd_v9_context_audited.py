"""Start the single approved rd_v9_context fullloss run with an early audit."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import common
import train

EVIDENCE = ROOT / "formal_launch_v9_context_01"
EXPECTED_VERIFIER_PARAMETERS = 11228
EXPECTED_CONTEXT_PARAMETERS = 5154
EXPECTED_REFINER_PARAMETERS = 3941
EXPECTED_FIELD_PARAMETERS = 30468959

original_build = train.build
original_step = train._finite_step
original_loss = train.loss_fn
hooks = []
forward_counts = {"verifier": 0, "refiner": 0}
loss_count = 0
step_count = 0
audit_complete = False


def emit(record):
    record = dict(record)
    record["unix_time"] = time.time()
    text = json.dumps(record, allow_nan=False)
    with (EVIDENCE / "activity.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()
    print("STARTUP_AUDIT", text, flush=True)


def cleanup_audit_hooks():
    """Remove instrumentation only; never marks the audit as successful.

    Cleanup runs from the ``finally`` block on every exit path, so it must be
    observation-free: a crash before the required observations must leave the
    audit records and receipts unmarked.
    """
    for handle in hooks:
        handle.remove()
    hooks.clear()
    train.build = original_build
    train._finite_step = original_step
    train.loss_fn = original_loss


def mark_audit_success():
    """Mark the startup audit successful, only from a satisfied maybe_finish."""
    global audit_complete
    if audit_complete:
        return
    audit_complete = True
    cleanup_audit_hooks()
    emit({"kind": "audit_complete", "loss_batches": loss_count,
          "optimizer_updates": step_count, "forward_counts": dict(forward_counts),
          "instrumentation_removed": True, "training_continues": True})


def maybe_finish_audit():
    if not audit_complete and loss_count >= 5 and step_count >= 2 and all(
            value >= 5 for value in forward_counts.values()):
        mark_audit_success()


def audited_build(*args, **kwargs):
    model = original_build(*args, **kwargs)
    if model.region_refiner is None or model.edit_verifier is None:
        raise AssertionError("combined run must construct SR refiner and verifier")
    counts = {
        "field": sum(parameter.numel() for parameter in model.parameters()),
        "refiner": sum(parameter.numel() for parameter in model.region_refiner.parameters()),
        "verifier": sum(parameter.numel() for parameter in model.edit_verifier.parameters()),
        "context": sum(parameter.numel() for parameter in model.edit_verifier.context_projection.parameters()) +
            sum(parameter.numel() for parameter in model.edit_verifier.cut_context.parameters()) +
            sum(parameter.numel() for parameter in model.edit_verifier.fill_context.parameters()),
    }
    if counts != {"field": EXPECTED_FIELD_PARAMETERS,
                  "refiner": EXPECTED_REFINER_PARAMETERS,
                  "verifier": EXPECTED_VERIFIER_PARAMETERS,
                  "context": EXPECTED_CONTEXT_PARAMETERS}:
        raise AssertionError(f"Unexpected parameter counts: {counts}")
    if model.edit_verifier.aggregation != "multi_scale_support":
        raise AssertionError("The formal run must use the multi-scale support aggregation")

    def observe_refiner(_module, _inputs, output):
        forward_counts["refiner"] += 1
        delta = output[0] if isinstance(output, tuple) else output
        if not common.torch.isfinite(delta).all().item():
            raise FloatingPointError("Non-finite SR delta during startup audit")
        if forward_counts["refiner"] <= 5:
            emit({"kind": "refiner_forward", "count": forward_counts["refiner"],
                  "delta_abs_mean": float(delta.detach().abs().mean()),
                  "delta_abs_max": float(delta.detach().abs().max())})
        maybe_finish_audit()

    def observe_verifier(_module, _inputs, output):
        forward_counts["verifier"] += 1
        if output.shape[1] != 2 or not common.torch.isfinite(output).all().item():
            raise AssertionError("Verifier output must be finite [B,2,H,W]")
        if forward_counts["verifier"] <= 5:
            emit({"kind": "verifier_forward", "count": forward_counts["verifier"],
                  "shape": list(output.shape),
                  "logit_abs_mean": float(output.detach().abs().mean())})
        maybe_finish_audit()

    hooks.extend((model.region_refiner.register_forward_hook(observe_refiner),
                  model.edit_verifier.register_forward_hook(observe_verifier)))
    emit({"kind": "model_built", "parameter_counts": counts,
          "verifier_loss_mode": common.VERIFIER_LOSS_MODE,
          "verifier_aggregation": model.edit_verifier.aggregation,
          "official_pretraining_only": True, "auto_test": False})
    return model


def audited_step(*args, **kwargs):
    global step_count
    model = args[0]
    verifier_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.edit_verifier.parameters()
        if parameter.grad is not None
    )
    refiner_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.region_refiner.parameters()
        if parameter.grad is not None
    )
    result = original_step(*args, **kwargs)
    step_count += 1
    if step_count <= 2:
        emit({"kind": "optimizer_update", "count": step_count,
              "verifier_gradient_l1": verifier_grad,
              "refiner_gradient_l1": refiner_grad})
    maybe_finish_audit()
    return result


def audited_loss(*args, **kwargs):
    global loss_count
    result = original_loss(*args, **kwargs)
    loss_count += 1
    if loss_count <= 5:
        if not isinstance(result, tuple) or len(result) != 2:
            raise AssertionError("Training loss must expose components for startup audit")
        emit({"kind": "loss_components", "count": loss_count,
              "components": result[1]})
    maybe_finish_audit()
    return result


def main():
    if not __debug__:
        raise RuntimeError("Startup audit requires assertions")
    cfg = common.config(ROOT / "config_protocol_b.yaml")
    if cfg["experiment"]["run_name"] != common.MAIN_RUN:
        raise ValueError("Only the approved combined run may start")
    with train.training_lock(cfg, purpose="training"):
        fingerprint = common.source_fingerprint()
        common.require_engineering(cfg, fingerprint)
        log, weights = train._output_paths(cfg)
        train._validate_outputs(log, weights, None)
        if EVIDENCE.exists():
            raise FileExistsError(f"Startup evidence already exists: {EVIDENCE}")
        EVIDENCE.mkdir(exist_ok=False)
        common.atomic_json(EVIDENCE / "launch.json", {
            "pid": os.getpid(), "start_unix": time.time(),
            "run_name": common.MAIN_RUN, "config": str(ROOT / "config_protocol_b.yaml"),
            "resume": False, "auto_test": False,
            "source_fingerprint": fingerprint,
        })
        train.build = audited_build
        train._finite_step = audited_step
        train.loss_fn = audited_loss
        try:
            device = common.torch.device("cuda:0")
            if device.type != "cuda" or not common.torch.cuda.is_available():
                raise RuntimeError("Formal training requires CUDA")
            common.torch.cuda.set_device(device)
            train._train_locked(cfg, device, resume=None)
            if not audit_complete:
                raise RuntimeError("Training completed before startup audit reached its required observations")
            code = 0
        except BaseException as exc:
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            try:
                epoch = 0
                if (weights / "last.pt").is_file():
                    checkpoint = common.torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
                    epoch = int(checkpoint.get("epoch", 0))
                train._status(cfg, log, weights, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              epoch, error=f"{type(exc).__name__}: {exc}")
            finally:
                emit({"kind": "launch_exit", "exit_code": code, "audit_complete": audit_complete})
            raise
        finally:
            # Cleanup never marks success: a crash before 5 losses / 2 updates /
            # 5+5 forwards leaves audit_complete False in the exit receipt, and
            # acceptance must read the observations below, not a boolean alone.
            cleanup_audit_hooks()
            common.atomic_json(EVIDENCE / "exit.json", {
                "exit_code": code, "audit_complete": audit_complete,
                "loss_batches": loss_count, "optimizer_updates": step_count,
                "forward_counts": dict(forward_counts),
                "finished_unix": time.time(), "auto_test": False,
            })
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Training interrupted; resume only from this run last.pt.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"HBGNet training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
