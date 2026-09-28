#!/usr/bin/env python3
"""Inspect the real eager vLLM worker model and optionally capture inference.

Run on the GPU host: python walk_model.py MODEL -o run/model_tree.json
Add --with-forward to capture module I/O and a short profiler trace.
"""

import argparse
import json
import os
import uuid
from pathlib import Path

from flow_support import inspect_method


def walk_modules(model):
    entries = []
    for name, module in model.named_modules():
        cls = type(module)
        entry = {
            "name": name or "__root__",
            "class": cls.__name__,
            "module": f"{cls.__module__}.{cls.__name__}",
            "init": inspect_method(cls.__init__),
            "forward": inspect_method(cls.forward),
            "weights": {},
            "buffers": {},
            "config": {},
            "implementations": {},
        }
        for field, tensors in (
            ("weights", module.named_parameters(recurse=False)),
            ("buffers", module.named_buffers(recurse=False)),
        ):
            for key, value in tensors:
                entry[field][key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "stride": list(value.stride()),
                }
        for key, value in vars(module).items():
            if not key.startswith("_") and isinstance(
                value, (str, int, float, bool, type(None))
            ):
                entry["config"][key] = value
        for key in ("quant_method", "impl", "attn_backend", "linear_method"):
            value = getattr(module, key, None)
            if value is None:
                continue
            target = value if isinstance(value, type) else type(value)
            entry["implementations"][key] = {
                "class": f"{target.__module__}.{target.__name__}",
                "evidence": "loaded_object",
                "methods": {
                    method: inspect_method(getattr(target, method))
                    for method in ("forward", "apply", "forward_cuda")
                    if hasattr(target, method)
                },
            }
        entries.append(entry)
    return entries


def extract_execution_order(forward_data):
    """Extract ordered submodule calls from AST forward ops."""
    if not forward_data or forward_data.get("evidence") == "unavailable":
        return []
    ops = forward_data.get("operations", [])
    order = []
    for op in ops:
        for call in op.get("calls", []):
            if call.startswith("self."):
                target = call[5:]
                if "." in target:
                    target = target.split(".")[0]
                if "(" in target:
                    target = target.split("(")[0]
                order.append({
                    "target": target,
                    "call": call,
                    "writes": op.get("writes", []),
                    "reads": op.get("reads", []),
                    "line": op.get("line", ""),
                })
    return order


def build_tree(entries):
    root = {"name": "model", "children": []}
    by_path = {"": root}
    for entry in entries:
        path = entry["name"]
        parent, _, leaf = path.rpartition(".")
        node = {**entry, "name": leaf, "path": path}
        fwd = node.get("forward")
        if fwd:
            eo = extract_execution_order(fwd)
            if eo:
                fwd["execution_order"] = eo
        by_path.get(parent, root).setdefault("children", []).append(node)
        by_path[path] = node
    return root


def tensor_metadata(value):
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
        }
    if isinstance(value, (list, tuple)):
        return [tensor_metadata(v) for v in value]
    if isinstance(value, dict):
        return {str(k): tensor_metadata(v) for k, v in value.items()}
    return {"type": type(value).__name__}


def install_recorder(model):
    """Annotate invocations without retaining activation tensors."""
    import threading

    import torch

    state = {"events": [], "handles": [], "local": threading.local()}
    model._architecture_capture = state

    def pre(path):
        def hook(module, args, kwargs):
            stack = getattr(state["local"], "stack", [])
            record = {
                "id": len(state["events"]),
                "module": path,
                "parent": stack[-1][0]["id"] if stack else None,
                "inputs": tensor_metadata(args),
                "kwargs": tensor_metadata(kwargs),
            }
            state["events"].append(record)
            context = torch.profiler.record_function(f"Module:{path}")
            context.__enter__()
            stack.append((record, context))
            state["local"].stack = stack

        return hook

    def post(module, args, output):
        record, context = state["local"].stack.pop()
        try:
            record["outputs"] = tensor_metadata(output)
        finally:
            context.__exit__(None, None, None)

    for name, module in model.named_modules():
        state["handles"].append(
            module.register_forward_pre_hook(pre(name or "__root__"), with_kwargs=True)
        )
        state["handles"].append(module.register_forward_hook(post, always_call=True))


def collect_recorder(model):
    state = model._architecture_capture
    for handle in state["handles"]:
        handle.remove()
    del model._architecture_capture
    return state["events"]


def snapshot(model):
    return {"modules": walk_modules(model), "worker_pid": os.getpid()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--with-forward", action="store_true")
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory to save torch profiler outputs. Defaults to <output_dir>/profiles"
    )
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--prompt", default="Explain what attention does.")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    if args.profile_dir:
        # Respect user's explicit path (e.g. samples/traces/profiles or custom out path)
        profile_dir = Path(args.profile_dir).resolve() / f"profiles-{run_id}"
    else:
        profile_dir = destination.parent / f"profiles-{run_id}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    options = {}
    if args.with_forward:
        options["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_record_shapes": True,
        }
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **options,
    )
    workers = llm.apply_model(snapshot)
    if len(workers) != 1:
        raise RuntimeError("This capture command currently supports one worker only")
    worker = workers[0]
    config = llm.llm_engine.vllm_config.model_config.hf_config.to_dict()
    invocations = []
    if args.with_forward:
        llm.generate([args.prompt], SamplingParams(max_tokens=1, temperature=0))
        llm.apply_model(install_recorder)
        try:
            llm.start_profile()
            try:
                llm.generate([args.prompt], SamplingParams(max_tokens=4, temperature=0))
            finally:
                llm.stop_profile()
        finally:
            invocations = llm.apply_model(collect_recorder)[0]
    result = {
        "schema_version": 2,
        "model": args.model,
        "source": "loaded_vllm_worker",
        "run_id": run_id,
        "worker_pid": worker["worker_pid"],
        "config": config,
        "tree": build_tree(worker["modules"]),
        "invocations": invocations,
        "profile_dir": str(profile_dir) if args.with_forward else None,
        "capture_mode": "eager",
        "limitations": [
            "Module I/O metadata is observed; source flow is static.",
            "No runtime tensor producer/consumer or alias graph is captured.",
            "Prefill/decode phase is not assigned to individual invocations.",
        ],
    }
    destination.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {destination}")
    if args.with_forward:
        print(f"Worker PID {worker['worker_pid']}; traces: {profile_dir}")


if __name__ == "__main__":
    main()
