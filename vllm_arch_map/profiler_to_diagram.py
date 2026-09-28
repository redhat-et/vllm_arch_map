#!/usr/bin/env python3
"""Build model_execution.json from raw Chrome trace via DFS.

Replaces capture.py + build_graph.py with one simple script.
Reads: profiles/*.json + model_tree.json
Writes: model_execution.json (same format model_viz.html expects)
"""
import json
import gzip
import statistics
import argparse
from pathlib import Path


GPU_DTYPES = {"c10::BFloat16", "c10::Half", "c10::Float", "c10::Double",
              "c10::Float8_e4m3fn", "c10::Float8_e5m2"}


def find_trace(traces_dir):
    profiles_dir = Path(traces_dir) / "profiles"
    for f in sorted(profiles_dir.glob("rank0*.json"), key=lambda p: p.stat().st_size, reverse=True):
        if f.stat().st_size > 500_000:
            return f
    for f in sorted(profiles_dir.glob("rank0*.json.gz"), key=lambda p: p.stat().st_size, reverse=True):
        if f.stat().st_size > 500_000:
            decompressed_path = f.with_suffix("")  # Removes the '.gz' extension
            if not decompressed_path.exists():
                print(f"Extracting {f.name} -> {decompressed_path.name}...")
                with gzip.open(f, "rb") as f_in:
                    with open(decompressed_path, "wb") as f_out:
                        f_out.write(f_in.read())
            return decompressed_path


def load_events(path):
    print(f"Loading {path} ({path.stat().st_size / 1e6:.1f} MB)...")
    with open(path) as f:
        raw = json.load(f)
    return raw.get("traceEvents", raw)


def categorize(events):
    cats = {}
    for e in events:
        cats.setdefault(e.get("cat", ""), []).append(e)
    return cats


def find_main_thread(cpu_ops):
    counts = {}
    for e in cpu_ops:
        k = (e["pid"], e["tid"])
        counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get)


def build_correlation_map(cats):
    """External id → kernel via cuda_runtime correlation."""
    corr_to_gpu = {}
    for e in cats.get("kernel", []) + cats.get("gpu_memcpy", []):
        c = (e.get("args") or {}).get("correlation")
        if c is not None:
            corr_to_gpu[c] = e
    ext_to_corrs = {}
    for e in cats.get("cuda_runtime", []):
        a = e.get("args") or {}
        ext, corr = a.get("External id"), a.get("correlation")
        if ext is not None and corr is not None:
            ext_to_corrs.setdefault(ext, []).append(corr)

    def find_kernel(ext_id):
        for c in ext_to_corrs.get(ext_id, []):
            if c in corr_to_gpu:
                return corr_to_gpu[c]
        return None
    return find_kernel


def has_gpu_input(args):
    for t in (args.get("Input type") or []):
        if t in GPU_DTYPES:
            return True
    return False


def gpu_label(op_name, args, kernel):
    if kernel:
        name = kernel["name"]
        dur = round(kernel.get("dur", 0), 1)
        short = name[:80] + "..." if len(name) > 80 else name
        return short, dur
    if has_gpu_input(args):
        return None, None  # GPU tensor metadata, no kernel
    return None, None


def build_dfs_tree(modules, cpu_ops, find_kernel, main_pid, main_tid):
    """DFS: sort by (ts, -dur), nest by time containment."""
    on_main = lambda e: e["pid"] == main_pid and e["tid"] == main_tid
    tree_input = sorted(
        [e for e in modules + cpu_ops
         if e.get("ph") == "X" and e.get("dur", 0) > 0 and on_main(e)],
        key=lambda e: (e["ts"], -e["dur"]),
    )

    roots = []
    stack = []
    for e in tree_input:
        end = e["ts"] + e["dur"]
        while stack and e["ts"] >= stack[-1]["_end"]:
            stack.pop()

        node = {"_e": e, "_end": end, "_children": []}
        if stack:
            stack[-1]["_children"].append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def raw_to_module_tree(raw_nodes, find_kernel, seq_start=0):
    """Convert DFS raw nodes → module_tree format for model_viz.html."""
    children = []
    ops = []
    seq = seq_start

    for rn in raw_nodes:
        e = rn["_e"]
        if e["cat"] == "python_function" and e["name"].startswith("nn.Module:"):
            cls_inst = e["name"][11:].strip()
            parts = cls_inst.rsplit("_", 1)
            cls = parts[0] if len(parts) == 2 and parts[1].isdigit() else cls_inst
            inst = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

            sub = raw_to_module_tree(rn["_children"], find_kernel)
            mod = {
                "class": cls,
                "instance": inst,
                "dur_us": round(e["dur"], 1),
                "children": sub["children"],
                "ops": sub["ops"],
                "seq": seq,
            }
            children.append(mod)
        elif e["cat"] == "cpu_op":
            args = e.get("args") or {}
            ext = args.get("External id")
            kernel = find_kernel(ext) if ext is not None else None
            op = {
                "op": e["name"],
                "dur_us": round(e["dur"], 1),
                "seq": seq,
            }
            # Dims
            dims = args.get("Input Dims")
            if dims:
                op["dims"] = " → ".join(f"[{','.join(str(x) for x in d)}]" for d in dims)
                op["args"] = [
                    {"name": f"in{i}", "shape": d, "dtype": ((args.get("Input type") or [None]*99)[i] or "").replace("c10::", "")}
                    for i, d in enumerate(dims)
                ]
            # GPU
            if kernel:
                kname = kernel["name"]
                short = kname[:80] + "..." if len(kname) > 80 else kname
                op["gpu"] = short
                op["gpu_dur_us"] = round(kernel.get("dur", 0), 1)
                ka = kernel.get("args") or {}
                op["gpu_role"] = _guess_role(e["name"], kname)
            elif has_gpu_input(args):
                op["gpu_note"] = "GPU tensor metadata (no kernel)"
            else:
                op["gpu_note"] = "CPU-only"

            # Sub-ops from children
            sub_ops = _build_sub_ops(rn["_children"], find_kernel)
            if sub_ops:
                op["sub_ops"] = sub_ops

            ops.append(op)
        seq += 1

    return {"children": children, "ops": ops}


def _build_sub_ops(raw_children, find_kernel):
    result = []
    for rn in raw_children:
        e = rn["_e"]
        if e["cat"] != "cpu_op":
            continue
        args = e.get("args") or {}
        ext = args.get("External id")
        kernel = find_kernel(ext) if ext is not None else None
        sub = {"op": e["name"], "dur_us": round(e["dur"], 1)}
        dims = args.get("Input Dims")
        if dims:
            sub["dims"] = " → ".join(f"[{','.join(str(x) for x in d)}]" for d in dims)
        if kernel:
            kname = kernel["name"]
            sub["gpu_display"] = kname[:80] + ("..." if len(kname) > 80 else "")
            sub["gpu_dur_us"] = round(kernel.get("dur", 0), 1)
        elif has_gpu_input(args):
            sub["gpu_note"] = "GPU tensor metadata (no kernel)"
        else:
            sub["gpu_note"] = "CPU-only"
        child_subs = _build_sub_ops(rn["_children"], find_kernel)
        if child_subs:
            sub["sub_ops"] = child_subs
        result.append(sub)
    return result


def _guess_role(cpu_name, kernel_name):
    kl = kernel_name.lower()
    if "gemm" in kl or "gemv" in kl or "cutlass" in kl:
        return "gemm"
    if "flash" in kl or "attention" in kl or "fmha" in kl:
        return "flash_attention"
    if "rms_norm" in kl or "layer_norm" in kl:
        return "rms_norm"
    if "silu" in kl or "gelu" in kl:
        return "silu_and_mul"
    if "rotary" in kl or "rope" in kl:
        return "rotary_embedding"
    if "cache" in kl or "reshape_and_cache" in kl:
        return "kv_cache"
    if "elementwise" in kl or "vectorized" in kl or "add" in kl:
        return "elementwise"
    if "embedding" in kl:
        return "embedding"
    if "topk" in kl or "sample" in kl or "softmax" in kl:
        return "sampling"
    return "other"


def collapse_layers(module):
    """Detect consecutive children with same class → collapse to template ×N."""
    children = module.get("children", [])
    if len(children) < 2:
        return

    # Recursively collapse children first
    for ch in children:
        collapse_layers(ch)

    # Find runs of same class
    groups = []
    i = 0
    while i < len(children):
        cls = children[i]["class"]
        run = [children[i]]
        j = i + 1
        while j < len(children) and children[j]["class"] == cls:
            run.append(children[j])
            j += 1
        groups.append(run)
        i = j

    new_children = []
    for run in groups:
        if len(run) < 3:
            new_children.extend(run)
            continue
        # Collapse: keep first as template, record per-layer stats
        template = run[0]
        template["repeat"] = len(run)
        durs = [r["dur_us"] for r in run]
        template["per_layer_dur"] = [round(d, 1) for d in durs]
        template["avg_dur_us"] = round(statistics.mean(durs), 1)
        template["min_dur_us"] = round(min(durs), 1)
        template["max_dur_us"] = round(max(durs), 1)
        template["all_layer_durs"] = template["per_layer_dur"]
        # Per-op and per-child stats across layers
        _add_cross_layer_stats(template, run)
        new_children.append(template)

    module["children"] = new_children


def _add_cross_layer_stats(template, all_layers):
    """Add avg/min/max across layers for ops and children."""
    # Ops stats
    for oi, op in enumerate(template.get("ops", [])):
        durs = []
        for layer in all_layers:
            layer_ops = layer.get("ops", [])
            if oi < len(layer_ops):
                durs.append(layer_ops[oi]["dur_us"])
        if len(durs) > 1:
            op["avg_us"] = round(statistics.mean(durs), 1)
            op["min_us"] = round(min(durs), 1)
            op["max_us"] = round(max(durs), 1)
            op["all_layer_durs"] = [round(d, 1) for d in durs]

    # Children stats
    for ci, ch in enumerate(template.get("children", [])):
        durs = []
        for layer in all_layers:
            layer_ch = layer.get("children", [])
            if ci < len(layer_ch):
                durs.append(layer_ch[ci]["dur_us"])
        if len(durs) > 1:
            ch["avg_dur_us"] = round(statistics.mean(durs), 1)
            ch["min_dur_us"] = round(min(durs), 1)
            ch["max_dur_us"] = round(max(durs), 1)
            ch["all_layer_durs"] = [round(d, 1) for d in durs]
            # Recurse: stats for grandchildren ops
            _add_cross_layer_stats(ch, [
                layer.get("children", [])[ci]
                for layer in all_layers
                if ci < len(layer.get("children", []))
            ])


def build_kernel_summary(cats):
    summary = {}
    for e in cats.get("kernel", []):
        name = e["name"]
        dur = e.get("dur", 0)
        if name not in summary:
            summary[name] = {"count": 0, "total_dur_us": 0, "role": _guess_role("", name)}
        summary[name]["count"] += 1
        summary[name]["total_dur_us"] = round(summary[name]["total_dur_us"] + dur, 1)
    return dict(sorted(summary.items(), key=lambda x: -x[1]["total_dur_us"])[:30])


def main(out_dir="samples",traces_dir="samples/traces"):
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out = out_dir_path / "model_execution.json"
    trace_path = find_trace(traces_dir)
    if not trace_path:
        print("No trace file found in profiles/")
        raise SystemExit(1)

    # Load model_tree for config
    mt_path = out_dir_path/ "model_tree.json"
    config = {}
    model_name = "?"
    device = "cuda"
    dtype = "bfloat16"
    engine = {}
    if mt_path.exists():
        mt = json.loads(mt_path.read_text())
        config = mt.get("config", {})
        model_name = mt.get("model", "?")
    else:
        print("Warning: model_tree.json not found, using defaults")

    # Parse trace
    events = load_events(trace_path)
    cats = categorize(events)
    main_pid, main_tid = find_main_thread(cats.get("cpu_op", []))
    find_kernel = build_correlation_map(cats)

    # Filter events
    modules = [e for e in cats.get("python_function", [])
               if (e.get("name") or "").startswith("nn.Module:")
               and e["pid"] == main_pid and e["tid"] == main_tid]
    cpu_ops = [e for e in cats.get("cpu_op", [])
               if e["pid"] == main_pid and e["tid"] == main_tid]

    print(f"  {len(events)} events | {len(modules)} modules | {len(cpu_ops)} cpu_ops | {len(cats.get('kernel', []))} kernels")

    # DFS tree
    raw_roots = build_dfs_tree(modules, cpu_ops, find_kernel, main_pid, main_tid)

    # Find the nn.Module root (the model forward)
    model_root = None
    pre_ops = []
    post_ops = []
    found = False
    for rn in raw_roots:
        e = rn["_e"]
        if not found and e["cat"] == "python_function" and e["name"].startswith("nn.Module:"):
            model_root = rn
            found = True
        elif not found:
            # Pre-model ops
            if e["cat"] == "cpu_op":
                args = e.get("args") or {}
                ext = args.get("External id")
                kernel = find_kernel(ext) if ext is not None else None
                op = {"op": e["name"], "dur_us": round(e["dur"], 1)}
                if kernel:
                    op["gpu"] = kernel["name"][:80]
                    op["gpu_dur_us"] = round(kernel.get("dur", 0), 1)
                elif has_gpu_input(args):
                    op["gpu_note"] = "GPU tensor metadata (no kernel)"
                else:
                    op["gpu_note"] = "CPU-only"
                pre_ops.append(op)
        else:
            # Post-model ops
            if e["cat"] == "cpu_op":
                args = e.get("args") or {}
                ext = args.get("External id")
                kernel = find_kernel(ext) if ext is not None else None
                dims = args.get("Input Dims")
                op = {"op": e["name"], "dur_us": round(e["dur"], 1)}
                if dims:
                    op["dims"] = " → ".join(f"[{','.join(str(x) for x in d)}]" for d in dims)
                if kernel:
                    op["gpu"] = kernel["name"][:80]
                    op["gpu_dur_us"] = round(kernel.get("dur", 0), 1)
                elif has_gpu_input(args):
                    op["gpu_note"] = "GPU tensor metadata (no kernel)"
                else:
                    op["gpu_note"] = "CPU-only"
                post_ops.append(op)

    if not model_root:
        print("Error: no nn.Module root found in trace")
        raise SystemExit(1)

    # Build module tree
    e = model_root["_e"]
    cls_inst = e["name"][11:].strip()
    parts = cls_inst.rsplit("_", 1)
    root_cls = parts[0] if len(parts) == 2 and parts[1].isdigit() else cls_inst

    sub = raw_to_module_tree(model_root["_children"], find_kernel)
    module_tree = {
        "class": root_cls,
        "instance": 0,
        "dur_us": round(e["dur"], 1),
        "children": sub["children"],
        "ops": sub["ops"],
    }

    # Collapse repeated layers
    collapse_layers(module_tree)

    # Count layers
    num_layers = 0
    for ch in module_tree.get("children", []):
        if ch.get("repeat"):
            num_layers = ch["repeat"]
            break

    # Kernel summary
    kernel_summary = build_kernel_summary(cats)

    # Output
    result = {
        "model": model_name,
        "device": device,
        "dtype": dtype,
        "engine": engine,
        "config": config,
        "mode": "prefill",
        "iterations": 1,
        "layers": num_layers,
        "ops_per_layer": 0,
        "layer_template": [],
        "per_layer": [],
        "pre_layers": [],
        "post_layers": [],
        "kernel_summary": kernel_summary,
        "module_tree": module_tree,
        "pre_model_ops": pre_ops,
        "post_model_ops": post_ops,
        "stats": {
            "total_cpu_ops": len(cpu_ops),
            "total_kernels": len(cats.get("kernel", [])),
            "total_nn_module_events": len(modules),
        },
    }

    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  module_tree: {root_cls} | {num_layers} layers | {len(kernel_summary)} kernel types")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="samples", help="Target output directory")
    parser.add_argument("--traces-dir", default="samples/traces", help="Target output directory")
    args = parser.parse_args()
    
    main(out_dir=args.out_dir, traces_dir=args.traces_dir)
