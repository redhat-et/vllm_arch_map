#!/usr/bin/env python3
"""Build model_execution.json from raw Chrome trace via DFS.

Replaces capture.py + build_graph.py with one simple script.
Reads: profiles/*.json + model_tree.json
Writes: model_execution.json (same format model_viz.html expects)

Per-request mode (--per-request-dir):
  Reads: request_001_*.json, request_002_*.json, ...
  Writes: per_request_profiles.json (array of per-request module trees)
"""
import copy
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
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as f:
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
    """External id → kernel via cuda_runtime correlation, with timestamp fallback."""
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

    # Build sorted kernel list for timestamp fallback
    all_kernels = sorted(
        [e for e in cats.get("kernel", []) if e.get("dur", 0) > 0],
        key=lambda e: e["ts"]
    )
    _used_kernels = set()

    def find_kernel(ext_id, cpu_ts=None, cpu_dur=None):
        # Primary: correlation-based
        for c in ext_to_corrs.get(ext_id, []):
            if c in corr_to_gpu:
                return corr_to_gpu[c]
        # Fallback: find longest kernel launched during this CPU op's window
        if cpu_ts is not None and cpu_dur is not None and cpu_dur > 1:
            cpu_end = cpu_ts + cpu_dur
            best = None
            for k in all_kernels:
                if k["ts"] > cpu_end:
                    break
                if k["ts"] >= cpu_ts and id(k) not in _used_kernels:
                    if best is None or k.get("dur", 0) > best.get("dur", 0):
                        best = k
            if best:
                _used_kernels.add(id(best))
                return best
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
        is_mod = (e["cat"] == "python_function" and e["name"].startswith("nn.Module:")) or \
                 (e["cat"] == "user_annotation" and e["name"].startswith("Module:"))
        if is_mod:
            if e["name"].startswith("nn.Module:"):
                cls_inst = e["name"][11:].strip()
            else:
                cls_inst = e["name"][7:].strip().rsplit(".", 1)[-1]
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
            kernel = find_kernel(ext, e.get("ts"), e.get("dur")) if ext is not None else None
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
        kernel = find_kernel(ext, e.get("ts"), e.get("dur")) if ext is not None else None
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
        # Collapse: deep-copy all layers before modifying template (avoids circular ref)
        all_layers_copy = copy.deepcopy(run)
        template = run[0]
        template["repeat"] = len(run)
        durs = [r["dur_us"] for r in run]
        template["per_layer_dur"] = [round(d, 1) for d in durs]
        template["dur_us"] = round(sum(durs), 1)
        template["avg_dur_us"] = round(statistics.mean(durs), 1)
        template["min_dur_us"] = round(min(durs), 1)
        template["max_dur_us"] = round(max(durs), 1)
        template["all_layer_durs"] = template["per_layer_dur"]
        # Keep all individual layers for drill-down (from copy, no circular ref)
        template["layers"] = all_layers_copy
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
            op["dur_us"] = round(sum(durs), 1)
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
            ch["dur_us"] = round(sum(durs), 1)
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


def process_trace(trace_path):
    """Process a single Chrome trace → module_tree + kernel_summary + stats.

    Returns (module_tree, kernel_summary, pre_ops, post_ops, stats) or None on error.
    """
    events = load_events(trace_path)
    cats = categorize(events)
    cpu_op_events = cats.get("cpu_op", [])
    if not cpu_op_events:
        print(f"  No cpu_op events in {trace_path.name}")
        return None

    main_pid, main_tid = find_main_thread(cpu_op_events)
    find_kernel = build_correlation_map(cats)

    # Module markers: prefer nn.Module: (has class names, collapses properly)
    # Module: from install_recorder used as fallback only if no nn.Module: markers
    modules = [e for e in cats.get("python_function", [])
               if (e.get("name") or "").startswith("nn.Module:")
               and e["pid"] == main_pid and e["tid"] == main_tid]
    if not modules:
        modules = [e for e in cats.get("user_annotation", [])
                   if (e.get("name") or "").startswith("Module:")
                   and e["pid"] == main_pid and e["tid"] == main_tid]
    cpu_ops = [e for e in cpu_op_events
               if e["pid"] == main_pid and e["tid"] == main_tid]

    print(f"  {len(events)} events | {len(modules)} modules | {len(cpu_ops)} cpu_ops | {len(cats.get('kernel', []))} kernels")

    raw_roots = build_dfs_tree(modules, cpu_ops, find_kernel, main_pid, main_tid)

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
            if e["cat"] == "cpu_op":
                args = e.get("args") or {}
                ext = args.get("External id")
                kernel = find_kernel(ext, e.get("ts"), e.get("dur")) if ext is not None else None
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
            if e["cat"] == "cpu_op":
                args = e.get("args") or {}
                ext = args.get("External id")
                kernel = find_kernel(ext, e.get("ts"), e.get("dur")) if ext is not None else None
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
        print(f"  No nn.Module root in {trace_path.name}")
        return None

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

    collapse_layers(module_tree)

    num_layers = 0
    for ch in module_tree.get("children", []):
        if ch.get("repeat"):
            num_layers = ch["repeat"]
            break

    kernel_summary = build_kernel_summary(cats)

    stats = {
        "total_cpu_ops": len(cpu_ops),
        "total_kernels": len(cats.get("kernel", [])),
        "total_nn_module_events": len(modules),
        "num_layers": num_layers,
        "root_class": root_cls,
    }

    return module_tree, kernel_summary, pre_ops, post_ops, stats


def _merge_module_node(template, all_versions):
    """Recursively merge per-request timing into one tree node."""
    template["per_request_dur_us"] = [round(v.get("dur_us", 0), 1) for v in all_versions]

    if template.get("repeat"):
        prl = []
        for v in all_versions:
            prl.append(v.get("per_layer_dur", v.get("all_layer_durs", [])))
        template["per_request_per_layer_dur"] = prl

        # Children inside collapsed layer also get per-request cross-layer data
        for ci, child in enumerate(template.get("children", [])):
            if child.get("all_layer_durs"):
                child_cross = []
                for v in all_versions:
                    vc = v.get("children", [])
                    if ci < len(vc):
                        child_cross.append(vc[ci].get("all_layer_durs", []))
                    else:
                        child_cross.append([])
                child["per_request_all_layer_durs"] = child_cross

        # Merge per-request timing into individual layers for drill-down
        if template.get("layers"):
            n_layers = len(template["layers"])
            for li in range(n_layers):
                layer = template["layers"][li]
                layer_durs = []
                for v in all_versions:
                    v_layers = v.get("layers", [])
                    if li < len(v_layers):
                        layer_durs.append(round(v_layers[li].get("dur_us", 0), 1))
                    else:
                        layer_durs.append(0)
                layer["per_request_dur_us"] = layer_durs
                # Merge children and ops timing within each layer
                for ci2, lch in enumerate(layer.get("children", [])):
                    ch_durs = []
                    for v in all_versions:
                        v_layers = v.get("layers", [])
                        if li < len(v_layers):
                            vc = v_layers[li].get("children", [])
                            ch_durs.append(round(vc[ci2].get("dur_us", 0), 1) if ci2 < len(vc) else 0)
                        else:
                            ch_durs.append(0)
                    lch["per_request_dur_us"] = ch_durs
                for oi2, lop in enumerate(layer.get("ops", [])):
                    op_durs = []
                    for v in all_versions:
                        v_layers = v.get("layers", [])
                        if li < len(v_layers):
                            vo = v_layers[li].get("ops", [])
                            op_durs.append(round(vo[oi2].get("dur_us", 0), 1) if oi2 < len(vo) else 0)
                        else:
                            op_durs.append(0)
                    lop["per_request_dur_us"] = op_durs

    for ci, child in enumerate(template.get("children", [])):
        child_versions = []
        for v in all_versions:
            vc = v.get("children", [])
            if ci < len(vc):
                child_versions.append(vc[ci])
            else:
                child_versions.append({"dur_us": 0, "children": [], "ops": []})
        _merge_module_node(child, child_versions)

    for oi, op in enumerate(template.get("ops", [])):
        op_durs = []
        gpu_durs = []
        has_gpu = False
        for v in all_versions:
            vo = v.get("ops", [])
            if oi < len(vo):
                op_durs.append(round(vo[oi].get("dur_us", 0), 1))
                gd = vo[oi].get("gpu_dur_us", 0)
                gpu_durs.append(round(gd, 1))
                if gd:
                    has_gpu = True
            else:
                op_durs.append(0)
                gpu_durs.append(0)
        op["per_request_dur_us"] = op_durs
        if has_gpu:
            op["per_request_gpu_dur_us"] = gpu_durs

        if op.get("sub_ops"):
            for si, sub in enumerate(op["sub_ops"]):
                sub_durs = []
                sub_gpu = []
                sub_has_gpu = False
                for v in all_versions:
                    vo = v.get("ops", [])
                    if oi < len(vo):
                        vs = vo[oi].get("sub_ops", [])
                        if si < len(vs):
                            sub_durs.append(round(vs[si].get("dur_us", 0), 1))
                            sgd = vs[si].get("gpu_dur_us", 0)
                            sub_gpu.append(round(sgd, 1))
                            if sgd:
                                sub_has_gpu = True
                            continue
                    sub_durs.append(0)
                    sub_gpu.append(0)
                sub["per_request_dur_us"] = sub_durs
                if sub_has_gpu:
                    sub["per_request_gpu_dur_us"] = sub_gpu


def merge_request_trees(trees, labels):
    """Merge N per-request module trees into one tree with per-request timing arrays.

    The tree structure is identical across requests (same model). Each node gets
    per_request_dur_us arrays so one tree can show timing for all requests.
    """
    import copy
    if not trees:
        return None
    merged = copy.deepcopy(trees[0])
    _merge_module_node(merged, trees)
    merged["request_labels"] = labels
    return merged


def process_per_request_traces(per_request_dir, out_dir):
    """Process all per-request Chrome traces → per_request_profiles.json.

    Merges all per-request trees into ONE tree with per-request timing arrays.
    """
    req_dir = Path(per_request_dir)
    out_path = Path(out_dir) / "per_request_profiles.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    traces = sorted(req_dir.glob("request_[0-9]*.json")) + sorted(req_dir.glob("request_[0-9]*.json.gz"))
    traces.sort(key=lambda p: p.name)
    if not traces:
        print(f"No request_*.json[.gz] files in {req_dir}")
        raise SystemExit(1)

    # Read metadata from request_metadata.json if present
    meta_path = req_dir / "request_metadata.json"
    custom_labels = None
    profile_mode = "eager"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        custom_labels = [r["label"] for r in meta.get("requests", [])]
        profile_mode = meta.get("mode", "eager")
        print(f"Using custom labels from {meta_path.name} (mode: {profile_mode})")

    print(f"Processing {len(traces)} per-request traces from {req_dir}...")

    trees = []
    labels = []
    request_ids = []
    kernel_summaries = []

    for i, trace_path in enumerate(traces):
        label = custom_labels[i] if custom_labels and i < len(custom_labels) else f"Req-{i + 1:03d}"
        print(f"\n[{label}] {trace_path.name}")

        result = process_trace(trace_path)
        if result is None:
            print(f"  Skipping {trace_path.name} (no module tree)")
            continue

        module_tree, kernel_summary, pre_ops, post_ops, stats = result

        req_id = trace_path.stem
        parts = req_id.split("_", 2)
        if len(parts) >= 3:
            req_id = parts[2]

        trees.append(module_tree)
        labels.append(label)
        request_ids.append(req_id)
        kernel_summaries.append(kernel_summary)

    if not trees:
        print("No valid traces processed")
        raise SystemExit(1)

    merged_tree = merge_request_trees(trees, labels)

    output = {
        "type": "per_request_profiles",
        "mode": profile_mode,
        "count": len(trees),
        "request_labels": labels,
        "request_ids": request_ids,
        "merged_tree": merged_tree,
        "per_request_kernel_summaries": [
            {"label": labels[i], "kernels": kernel_summaries[i]}
            for i in range(len(trees))
        ],
    }

    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(trees)} requests merged into one tree")


def main(out_dir="samples", traces_dir="samples/traces", per_request_dir=None):
    if per_request_dir:
        process_per_request_traces(per_request_dir, out_dir)
        return

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out = out_dir_path / "model_execution.json"
    trace_path = find_trace(traces_dir)
    if not trace_path:
        print("No trace file found in profiles/")
        raise SystemExit(1)

    mt_path = out_dir_path / "model_tree.json"
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

    print(f"Processing {trace_path.name}...")
    result = process_trace(trace_path)
    if result is None:
        print("Error: could not process trace")
        raise SystemExit(1)

    module_tree, kernel_summary, pre_ops, post_ops, stats = result

    output = {
        "model": model_name,
        "device": device,
        "dtype": dtype,
        "engine": engine,
        "config": config,
        "mode": "prefill",
        "iterations": 1,
        "layers": stats["num_layers"],
        "ops_per_layer": 0,
        "layer_template": [],
        "per_layer": [],
        "pre_layers": [],
        "post_layers": [],
        "kernel_summary": kernel_summary,
        "module_tree": module_tree,
        "pre_model_ops": pre_ops,
        "post_model_ops": post_ops,
        "stats": stats,
    }

    out.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  module_tree: {stats['root_class']} | {stats['num_layers']} layers | {len(kernel_summary)} kernel types")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="samples", help="Target output directory")
    parser.add_argument("--traces-dir", default="samples/traces", help="Traces directory (profiles/ subdir)")
    parser.add_argument("--per-request-dir", default=None,
                        help="Directory of per-request Chrome traces from profile_serve.py")
    args = parser.parse_args()

    main(out_dir=args.out_dir, traces_dir=args.traces_dir, per_request_dir=args.per_request_dir)
