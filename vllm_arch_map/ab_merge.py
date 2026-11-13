#!/usr/bin/env python3
"""Merge per_request_profiles.json (A/B) + kernel_diff.json into ab_compare.json.

Combines:
  - Module tree from torch.profiler (per_request_profiles.json) — hierarchy + ops
  - Kernel A/B timing from nsys (kernel_diff.json) — accurate GPU timing
  - Bench metrics from ab_bench.py (ab_bench.json) — latency comparison

The module→kernel mapping comes from torch.profiler's CUDA launch correlation
(traced, not pattern-matched). Kernel names are normalized to bridge formatting
differences between torch.profiler and nsys (template arg casts, mangling).

Usage:
  python ab_merge.py samples/cutlass/per_request_profiles.json \\
                      samples/b12x/per_request_profiles.json \\
                      --kernel-diff samples/kernel_diff.json \\
                      --bench samples/ab/ab_bench.json \\
                      -o samples/ab_compare.json
"""
import argparse
import json
import math
import re
from pathlib import Path


def _sanitize(obj):
    """Replace Infinity/NaN with None so json.dumps produces valid JSON."""
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ── Kernel name normalization ──
# torch.profiler and nsys render C++ template args differently:
#   profiler: true/false, 8, 128u
#   nsys:     (bool)1/(bool)0, (int)8, (unsigned int)128
# These normalizers strip type casts and unify bool/int representations.

def _normalize_kernel(name):
    """Normalize a kernel name for matching across profiler/nsys."""
    s = name
    s = re.sub(r'\(bool\)', '', s)
    s = re.sub(r'\(int\)', '', s)
    s = re.sub(r'\(unsigned int\)', '', s)
    s = s.replace('true', '1').replace('false', '0')
    s = re.sub(r'(\d+)u\b', r'\1', s)
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'\bT\d+\b', '', s)
    return s


def _extract_kernel_fn(name):
    """Extract the core kernel function name, stripping std::enable_if wrappers."""
    m = re.search(r'(\w+::[\w:]+_kernel\w*<[^>]*>)', name)
    if m:
        return m.group(1)
    m = re.search(r'(\w+::[\w:]+_kernel\w*)', name)
    if m:
        return m.group(1)
    return name


def _match_key(name):
    """Produce a normalized key for matching a kernel name."""
    if name.startswith('_Z'):
        return ('mangled', name)
    fn = _extract_kernel_fn(name)
    return ('norm', _normalize_kernel(fn)[:100])


def build_nsys_index(kernel_diff):
    """Build lookup from normalized kernel name → nsys entry."""
    if not kernel_diff:
        return {}, {}
    idx_norm = {}
    idx_mangled = {}
    for k in kernel_diff.get("kernels", []):
        key_type, key_val = _match_key(k["name"])
        if key_type == 'mangled':
            idx_mangled[key_val] = k
        else:
            idx_norm[key_val] = k
        if k.get("mangled_name") and k["mangled_name"].startswith("_Z"):
            idx_mangled[k["mangled_name"]] = k
        idx_norm.setdefault(('prefix', _normalize_kernel(k["name"])[:50]), k)
    return idx_norm, idx_mangled


def match_nsys_kernel(gpu_full, idx_norm, idx_mangled):
    """Match a profiler gpu_full name to an nsys kernel entry."""
    if not gpu_full:
        return None
    key_type, key_val = _match_key(gpu_full)
    if key_type == 'mangled':
        hit = idx_mangled.get(key_val)
        if hit:
            return hit
        return None
    hit = idx_norm.get(key_val)
    if hit:
        return hit
    prefix_key = ('prefix', _normalize_kernel(gpu_full)[:50])
    return idx_norm.get(prefix_key)


def _gpu_full_list(op):
    """Get all GPU kernel names from an op (supports multi-kernel dispatch)."""
    if not op:
        return []
    return op.get("gpu_full_list") or ([op["gpu_full"]] if op.get("gpu_full") else [])


def _collect_all_ops(node_a, node_b, idx_norm, idx_mangled):
    """First pass: collect all ops with their nsys matches to count sharing."""
    share_count_a = {}
    share_count_b = {}

    def _walk(na, nb):
        oa_list = na.get("ops", [])
        ob_list = nb.get("ops", [])
        max_seq = max(
            max((op.get("seq", i) for i, op in enumerate(oa_list)), default=-1),
            max((op.get("seq", i) for i, op in enumerate(ob_list)), default=-1),
        )
        ia = {op.get("seq", i): op for i, op in enumerate(oa_list)}
        ib = {op.get("seq", i): op for i, op in enumerate(ob_list)}
        for seq in range(max_seq + 1):
            oa, ob = ia.get(seq), ib.get(seq)
            if not oa and not ob:
                continue
            for gf in _gpu_full_list(oa):
                nsys = match_nsys_kernel(gf, idx_norm, idx_mangled)
                if nsys:
                    share_count_a[nsys["name"]] = share_count_a.get(nsys["name"], 0) + 1
            for gf in _gpu_full_list(ob):
                nsys = match_nsys_kernel(gf, idx_norm, idx_mangled)
                if nsys:
                    share_count_b[nsys["name"]] = share_count_b.get(nsys["name"], 0) + 1

        ca = na.get("children", [])
        cb = nb.get("children", [])
        ma, mb = {}, {}
        for c in ca:
            ma.setdefault(c.get("class", ""), []).append(c)
        for c in cb:
            mb.setdefault(c.get("class", ""), []).append(c)
        all_cls = list(dict.fromkeys(
            [c.get("class", "") for c in ca] + [c.get("class", "") for c in cb]
        ))
        for cn in all_cls:
            la, lb = ma.get(cn, []), mb.get(cn, [])
            for i in range(max(len(la), len(lb))):
                _walk(
                    la[i] if i < len(la) else {"class": cn},
                    lb[i] if i < len(lb) else {"class": cn},
                )

    _walk(node_a, node_b)
    return share_count_a, share_count_b


def merge_ops(ops_a, ops_b, idx_norm, idx_mangled, share_a, share_b):
    """Merge ops from A and B trees by seq order, adding nsys timing."""
    merged = []
    max_seq = max(
        max((op.get("seq", i) for i, op in enumerate(ops_a)), default=-1),
        max((op.get("seq", i) for i, op in enumerate(ops_b)), default=-1),
    )

    idx_a = {op.get("seq", i): op for i, op in enumerate(ops_a)}
    idx_b = {op.get("seq", i): op for i, op in enumerate(ops_b)}

    for seq in range(max_seq + 1):
        oa = idx_a.get(seq)
        ob = idx_b.get(seq)
        if not oa and not ob:
            continue

        entry = {
            "op": (oa or ob).get("op", ""),
            "dur_us_a": round(oa.get("dur_us", 0), 1) if oa else 0,
            "dur_us_b": round(ob.get("dur_us", 0), 1) if ob else 0,
        }

        gpu_dur_a = oa.get("gpu_dur_us", 0) if oa else 0
        gpu_dur_b = ob.get("gpu_dur_us", 0) if ob else 0
        if gpu_dur_a or gpu_dur_b:
            entry["gpu_dur_us_a"] = round(gpu_dur_a or 0, 1)
            entry["gpu_dur_us_b"] = round(gpu_dur_b or 0, 1)

        gpu_a = oa.get("gpu", "") if oa else ""
        gpu_b = ob.get("gpu", "") if ob else ""
        if gpu_a or gpu_b:
            entry["gpu_a"] = gpu_a
            entry["gpu_b"] = gpu_b

        gpu_full_a = oa.get("gpu_full", "") if oa else ""
        gpu_full_b = ob.get("gpu_full", "") if ob else ""
        if gpu_full_a or gpu_full_b:
            entry["gpu_full_a"] = gpu_full_a
            entry["gpu_full_b"] = gpu_full_b

        gf_list_a = _gpu_full_list(oa)
        gf_list_b = _gpu_full_list(ob)

        nsys_total_a = 0
        nsys_names_a = []
        for gf in gf_list_a:
            nsys_hit = match_nsys_kernel(gf, idx_norm, idx_mangled)
            if nsys_hit:
                n_share = share_a.get(nsys_hit["name"], 1)
                nsys_total_a += nsys_hit["total_ms_a"] / n_share
                nsys_names_a.append(nsys_hit["name"])

        nsys_total_b = 0
        nsys_names_b = []
        for gf in gf_list_b:
            nsys_hit = match_nsys_kernel(gf, idx_norm, idx_mangled)
            if nsys_hit:
                n_share = share_b.get(nsys_hit["name"], 1)
                nsys_total_b += nsys_hit["total_ms_b"] / n_share
                nsys_names_b.append(nsys_hit["name"])

        if nsys_total_a > 0:
            entry["nsys_ms_a"] = round(nsys_total_a, 2)
            entry["nsys_name_a"] = nsys_names_a[0]
            if len(nsys_names_a) > 1:
                entry["nsys_names_a"] = nsys_names_a
            if any(share_a.get(n, 1) > 1 for n in nsys_names_a):
                entry["nsys_shared_a"] = max(share_a.get(n, 1) for n in nsys_names_a)
        if nsys_total_b > 0:
            entry["nsys_ms_b"] = round(nsys_total_b, 2)
            entry["nsys_name_b"] = nsys_names_b[0]
            if len(nsys_names_b) > 1:
                entry["nsys_names_b"] = nsys_names_b
            if any(share_b.get(n, 1) > 1 for n in nsys_names_b):
                entry["nsys_shared_b"] = max(share_b.get(n, 1) for n in nsys_names_b)

        merged.append(entry)

    return merged


def merge_node(node_a, node_b, idx_norm, idx_mangled, share_a, share_b):
    """Merge two module tree nodes into one with A/B timing + ops."""
    cls = node_a.get("class", node_b.get("class", "?"))
    repeat = node_a.get("repeat", node_b.get("repeat"))

    dur_a = node_a.get("dur_us", 0)
    dur_b = node_b.get("dur_us", 0)

    ops_a = node_a.get("ops", [])
    ops_b = node_b.get("ops", [])

    gpu_us_a = sum(op.get("gpu_dur_us", 0) for op in ops_a if op.get("gpu_dur_us"))
    gpu_us_b = sum(op.get("gpu_dur_us", 0) for op in ops_b if op.get("gpu_dur_us"))

    merged_ops = merge_ops(ops_a, ops_b, idx_norm, idx_mangled, share_a, share_b)

    # Compute nsys totals for this node's own ops
    nsys_a = sum(op.get("nsys_ms_a", 0) for op in merged_ops)
    nsys_b = sum(op.get("nsys_ms_b", 0) for op in merged_ops)

    result = {
        "class": cls,
        "dur_us_a": round(dur_a, 1),
        "dur_us_b": round(dur_b, 1),
        "gpu_us_a": round(gpu_us_a, 1),
        "gpu_us_b": round(gpu_us_b, 1),
        "nsys_ms_a": round(nsys_a, 2),
        "nsys_ms_b": round(nsys_b, 2),
        "children": [],
    }
    if repeat:
        result["repeat"] = repeat
    if merged_ops:
        result["ops"] = merged_ops

    children_a = node_a.get("children", [])
    children_b = node_b.get("children", [])

    map_a = {}
    for c in children_a:
        key = c.get("class", "")
        map_a.setdefault(key, []).append(c)

    map_b = {}
    for c in children_b:
        key = c.get("class", "")
        map_b.setdefault(key, []).append(c)

    all_classes = list(dict.fromkeys(
        [c.get("class", "") for c in children_a] +
        [c.get("class", "") for c in children_b]
    ))

    for cls_name in all_classes:
        ca_list = map_a.get(cls_name, [])
        cb_list = map_b.get(cls_name, [])
        pairs = max(len(ca_list), len(cb_list))
        for i in range(pairs):
            ca = ca_list[i] if i < len(ca_list) else {"class": cls_name}
            cb = cb_list[i] if i < len(cb_list) else {"class": cls_name}
            result["children"].append(merge_node(ca, cb, idx_norm, idx_mangled, share_a, share_b))

    return result


def sum_recursive(node, key_a="nsys_ms_a", key_b="nsys_ms_b"):
    """Sum a timing field across a node and its children."""
    total = (node.get(key_a, 0), node.get(key_b, 0))
    for child in node.get("children", []):
        ca, cb = sum_recursive(child, key_a, key_b)
        total = (total[0] + ca, total[1] + cb)
    return total


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("profile_a", help="per_request_profiles.json for config A")
    parser.add_argument("profile_b", help="per_request_profiles.json for config B")
    parser.add_argument("--kernel-diff", default=None,
                        help="kernel_diff.json from ab_nsys.py (nsys timing)")
    parser.add_argument("--bench", default=None,
                        help="ab_bench.json from ab_bench.py (latency metrics)")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("-o", "--output", default="ab_compare.json")
    args = parser.parse_args()

    with open(args.profile_a) as f:
        pa = json.load(f)
    with open(args.profile_b) as f:
        pb = json.load(f)

    kernel_diff_raw = None
    kernel_diff_workloads = None
    kernel_diff_combined = None
    if args.kernel_diff:
        with open(args.kernel_diff) as f:
            kernel_diff_raw = json.load(f)
        if "workloads" in kernel_diff_raw:
            kernel_diff_workloads = kernel_diff_raw["workloads"]
            all_kernels = {}
            for wl_diff in kernel_diff_workloads.values():
                for k in wl_diff.get("kernels", []):
                    name = k["name"]
                    if name not in all_kernels:
                        all_kernels[name] = {**k, "total_ms_a": 0, "total_ms_b": 0,
                                             "instances_a": 0, "instances_b": 0}
                    all_kernels[name]["total_ms_a"] += k.get("total_ms_a", 0)
                    all_kernels[name]["total_ms_b"] += k.get("total_ms_b", 0)
                    all_kernels[name]["instances_a"] += k.get("instances_a", 0)
                    all_kernels[name]["instances_b"] += k.get("instances_b", 0)
            combined_kernels = sorted(all_kernels.values(),
                                      key=lambda k: max(k["total_ms_a"], k["total_ms_b"]),
                                      reverse=True)
            kernel_diff_combined = {
                "label_a": kernel_diff_raw.get("label_a", "A"),
                "label_b": kernel_diff_raw.get("label_b", "B"),
                "total_ms_a": sum(k["total_ms_a"] for k in combined_kernels),
                "total_ms_b": sum(k["total_ms_b"] for k in combined_kernels),
                "kernels": combined_kernels,
            }
        else:
            kernel_diff_combined = kernel_diff_raw

    bench = None
    if args.bench:
        with open(args.bench) as f:
            bench = json.load(f)

    label_a = args.label_a or Path(args.profile_a).parent.name
    label_b = args.label_b or Path(args.profile_b).parent.name

    idx_norm, idx_mangled = build_nsys_index(kernel_diff_combined)
    share_a, share_b = _collect_all_ops(
        pa["merged_tree"], pb["merged_tree"], idx_norm, idx_mangled
    )
    merged = merge_node(pa["merged_tree"], pb["merged_tree"],
                        idx_norm, idx_mangled, share_a, share_b)

    total_gpu_a, total_gpu_b = sum_recursive(merged, "gpu_us_a", "gpu_us_b")
    total_nsys_a, total_nsys_b = sum_recursive(merged, "nsys_ms_a", "nsys_ms_b")

    output = {
        "type": "ab_compare",
        "label_a": label_a,
        "label_b": label_b,
        "model": pa["merged_tree"].get("class", ""),
        "total_gpu_us_a": round(total_gpu_a, 1),
        "total_gpu_us_b": round(total_gpu_b, 1),
        "total_nsys_ms_a": round(total_nsys_a, 2),
        "total_nsys_ms_b": round(total_nsys_b, 2),
        "merged_tree": merged,
    }

    if kernel_diff_combined:
        output["nsys"] = kernel_diff_combined
    if kernel_diff_workloads:
        output["nsys_workloads"] = kernel_diff_workloads

    if bench:
        output["bench"] = bench.get("workloads", {})

    output = _sanitize(output)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"Merged {label_a} vs {label_b}")
    delta_gpu = (total_gpu_b - total_gpu_a) / total_gpu_a * 100 if total_gpu_a > 0 else 0
    print(f"  Profiler GPU: {total_gpu_a/1e3:.1f}ms ({label_a}) -> {total_gpu_b/1e3:.1f}ms ({label_b}) [{delta_gpu:+.1f}%]")
    if kernel_diff_combined:
        na, nb = kernel_diff_combined["total_ms_a"], kernel_diff_combined["total_ms_b"]
        nd = (nb - na) / na * 100 if na > 0 else 0
        print(f"  nsys total:   {na:.1f}ms ({label_a}) -> {nb:.1f}ms ({label_b}) [{nd:+.1f}%]")
        print(f"  nsys matched: {total_nsys_a:.1f}ms ({label_a}) -> {total_nsys_b:.1f}ms ({label_b})")
        unmatched_a = na - total_nsys_a
        unmatched_b = nb - total_nsys_b
        print(f"  nsys unmatched: {unmatched_a:.1f}ms ({label_a}), {unmatched_b:.1f}ms ({label_b})")
        print(f"  nsys kernels: {len(kernel_diff_combined.get('kernels', []))}")
    if kernel_diff_workloads:
        print(f"  nsys workloads:")
        for wname, wdiff in kernel_diff_workloads.items():
            wa, wb = wdiff["total_ms_a"], wdiff["total_ms_b"]
            wd = (wb - wa) / wa * 100 if wa > 0 else 0
            print(f"    {wname:>8}: {wa:.1f}ms ({label_a}) -> {wb:.1f}ms ({label_b}) [{wd:+.1f}%]")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
