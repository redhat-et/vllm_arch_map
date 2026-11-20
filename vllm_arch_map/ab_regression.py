#!/usr/bin/env python3
"""A/B regression analysis: classify kernel regressions and select NCU targets.

Cross-references nsys kernel diff with dispatch chains to find:
  - [FW]     framework issue: B has extra ops (transpose, dtype, copy)
  - [KN-B]   B-specific kernel (different kernel than A for same op) → NCU target
  - [SHARED] same kernel in both backends, not actionable for backend dev

Usage:
  python ab_regression.py \
      --kernel-diff traces/ab/nsys/kernel_diff_all.json \
      --dispatch-dir traces/ab/ \
      --label-a cutlass --label-b b12x \
      -o traces/ab/ab_regression_report.json
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path


def _short(name, maxlen=80):
    s = re.sub(r'void\s+', '', name)
    s = re.sub(r'<[^>]*>', '<…>', s)
    if len(s) > maxlen:
        s = s[:maxlen - 3] + '...'
    return s


def build_kernel_index(chains):
    """kernel name → parent op name."""
    idx = {}
    for parent_op, chain in chains.items():
        for child in chain.get("children", []):
            for call in child.get("calls", []):
                if call.get("kernel") and call["kernel"] not in idx:
                    idx[call["kernel"]] = parent_op
    return idx


def build_op_kernels(chains):
    """parent op → set of kernel names dispatched."""
    op_k = {}
    for parent_op, chain in chains.items():
        kernels = set()
        for child in chain.get("children", []):
            for call in child.get("calls", []):
                if call.get("kernel"):
                    kernels.add(call["kernel"])
        if kernels:
            op_k[parent_op] = kernels
    return op_k


def _kernel_base(name):
    """Strip void prefix and template args for base-name comparison."""
    s = re.sub(r'^void\s+', '', name)
    idx = s.find('<')
    if idx > 0:
        s = s[:idx]
    return s.strip()


def find_b_specific_kernels(chains_a, chains_b):
    """Find kernels in B whose base name doesn't appear in A at all."""
    ok_a = build_op_kernels(chains_a)
    ok_b = build_op_kernels(chains_b)
    all_a_bases = set()
    for kernels in ok_a.values():
        for k in kernels:
            all_a_bases.add(_kernel_base(k))

    b_specific = set()
    for parent_op, b_kernels in ok_b.items():
        for k in b_kernels:
            if _kernel_base(k) not in all_a_bases:
                b_specific.add(k)
    return b_specific


def load_shapes(profile_path):
    """Load op → shapes mapping from per_request_profiles.json."""
    if not profile_path:
        return {}
    with open(profile_path) as f:
        data = json.load(f)

    shapes = {}

    def _walk(node):
        for op in node.get("ops", []):
            name = op.get("op", "")
            if name and op.get("args"):
                shapes[name] = op["args"]
        for child in node.get("children", []):
            _walk(child)

    tree = data.get("merged_tree", data)
    if isinstance(data, list):
        for entry in data:
            _walk(entry.get("merged_tree", entry))
    else:
        _walk(tree)
    return shapes


def find_extra_ops(chains_a, chains_b, parent_a, parent_b):
    """Find child ops in B's chain that A doesn't have."""
    if not parent_a or not parent_b:
        return []
    ca = chains_a.get(parent_a, {})
    cb = chains_b.get(parent_b, {})
    ops_a = {c["op"] for c in ca.get("children", [])}
    return [
        {"op": c["op"], "dur_us": c.get("dur_us", 0)}
        for c in cb.get("children", [])
        if c["op"] not in ops_a
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel-diff", required=True)
    parser.add_argument("--dispatch-dir", default=None)
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--shapes", default=None,
                        help="per_request_profiles.json from profile_serve.py (for tensor shapes)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("-o", "--output", default="ab_regression_report.json")
    args = parser.parse_args()

    with open(args.kernel_diff) as f:
        kdiff = json.load(f)

    label_a = args.label_a or kdiff.get("label_a", "A")
    label_b = args.label_b or kdiff.get("label_b", "B")

    op_shapes = load_shapes(args.shapes)

    dc_by_wl = {}
    if args.dispatch_dir:
        for fp in glob.glob(str(Path(args.dispatch_dir) / "dispatch_chain_*.json")):
            stem = Path(fp).stem.replace("dispatch_chain_", "")
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            backend, workload = parts
            dc_by_wl.setdefault(workload, {})
            with open(fp) as f:
                dc_by_wl[workload][backend] = json.load(f)

    workloads = kdiff.get("workloads", {"all": kdiff})
    report = {"label_a": label_a, "label_b": label_b, "workloads": {}, "ncu_kernels": []}

    for wname, wdiff in workloads.items():
        chains_a = dc_by_wl.get(wname, {}).get(label_a, {})
        chains_b = dc_by_wl.get(wname, {}).get(label_b, {})
        idx_a = build_kernel_index(chains_a)
        idx_b = build_kernel_index(chains_b)
        b_specific = find_b_specific_kernels(chains_a, chains_b)
        a_kernel_bases = {
            _kernel_base(k["name"]) for k in wdiff.get("kernels", [])
            if k.get("side") in ("a_only", "both")
        }

        regressions = [
            k for k in wdiff.get("kernels", [])
            if k.get("side") != "a_only"
            and k.get("total_ms_b", 0) - k.get("total_ms_a", 0) > 0.01
        ]
        regressions.sort(key=lambda k: k["total_ms_b"] - k["total_ms_a"], reverse=True)
        regressions = regressions[:args.top]

        results = []
        for k in regressions:
            name = k["name"]
            parent_a = idx_a.get(name)
            parent_b = idx_b.get(name)
            extra = find_extra_ops(chains_a, chains_b, parent_a, parent_b)

            entry = {
                "kernel": _short(name),
                "kernel_full": name,
                "ms_a": k.get("total_ms_a", 0),
                "ms_b": k.get("total_ms_b", 0),
                "delta_ms": round(k["total_ms_b"] - k["total_ms_a"], 2),
                "parent_op_a": parent_a,
                "parent_op_b": parent_b,
            }

            if extra:
                entry["category"] = "FW"
                entry["extra_ops"] = extra
            elif name in b_specific or (k.get("side") == "b_only" and _kernel_base(name) not in a_kernel_bases):
                entry["category"] = "KN-B"
                report["ncu_kernels"].append(name)
            else:
                entry["category"] = "SHARED"

            # Attach dispatch chain with shapes for B
            chain_b = chains_b.get(parent_b, {}) if parent_b else {}
            if chain_b.get("children"):
                dispatch = []
                for child in chain_b["children"]:
                    step = {"op": child["op"], "dur_us": child.get("dur_us", 0)}
                    shapes = op_shapes.get(child["op"])
                    if shapes:
                        step["args"] = shapes
                    for call in child.get("calls", []):
                        if call.get("kernel"):
                            step["kernel"] = _short(call["kernel"], 60)
                    dispatch.append(step)
                entry["dispatch"] = dispatch

            results.append(entry)

        total_a = wdiff.get("total_ms_a", 0)
        total_b = wdiff.get("total_ms_b", 0)
        report["workloads"][wname] = {"total_ms_a": total_a, "total_ms_b": total_b, "regressions": results}

        print(f"\n  {wname}: {total_a:.1f}ms ({label_a}) → {total_b:.1f}ms ({label_b})")
        for r in results:
            tag = r["category"]
            extra_str = ""
            if r.get("extra_ops"):
                extra_str = " ← " + ", ".join(e["op"] for e in r["extra_ops"])
            print(f"    [{tag:>6}] +{r['delta_ms']:.2f}ms  {r['kernel']}{extra_str}")

    # Capture GPU info via torch (works in containers without nvidia-smi)
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        report["hardware"] = {
            "gpu": props.name,
            "compute_cap": f"{props.major}.{props.minor}",
            "memory_mb": props.total_memory // (1024 * 1024),
            "sm_count": props.multi_processor_count,
        }
    except Exception:
        pass

    # Deduplicate NCU targets
    report["ncu_kernels"] = sorted(set(report["ncu_kernels"]))
    report["legend"] = {
        "FW": "Framework issue — B has extra ops in its dispatch path that A doesn't. Fix the dispatch path.",
        "KN-B": "B-specific kernel — different kernel than A for the same op. NCU target for kernel optimization.",
        "SHARED": "Same kernel in both backends. Regression is not caused by the backend change.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    n_fw = sum(1 for w in report["workloads"].values() for r in w["regressions"] if r["category"] == "framework")
    n_knb = sum(1 for w in report["workloads"].values() for r in w["regressions"] if r["category"] == "kernel_b")
    n_sh = sum(1 for w in report["workloads"].values() for r in w["regressions"] if r["category"] == "shared")
    print(f"\n  Legend:")
    print(f"    [    FW] Framework — B has extra ops that A doesn't. Fix dispatch path.")
    print(f"    [  KN-B] B-specific kernel — different kernel than A. NCU target.")
    print(f"    [SHARED] Same kernel in both backends. Not caused by backend change.")
    print(f"\n  {n_fw} framework, {n_knb} kernel-B (NCU targets), {n_sh} shared")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
