#!/usr/bin/env python3
"""Extract full dispatch chain from nsys sqlite with NVTX annotations.

For each NVTX op, shows the complete chain:
  NVTX op → child ops → CUDA runtime calls → GPU kernels/memcpy/sync

How it works:

  nsys sqlite (with --pytorch=autograd-nvtx)
  │
  ├─ NVTX_EVENTS table
  │   └─ vllm::flashinfer_mm_fp4  (start=T0, end=T1, globalTid=X)
  │       ├─ aten::rand            (start>T0, end<T1, same globalTid)
  │       ├─ aten::to              (nested by time + thread)
  │       └─ aten::copy_           (nested by time + thread)
  │
  ├─ CUPTI_ACTIVITY_KIND_RUNTIME table
  │   └─ cudaLaunchKernel  (start in [child.start, child.end], globalTid=X)
  │      correlationId=1637533 ──────────────────────┐  (exact, not time-based)
  │                                                   │
  ├─ CUPTI_ACTIVITY_KIND_KERNEL table                 │
  │   └─ unrolled_elementwise_kernel  ◄───────────────┘  correlationId=1637533
  │
  └─ CUPTI_ACTIVITY_KIND_MEMCPY table
      └─ memcpy 41MB  ◄──── correlationId from cudaMemcpyAsync

  Linkage:
    NVTX parent → NVTX child:  same globalTid + time containment (strict nesting)
    NVTX child  → RUNTIME:     same globalTid + time containment (single thread)
    RUNTIME     → GPU:         correlationId (exact, queue-based)

Usage:
  python extract_dispatch_chain.py cutlass_prefill.sqlite -o dispatch_chain.json
  python extract_dispatch_chain.py cutlass_prefill.sqlite --op "flashinfer_mm"
  python extract_dispatch_chain.py a.sqlite b.sqlite --diff -o dispatch_diff.json
"""
import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


def _clean_op_name(text):
    text = re.sub(r',\s*op_id\s*=\s*\d+', '', text)
    text = re.sub(r',\s*seq\s*=\s*\d+', '', text).strip()
    return text


def _clean_runtime_name(name):
    return re.sub(r'_v\d+$', '', name)


def _short_kernel(name, maxlen=120):
    s = re.sub(r'void\s+', '', name)
    s = re.sub(r'<[^>]*>', '<…>', s)
    s = re.sub(r'<…>(<…>)', '<…>', s)
    if len(s) > maxlen:
        s = s[:maxlen - 3] + '...'
    return s


def extract_chains(sqlite_path, op_filter=None):
    conn = sqlite3.connect(sqlite_path)

    nvtx_count = conn.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0]
    if nvtx_count == 0:
        raise ValueError("No NVTX events — was nsys run with --pytorch=autograd-nvtx?")

    all_nvtx = conn.execute("""
        SELECT start, end, text, globalTid FROM NVTX_EVENTS
        WHERE text IS NOT NULL AND text != '' AND (end - start) > 1000
        ORDER BY start
    """).fetchall()

    top_ops = []
    for s, e, text, tid in all_nvtx:
        clean = _clean_op_name(text)
        if not (clean.startswith("vllm::") or clean.startswith("_C::") or
                clean.startswith("_moe_C::") or clean.startswith("aten::")):
            continue
        if op_filter and op_filter.lower() not in clean.lower():
            continue
        top_ops.append((s, e, clean, tid))

    op_groups = defaultdict(list)
    for s, e, name, tid in top_ops:
        op_groups[name].append((s, e, tid))

    chains = {}
    for op_name, instances in sorted(op_groups.items()):
        s, e, tid = max(instances, key=lambda x: x[1] - x[0])

        children = conn.execute("""
            SELECT start, end, text FROM NVTX_EVENTS
            WHERE start > ? AND end < ? AND globalTid = ?
            AND text IS NOT NULL AND text != ''
            ORDER BY start
        """, (s, e, tid)).fetchall()

        child_ops = []
        for cs, ce, ctext in children:
            cname = _clean_op_name(ctext)
            is_nested = any(
                cs > prev_s and ce < prev_e
                for prev_s, prev_e, _ in child_ops
            )
            if is_nested:
                continue

            runtime_calls = conn.execute("""
                SELECT s.value, r.correlationId, r."end" - r."start" as dur_ns
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds s ON r.nameId = s.id
                WHERE r.start >= ? AND r.start <= ? AND r.globalTid = ?
                ORDER BY r.start
            """, (cs, ce, tid)).fetchall()

            calls = []
            for rname, cid, dur_ns in runtime_calls:
                rclean = _clean_runtime_name(rname)
                call = {"api": rclean, "dur_us": round(dur_ns / 1e3, 1)}

                kern = conn.execute("""
                    SELECT s.value, k."end" - k."start"
                    FROM CUPTI_ACTIVITY_KIND_KERNEL k
                    JOIN StringIds s ON k.demangledName = s.id
                    WHERE k.correlationId = ?
                """, (cid,)).fetchone()
                if kern:
                    call["kernel"] = kern[0]
                    call["kernel_short"] = _short_kernel(kern[0])
                    call["kernel_us"] = round(kern[1] / 1e3, 1)

                mem = conn.execute("""
                    SELECT bytes, m."end" - m."start"
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
                    WHERE m.correlationId = ?
                """, (cid,)).fetchone()
                if mem:
                    call["memcpy_bytes"] = mem[0]
                    call["memcpy_us"] = round(mem[1] / 1e3, 1)

                calls.append(call)

            child_ops.append((cs, ce, {
                "op": cname,
                "dur_us": round((ce - cs) / 1e3, 1),
                "calls": calls,
            }))

        runtime_direct = conn.execute("""
            SELECT s.value, r.correlationId, r."end" - r."start" as dur_ns
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON r.nameId = s.id
            WHERE r.start >= ? AND r.start <= ? AND r.globalTid = ?
            ORDER BY r.start
        """, (s, e, tid)).fetchall()

        child_ranges = [(cs, ce) for cs, ce, _ in child_ops]
        direct_calls = []
        for rname, cid, dur_ns in runtime_direct:
            in_child = any(cs <= cid <= ce for cs, ce in child_ranges)
            if in_child:
                continue

        chain_entry = {
            "op": op_name,
            "instances": len(instances),
            "dur_us": round((e - s) / 1e3, 1),
            "children": [c[2] for c in child_ops],
        }

        total_kernel_us = 0
        total_calls = 0
        for child_data in chain_entry["children"]:
            for call in child_data.get("calls", []):
                total_calls += 1
                total_kernel_us += call.get("kernel_us", 0)
        chain_entry["total_kernel_us"] = round(total_kernel_us, 1)
        chain_entry["total_runtime_calls"] = total_calls

        chains[op_name] = chain_entry

    conn.close()
    return chains


def diff_chains(chains_a, chains_b, label_a="A", label_b="B"):
    all_ops = set(chains_a.keys()) | set(chains_b.keys())
    diff = {}
    for op in sorted(all_ops):
        ca = chains_a.get(op)
        cb = chains_b.get(op)
        entry = {"op": op}
        if ca and cb:
            entry["side"] = "both"
            entry[label_a] = ca
            entry[label_b] = cb
            children_a = {c["op"] for c in ca.get("children", [])}
            children_b = {c["op"] for c in cb.get("children", [])}
            if children_a != children_b:
                entry["divergent_children"] = {
                    f"only_{label_a}": sorted(children_a - children_b),
                    f"only_{label_b}": sorted(children_b - children_a),
                }
        elif ca:
            entry["side"] = f"{label_a}_only"
            entry[label_a] = ca
        else:
            entry["side"] = f"{label_b}_only"
            entry[label_b] = cb
        diff[op] = entry
    return diff


def print_chain(chains):
    for op_name, chain in chains.items():
        dur_ms = chain["dur_us"] / 1000
        print(f"\n{op_name}  ({dur_ms:.2f}ms, {chain['instances']}x)")
        for child in chain.get("children", []):
            calls_with_gpu = [c for c in child.get("calls", []) if "kernel" in c or "memcpy_bytes" in c]
            gpu_str = ""
            if calls_with_gpu:
                parts = []
                for c in calls_with_gpu:
                    if "kernel" in c:
                        parts.append(f"{c['kernel_short']} ({c['kernel_us']}us)")
                    elif "memcpy_bytes" in c:
                        parts.append(f"memcpy {c['memcpy_bytes']}B ({c['memcpy_us']}us)")
                gpu_str = " → " + ", ".join(parts[:2])
                if len(parts) > 2:
                    gpu_str += f" +{len(parts)-2} more"
            n_calls = len(child.get("calls", []))
            print(f"  {child['op']:<40} {child['dur_us']:>8.1f}us  {n_calls} calls{gpu_str}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sqlite", nargs="+", help="nsys sqlite(s) with NVTX annotations")
    parser.add_argument("--op", default=None, help="Filter ops by name substring")
    parser.add_argument("--diff", action="store_true", help="Diff two sqlites")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    if args.diff and len(args.sqlite) == 2:
        print(f"Loading {args.sqlite[0]}...")
        chains_a = extract_chains(args.sqlite[0], args.op)
        print(f"  {len(chains_a)} ops")
        print(f"Loading {args.sqlite[1]}...")
        chains_b = extract_chains(args.sqlite[1], args.op)
        print(f"  {len(chains_b)} ops")

        result = diff_chains(chains_a, chains_b, args.label_a, args.label_b)

        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2))
            print(f"Wrote {args.output}")
        else:
            for op, entry in result.items():
                print(f"\n{'='*60}")
                print(f"{op}  [{entry['side']}]")
                if entry.get("divergent_children"):
                    dc = entry["divergent_children"]
                    for k, v in dc.items():
                        if v:
                            print(f"  {k}: {v}")
                for label in [args.label_a, args.label_b]:
                    if label in entry:
                        print(f"\n  [{label}]")
                        print_chain({op: entry[label]})
    else:
        print(f"Loading {args.sqlite[0]}...")
        chains = extract_chains(args.sqlite[0], args.op)
        print(f"  {len(chains)} ops")

        if args.output:
            Path(args.output).write_text(json.dumps(chains, indent=2))
            print(f"Wrote {args.output}")
        else:
            print_chain(chains)


if __name__ == "__main__":
    main()
