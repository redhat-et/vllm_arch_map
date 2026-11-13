#!/usr/bin/env python3
"""Extract op→kernel structural map from an nsys sqlite with NVTX annotations.

Requires nsys capture with --pytorch=autograd-nvtx.
One capture is enough — the structure is static for a given model.

Usage:
  python extract_structural_map.py cutlass_prefill.sqlite -o structural_map.json
"""
import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


def extract_map(sqlite_path):
    conn = sqlite3.connect(sqlite_path)

    nvtx_count = conn.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0]
    if nvtx_count == 0:
        raise ValueError("No NVTX events — was nsys run with --pytorch=autograd-nvtx?")

    all_ops = conn.execute("""
        SELECT text, start, end
        FROM NVTX_EVENTS
        WHERE (end - start) > 1000
        AND text IS NOT NULL AND text != ''
        ORDER BY start
    """).fetchall()

    op_groups = defaultdict(list)
    for text, s, e in all_ops:
        clean = re.sub(r',\s*op_id\s*=\s*\d+', '', text)
        clean = re.sub(r',\s*seq\s*=\s*\d+', '', clean).strip()
        op_groups[clean].append((s, e))

    structural_map = {}
    for op_name, instances in sorted(op_groups.items()):
        s, e = max(instances, key=lambda x: x[1] - x[0])

        kernels = conn.execute("""
            SELECT s_d.value as name, (k.end - k.start) as dur_ns
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON r.correlationId = k.correlationId
            JOIN StringIds s_d ON k.demangledName = s_d.id
            WHERE r.start >= ? AND r.start <= ?
            AND r.nameId IN (SELECT id FROM StringIds WHERE value LIKE 'cudaLaunchKernel%')
            ORDER BY k.start
        """, (s, e)).fetchall()

        if kernels:
            kernel_names = list(dict.fromkeys(k[0] for k in kernels))
            structural_map[op_name] = kernel_names

    conn.close()
    return structural_map


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sqlite", help="nsys sqlite with NVTX annotations")
    parser.add_argument("-o", "--output", default="structural_map.json")
    args = parser.parse_args()

    print(f"Loading {args.sqlite}...")
    smap = extract_map(args.sqlite)

    vllm_ops = {k: v for k, v in smap.items()
                if k.startswith("vllm::") or k.startswith("_C::") or k.startswith("_moe_C::")}
    aten_ops = {k: v for k, v in smap.items() if k.startswith("aten::")}
    other_ops = {k: v for k, v in smap.items()
                 if k not in vllm_ops and k not in aten_ops}

    output = {
        "type": "structural_map",
        "source": str(args.sqlite),
        "vllm_ops": vllm_ops,
        "aten_ops": aten_ops,
        "other_ops": other_ops,
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote {args.output}")
    print(f"  vllm ops: {len(vllm_ops)} ({sum(len(v) for v in vllm_ops.values())} kernel types)")
    print(f"  aten ops: {len(aten_ops)} ({sum(len(v) for v in aten_ops.values())} kernel types)")
    print(f"  other:    {len(other_ops)} ({sum(len(v) for v in other_ops.values())} kernel types)")

    for op, kerns in sorted(vllm_ops.items()):
        print(f"\n  {op}: {len(kerns)} kernels")
        for k in kerns:
            print(f"    {k[:100]}")


if __name__ == "__main__":
    main()
