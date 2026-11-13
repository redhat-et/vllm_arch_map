#!/usr/bin/env python3
"""A/B nsys profiling: per-workload kernel breakdown for two vLLM backend configs.

Runs one server+nsys cycle per workload per config, producing clean per-workload
sqlite files with no timestamp-based splitting needed.

Output files (in nsys/ subdir):
  {label}_{workload}.nsys-rep / .sqlite   — raw nsys trace per workload
  kernel_diff_{workload}.json             — A/B kernel diff per workload
  kernel_diff_all.json                    — combined with workload key

Metrics from nsys runs are NOT representative (profiling overhead).
Use ab_bench.py for clean metrics.

Usage:
  # From JSON config
  python ab_nsys.py --config ab_config.json -o traces/ab/

  # From CLI
  python ab_nsys.py \
      --model nvidia/Qwen3-30B-A3B-NVFP4 \
      --config-a "cutlass: --linear-backend cutlass" \
      --config-b "b12x: --linear-backend flashinfer_b12x --moe-backend flashinfer_b12x --enforce-eager" \
      --common-flags "--trust-remote-code --gpu-memory-utilization 0.8" \
      --workload "prefill:1024,1,50" "decode:16,256,50" "mix:512,128,50" \
      -o traces/ab/

Config JSON format: same as ab_bench.py (see ab_bench.py --help).
"""
import argparse
import json
import subprocess
from pathlib import Path

from utils import (
    DEFAULT_PORT, DEFAULT_WORKLOADS,
    build_server_cmd, start_server_nsys, wait_ready, kill_server,
    build_bench_cmd, load_ab_config,
)


def export_nsys_sqlite(nsys_rep_path):
    sqlite_path = Path(str(nsys_rep_path).replace(".nsys-rep", ".sqlite"))
    if sqlite_path.exists():
        sqlite_path.unlink()
    subprocess.run(
        ["nsys", "export", "--type=sqlite", "--output", str(sqlite_path),
         str(nsys_rep_path)],
        check=True,
        capture_output=True,
    )
    return sqlite_path


def run_workload(model, common_flags, config_flags, workload_name, workload,
                 output_dir, label, port):
    """Run one workload under its own nsys trace. Returns sqlite path or None."""
    server_cmd = build_server_cmd(model, common_flags, config_flags, port)
    nsys_dir = output_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    nsys_output = nsys_dir / f"{label}_{workload_name}"

    print(f"\n  [{label}/{workload_name}] input={workload['input-len']} "
          f"output={workload['output-len']} prompts={workload['num-prompts']}")

    proc = start_server_nsys(server_cmd, nsys_output)
    try:
        print(f"  [{label}/{workload_name}] Waiting for server...")
        wait_ready(proc)
        if proc.poll() is not None:
            print(f"  ERROR [{label}/{workload_name}] Server died before bench (rc={proc.returncode})")
            return None
        print(f"  [{label}/{workload_name}] Server ready, running bench...")
        bench_cmd = build_bench_cmd(model, workload, port)
        result = subprocess.run(bench_cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"  WARNING [{label}/{workload_name}] bench failed (rc={result.returncode}):")
            for line in (result.stderr or result.stdout or "").strip().splitlines()[-10:]:
                print(f"    {line}")
        else:
            print(f"  [{label}/{workload_name}] Bench done.")
    finally:
        print(f"  [{label}/{workload_name}] Stopping server (nsys finalizing)...")
        kill_server(proc)

    nsys_rep = Path(f"{nsys_output}.nsys-rep")
    if not nsys_rep.exists():
        print(f"  WARNING: {nsys_rep} not found")
        return None

    print(f"  [{label}/{workload_name}] Exporting sqlite...")
    sqlite_path = export_nsys_sqlite(nsys_rep)
    print(f"  [{label}/{workload_name}] → {sqlite_path}")
    return str(sqlite_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config-a", type=str, default=None,
                        help='"label: --flag1 val1 --flag2"')
    parser.add_argument("--config-b", type=str, default=None,
                        help='"label: --flag1 val1 --flag2"')
    parser.add_argument("--common-flags", type=str, default="")
    parser.add_argument("--workload", type=str, nargs="+", default=None,
                        help='"prefill:1024,1,50" or just "decode" for defaults')
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("-o", "--output-dir", type=str, default="traces/ab")
    args = parser.parse_args()

    if not args.config and (not args.model or not args.config_a or not args.config_b):
        parser.error("--model, --config-a, --config-b required without --config")

    model, common_flags, label_a, flags_a, label_b, flags_b, workloads = load_ab_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model}")
    print(f"Config A [{label_a}]: {flags_a}")
    print(f"Config B [{label_b}]: {flags_b}")
    print(f"Common: {common_flags}")
    print(f"Workloads: {list(workloads.keys())}")
    print(f"Output: {output_dir}")
    print(f"Strategy: separate nsys trace per workload "
          f"({len(workloads) * 2} server starts)")

    from diff_kernels import load_kernels, merge, write_json, load_gpu_overhead

    all_diffs = {}
    all_results = {
        "model": model, "label_a": label_a, "label_b": label_b,
        "workloads": {},
    }

    for wname, workload in workloads.items():
        print(f"\n{'='*60}")
        print(f"  Workload: {wname}")
        print(f"{'='*60}")

        sqlite_a = run_workload(model, common_flags, flags_a, wname, workload,
                                output_dir, label_a, args.port)
        sqlite_b = run_workload(model, common_flags, flags_b, wname, workload,
                                output_dir, label_b, args.port)

        wl_result = {
            "nsys_sqlite_a": sqlite_a,
            "nsys_sqlite_b": sqlite_b,
        }

        if sqlite_a and sqlite_b:
            kernels_a = load_kernels(sqlite_a)
            kernels_b = load_kernels(sqlite_b)
            rows = merge(kernels_a, kernels_b)
            diff = write_json(rows, kernels_a, kernels_b, label_a, label_b)

            overhead_a = load_gpu_overhead(sqlite_a)
            overhead_b = load_gpu_overhead(sqlite_b)
            if overhead_a:
                diff["overhead_a"] = overhead_a
            if overhead_b:
                diff["overhead_b"] = overhead_b

            diff_path = output_dir / "nsys" / f"kernel_diff_{wname}.json"
            diff_path.write_text(json.dumps(diff, indent=2))
            wl_result["kernel_diff_path"] = str(diff_path)

            total_a = diff["total_ms_a"]
            total_b = diff["total_ms_b"]
            if total_a > 0:
                delta = (total_b - total_a) / total_a * 100
                print(f"\n  [{wname}] GPU kernel time: {total_a:.1f}ms ({label_a}) → "
                      f"{total_b:.1f}ms ({label_b}) [{delta:+.1f}%]")

            print(f"  [{wname}] Top 5 kernel changes:")
            for k in diff["kernels"][:5]:
                name = k["name"][:50]
                delta_str = f"{k['delta_pct']:+.1f}%" if k["delta_pct"] is not None else "new"
                side = f" [{k['side']}]" if k["side"] != "both" else ""
                print(f"    {delta_str:>8}  {k['total_ms_a']:.1f}→{k['total_ms_b']:.1f}ms  "
                      f"{name}{side}")

            all_diffs[wname] = diff

        all_results["workloads"][wname] = wl_result

    if all_diffs:
        combined = {
            "label_a": label_a,
            "label_b": label_b,
            "workloads": all_diffs,
        }
        combined_path = output_dir / "nsys" / "kernel_diff_all.json"
        combined_path.write_text(json.dumps(combined, indent=2))
        all_results["kernel_diff_all_path"] = str(combined_path)
        print(f"\nCombined kernel diff → {combined_path}")

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for wname, diff in all_diffs.items():
        total_a = diff["total_ms_a"]
        total_b = diff["total_ms_b"]
        delta = (total_b - total_a) / total_a * 100 if total_a > 0 else 0
        print(f"  {wname:>8}: {total_a:.1f}ms ({label_a}) → "
              f"{total_b:.1f}ms ({label_b}) [{delta:+.1f}%]")

    summary_path = output_dir / "ab_nsys.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
