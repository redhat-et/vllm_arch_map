#!/usr/bin/env python3
"""A/B benchmark: clean metrics for two vLLM backend configs.

Runs vLLM server + vllm bench serve for each config, captures TTFT, TPOT,
throughput, ITL for prefill/decoded/mixed. No profiling overhead.

Usage:
  # From JSON config
  python ab_bench.py --config ab_config.json -o traces/ab/

  # From CLI
  python ab_bench.py \
      --model nvidia/Qwen3-30B-A3B-NVFP4 \
      --config-a "cutlass: --linear-backend cutlass" \
      --config-b "b12x: --linear-backend flashinfer_b12x --moe-backend flashinfer_b12x --enforce-eager" \
      --common-flags "--trust-remote-code --gpu-memory-utilization 0.8" \
      --workload "prefill:1024,1,50" "decode:16,256,50" "mix:512,128,50" \
      -o traces/ab/

Config JSON format:
  {
    "model": "nvidia/Qwen3-30B-A3B-NVFP4",
    "common_flags": {
      "trust-remote-code": true,
      "gpu-memory-utilization": 0.8
    },
    "configs": {
      "cutlass": { "linear-backend": "cutlass" },
      "b12x": {
        "linear-backend": "flashinfer_b12x",
        "moe-backend": "flashinfer_b12x",
        "enforce-eager": true
      }
    },
    "workloads": {
      "prefill": {"input-len": 1024, "output-len": 1, "num-prompts": 50},
      "decode":  {"input-len": 16,   "output-len": 256, "num-prompts": 50},
      "mix":     {"input-len": 512,  "output-len": 128, "num-prompts": 50}
    }
  }
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

from utils import (
    DEFAULT_PORT, DEFAULT_WORKLOADS,
    build_server_cmd, start_server, wait_ready, kill_server,
    build_bench_cmd, load_ab_config,
)


def parse_bench_output(text):
    metrics = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, _, value = line.rpartition(":")
        name = name.strip()
        value = value.strip()
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def run_config(model, common_flags, config_flags, workloads, label, port):
    server_cmd = build_server_cmd(model, common_flags, config_flags, port)

    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"  Workloads: {list(workloads.keys())}")
    print(f"{'='*60}")

    proc = start_server(server_cmd)
    results = {}
    try:
        print(f"  Waiting for server...")
        wait_ready(proc)
        print(f"  Server ready.")

        for wname, workload in workloads.items():
            print(f"\n  [{wname}] input={workload['input-len']} "
                  f"output={workload['output-len']} prompts={workload['num-prompts']}")

            bench_cmd = build_bench_cmd(model, workload, port)
            result = subprocess.run(bench_cmd, capture_output=True, text=True, timeout=1800)
            bench_text = result.stdout + result.stderr
            metrics = parse_bench_output(bench_text)
            print(f"  [{wname}] done.")

            results[wname] = metrics
    finally:
        print(f"  Stopping server...")
        kill_server(proc)
        print(f"  Server stopped.")

    return results


def compare_metrics(metrics_a, metrics_b, label_a, label_b):
    comparison = {}
    all_keys = set(metrics_a) | set(metrics_b)
    for key in sorted(all_keys):
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if va == vb:
                delta_pct = 0.0
            elif va > 0 and math.isfinite(va) and math.isfinite(vb):
                delta_pct = (vb - va) / va * 100
            else:
                continue
            comparison[key] = {
                label_a: va,
                label_b: vb,
                "delta_pct": round(delta_pct, 1),
            }
    return comparison


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

    results_a = run_config(model, common_flags, flags_a, workloads, label_a, args.port)
    results_b = run_config(model, common_flags, flags_b, workloads, label_b, args.port)

    summary = {
        "model": model,
        "workloads": {},
        label_a: results_a,
        label_b: results_b,
    }

    for wname in workloads:
        summary["workloads"][wname] = {
            "config": workloads[wname],
            "comparison": compare_metrics(
                results_a[wname], results_b[wname], label_a, label_b),
        }

    summary_path = output_dir / "ab_bench.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults → {summary_path}")

    print(f"\nMetrics comparison:")
    for wname, wr in summary["workloads"].items():
        mc = wr.get("comparison", {})
        if mc:
            print(f"\n  [{wname}]")
            for key, vals in mc.items():
                va, vb = vals[label_a], vals[label_b]
                if vals["delta_pct"] == 0.0:
                    print(f"    {key}: {va} → {vb} (match)")
                else:
                    is_throughput = "throughput" in key.lower()
                    if is_throughput:
                        direction = "better" if vals["delta_pct"] > 0 else "worse"
                    else:
                        direction = "better" if vals["delta_pct"] < 0 else "worse"
                    print(f"    {key}: {va} → {vb} "
                          f"({vals['delta_pct']:+.1f}% {direction})")


if __name__ == "__main__":
    main()
