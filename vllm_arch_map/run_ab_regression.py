#!/usr/bin/env python3
"""A/B Regression Pipeline: nsys → dispatch chains → classify → NCU → report.

Steps:
  1. A/B benchmark metrics       (ab_bench.py → TTFT, TPOT, throughput)
  2. A/B nsys profiling          (ab_nsys.py)
  3. Extract dispatch chains     (extract_dispatch_chain.py)
  4. Profile B for shapes        (profile_serve.py + profiler_to_diagram.py)
  5. Classify regressions        (ab_regression.py)
  6. NCU on B-specific kernels   (ncu_profile.py, requires sudo)

Usage:
  python run_ab_regression.py --config ab_config.json -o traces/ab/

python run_ab_regression.py \
    --model nvidia/Qwen3-30B-A3B-NVFP4 \
    --config-a "cutlass: --linear-backend cutlass" \
    --config-b "b12x: --linear-backend flashinfer_b12x --moe-backend flashinfer_b12x --enforce-eager" \
    --common-flags "--trust-remote-code --gpu-memory-utilization 0.8" \
    --workload "prefill:1024,1,50" "decode:16,256,50" \
    --ncu \
    -o traces/ab/
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
import os

import csv

NCU_METRICS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "gpu__time_duration.sum": "duration_ns",
    "l1tex__throughput.avg.pct_of_peak_sustained_active": "l1_throughput_pct",
}


def parse_ncu_csv(csv_path):
    """Extract key metrics per kernel from NCU raw CSV (columnar format)."""
    kernels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        col_map = {}
        for col in reader.fieldnames:
            for metric_key, metric_name in NCU_METRICS.items():
                if metric_key in col:
                    col_map[col] = metric_name
                    break
        for row in reader:
            kname = row.get("Kernel Name", "")
            if not kname:
                continue
            if kname not in kernels:
                kernels[kname] = {}
            for col, name in col_map.items():
                try:
                    kernels[kname][name] = round(float(row[col]), 2)
                except (ValueError, TypeError):
                    pass

    for metrics in kernels.values():
        compute = metrics.get("compute_throughput_pct", 0)
        memory = metrics.get("memory_throughput_pct", 0)
        if memory > compute:
            metrics["bottleneck"] = "memory"
        elif max(compute, memory) < 20:
            metrics["bottleneck"] = "latency"
        else:
            metrics["bottleneck"] = "compute"
    return kernels

def run(cmd, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    print(f"  $ {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config-a", type=str, default=None)
    parser.add_argument("--config-b", type=str, default=None)
    parser.add_argument("--common-flags", type=str, default="")
    parser.add_argument("--workload", type=str, nargs="+", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--ncu", action="store_true", help="Run NCU on B-specific kernels")
    parser.add_argument("-o", "--output-dir", type=str, default="traces/ab")
    args = parser.parse_args()

    py = sys.executable
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    user_tmp = Path.home() / "tmp"
    user_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(user_tmp)

    # Build shared flags for ab_bench / ab_nsys (same args)
    shared_cmd = ["-o", str(output_dir), "--port", str(args.port)]
    
    if args.config:
        shared_cmd += ["--config", args.config]
    else:
        shared_cmd += ["--model", args.model, "--config-a", args.config_a,
                       "--config-b", args.config_b]
        if args.common_flags:
            shared_cmd += ["--common-flags", args.common_flags]
        if args.workload:
            shared_cmd += ["--workload"] + args.workload

    # Step 1: A/B benchmark (clean metrics, no profiling overhead)
    bench_cmd = [py, str(script_dir / "ab_bench.py")] + shared_cmd
    if not run(bench_cmd, "Step 1: A/B benchmark metrics (TTFT, TPOT, throughput)"):
        sys.exit(1)

    # Step 2: A/B nsys
    nsys_cmd = [py, str(script_dir / "ab_nsys.py")] + shared_cmd
    if not run(nsys_cmd, "Step 2: A/B nsys profiling"):
        sys.exit(1)

    # Step 3: Extract dispatch chains
    nsys_dir = output_dir / "nsys"
    for sqlite in sorted(nsys_dir.glob("*.sqlite")):
        name = sqlite.stem
        out_json = output_dir / f"dispatch_chain_{name}.json"
        run([py, str(script_dir / "extract_dispatch_chain.py"), str(sqlite), "-o", str(out_json)],
            f"Step 3: Extract dispatch chain → {name}")

    # Read labels from ab_nsys output
    ab_nsys_path = output_dir / "ab_nsys.json"
    if ab_nsys_path.exists():
        with open(ab_nsys_path) as f:
            ab_meta = json.load(f)
        label_a = ab_meta.get("label_a", "A")
        label_b = ab_meta.get("label_b", "B")
    else:
        ab_meta = {}
        label_a, label_b = "A", "B"

    # Step 4: Profile backend B for shapes
    profile_b_dir = output_dir / "profile_b"
    shapes_path = profile_b_dir / "per_request_profiles.json"
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        model = cfg["model"]
        configs = cfg["configs"]
        config_b_name = list(configs.keys())[1]
        config_b_flags = configs[config_b_name]
        common_flags_dict = cfg.get("common_flags", {})
    else:
        model = args.model
        config_b_flags = {}
        common_flags_dict = {}

    profile_cmd = [py, str(script_dir / "profile_serve.py"),
                   model, "--profile-dir", str(profile_b_dir),
                   "--num-prompts", "1",
                   "--input-len", "1024", "--output-len", "1"]
    # Pass backend B flags and common flags
    if args.config:
        for k, v in {**common_flags_dict, **config_b_flags}.items():
            flag = f"--{k}" if not k.startswith("--") else k
            if isinstance(v, bool) and v:
                profile_cmd.append(flag)
            elif not isinstance(v, bool):
                profile_cmd.extend([flag, str(v)])
    elif args.config_b:
        # Extract flags from "label: --flag1 val1 --flag2" format
        parts = args.config_b.split(":", 1)
        if len(parts) == 2:
            import shlex
            profile_cmd += shlex.split(parts[1].strip())
        if args.common_flags:
            import shlex
            profile_cmd += shlex.split(args.common_flags)

    run(profile_cmd, "Step 4: Profile B for shapes/dtypes")

    # Run profiler_to_diagram to get per_request_profiles.json
    if not shapes_path.exists():
        run([py, str(script_dir / "profiler_to_diagram.py"),
             "--per-request-dir", str(profile_b_dir),
             "--out_dir", str(profile_b_dir)],
            "Step 4b: Extract shapes from traces")

    # Step 5: Classify regressions
    kdiff_path = nsys_dir / "kernel_diff_all.json"
    report_path = output_dir / "ab_regression_report.json"
    if kdiff_path.exists():
        classify_cmd = [py, str(script_dir / "ab_regression.py"),
             "--kernel-diff", str(kdiff_path),
             "--dispatch-dir", str(output_dir),
             "--label-a", label_a, "--label-b", label_b,
             "--top", str(args.top),
             "-o", str(report_path)]
        if shapes_path.exists():
            classify_cmd += ["--shapes", str(shapes_path)]
        run(classify_cmd, "Step 5: Classify regressions")

    # Step 6: NCU on B-specific kernels only
    if args.ncu and report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        ncu_targets = report.get("ncu_kernels", [])
        if ncu_targets:
            filters = set()
            for name in ncu_targets:
                s = re.sub(r'^void\s+', '', name)
                base = re.match(r'([^<(]+)', s)
                base = base.group(1).strip() if base else s
                if len(base) > 80:
                    parts = re.findall(r'[A-Z][a-zA-Z0-9]{4,}', base)
                    if parts:
                        filters.add(parts[0])
                    else:
                        filters.add(base.split('_')[-1])
                elif '::' in base:
                    func = base.rsplit('::', 1)[-1]
                    if func:
                        filters.add(func)
                elif base:
                    filters.add(base)
            kernel_regex = "|".join(sorted(filters))
            model = ab_meta.get("model", args.model)

            if args.config:
                with open(args.config) as f:
                    cfg = json.load(f)
                workloads = cfg.get("workloads", {})
            elif args.workload:
                sys.path.insert(0, str(script_dir))
                from utils import parse_workload_arg
                workloads = {}
                for w in args.workload:
                    wname, wl = parse_workload_arg(w)
                    workloads[wname] = wl
            else:
                workloads = {"prefill": {"input-len": 1024, "output-len": 1}}

            import shutil
            ncu_path = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
            venv = os.environ.get("VIRTUAL_ENV", "")
            venv_python = f"{venv}/bin/python" if venv else py

            print(f"\n  NCU targets: {len(ncu_targets)} B-specific kernels")
            print(f"  Kernel regex: {kernel_regex}")
            print(f"  VIRTUAL_ENV: {venv}")
            print(f"  Python: {venv_python}")

            # Build backend B flags for ncu_profile.py
            b_flags = []
            if args.config:
                for k, v in {**common_flags_dict, **config_b_flags}.items():
                    flag = f"--{k}" if not k.startswith("--") else k
                    if isinstance(v, bool) and v:
                        b_flags.append(flag)
                    elif not isinstance(v, bool):
                        #val = "0.25" if flag == "--gpu-memory-utilization" else str(v)
                        b_flags.extend([flag, str(v)])

            elif args.config_b:
                import shlex
                parts = args.config_b.split(":", 1)
                if len(parts) == 2:
                    b_flags = shlex.split(parts[1].strip())
                if args.common_flags:
                    b_flags += shlex.split(args.common_flags)
            print(b_flags)
            ncu_profile_script = str(script_dir / "ncu_profile.py")

            ncu_t0 = time.time()
            for wname, wl in workloads.items():
                ncu_out = output_dir / f"ncu_{label_b}_{wname}"
                ncu_rep = Path(f"{ncu_out}.ncu-rep")
                if ncu_rep.exists():
                    print(f"\n  Skipping {wname} — {ncu_rep} already exists")
                    continue

                wl_t0 = time.time()
                input_len = wl.get("input-len", 1024)
                output_len = wl.get("output-len", 1)

                ncu_cmd = [
                    "systemd-run", "--user", "--scope", "-p", "MemoryMax=80G",
                    "sudo", "env",
                    f"HOME={str(Path.home())}",
                    f"PATH={venv}/bin:{os.environ.get('PATH', '')}",
                    f"VIRTUAL_ENV={venv}",
                    f"HF_HOME={os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))}",
                    f"TMPDIR={os.environ.get('TMPDIR', '/tmp')}",
                    f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '3')}",
                    ncu_path,
                    "--profile-from-start", "0",
                    "--target-processes", "application-only",
                    "--kernel-name", f"regex:{kernel_regex}",
                    "--launch-skip", "0",
                    "--launch-count", "10",
                    "--set", "full",
                    "-o", str(ncu_out),
                    venv_python, ncu_profile_script, model,
                    "--input-len", str(input_len),
                    "--output-len", str(output_len),
                ] + b_flags

                print(f"\n{'='*60}")
                print(f"  Step 6: NCU {label_b}/{wname} (input={input_len}, output={output_len})")
                print(f"{'='*60}")
                result = subprocess.run(ncu_cmd)
                if result.returncode != 0:
                    print(f"  WARNING: NCU {wname} failed (rc={result.returncode})")

                wl_elapsed = time.time() - wl_t0
                if ncu_rep.exists():
                    csv_path = output_dir / f"ncu_{label_b}_{wname}.csv"
                    subprocess.run([ncu_path, "--import", str(ncu_rep),
                                    "--csv", "--page", "raw"],
                                   stdout=open(csv_path, "w"))
                    print(f"  NCU CSV → {csv_path}")
                else:
                    print(f"  WARNING: {ncu_rep} not found")
                print(f"  NCU {wname} took {wl_elapsed/60:.1f} min")
            ncu_total = time.time() - ncu_t0
            print(f"\n  NCU total: {ncu_total/60:.1f} min")
            # Attach NCU metrics to report
            ncu_data = {}
            for wname in workloads:
                csv_path = output_dir / f"ncu_{label_b}_{wname}.csv"
                if csv_path.exists():
                    ncu_data[wname] = parse_ncu_csv(str(csv_path))
                    report.setdefault("ncu_csv", {})[wname] = str(csv_path)

            if ncu_data:
                ncu_summary = {}
                for wname, metrics_by_kernel in ncu_data.items():
                    for kname, metrics in metrics_by_kernel.items():
                        ncu_summary[f"{wname}/{kname[:80]}"] = metrics
                report["ncu_metrics"] = ncu_summary
                report_path.write_text(json.dumps(report, indent=2))
                print(f"\n  NCU metrics added to report ({len(ncu_summary)} kernel profiles)")
        else:
            print("\n  No B-specific kernels found — skipping NCU")

    print(f"\n{'='*60}")
    print(f"  Done! Report → {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
