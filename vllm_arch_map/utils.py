"""Shared utilities for A/B benchmark and nsys profiling scripts."""
import os
import shlex
import signal
import subprocess
import sys
import threading
import time


DEFAULT_PORT = 8000
SERVER_STARTUP_TIMEOUT = 600
READY_PATTERN = "Application startup complete"

DEFAULT_WORKLOADS = {
    "prefill": {"input-len": 1024, "output-len": 1, "num-prompts": 50},
    "decode":  {"input-len": 16,   "output-len": 256, "num-prompts": 50},
    "mix":     {"input-len": 512,  "output-len": 128, "num-prompts": 50},
}


# ── Server lifecycle ─────────────────────────────────────────────────

def build_server_cmd(model, common_flags, config_flags, port):
    cmd = ["vllm", "serve", model, "--port", str(port)]
    for key, val in {**common_flags, **config_flags}.items():
        flag = f"--{key}" if not key.startswith("--") else key
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])
    return cmd


def start_server(server_cmd):
    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        preexec_fn=os.setsid,
    )
    return proc


def start_server_nsys(server_cmd, nsys_output):
    nsys_cmd = ["nsys", "profile",
                "--pytorch=autograd-nvtx",
                "-o", str(nsys_output),
                "--trace", "cuda,nvtx",
                "--cuda-graph-trace=node",
                "--sample=none",
                "--force-overwrite=true"]
    full_cmd = nsys_cmd + server_cmd
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        preexec_fn=os.setsid,
    )
    return proc


def wait_ready(proc, timeout=SERVER_STARTUP_TIMEOUT):
    ready = threading.Event()
    died = threading.Event()

    def _read_stdout():
        for line in proc.stdout:
            text = line.decode("utf-8", errors="replace")
            sys.stdout.write(f"  [server] {text}")
            if READY_PATTERN in text:
                ready.set()
        died.set()

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()

    deadline = time.time() + timeout
    while not ready.is_set():
        if died.is_set() or proc.poll() is not None:
            raise RuntimeError("Server process exited before becoming ready")
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"Server not ready after {timeout}s")
        ready.wait(timeout=min(remaining, 1.0))

    return reader


def kill_server(proc):
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except Exception:
            pass


# ── Bench client ─────────────────────────────────────────────────────

def build_bench_cmd(model, workload, port):
    cmd = ["vllm", "bench", "serve",
           "--model", model,
           "--port", str(port),
           "--input-len", str(workload["input-len"]),
           "--output-len", str(workload["output-len"]),
           "--num-prompts", str(workload["num-prompts"])]
    if "dataset-name" in workload:
        cmd.extend(["--dataset-name", workload["dataset-name"]])
    return cmd


# ── Config parsing ───────────────────────────────────────────────────

def parse_cli_config(config_str):
    parts = config_str.split(":", 1)
    if len(parts) == 2:
        label = parts[0].strip()
        flags_str = parts[1].strip()
    else:
        label = "config"
        flags_str = config_str.strip()

    flags = {}
    tokens = shlex.split(flags_str)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags[key] = tokens[i + 1]
                i += 2
            else:
                flags[key] = True
                i += 1
        else:
            i += 1
    return label, flags


def parse_common_flags(flags_str):
    if not flags_str:
        return {}
    _, flags = parse_cli_config(f"_: {flags_str}")
    return flags


def parse_workload_arg(workload_str):
    """Parse 'name:input_len,output_len,num_prompts' or just 'name' (uses defaults)."""
    if ":" in workload_str:
        name, params = workload_str.split(":", 1)
        parts = params.split(",")
        if len(parts) != 3:
            raise ValueError(f"Workload '{workload_str}' must be name:input_len,output_len,num_prompts")
        return name.strip(), {
            "input-len": int(parts[0]),
            "output-len": int(parts[1]),
            "num-prompts": int(parts[2]),
        }
    name = workload_str.strip()
    if name not in DEFAULT_WORKLOADS:
        raise ValueError(f"Unknown workload '{name}'. Use prefill/decode/mix or name:in,out,n")
    return name, DEFAULT_WORKLOADS[name].copy()


def load_ab_config(args):
    """Parse config from --config JSON or CLI args. Returns (model, common_flags, label_a, flags_a, label_b, flags_b, workloads)."""
    if args.config:
        import json
        with open(args.config) as f:
            cfg = json.load(f)
        model = cfg["model"]
        common_flags = cfg.get("common_flags", {})
        configs = cfg["configs"]
        config_names = list(configs.keys())
        label_a, flags_a = config_names[0], configs[config_names[0]]
        label_b, flags_b = config_names[1], configs[config_names[1]]
        workloads = cfg.get("workloads", DEFAULT_WORKLOADS)
    else:
        model = args.model
        common_flags = parse_common_flags(args.common_flags)
        label_a, flags_a = parse_cli_config(args.config_a)
        label_b, flags_b = parse_cli_config(args.config_b)

        if args.workload:
            workloads = {}
            for w in args.workload:
                name, wl = parse_workload_arg(w)
                workloads[name] = wl
        else:
            workloads = DEFAULT_WORKLOADS

    return model, common_flags, label_a, flags_a, label_b, flags_b, workloads
