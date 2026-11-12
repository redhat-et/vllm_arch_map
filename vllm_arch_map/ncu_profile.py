#!/usr/bin/env python3
"""Prepare a vLLM workload for ncu/nsys kernel profiling.

Uses the same dataset API as `vllm bench serve` so profiled prompts
match your benchmark exactly. Runs warmup outside the profiling window,
then captures target inference inside cudaProfilerStart/Stop.

Usage:
  # Prefill — 1024 random tokens in, 1 token out
  python ncu_profile.py Qwen/Qwen3-8B --input-len 1024 --output-len 1

  # Decode — short prefill, 128 decode steps
  python ncu_profile.py Qwen/Qwen3-8B --input-len 32 --output-len 128

  # FlashInfer backend
  python ncu_profile.py Qwen/Qwen3-8B --attention-backend FLASHINFER --input-len 1024 --output-len 1

  # Use workload data from benchmark
  python ncu_profile.py Qwen/Qwen3-8B --workload-json samples/workload_data.json

Wrap with ncu:
  PYTHON_PATH=$(which python)
  systemd-run --user --scope -p MemoryMax=80G \\
  sudo /usr/local/cuda/bin/ncu \\
    --profile-from-start 0 --target-processes application-only \\
    --kernel-name "regex:flash_fwd_splitkv|elementwise_kernel" \\
    --set full --nvtx -f \\
    -o ~/lkesem/vllm_arch_map/traces/ab/ncu_fa_prefill \\
    $PYTHON_PATH ncu_profile.py Qwen/Qwen3-8B \\
    --attention-backend FLASH_ATTN --input-len 1024 --output-len 1
"""
import argparse
import json
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _pick_representatives(workload_path):
    with open(workload_path) as f:
        data = json.load(f)
    requests = data.get("requests", [])
    if not requests:
        raise ValueError(f"No requests in {workload_path}")

    by_latency = sorted(requests, key=lambda r: r.get("latency_ms", 0))
    n = len(by_latency)
    return [
        {"label": "Fast (p10)", "input_tokens": by_latency[max(0, int(n * 0.1))].get("input_tokens", 256),
         "output_tokens": by_latency[max(0, int(n * 0.1))].get("output_tokens", 64)},
        {"label": "Typical (p50)", "input_tokens": by_latency[n // 2].get("input_tokens", 512),
         "output_tokens": by_latency[n // 2].get("output_tokens", 128)},
        {"label": "Slow (p90)", "input_tokens": by_latency[min(n - 1, int(n * 0.9))].get("input_tokens", 768),
         "output_tokens": by_latency[min(n - 1, int(n * 0.9))].get("output_tokens", 160)},
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--attention-backend", type=str, default=None,
                        help="FLASHINFER, FLASH_ATTN, etc.")
    parser.add_argument("--warmup-tokens", type=int, default=2,
                        help="Warmup output tokens (outside profiling window)")
    parser.add_argument("--workload-json", type=str, default=None,
                        help="Path to workload_data.json — auto-picks representatives")

    # Dataset args — same as vllm bench serve
    from vllm.benchmarks.datasets import add_dataset_parser
    add_dataset_parser(parser)
    parser.set_defaults(num_prompts=1, dataset_name="random")

    # Convenience aliases
    parser.add_argument("--input-len", type=int, default=None,
                        help="Maps to --random-input-len")
    parser.add_argument("--output-len", type=int, default=None,
                        help="Maps to --random-output-len")

    args = parser.parse_args()

    if args.input_len is not None:
        for attr in ('random_input_len', 'sonnet_input_len'):
            if hasattr(args, attr):
                setattr(args, attr, args.input_len)
    if args.output_len is not None:
        for attr in ('random_output_len', 'sonnet_output_len', 'sharegpt_output_len',
                      'custom_output_len', 'hf_output_len'):
            if hasattr(args, attr):
                setattr(args, attr, args.output_len)

    import torch
    from vllm import LLM, SamplingParams

    if args.attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    llm_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(f"[ncu_profile] Loading {args.model}...")
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Build prompts
    if args.workload_json:
        representatives = _pick_representatives(args.workload_json)
        prompts = []
        max_tokens_per_req = []
        for rep in representatives:
            from vllm.benchmarks.datasets import get_samples
            args.random_input_len = rep["input_tokens"]
            args.random_output_len = rep["output_tokens"]
            args.num_prompts = 1
            if not hasattr(args, 'backend'):
                args.backend = "openai-completions"
            samples = get_samples(args, tokenizer)
            prompts.append(samples[0].prompt if isinstance(samples[0].prompt, str) else str(samples[0].prompt))
            max_tokens_per_req.append(rep["output_tokens"])
            actual = len(tokenizer.encode(prompts[-1]))
            print(f"[ncu_profile] {rep['label']}: {actual} input → {rep['output_tokens']} output tokens")
    else:
        from vllm.benchmarks.datasets import get_samples
        if not hasattr(args, 'backend'):
            args.backend = "openai-completions"
        samples = get_samples(args, tokenizer)
        prompts = [s.prompt if isinstance(s.prompt, str) else str(s.prompt)
                   for s in samples[:args.num_prompts]]
        out_len = args.output_len or getattr(args, 'random_output_len', 32) or 32
        max_tokens_per_req = [out_len] * len(prompts)
        for i, p in enumerate(prompts, 1):
            toks = len(tokenizer.encode(p))
            print(f"[ncu_profile] Prompt {i}: {toks} input → {out_len} output tokens")

    # Warmup outside profiling window
    print("[ncu_profile] Warmup...")
    llm.generate(["Hello"], SamplingParams(max_tokens=args.warmup_tokens, temperature=0))
    print("[ncu_profile] Warmup done.")

    # Profile window
    print("[ncu_profile] cudaProfilerStart — ncu capture begins")
    torch.cuda.cudart().cudaProfilerStart()

    params = [SamplingParams(max_tokens=mt, temperature=0) for mt in max_tokens_per_req]
    outputs = llm.generate(prompts, params)

    torch.cuda.cudart().cudaProfilerStop()
    print("[ncu_profile] cudaProfilerStop — ncu capture ends")

    for i, out in enumerate(outputs):
        text = out.outputs[0].text[:60].replace("\n", " ")
        print(f"[ncu_profile] Output {i+1}: {text}...")

    backend = args.attention_backend or "default"
    input_len = args.input_len or getattr(args, 'random_input_len', '?')
    output_len = args.output_len or getattr(args, 'random_output_len', '?')
    print(f"[ncu_profile] Done. Model={args.model}, backend={backend}, "
          f"batch={len(prompts)}, input={input_len}, output={output_len}")


if __name__ == "__main__":
    main()
