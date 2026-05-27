#!/usr/bin/env python3
"""
Addendum A: Free-run generation for divergence analysis.

Generates max_tokens tokens from a given model using the same 8 WikiText
prompts from Exp 11.  No teacher-forcing — pure autoregressive free run
at temperature=0, seed=42, top_p=1.0 (greedy equivalent).

Saves:
  <output>.json   per-prompt generated token id sequences + decoded text
  <output>.txt    human-readable text diffs for quick inspection

Usage:
  python3 free_run_generate.py \
    --label bf16-freerun \
    --model Qwen/Qwen3.6-27B \
    --token-source /sweep/refs/wikitext-tokens-multi.safetensors \
    --output /sweep/addenda/A-free-run-divergence/results/bf16-freerun.json \
    --max-tokens 512 \
    [--quantization none] \
    [--speculative-config '{"method":"mtp","num_speculative_tokens":3}'] \
    [--enforce-eager] \
    [--language-model-only] \
    [...vllm engine flags...]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import safetensors.torch
import torch
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--token-source", required=True,
                   help="Safetensors with 2-D token array (num_prompts x prompt_len)")
    p.add_argument("--output", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    # vllm engine flags
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--load-format", default="auto")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--quantization", default="none")
    p.add_argument("--attention-backend", default="FLASHINFER")
    p.add_argument("--moe-backend", default="auto")
    p.add_argument("--speculative-config", default=None,
                   help="JSON string, e.g. '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--disable-custom-all-reduce", action="store_true")
    p.add_argument("--hf-overrides", default="{}")
    return p.parse_args()


def load_prompts(token_source: str):
    """Load multi-prompt token array from safetensors. Returns list of token lists."""
    st = safetensors.torch.load_file(token_source)
    # Expect key 'tokens' with shape (num_prompts, prompt_len)
    key = "tokens"
    if key not in st:
        key = list(st.keys())[0]
    tokens = st[key]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    return [tokens[i].tolist() for i in range(tokens.shape[0])]


def build_llm(args):
    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        load_format=args.load_format,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        quantization=None if args.quantization == "none" else args.quantization,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        seed=args.seed,
    )
    if args.language_model_only:
        kwargs["language_model_only"] = True
    if args.speculative_config:
        kwargs["speculative_config"] = json.loads(args.speculative_config)
    if args.hf_overrides and args.hf_overrides.strip() not in ("{}", ""):
        kwargs["hf_overrides"] = json.loads(args.hf_overrides)
    # Backend overrides via env
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend)
    return LLM(**kwargs)


def main():
    args = parse_args()
    print(f"[free_run_generate] label={args.label} model={args.model} max_tokens={args.max_tokens}")

    prompt_tokens = load_prompts(args.token_source)
    print(f"  loaded {len(prompt_tokens)} prompts, len={[len(p) for p in prompt_tokens]}")

    llm = build_llm(args)
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
        max_tokens=args.max_tokens,
        ignore_eos=True,   # fixed-length rollouts for clean comparison
        # No logprobs needed for free-run — just tokens
    )

    t0 = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": toks} for toks in prompt_tokens],
        sampling_params=sampling,
        use_tqdm=True,
    )
    elapsed = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  generated {total_tokens} tokens in {elapsed:.1f}s  ({total_tokens/elapsed:.1f} tok/s)")

    results = {
        "label": args.label,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "kv_cache_dtype": args.kv_cache_dtype,
        "quantization": args.quantization,
        "speculative_config": args.speculative_config,
        "elapsed_s": round(elapsed, 2),
        "tok_per_s": round(total_tokens / elapsed, 2),
        "prompts": [],
    }

    for i, out in enumerate(outputs):
        gen_ids = list(out.outputs[0].token_ids)
        gen_text = out.outputs[0].text
        prompt_text = tokenizer.decode(prompt_tokens[i], skip_special_tokens=True)
        results["prompts"].append({
            "prompt_idx": i,
            "prompt_len": len(prompt_tokens[i]),
            "prompt_text_snippet": prompt_text[:120] + "..." if len(prompt_text) > 120 else prompt_text,
            "generated_token_ids": gen_ids,
            "generated_text": gen_text,
            "generated_len": len(gen_ids),
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
