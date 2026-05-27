#!/usr/bin/env python3
"""Pre-tokenize WikiText-2-raw-v1 test split into safetensors format
that the GLM scripts can consume via --token-source.

The GLM scripts try to load WikiText themselves but `datasets` is
not installed in repne/vllm:v13. Running tokenization on host and
mounting the result is cleaner than rebuilding the container image.

Output schema:
    prompt_token_ids:        (num_prompts, prompt_len) int64
    teacher_force_token_ids: (num_prompts, max_tokens) int64

This matches what _load_token_source() in decode_logprob_kld{,_multi}.py expects.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--prompt-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    args = p.parse_args()

    print(f"Loading {args.dataset}/{args.config} {args.split}...")
    ds = load_dataset(args.dataset, args.config, split=args.split)
    text = "\n\n".join(x["text"] for x in ds if x.get("text"))
    print(f"  loaded {len(ds)} rows, {len(text):,} chars")

    print(f"Tokenizing with {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"  tokenized to {len(ids):,} tokens")

    stride = args.prompt_len + args.max_tokens
    needed = args.num_prompts * stride
    if len(ids) < needed:
        print(
            f"ERROR: WikiText test split has {len(ids)} tokens but need "
            f"{needed} ({args.num_prompts} prompts × {stride} tokens each).",
            file=sys.stderr,
        )
        return 1

    prompts: list[list[int]] = []
    teachers: list[list[int]] = []
    for i in range(args.num_prompts):
        start = i * stride
        prompts.append(ids[start : start + args.prompt_len])
        teachers.append(ids[start + args.prompt_len : start + stride])

    pt = torch.tensor(prompts, dtype=torch.int64)
    tt = torch.tensor(teachers, dtype=torch.int64)
    print(f"  prompts shape: {tuple(pt.shape)}")
    print(f"  teachers shape: {tuple(tt.shape)}")

    # Sanity: verify all prompts are different
    for i in range(args.num_prompts):
        for j in range(i + 1, args.num_prompts):
            if torch.equal(pt[i], pt[j]):
                print(
                    f"WARNING: prompt {i} == prompt {j} (duplicate prompts!)",
                    file=sys.stderr,
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Schema: single script expects 1-D, multi script expects 2-D.
    # Switch based on num_prompts.
    if args.num_prompts == 1:
        save_file(
            {
                "prompt_token_ids": pt[0],          # (prompt_len,)
                "teacher_force_token_ids": tt[0],   # (max_tokens,)
            },
            str(out),
        )
    else:
        save_file(
            {
                "prompt_token_ids": pt,             # (num_prompts, prompt_len)
                "teacher_force_token_ids": tt,      # (num_prompts, max_tokens)
            },
            str(out),
        )
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
