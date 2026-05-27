#!/usr/bin/env python3
"""Collect and compare teacher-forced decode full-vocab KLD for many prompts."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


DEFAULT_HF_OVERRIDES = {
    "index_topk_pattern": (
        "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSS"
        "FSFFFSFSSSFSFFSFFSSS"
    )
}


def _make_token_ids(model: str, min_len: int) -> list[int]:
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(x["text"] for x in ds if x.get("text"))
    except Exception:
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This fallback prompt is only used when the WikiText cache is "
            "unavailable. "
        ) * 16384

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < min_len:
        reps = (min_len // max(1, len(ids))) + 1
        ids = (ids * reps)[:min_len]
    return ids[:min_len]


def _load_token_source(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = load_file(path)
    prompt_ids = tensors["prompt_token_ids"].to(torch.int64)
    teacher_force_ids = tensors["teacher_force_token_ids"].to(torch.int64)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if teacher_force_ids.ndim == 1:
        teacher_force_ids = teacher_force_ids.unsqueeze(0)
    return prompt_ids, teacher_force_ids


def _make_prompts(
    *,
    model: str,
    num_prompts: int,
    prompt_len: int,
    max_tokens: int,
    token_source: str | None,
) -> tuple[list[list[int]], list[list[int]]]:
    if token_source:
        prompt_tensor, teacher_tensor = _load_token_source(token_source)
        if prompt_tensor.shape[0] < num_prompts:
            raise ValueError(
                f"token source has {prompt_tensor.shape[0]} prompts, "
                f"need {num_prompts}"
            )
        if prompt_tensor.shape[1] != prompt_len:
            raise ValueError(
                f"token source prompt length {prompt_tensor.shape[1]} != {prompt_len}"
            )
        if teacher_tensor.shape[1] < max_tokens:
            raise ValueError(
                f"token source teacher length {teacher_tensor.shape[1]} < {max_tokens}"
            )
        prompts = prompt_tensor[:num_prompts].tolist()
        teachers = teacher_tensor[:num_prompts, :max_tokens].tolist()
        return (
            [[int(x) for x in row] for row in prompts],
            [[int(x) for x in row] for row in teachers],
        )

    stride = prompt_len + max_tokens
    ids = _make_token_ids(model, num_prompts * stride)
    prompts = []
    teachers = []
    for i in range(num_prompts):
        start = i * stride
        prompts.append(ids[start : start + prompt_len])
        teachers.append(ids[start + prompt_len : start + stride])
    return prompts, teachers


def _vocab_size_from_llm(llm: Any, logprobs_obj: Any) -> int:
    model_config = llm.llm_engine.model_config
    if hasattr(model_config, "get_vocab_size"):
        return int(model_config.get_vocab_size())
    vocab_size = getattr(model_config, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    return int(max(logprobs_obj.token_ids) + 1)


def _position_to_dense(logprobs_obj: Any, pos: int, vocab_size: int) -> torch.Tensor:
    dense = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    if hasattr(logprobs_obj, "start_indices"):
        start = logprobs_obj.start_indices[pos]
        end = logprobs_obj.end_indices[pos]
        token_ids = torch.tensor(logprobs_obj.token_ids[start:end], dtype=torch.long)
        values = torch.tensor(logprobs_obj.logprobs[start:end], dtype=torch.float32)
        valid = (token_ids >= 0) & (token_ids < vocab_size)
        dense[token_ids[valid]] = values[valid]
        return dense

    one_pos = logprobs_obj[pos]
    for token_id, lp in one_pos.items():
        if 0 <= int(token_id) < vocab_size:
            dense[int(token_id)] = float(lp.logprob)
    return dense


def _llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    hf_overrides = json.loads(args.hf_overrides) if args.hf_overrides else {}
    kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "load_format": args.load_format,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max(
            args.prompt_len + args.max_tokens + 16, args.max_model_len
        ),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "attention_backend": args.attention_backend,
        "max_logprobs": -1,
        "disable_log_stats": True,
    }
    if args.quantization.lower() not in ("", "none", "null"):
        kwargs["quantization"] = args.quantization
    if args.moe_backend.lower() not in ("", "auto", "none", "null"):
        kwargs["moe_backend"] = args.moe_backend
    if args.cpu_offload_gb is not None:
        kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    if args.teacher_force:
        kwargs["logits_processors"] = [
            "teacher_force_logits_processor:TeacherForceLogitsProcessor"
        ]
    if hf_overrides:
        kwargs["hf_overrides"] = hf_overrides
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    if args.speculative_config:
        kwargs["speculative_config"] = json.loads(args.speculative_config)
    if args.language_model_only:
        kwargs["language_model_only"] = True
    return kwargs


def collect(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    prompts, teachers = _make_prompts(
        model=args.model,
        num_prompts=args.num_prompts,
        prompt_len=args.prompt_len,
        max_tokens=args.max_tokens,
        token_source=args.token_source,
    )
    print(
        "collect_start",
        json.dumps(
            {
                "label": args.label,
                "model": args.model,
                "num_prompts": args.num_prompts,
                "prompt_len": args.prompt_len,
                "max_tokens": args.max_tokens,
                "teacher_force": args.teacher_force,
                "token_source": args.token_source,
                "env": {
                    "VLLM_B12X_FORCE_MOE_A16": os.getenv(
                        "VLLM_B12X_FORCE_MOE_A16"
                    ),
                    "B12X_MOE_FORCE_A16": os.getenv("B12X_MOE_FORCE_A16"),
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    t0 = time.time()
    llm = LLM(**_llm_kwargs(args))
    all_rows = []
    all_generated = []
    vocab_size = None
    for i, (prompt_ids, teacher_ids) in enumerate(zip(prompts, teachers)):
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=args.max_tokens,
            min_tokens=args.max_tokens,
            ignore_eos=True,
            logprobs=-1,
            flat_logprobs=True,
            detokenize=False,
            skip_special_tokens=False,
            seed=0,
            extra_args=(
                {"teacher_force_token_ids": teacher_ids[: args.max_tokens]}
                if args.teacher_force
                else None
            ),
        )
        outputs = llm.generate([{"prompt_token_ids": prompt_ids}], params)
        completion = outputs[0].outputs[0]
        logprobs_obj = completion.logprobs
        if logprobs_obj is None:
            raise RuntimeError(f"prompt {i}: vLLM returned no decode logprobs")
        if vocab_size is None:
            vocab_size = _vocab_size_from_llm(llm, logprobs_obj)
        rows = [
            _position_to_dense(logprobs_obj, pos, vocab_size)
            for pos in range(len(logprobs_obj))
        ]
        all_rows.append(torch.stack(rows, dim=0))
        all_generated.append(torch.tensor(completion.token_ids, dtype=torch.int64))
        print(
            "prompt_done",
            json.dumps({"prompt_index": i, "positions": len(rows)}, sort_keys=True),
            flush=True,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "logprobs": torch.stack(all_rows, dim=0),
        "generated_token_ids": torch.stack(all_generated, dim=0),
        "prompt_token_ids": torch.tensor(prompts, dtype=torch.int64),
        "teacher_force_token_ids": torch.tensor(teachers, dtype=torch.int64),
    }
    save_file(tensors, str(out_path))
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "model": args.model,
                "output": str(out_path),
                "num_prompts": args.num_prompts,
                "prompt_len": args.prompt_len,
                "max_tokens": args.max_tokens,
                "num_logprob_positions": args.max_tokens,
                "vocab_size": vocab_size,
                "elapsed_sec": time.time() - t0,
                "env": {
                    "VLLM_B12X_FORCE_MOE_A16": os.getenv(
                        "VLLM_B12X_FORCE_MOE_A16"
                    ),
                    "B12X_MOE_FORCE_A16": os.getenv("B12X_MOE_FORCE_A16"),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("collect_done", meta_path, flush=True)


def _kl_rows(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    p = log_p.exp()
    terms = p * (log_p - log_q)
    terms = torch.where(p > 0, terms, torch.zeros_like(terms))
    return terms.sum(dim=-1)


def compare(args: argparse.Namespace) -> None:
    a = load_file(args.a)
    b = load_file(args.b)
    logp_a = a["logprobs"].to(torch.float32)
    logp_b = b["logprobs"].to(torch.float32)
    if logp_a.ndim == 2:
        logp_a = logp_a.unsqueeze(0)
    if logp_b.ndim == 2:
        logp_b = logp_b.unsqueeze(0)
    toks_a = a["generated_token_ids"].to(torch.int64)
    toks_b = b["generated_token_ids"].to(torch.int64)
    if toks_a.ndim == 1:
        toks_a = toks_a.unsqueeze(0)
    if toks_b.ndim == 1:
        toks_b = toks_b.unsqueeze(0)

    num_prompts = min(logp_a.shape[0], logp_b.shape[0])
    max_positions = min(logp_a.shape[1], logp_b.shape[1])
    start_pos = args.skip_prefill_next
    valid = []
    for prompt_idx in range(num_prompts):
        ta = toks_a[prompt_idx].tolist()
        tb = toks_b[prompt_idx].tolist()
        for pos in range(start_pos, max_positions):
            if ta[:pos] != tb[:pos]:
                break
            valid.append((prompt_idx, pos))

    if not valid:
        raise RuntimeError("No common-prefix decode positions")

    prompt_idx = torch.tensor([x[0] for x in valid], dtype=torch.long)
    pos_idx = torch.tensor([x[1] for x in valid], dtype=torch.long)
    pa = logp_a[prompt_idx, pos_idx]
    pb = logp_b[prompt_idx, pos_idx]
    kl_a_b = _kl_rows(pa, pb)
    kl_b_a = _kl_rows(pb, pa)
    log_m = torch.logaddexp(pa, pb) - math.log(2.0)
    js = 0.5 * _kl_rows(pa, log_m) + 0.5 * _kl_rows(pb, log_m)

    per_prompt = []
    for prompt in range(num_prompts):
        mask = prompt_idx == prompt
        if not bool(mask.any()):
            continue
        per_prompt.append(
            {
                "prompt_index": prompt,
                "num_positions": int(mask.sum().item()),
                "kl_a_to_b_mean": float(kl_a_b[mask].mean().item()),
                "kl_b_to_a_mean": float(kl_b_a[mask].mean().item()),
                "js_mean": float(js[mask].mean().item()),
            }
        )

    result = {
        "a": args.a,
        "b": args.b,
        "num_prompts": num_prompts,
        "num_positions": len(valid),
        "skip_prefill_next": start_pos,
        "kl_a_to_b_mean": float(kl_a_b.mean().item()),
        "kl_b_to_a_mean": float(kl_b_a.mean().item()),
        "js_mean": float(js.mean().item()),
        "kl_a_to_b_max": float(kl_a_b.max().item()),
        "kl_b_to_a_max": float(kl_b_a.max().item()),
        "js_max": float(js.max().item()),
        "per_prompt": per_prompt,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--label", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--num-prompts", type=int, default=8)
    c.add_argument("--prompt-len", type=int, default=2048)
    c.add_argument("--max-tokens", type=int, default=64)
    c.add_argument("--tensor-parallel-size", type=int, default=8)
    c.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    c.add_argument("--dtype", default="bfloat16")
    c.add_argument("--kv-cache-dtype", default="fp8")
    c.add_argument("--load-format", default="fastsafetensors")
    c.add_argument("--max-model-len", type=int, default=4096)
    c.add_argument("--max-num-batched-tokens", type=int, default=2048)
    c.add_argument("--quantization", default="modelopt_fp4")
    c.add_argument("--attention-backend", default="B12X_MLA_SPARSE")
    c.add_argument("--moe-backend", default="b12x")
    c.add_argument("--cpu-offload-gb", type=float, default=None)
    c.add_argument("--hf-overrides", default=json.dumps(DEFAULT_HF_OVERRIDES))
    c.add_argument("--teacher-force", action="store_true")
    c.add_argument("--token-source", default=None)
    c.add_argument("--enforce-eager", action="store_true")
    c.add_argument("--disable-custom-all-reduce", action="store_true")
    c.add_argument("--speculative-config", default=None,
                   help="JSON string for vLLM speculative_config, e.g. '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'")
    c.add_argument("--language-model-only", action="store_true",
                   help="Disable multimodal inputs (sets all modality limits to 0). Required for text-only quant variants of multimodal base models.")
    c.set_defaults(func=collect)

    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--skip-prefill-next", type=int, default=1)
    p.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
