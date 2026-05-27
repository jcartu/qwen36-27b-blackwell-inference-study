#!/usr/bin/env python3
"""Collect and compare decode-step full-vocab logprobs from vLLM.

The existing score_mode KLD script measures prompt/prefill logits. This helper
generates a short greedy continuation and stores the full-vocab logprobs for
generated positions. Position 0 is the first generated token after prefill, so
it is excluded from decode KLD by default; positions >=1 require the KV-cache
decode path.
"""

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
        ) * 2048

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < min_len:
        reps = (min_len // max(1, len(ids))) + 1
        ids = (ids * reps)[:min_len]
    return ids[:min_len]


def _load_token_source(path: str) -> tuple[list[int], list[int] | None]:
    tensors = load_file(path)
    prompt_ids = tensors["prompt_token_ids"].tolist()
    teacher_force_ids = None
    if "teacher_force_token_ids" in tensors:
        teacher_force_ids = tensors["teacher_force_token_ids"].tolist()
    return [int(x) for x in prompt_ids], (
        [int(x) for x in teacher_force_ids] if teacher_force_ids is not None else None
    )


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


def collect(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    if args.token_source:
        prompt_ids, source_teacher_force_ids = _load_token_source(args.token_source)
        if len(prompt_ids) != args.prompt_len:
            raise ValueError(
                f"token source prompt length {len(prompt_ids)} != {args.prompt_len}"
            )
        if args.teacher_force and source_teacher_force_ids is None:
            raise ValueError("token source does not contain teacher_force_token_ids")
        teacher_force_ids = (
            source_teacher_force_ids[: args.max_tokens]
            if source_teacher_force_ids is not None
            else None
        )
    else:
        token_ids = _make_token_ids(args.model, args.prompt_len + args.max_tokens)
        prompt_ids = token_ids[: args.prompt_len]
        teacher_force_ids = token_ids[
            args.prompt_len : args.prompt_len + args.max_tokens
        ]
    if args.teacher_force and (
        teacher_force_ids is None or len(teacher_force_ids) < args.max_tokens
    ):
        raise ValueError("not enough teacher-force token ids for max_tokens")

    hf_overrides = json.loads(args.hf_overrides) if args.hf_overrides else {}

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "load_format": args.load_format,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max(args.prompt_len + args.max_tokens + 16, args.max_model_len),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "attention_backend": args.attention_backend,
        "max_logprobs": -1,
        "disable_log_stats": True,
    }
    if args.quantization.lower() not in ("", "none", "null"):
        llm_kwargs["quantization"] = args.quantization
    if args.moe_backend.lower() not in ("", "auto", "none", "null"):
        llm_kwargs["moe_backend"] = args.moe_backend
    if args.cpu_offload_gb is not None:
        llm_kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    if args.teacher_force:
        llm_kwargs["logits_processors"] = [
            "teacher_force_logits_processor:TeacherForceLogitsProcessor"
        ]
    if hf_overrides:
        llm_kwargs["hf_overrides"] = hf_overrides
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.disable_custom_all_reduce:
        llm_kwargs["disable_custom_all_reduce"] = True
    if args.speculative_config:
        llm_kwargs["speculative_config"] = json.loads(args.speculative_config)
    if args.language_model_only:
        llm_kwargs["language_model_only"] = True

    print("collect_start", json.dumps({
        "label": args.label,
        "model": args.model,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "teacher_force": args.teacher_force,
        "token_source": args.token_source,
        "env": {
            "VLLM_B12X_FORCE_MOE_A16": os.getenv("VLLM_B12X_FORCE_MOE_A16"),
            "B12X_MOE_FORCE_A16": os.getenv("B12X_MOE_FORCE_A16"),
            "VLLM_B12X_MOE_DECODE_A16": os.getenv("VLLM_B12X_MOE_DECODE_A16"),
        },
    }, sort_keys=True), flush=True)

    t0 = time.time()
    llm = LLM(**llm_kwargs)
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
            {"teacher_force_token_ids": teacher_force_ids[: args.max_tokens]}
            if args.teacher_force
            else None
        ),
    )
    outputs = llm.generate([{"prompt_token_ids": prompt_ids}], params)
    completion = outputs[0].outputs[0]
    gen_token_ids = list(completion.token_ids)
    logprobs_obj = completion.logprobs
    if logprobs_obj is None:
        raise RuntimeError("vLLM returned no decode logprobs")

    vocab_size = _vocab_size_from_llm(llm, logprobs_obj)
    rows = []
    for pos in range(len(logprobs_obj)):
        rows.append(_position_to_dense(logprobs_obj, pos, vocab_size))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "logprobs": torch.stack(rows, dim=0),
        "generated_token_ids": torch.tensor(gen_token_ids, dtype=torch.int64),
        "prompt_token_ids": torch.tensor(prompt_ids, dtype=torch.int64),
    }
    if args.teacher_force:
        tensors["teacher_force_token_ids"] = torch.tensor(
            teacher_force_ids[: args.max_tokens], dtype=torch.int64
        )
    save_file(tensors, str(out_path))
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps({
        "label": args.label,
        "model": args.model,
        "output": str(out_path),
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "num_logprob_positions": len(rows),
        "generated_token_ids": gen_token_ids,
        "teacher_force_token_ids": (
            teacher_force_ids[: args.max_tokens] if args.teacher_force else None
        ),
        "vocab_size": vocab_size,
        "elapsed_sec": time.time() - t0,
        "env": {
            "VLLM_B12X_FORCE_MOE_A16": os.getenv("VLLM_B12X_FORCE_MOE_A16"),
            "B12X_MOE_FORCE_A16": os.getenv("B12X_MOE_FORCE_A16"),
            "VLLM_B12X_MOE_DECODE_A16": os.getenv("VLLM_B12X_MOE_DECODE_A16"),
        },
    }, indent=2, sort_keys=True))
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
    toks_a = a["generated_token_ids"].tolist()
    toks_b = b["generated_token_ids"].tolist()

    max_positions = min(logp_a.shape[0], logp_b.shape[0])
    start_pos = args.skip_prefill_next
    valid_positions = []
    for pos in range(start_pos, max_positions):
        # Logprobs at generated position `pos` are conditioned on all generated
        # tokens before `pos`. The sampled token at `pos` may differ; that only
        # affects later positions.
        if toks_a[:pos] != toks_b[:pos]:
            break
        valid_positions.append(pos)

    if not valid_positions:
        raise RuntimeError(
            "No common-prefix decode positions. First generated tokens differ: "
            f"{toks_a[:4]} vs {toks_b[:4]}"
        )

    idx = torch.tensor(valid_positions, dtype=torch.long)
    pa = logp_a.index_select(0, idx)
    pb = logp_b.index_select(0, idx)
    kl_a_b = _kl_rows(pa, pb)
    kl_b_a = _kl_rows(pb, pa)
    log_m = torch.logaddexp(pa, pb) - math.log(2.0)
    js = 0.5 * _kl_rows(pa, log_m) + 0.5 * _kl_rows(pb, log_m)

    result = {
        "a": args.a,
        "b": args.b,
        "positions": valid_positions,
        "num_positions": len(valid_positions),
        "generated_token_ids_a": toks_a,
        "generated_token_ids_b": toks_b,
        "kl_a_to_b_mean": float(kl_a_b.mean().item()),
        "kl_a_to_b_per_pos": [float(x) for x in kl_a_b.tolist()],
        "kl_b_to_a_mean": float(kl_b_a.mean().item()),
        "kl_b_to_a_per_pos": [float(x) for x in kl_b_a.tolist()],
        "js_mean": float(js.mean().item()),
        "js_per_pos": [float(x) for x in js.tolist()],
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
    c.add_argument("--prompt-len", type=int, default=2048)
    c.add_argument("--max-tokens", type=int, default=9)
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
