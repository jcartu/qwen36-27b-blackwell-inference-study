#!/usr/bin/env python3
"""Smoke-test loading + generating with a given vLLM quantization."""
import os
import sys
import time

from vllm import LLM, SamplingParams


def main() -> int:
    label = os.environ["SMOKE_LABEL"]
    model = os.environ["SMOKE_MODEL"]
    quant_raw = os.environ["SMOKE_QUANT"]
    eager = os.environ.get("SMOKE_EAGER", "0") == "1"

    quant = None if quant_raw == "none" else quant_raw
    print(
        f"[smoke] label={label} model={model} quant={quant} eager={eager}",
        flush=True,
    )

    prompt_text = "The quick brown fox"

    t_load = time.time()
    try:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
        llm = LLM(
            model=model,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
            max_model_len=2048,
            max_num_batched_tokens=2048,
            quantization=quant,
            enforce_eager=eager,
            disable_custom_all_reduce=True,
            seed=42,
            language_model_only=True,
            trust_remote_code=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] LOAD_FAIL: {type(e).__name__}: {e}", flush=True)
        return 2
    print(f"[smoke] LOAD_OK in {time.time() - t_load:.1f}s", flush=True)

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=42,
        max_tokens=32,
        ignore_eos=True,
    )

    t_gen = time.time()
    try:
        outputs = llm.generate([prompt_text], sampling_params=sampling, use_tqdm=False)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] GEN_FAIL: {type(e).__name__}: {e}", flush=True)
        return 3
    elapsed = time.time() - t_gen
    n_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
    sample = outputs[0].outputs[0].text
    print(
        f"[smoke] GEN_OK {n_tok} tok in {elapsed:.2f}s ({n_tok / elapsed:.1f} tok/s)",
        flush=True,
    )
    print(f"[smoke] SAMPLE: {sample!r}", flush=True)
    print("[smoke] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
