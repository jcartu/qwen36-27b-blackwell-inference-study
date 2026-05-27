"""vLLM logits processor for teacher-forced decode KLD collection.

The vLLM v1 sampler records raw logprobs before custom logits processors are
applied. This processor only changes the sampled token, so the returned
full-vocab logprobs remain the model's real decode distribution.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class TeacherForceLogitsProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        extra_args = sampling_params.extra_args or {}
        token_ids = extra_args.get("teacher_force_token_ids")
        if token_ids is None:
            return None
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise ValueError("teacher_force_token_ids must be a sequence of integers")
        if not token_ids:
            raise ValueError("teacher_force_token_ids must not be empty")
        for token_id in token_ids:
            if not isinstance(token_id, int) or token_id < 0:
                raise ValueError(
                    "teacher_force_token_ids must contain non-negative integers"
                )
        return None

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        extra_args = params.extra_args or {}
        token_ids = extra_args.get("teacher_force_token_ids")
        if token_ids is None:
            return None
        forced_ids = [int(x) for x in token_ids]

        def force_next(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            pos = len(output_ids)
            if pos >= len(forced_ids):
                return logits
            forced_id = forced_ids[pos]
            logits.fill_(float("-inf"))
            logits[forced_id] = 0.0
            return logits

        return force_next
