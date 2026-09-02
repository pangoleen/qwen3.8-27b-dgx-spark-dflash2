# r0b0tlab compat shim for the DFlash2 overlay (sgl-project/sglang #35371 @ c14312a)
# on the pinned day-0 image (sglang 0.0.0.dev0+qwen38.27b.g561c8f3).
#
# The day-0 image fork predates two upstream additions the DFlash2 worker uses:
#   - logprob_processor.compute_spec_logprobs (new function; no image-side callers)
# This module ports the upstream function verbatim. Its only helpers
# (get_top_logprobs, get_token_ids_logprobs) already exist in the image's
# logprob_processor with identical DECODE-stage signatures.
from typing import List, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    get_token_ids_logprobs,
    get_top_logprobs,
)
from sglang.srt.environ import envs


def compute_spec_logprobs(
    batch,
    logits_output: LogitsProcessorOutput,
    predict: torch.Tensor,
    *,
    accept_index: Optional[torch.Tensor] = None,
    chain_stride: Optional[int] = None,
):
    assert (accept_index is None) != (
        chain_stride is None
    ), "pass exactly one of accept_index / chain_stride"

    bs = len(batch.seq_lens)
    next_token_logits = logits_output.next_token_logits

    if accept_index is not None:
        max_accept = accept_index.shape[1]
        flat_accept_idx = accept_index.long().reshape(-1)
        gathered_logits = next_token_logits[flat_accept_idx]
        accepted_token_ids = predict[flat_accept_idx]
    else:
        max_accept = chain_stride
        # Guards the layout contract the identity gather rests on: out token
        # (b, j) must come from logits row b * stride + j.
        assert next_token_logits.shape[0] == bs * max_accept, (
            f"chain layout expects {bs * max_accept} logits rows, got "
            f"{next_token_logits.shape[0]}"
        )
        gathered_logits = next_token_logits
        accepted_token_ids = predict

    if batch.sampling_info.is_all_greedy or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        gathered_logprobs = torch.nn.functional.log_softmax(gathered_logits, dim=-1)
    else:
        temperatures = torch.repeat_interleave(
            batch.sampling_info.temperatures,
            max_accept,
            dim=0,
        )
        gathered_logprobs = torch.nn.functional.log_softmax(
            gathered_logits / temperatures, dim=-1
        )
    gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

    logits_output.next_token_logprobs = gathered_logprobs.gather(
        1, accepted_token_ids.long().view(-1, 1)
    ).view(bs, max_accept)

    if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
        top_logprobs_nums_expanded: List[int] = [
            num for num in batch.top_logprobs_nums for _ in range(max_accept)
        ]
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True
        )

    if batch.token_ids_logprobs and any(
        x is not None for x in batch.token_ids_logprobs
    ):
        token_ids_logprobs_expanded = [
            ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
        ]
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True
        )
