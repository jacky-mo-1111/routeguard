from __future__ import annotations

from typing import Iterable, Optional

import torch
from transformers import LogitsProcessor

from . import logging


logger = logging.get_logger(__name__)


class TagDebugLogitsProcessor(LogitsProcessor):
    r"""When the last generated token matches `trigger_id`, mask logits to only allow `candidate_ids`."""

    def __init__(self, trigger_id: int, candidate_ids: Iterable[int]) -> None:
        self.trigger_id = int(trigger_id)
        self.candidate_ids = list(dict.fromkeys(int(t) for t in candidate_ids))  # drop duplicates, preserve order
        if not self.candidate_ids:
            raise ValueError("`candidate_ids` must contain at least one token id.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.numel() == 0:
            return scores

        batch_mask = input_ids[:, -1] == self.trigger_id
        if not torch.any(batch_mask):
            return scores

        allowed = torch.as_tensor(self.candidate_ids, device=scores.device, dtype=torch.long)
        finfo_min = torch.finfo(scores.dtype).min
        next_scores = scores.clone()

        restricted_scores = next_scores[batch_mask]
        restricted_scores.fill_(finfo_min)
        restricted_scores[:, allowed] = scores[batch_mask][:, allowed]
        next_scores[batch_mask] = restricted_scores
        return next_scores


def build_tag_debug_processor(
    tokenizer,
    trigger_token: str,
    candidate_tokens: Optional[list[str]],
) -> tuple[Optional[TagDebugLogitsProcessor], list[str]]:
    r"""Convert tokens to ids and build TagDebugLogitsProcessor.

    Returns (processor_or_None, missing_tokens).
    """

    if not trigger_token or not candidate_tokens:
        return None, []

    trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)
    candidate_ids = []
    missing_tokens: list[str] = []
    for token in candidate_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            missing_tokens.append(token)
        else:
            candidate_ids.append(token_id)

    if trigger_id == tokenizer.unk_token_id:
        logger.warning_rank0(f"Trigger token `{trigger_token}` is not in tokenizer vocab; tag routing disabled.")
        return None, missing_tokens

    if not candidate_ids:
        logger.warning_rank0("No valid candidate tokens for tag routing; feature disabled.")
        return None, missing_tokens

    try:
        processor = TagDebugLogitsProcessor(trigger_id, candidate_ids)
    except ValueError:
        processor = None

    return processor, missing_tokens

