from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
from transformers import LogitsProcessor

from . import logging


logger = logging.get_logger(__name__)


class TagDebugLogitsProcessor(LogitsProcessor):
    r"""When the last generated token matches `trigger_id`, mask logits to only allow `candidate_ids`.

    If `phrase_ids` is provided and the sequence before trigger ends with that phrase, use `override_candidate_ids`.
    """

    def __init__(
        self,
        trigger_id: int,
        candidate_ids: Iterable[int],
        phrase_ids: Optional[Sequence[int]] = None,
        override_candidate_ids: Optional[Iterable[int]] = None,
    ) -> None:
        self.trigger_id = int(trigger_id)
        self.candidate_ids = list(dict.fromkeys(int(t) for t in candidate_ids))  # drop duplicates, preserve order
        self.phrase_ids = list(phrase_ids) if phrase_ids else None
        self.override_candidate_ids = (
            list(dict.fromkeys(int(t) for t in override_candidate_ids)) if override_candidate_ids else None
        )
        if not self.candidate_ids:
            raise ValueError("`candidate_ids` must contain at least one token id.")
        if self.phrase_ids and not self.override_candidate_ids:
            raise ValueError("`override_candidate_ids` is required when `phrase_ids` is set.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.numel() == 0:
            return scores

        batch_mask = input_ids[:, -1] == self.trigger_id
        if not torch.any(batch_mask):
            return scores

        base_allowed = torch.as_tensor(self.candidate_ids, device=scores.device, dtype=torch.long)

        finfo_min = torch.finfo(scores.dtype).min
        next_scores = scores.clone()

        restricted_scores = next_scores[batch_mask]
        restricted_scores.fill_(finfo_min)

        # Default assignment
        restricted_scores[:, base_allowed] = scores[batch_mask][:, base_allowed]

        # Optional override: phrase immediately before trigger
        if self.phrase_ids and self.override_candidate_ids:
            phrase_len = len(self.phrase_ids)
            if input_ids.size(1) >= phrase_len + 1:  # +1 for trigger
                before_trigger = input_ids[:, -(phrase_len + 1) : -1]
                phrase_tensor = torch.as_tensor(self.phrase_ids, device=input_ids.device, dtype=input_ids.dtype)
                match_mask = torch.all(before_trigger == phrase_tensor, dim=1) & batch_mask
                if torch.any(match_mask):
                    override_allowed = torch.as_tensor(
                        self.override_candidate_ids, device=scores.device, dtype=torch.long
                    )
                    restricted_scores_override = next_scores[match_mask]
                    restricted_scores_override.fill_(finfo_min)
                    restricted_scores_override[:, override_allowed] = scores[match_mask][:, override_allowed]
                    restricted_scores[match_mask[batch_mask]] = restricted_scores_override

        next_scores[batch_mask] = restricted_scores
        return next_scores


class TagDebugForceEosProcessor(LogitsProcessor):
    r"""When the last generated token is in `candidate_ids`, force next token to `eos_ids`."""

    def __init__(self, candidate_ids: Iterable[int], eos_ids: Sequence[int]) -> None:
        self.candidate_ids = set(int(t) for t in candidate_ids)
        self.eos_ids = list(dict.fromkeys(int(t) for t in eos_ids))  # drop duplicates, preserve order
        if not self.candidate_ids:
            raise ValueError("`candidate_ids` must contain at least one token id.")
        if not self.eos_ids:
            raise ValueError("`eos_ids` must contain at least one token id.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.numel() == 0:
            return scores

        batch_mask = torch.isin(input_ids[:, -1], torch.as_tensor(list(self.candidate_ids), device=input_ids.device))
        if not torch.any(batch_mask):
            return scores

        allowed = torch.as_tensor(self.eos_ids, device=scores.device, dtype=torch.long)
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
    phrase: Optional[str] = None,
    override_candidate_tokens: Optional[list[str]] = None,
) -> tuple[Optional[TagDebugLogitsProcessor], list[str]]:
    r"""Convert tokens to ids and build TagDebugLogitsProcessor.

    Returns (processor_or_None, missing_tokens).
    """

    if not trigger_token or not candidate_tokens:
        return None, []

    trigger_id = tokenizer.convert_tokens_to_ids(trigger_token)
    candidate_ids = []
    missing_tokens: list[str] = []

    def tokens_to_ids(tokens: list[str]):
        ids = []
        miss = []
        for token in tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id == tokenizer.unk_token_id:
                miss.append(token)
            else:
                ids.append(token_id)
        return ids, miss

    candidate_ids, miss = tokens_to_ids(candidate_tokens)
    missing_tokens += miss

    phrase_ids = tokenizer.encode(phrase, add_special_tokens=False) if phrase else None
    override_ids = None
    if override_candidate_tokens:
        override_ids, miss = tokens_to_ids(override_candidate_tokens)
        missing_tokens += miss

    if trigger_id == tokenizer.unk_token_id:
        logger.warning_rank0(f"Trigger token `{trigger_token}` is not in tokenizer vocab; tag routing disabled.")
        return None, missing_tokens

    if not candidate_ids:
        logger.warning_rank0("No valid candidate tokens for tag routing; feature disabled.")
        return None, missing_tokens

    try:
        processor = TagDebugLogitsProcessor(trigger_id, candidate_ids, phrase_ids, override_ids)
    except ValueError:
        processor = None

    return processor, missing_tokens


def build_tag_debug_force_eos_processor(
    tokenizer,
    candidate_tokens: Optional[list[str]],
    eos_ids: Sequence[int],
) -> tuple[Optional[TagDebugForceEosProcessor], list[str]]:
    r"""Force eos right after any candidate token. Returns (processor_or_None, missing_tokens)."""

    if not candidate_tokens:
        return None, []

    candidate_ids = []
    missing_tokens: list[str] = []
    for token in candidate_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            missing_tokens.append(token)
        else:
            candidate_ids.append(token_id)

    if not candidate_ids:
        logger.warning_rank0("No valid candidate tokens for tag routing; eos forcing disabled.")
        return None, missing_tokens

    try:
        processor = TagDebugForceEosProcessor(candidate_ids, eos_ids)
    except ValueError:
        processor = None

    return processor, missing_tokens

