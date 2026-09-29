"""Reference-model DPO and answer-letter margin losses."""

from __future__ import annotations

import torch


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO (Rafailov et al.) on sequence log-probabilities. Returns a per-row loss."""
    logits = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return -torch.nn.functional.logsigmoid(logits)


def completion_token_mask(attention_mask: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    """Mask over shifted labels: 1 on completion tokens, 0 on prompt and padding.

    `attention_mask` is aligned with `input_ids` (batch, seq). The returned mask
    is aligned with next-token labels (batch, seq - 1). A label at index j
    predicts token j + 1, so completion starts at j = prompt_len - 1.
    """
    shifted = attention_mask[:, 1:].to(dtype=torch.float32)
    positions = torch.arange(shifted.size(1), device=shifted.device).unsqueeze(0)
    start = (prompt_lens.to(device=shifted.device) - 1).clamp(min=0).unsqueeze(1)
    return shifted * (positions >= start).to(dtype=shifted.dtype)


def sequence_logprob(token_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (token_logp * mask).sum(dim=-1)


def letter_margin_loss(
    answer_logits: torch.Tensor,
    gold_ids: torch.Tensor,
    other_ids: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Hinge that keeps the gold letter logit above every other letter logit."""
    gold = answer_logits.gather(1, gold_ids.view(-1, 1)).squeeze(1)
    others = answer_logits.index_select(1, other_ids)
    gap = gold.unsqueeze(1) - others
    return torch.relu(torch.tensor(margin, device=answer_logits.device, dtype=gap.dtype) - gap).mean()
