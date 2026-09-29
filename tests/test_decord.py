from option_mismatch.decord import argmax_letter, blend_letter_scores, commit_letter, majority_letter, phase2_prefix, select_decord
from option_mismatch.losses import completion_token_mask, dpo_loss, letter_margin_loss


def test_blend_pulls_toward_inferred_letter():
    base = {"A": -1.0, "B": -0.1, "C": -2.0}
    assert argmax_letter(blend_letter_scores(base, "A", 0.0)) == "B"
    assert argmax_letter(blend_letter_scores(base, "A", 1.0)) == "A"


def test_blend_ignores_empty_inference():
    base = {"A": -1.0, "B": -0.2}
    assert blend_letter_scores(base, "", 2.0) == base


def test_select_decord_uses_highest_blended_trace():
    traces = [
        {"letter_logprobs": {"A": -0.2, "B": -1.0}, "inferred_letter": "B", "pred": "A"},
        {"letter_logprobs": {"A": -2.0, "B": -0.3}, "inferred_letter": "B", "pred": "B"},
    ]
    chosen = select_decord(traces, alpha=1.0)
    assert chosen["letter"] == "B"


def test_commit_keeps_emitted_letter_when_no_number_is_inferred():
    assert commit_letter({"A": -0.1, "B": -2.0}, "", "B", 4.0) == "B"


def test_majority_letter_breaks_ties_by_letter_order():
    assert majority_letter(["C", "A", "A", "C"]) == "A"


def test_phase2_prefix_stops_at_final_answer():
    text = "Total is 125.\nFinal answer: C"
    assert phase2_prefix(text, ["A)125", "B)1"]).endswith("125.")


def test_completion_mask_drops_prompt_tokens():
    import torch

    attention = torch.tensor([[1, 1, 1, 1, 1]])
    mask = completion_token_mask(attention, torch.tensor([3]))
    assert mask.tolist() == [[0.0, 0.0, 1.0, 1.0]]


def test_dpo_loss_is_small_when_policy_prefers_chosen():
    import torch

    loss = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([-20.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=0.1,
    )
    assert float(loss) < 0.2


def test_letter_margin_is_zero_when_gold_already_wins():
    import torch

    logits = torch.tensor([[0.0, 5.0, 1.0]])
    loss = letter_margin_loss(logits, torch.tensor([1]), torch.tensor([0, 2]), margin=1.0)
    assert float(loss) == 0.0
