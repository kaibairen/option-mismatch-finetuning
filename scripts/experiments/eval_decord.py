"""Greedy accuracy and training-free DECORD on one checkpoint."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from option_mismatch.behavior import analyze_behavior, summarize_behavior
from option_mismatch.decord import letter_logprobs, phase2_prefix, select_decord
from option_mismatch.io_utils import read_jsonl, write_json
from option_mismatch.model_runtime import apply_chat, load_tokenizer
from option_mismatch.probe import generate_solution
from option_mismatch.prompts import chat_messages, format_mcq

ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)


def load_model(model_dir: str, adapter: str):
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, trust_remote_code=True)
    model.to("cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    return model


def score_split(model, tokenizer, rows: list[dict], max_new_tokens: int) -> dict:
    greedy_rows = []
    by_alpha = {str(alpha): [] for alpha in ALPHAS}
    examples = []
    for row in tqdm(rows, desc="eval"):
        options = list(row["options"])
        gold = str(row.get("correct") or "").strip().upper()[:1]
        prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], options), diligent=True))
        generation = generate_solution(model, tokenizer, prompt, max_new_tokens)
        behavior = analyze_behavior(generation, options, gold)
        greedy_rows.append(behavior)
        prefix = f"{prompt}{phase2_prefix(generation, options)}\nFinal answer:"
        logprobs = letter_logprobs(model, tokenizer, prefix)
        trace = {
            "letter_logprobs": logprobs,
            "inferred_letter": behavior.get("inferred_letter") or "",
            "pred": behavior.get("pred") or "",
            "generation": generation,
        }
        decord_letters = {}
        for alpha in ALPHAS:
            chosen = select_decord([trace], alpha=alpha)
            letter = chosen["letter"]
            decord_letters[str(alpha)] = letter
            by_alpha[str(alpha)].append(
                {
                    "letter_correct": letter == gold and bool(letter),
                    "mismatch": bool(behavior.get("inferred_letter")) and bool(letter) and behavior["inferred_letter"] != letter,
                    "format_ok": True,
                    "number_match": behavior.get("number_match", False),
                }
            )
        if len(examples) < 8:
            examples.append(
                {
                    "gold": gold,
                    "pred": behavior.get("pred"),
                    "inferred": behavior.get("inferred_letter"),
                    "decord": decord_letters,
                    "generation": generation[:500],
                }
            )
    return {
        "n": len(rows),
        "greedy": summarize_behavior(greedy_rows),
        "decord": {alpha: summarize_behavior(rows_alpha) for alpha, rows_alpha in by_alpha.items()},
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.test_jsonl, limit=args.limit)
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, args.adapter)
    report = score_split(model, tokenizer, rows, args.max_new_tokens)
    report["adapter"] = args.adapter or "base"
    report["test_jsonl"] = args.test_jsonl
    write_json(args.output_json, report)
    print({"greedy": report["greedy"], "decord": {k: v["accuracy"] for k, v in report["decord"].items()}}, flush=True)


if __name__ == "__main__":
    main()
