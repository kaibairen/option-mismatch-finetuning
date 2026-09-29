"""Greedy accuracy and training-free DECORD on one checkpoint."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from option_mismatch.behavior import analyze_behavior, summarize_behavior
from option_mismatch.decord import commit_letter, letter_logprobs, majority_letter, phase2_prefix
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


def sample_solution(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.9
    model.generation_config.top_k = 50
    out = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def letters_for_generation(model, tokenizer, prompt: str, generation: str, options: list[str], gold: str) -> dict:
    behavior = analyze_behavior(generation, options, gold)
    prefix = f"{prompt}{phase2_prefix(generation, options)}\nFinal answer:"
    logprobs = letter_logprobs(model, tokenizer, prefix)
    inferred = behavior.get("inferred_letter") or ""
    pred = behavior.get("pred") or ""
    chosen = {str(alpha): commit_letter(logprobs, inferred, pred, alpha) for alpha in ALPHAS}
    return {"behavior": behavior, "chosen": chosen, "generation": generation}


def score_split(model, tokenizer, rows: list[dict], max_new_tokens: int, decord_samples: int) -> dict:
    greedy_rows = []
    by_alpha = {str(alpha): [] for alpha in ALPHAS}
    majority = {str(alpha): [] for alpha in ALPHAS}
    examples = []
    for row in tqdm(rows, desc="eval"):
        options = list(row["options"])
        gold = str(row.get("correct") or "").strip().upper()[:1]
        prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], options), diligent=True))
        generations = [generate_solution(model, tokenizer, prompt, max_new_tokens)]
        for _ in range(max(decord_samples - 1, 0)):
            generations.append(sample_solution(model, tokenizer, prompt, max_new_tokens))
        scored = [letters_for_generation(model, tokenizer, prompt, text, options, gold) for text in generations]
        behavior = scored[0]["behavior"]
        greedy_rows.append(behavior)
        per_alpha = {str(alpha): [] for alpha in ALPHAS}
        for item in scored:
            for alpha, letter in item["chosen"].items():
                per_alpha[alpha].append(letter)
        decord_letters = {}
        for alpha, letters in per_alpha.items():
            letter = letters[0]
            voted = majority_letter(letters)
            decord_letters[alpha] = {"first": letter, "majority": voted}
            by_alpha[alpha].append(
                {
                    "letter_correct": letter == gold and bool(letter),
                    "mismatch": bool(behavior.get("inferred_letter")) and bool(letter) and behavior["inferred_letter"] != letter,
                    "format_ok": bool(behavior.get("format_ok")),
                    "number_match": bool(behavior.get("number_match")),
                }
            )
            majority[alpha].append(
                {
                    "letter_correct": voted == gold and bool(voted),
                    "mismatch": False,
                    "format_ok": True,
                    "number_match": bool(behavior.get("number_match")),
                }
            )
        if len(examples) < 8:
            examples.append(
                {
                    "gold": gold,
                    "pred": behavior.get("pred"),
                    "inferred": behavior.get("inferred_letter"),
                    "decord": decord_letters,
                    "generation": scored[0]["generation"][:500],
                }
            )
    report = {
        "n": len(rows),
        "greedy": summarize_behavior(greedy_rows),
        "decord": {alpha: summarize_behavior(rows_alpha) for alpha, rows_alpha in by_alpha.items()},
        "examples": examples,
    }
    if decord_samples > 1:
        report["decord_majority"] = {alpha: summarize_behavior(rows_alpha) for alpha, rows_alpha in majority.items()}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--decord-samples", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.test_jsonl, limit=args.limit)
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, args.adapter)
    report = score_split(model, tokenizer, rows, args.max_new_tokens, args.decord_samples)
    report["adapter"] = args.adapter or "base"
    report["test_jsonl"] = args.test_jsonl
    write_json(args.output_json, report)
    summary = {"greedy": report["greedy"], "decord": {k: v["accuracy"] for k, v in report["decord"].items()}}
    if "decord_majority" in report:
        summary["decord_majority"] = {k: v["accuracy"] for k, v in report["decord_majority"].items()}
    print(summary, flush=True)


if __name__ == "__main__":
    main()
