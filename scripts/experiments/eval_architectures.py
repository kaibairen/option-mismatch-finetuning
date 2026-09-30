"""Prompting-architecture baselines on one frozen checkpoint.

Architectures are direct answer, zero-shot chain-of-thought, plan-and-solve,
and one-pass reflection. Careful chain-of-thought is the existing base prompt
and is not regenerated here.
"""

from __future__ import annotations

import argparse

from tqdm import tqdm

from option_mismatch.behavior import analyze_behavior, summarize_behavior
from option_mismatch.io_utils import read_jsonl, write_json
from option_mismatch.model_runtime import apply_chat, load_tokenizer
from option_mismatch.probe import generate_solution

SYSTEM = "You solve multiple-choice algebra questions."


def question_block(row: dict) -> str:
    options = "\n".join(str(opt) for opt in row["options"])
    return f"{row['question'].strip()}\n\nOptions:\n{options}"


def user_text(row: dict, instruction: str) -> str:
    return f"{question_block(row)}\n\n{instruction}"


INSTRUCTIONS = {
    "direct": "Reply with only the line `Final answer: X`, where X is A, B, C, D, or E.",
    "cot": "Let's think step by step. End with the line `Final answer: X`, where X is A, B, C, D, or E.",
    "plan": (
        "Let's first understand the problem and devise a plan. "
        "Then let's carry out the plan and solve the problem step by step. "
        "End with the line `Final answer: X`, where X is A, B, C, D, or E."
    ),
}


def chat(tokenizer, user: str) -> str:
    return apply_chat(tokenizer, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])


def score_rows(model, tokenizer, rows: list[dict], name: str, max_new_tokens: int) -> dict:
    details = []
    for row in tqdm(rows, desc=name):
        gold = str(row.get("correct") or "")
        options = list(row["options"])
        if name == "reflect":
            first = generate_solution(model, tokenizer, chat(tokenizer, user_text(row, INSTRUCTIONS["cot"])), max(64, max_new_tokens // 2))
            review = (
                f"{question_block(row)}\n\nYour previous solution:\n{first}\n\n"
                "Check the arithmetic and whether the final letter matches the value you computed. "
                "Correct any mistake. End with the line `Final answer: X`."
            )
            generation = generate_solution(model, tokenizer, chat(tokenizer, review), max(64, max_new_tokens // 2))
        else:
            generation = generate_solution(model, tokenizer, chat(tokenizer, user_text(row, INSTRUCTIONS[name])), max_new_tokens)
            first = ""
        behavior = analyze_behavior(generation, options, gold)
        details.append(behavior)
    summary = summarize_behavior(details)
    examples = []
    for row, behavior in list(zip(rows, details))[:4]:
        examples.append({"gold": behavior["gold"], "pred": behavior["pred"], "generation": behavior.get("phase2_text", "")[:240]})
    return {"name": name, "summary": summary, "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--names", default="direct,cot,plan,reflect")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM
    import torch

    rows = read_jsonl(args.test_jsonl, limit=args.limit)
    tokenizer = load_tokenizer(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float16, trust_remote_code=True)
    model.to("cuda")
    model.eval()
    model.config.use_cache = True
    reports = {}
    for name in [part.strip() for part in args.names.split(",") if part.strip()]:
        reports[name] = score_rows(model, tokenizer, rows, name, args.max_new_tokens)
        print(name, reports[name]["summary"], flush=True)
    write_json(args.output_json, {"n": len(rows), "architectures": reports})


if __name__ == "__main__":
    main()
