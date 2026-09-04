"""Build real preference pairs from gold rationales and model traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from option_mismatch.behavior import analyze_behavior
from option_mismatch.io_utils import load_yaml, read_jsonl, write_jsonl
from option_mismatch.model_runtime import apply_chat, load_causal_lm, load_tokenizer, resolve_model_name
from option_mismatch.probe import generate_solution
from option_mismatch.prompts import chat_messages, format_mcq
from option_mismatch.span import phase3_token_start


def chosen_text(row: dict[str, Any]) -> str:
    rationale = (row.get("rationale") or "").strip()
    gold = str(row.get("correct") or "A").strip().upper()[:1]
    if "Final answer" not in rationale:
        rationale = f"{rationale}\nFinal answer: {gold}".strip()
    return rationale


def build_preferences(cfg_path: str, overrides: dict | None = None) -> list[dict[str, Any]]:
    cfg = load_yaml(cfg_path)
    overrides = overrides or {}
    if overrides.get("local_model_dir"):
        cfg["local_model_dir"] = overrides["local_model_dir"]
    train_path = overrides.get("train_jsonl") or cfg["data"].get("train_jsonl") or cfg["data"]["test_jsonl"]
    rows = read_jsonl(train_path, limit=int(overrides.get("max_train_samples") or cfg["train"].get("max_train_samples") or 100))
    model_name = resolve_model_name(cfg)
    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(model_name)
    max_new = int(cfg["data"].get("max_new_tokens", 128))
    pairs = []
    for row in tqdm(rows, desc="pref"):
        options = list(row["options"])
        prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], options), diligent=True))
        chosen = chosen_text(row)
        rejected = generate_solution(model, tokenizer, prompt, max_new)
        beh = analyze_behavior(rejected, options, row.get("correct", ""))
        if beh["letter_correct"] and not beh["mismatch"]:
            careless_prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], options), diligent=False))
            rejected = generate_solution(model, tokenizer, careless_prompt, max_new)
            beh = analyze_behavior(rejected, options, row.get("correct", ""))
        chosen_span = phase3_token_start(tokenizer, chosen, options=options)
        rejected_span = phase3_token_start(tokenizer, rejected, options=options)
        pairs.append(
            {
                "question": row["question"],
                "options": options,
                "correct": row.get("correct"),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_phase3_token": chosen_span["token_start"],
                "rejected_phase3_token": rejected_span["token_start"],
                "rejected_behavior": beh,
            }
        )
    out = Path(overrides.get("output_jsonl") or "data/processed/aqua_pref_100.jsonl")
    write_jsonl(out, pairs)
    return pairs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_0.5b_2080.yaml")
    parser.add_argument("--local-model-dir", default="")
    parser.add_argument("--train-jsonl", default="")
    parser.add_argument("--output-jsonl", default="data/processed/aqua_pref_100.jsonl")
    parser.add_argument("--max-train-samples", type=int, default=100)
    args = parser.parse_args()
    pairs = build_preferences(
        args.config,
        {
            "local_model_dir": args.local_model_dir,
            "train_jsonl": args.train_jsonl,
            "output_jsonl": args.output_jsonl,
            "max_train_samples": args.max_train_samples,
        },
    )
    print({"n": len(pairs), "path": args.output_jsonl})


if __name__ == "__main__":
    main()
