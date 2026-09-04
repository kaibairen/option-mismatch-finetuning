"""Score letter accuracy on AQUA-RAT after optional LoRA repair."""

from __future__ import annotations

import argparse
import re

from peft import PeftModel

from option_mismatch.io_utils import load_yaml, read_jsonl, write_json
from option_mismatch.model_runtime import apply_chat, load_causal_lm, load_tokenizer, resolve_model_name
from option_mismatch.prompts import chat_messages, format_mcq

LETTER_RE = re.compile(r"Final answer:\s*([A-E])", re.IGNORECASE)


def extract_letter(text: str) -> str:
    match = LETTER_RE.search(text)
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b([A-E])\b", text.upper())
    return letters[-1] if letters else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_0.5b_2080.yaml")
    parser.add_argument("--adapter", default="")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    model_name = resolve_model_name(cfg)
    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(model_name)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    rows = read_jsonl(cfg["data"]["test_jsonl"], limit=int(cfg["eval"]["num_samples"]))
    correct = 0
    details = []
    for row in rows:
        prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], row["options"]), diligent=True))
        encoded = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        out = model.generate(**encoded, max_new_tokens=int(cfg["data"]["max_new_tokens"]), do_sample=False, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = extract_letter(text)
        ok = pred == str(row.get("correct", "")).upper()[:1]
        correct += int(ok)
        details.append({"correct": row.get("correct"), "pred": pred, "ok": ok, "generation": text})

    report = {"n": len(rows), "accuracy": (correct / len(rows)) if rows else 0.0, "details": details}
    write_json(cfg["eval"]["output_json"], report)
    print({"n": report["n"], "accuracy": report["accuracy"]})


if __name__ == "__main__":
    main()
