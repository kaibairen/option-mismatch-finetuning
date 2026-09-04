"""Score letter accuracy, mismatch, and optional NDI after an adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from peft import PeftModel

from option_mismatch.behavior import analyze_behavior, summarize_behavior
from option_mismatch.io_utils import load_yaml, read_jsonl, write_json
from option_mismatch.model_runtime import apply_chat, load_causal_lm, load_tokenizer, resolve_model_name
from option_mismatch.probe import generate_solution, hidden_at_layer
from option_mismatch.prompts import chat_messages, format_mcq
from option_mismatch.span import phase3_token_start, split_ndi
from option_mismatch.stats import paired_drift_tests


def evaluate(cfg_path: str, *, adapter: str = "", output_json: str = "", test_jsonl: str = "") -> dict:
    cfg = load_yaml(cfg_path)
    model_name = resolve_model_name(cfg)
    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(model_name)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    rows = read_jsonl(test_jsonl or cfg["data"]["test_jsonl"], limit=int(cfg["eval"]["num_samples"]))
    layer = int(cfg["probe_layer"])
    vec_path = Path(cfg["probe"].get("output_npz") or "results/stage_a_vectors.npz")
    v = None
    if vec_path.exists():
        blob = np.load(vec_path)
        key = f"task_v_{layer}"
        v = blob[key] if key in blob.files else None
    details = []
    p2s, p3s = [], []
    for row in rows:
        prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], row["options"]), diligent=True))
        text = generate_solution(model, tokenizer, prompt, int(cfg["data"]["max_new_tokens"]))
        beh = analyze_behavior(text, row["options"], row.get("correct", ""))
        span = phase3_token_start(tokenizer, text, options=list(row["options"]))
        rec = {"correct": row.get("correct"), "generation": text, **beh, "span_rule": span["rule"]}
        if v is not None:
            import torch

            device = next(model.parameters()).device
            full = tokenizer(prompt + text, return_tensors="pt").to(device)
            out = model(**full, output_hidden_states=True, use_cache=False)
            h = hidden_at_layer(out, layer)[0].float()
            h = torch.nn.functional.normalize(h, dim=-1)
            ndi = torch.matmul(h, torch.tensor(v, device=h.device, dtype=h.dtype)).cpu().numpy()
            plen = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            gen_ndi = ndi[max(plen - 1, 0) :]
            p2, p3 = split_ndi(gen_ndi, span["token_start"])
            rec["phase2"] = p2
            rec["phase3"] = p3
            if p2 == p2 and p3 == p3:
                p2s.append(p2)
                p3s.append(p3)
        details.append(rec)
    summary = summarize_behavior(details)
    if p2s:
        summary["ndi"] = paired_drift_tests(np.asarray(p2s), np.asarray(p3s))
    report = {"adapter": adapter or "base", "summary": summary, "details": details}
    write_json(output_json or cfg["eval"]["output_json"], report)
    print(summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_0.5b_2080.yaml")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--test-jsonl", default="")
    args = parser.parse_args()
    evaluate(args.config, adapter=args.adapter, output_json=args.output_json, test_jsonl=args.test_jsonl)


if __name__ == "__main__":
    main()
