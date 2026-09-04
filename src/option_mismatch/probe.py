"""Layer-16 NDI probe for H1: Phase II diligence vs Phase III collapse."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from option_mismatch.io_utils import load_yaml, read_jsonl, write_json
from option_mismatch.model_runtime import apply_chat, load_causal_lm, load_tokenizer, resolve_model_name
from option_mismatch.prompts import DILIGENCE_PAIRS, chat_messages, format_mcq
from option_mismatch.stats import paired_drift_tests


def hidden_at_layer(outputs, layer_1indexed: int) -> torch.Tensor:
    # hidden_states[0] = embedding; hidden_states[i] = block i
    states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs
    return states[layer_1indexed]


@torch.no_grad()
def last_token_hidden(model, tokenizer, text: str, layer: int) -> np.ndarray:
    device = next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    h = hidden_at_layer(out, layer)[0, -1]
    return torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()


@torch.no_grad()
def token_ndi_trace(model, tokenizer, prompt: str, generation: str, layer: int, v_dil: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    full = prompt + generation
    encoded = tokenizer(full, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    h = hidden_at_layer(out, layer)[0].float()
    h = torch.nn.functional.normalize(h, dim=-1)
    v = torch.tensor(v_dil, device=h.device, dtype=h.dtype)
    ndi = torch.matmul(h, v).cpu().numpy()
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    gen_ndi = ndi[max(prompt_len - 1, 0) :]
    return gen_ndi


def build_diligence_vector(model, tokenizer, layer: int, pairs: list[tuple[str, str]]) -> np.ndarray:
    diligent, careless = [], []
    for pos, neg in pairs:
        diligent.append(last_token_hidden(model, tokenizer, apply_chat(tokenizer, chat_messages(pos, diligent=True)), layer))
        careless.append(last_token_hidden(model, tokenizer, apply_chat(tokenizer, chat_messages(neg, diligent=False)), layer))
    v = np.mean(np.stack(diligent), axis=0) - np.mean(np.stack(careless), axis=0)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise RuntimeError("Degenerate diligence vector")
    return v / norm


def split_phases(ndi: np.ndarray, phase3_frac: float) -> tuple[float, float]:
    if len(ndi) == 0:
        return float("nan"), float("nan")
    cut = max(1, int(len(ndi) * (1.0 - phase3_frac)))
    phase2 = ndi[:cut]
    phase3 = ndi[cut:] if cut < len(ndi) else ndi[-1:]
    return float(np.mean(phase2)), float(np.mean(phase3))


def generate_solution(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def run_probe(cfg_path: str, overrides: dict | None = None) -> dict:
    cfg = load_yaml(cfg_path)
    overrides = overrides or {}
    if overrides.get("model_name"):
        cfg["model_name"] = overrides["model_name"]
        cfg["local_model_dir"] = ""
    if overrides.get("local_model_dir"):
        cfg["local_model_dir"] = overrides["local_model_dir"]
    if overrides.get("probe_layer") is not None:
        cfg["probe_layer"] = int(overrides["probe_layer"])
    data_cfg = cfg["data"]
    probe_cfg = cfg["probe"]
    if overrides.get("test_jsonl"):
        data_cfg["test_jsonl"] = overrides["test_jsonl"]
    if overrides.get("num_samples") is not None:
        data_cfg["num_probe_samples"] = int(overrides["num_samples"])
    if overrides.get("max_new_tokens") is not None:
        data_cfg["max_new_tokens"] = int(overrides["max_new_tokens"])
    if overrides.get("output_json"):
        probe_cfg["output_json"] = overrides["output_json"]
    if overrides.get("output_npz"):
        probe_cfg["output_npz"] = overrides["output_npz"]

    model_name = resolve_model_name(cfg)
    layer = int(cfg["probe_layer"])
    rows = read_jsonl(data_cfg["test_jsonl"], limit=int(data_cfg["num_probe_samples"]))
    if not rows:
        raise FileNotFoundError(f"No probe rows in {data_cfg['test_jsonl']}; run scripts/prepare_data.py")

    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(model_name)
    pairs = DILIGENCE_PAIRS[: int(probe_cfg.get("diligence_pairs", len(DILIGENCE_PAIRS)))]
    v_dil = build_diligence_vector(model, tokenizer, layer, pairs)

    phase2_vals, phase3_vals, traces = [], [], []
    for row in tqdm(rows, desc="probe"):
        prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], row["options"]), diligent=True))
        generation = generate_solution(model, tokenizer, prompt, int(data_cfg["max_new_tokens"]))
        ndi = token_ndi_trace(model, tokenizer, prompt, generation, layer, v_dil)
        p2, p3 = split_phases(ndi, float(data_cfg["phase3_token_frac"]))
        phase2_vals.append(p2)
        phase3_vals.append(p3)
        traces.append({"question": row["question"][:160], "correct": row.get("correct"), "generation": generation, "ndi": ndi.tolist(), "phase2": p2, "phase3": p3})

    stats = paired_drift_tests(np.asarray(phase2_vals, dtype=np.float64), np.asarray(phase3_vals, dtype=np.float64))
    out_npz = Path(probe_cfg["output_npz"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, diligence_vector=v_dil, layer=np.array([layer]), phase2=np.asarray(phase2_vals), phase3=np.asarray(phase3_vals))
    report = {"model": model_name, "layer": layer, "stats": stats, "traces": traces}
    write_json(probe_cfg["output_json"], report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_0.5b_2080.yaml")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--local-model-dir", default="")
    parser.add_argument("--probe-layer", type=int, default=None)
    parser.add_argument("--test-jsonl", default="")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--protocol", action="store_true", help="Run Stage A content-split protocol")
    args = parser.parse_args()
    if args.protocol:
        from option_mismatch.protocol import run_stage_a

        report = run_stage_a(
            args.config,
            {
                "local_model_dir": args.local_model_dir,
                "probe_layer": args.probe_layer,
                "test_jsonl": args.test_jsonl,
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
                "output_json": args.output_json,
                "output_npz": args.output_npz,
            },
        )
        print(json_pretty({k: report[k] for k in ("primary_layer", "n_eval", "behavior", "h1_supported", "control_artifact") if k in report}))
        print(json_pretty(report.get("layers", {}).get(str(report.get("primary_layer")), {})))
        return

    report = run_probe(
        args.config,
        {
            "model_name": args.model_name,
            "local_model_dir": args.local_model_dir,
            "probe_layer": args.probe_layer,
            "test_jsonl": args.test_jsonl,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "output_json": args.output_json,
            "output_npz": args.output_npz,
        },
    )
    print(json_pretty(report["stats"]))


def json_pretty(payload) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
