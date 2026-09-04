"""Stage A protocol: generate, content-split, multi-layer NDI, behavior, controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from option_mismatch.behavior import analyze_behavior, summarize_behavior
from option_mismatch.controls import permute_vector, random_unit_vector, shuffle_options, zscore
from option_mismatch.io_utils import load_yaml, read_jsonl, write_json, write_jsonl
from option_mismatch.model_runtime import apply_chat, load_causal_lm, load_tokenizer, resolve_model_name
from option_mismatch.probe import build_diligence_vector, generate_solution, hidden_at_layer, last_token_hidden
from option_mismatch.prompts import DILIGENCE_PAIRS, chat_messages, format_mcq
from option_mismatch.span import phase3_token_start, split_ndi
from option_mismatch.stats import paired_drift_tests

DEFAULT_LAYERS = {
    "0.5b": [8, 12, 16, 20, 24],
    "1.5b": [10, 14, 18, 22, 28],
}


def infer_layer_set(cfg: dict[str, Any]) -> list[int]:
    name = str(cfg.get("model_name", "")).lower()
    if "1.5b" in name:
        return list(DEFAULT_LAYERS["1.5b"])
    return list(DEFAULT_LAYERS["0.5b"])


def build_task_vector(model, tokenizer, rows: list[dict], layer: int) -> np.ndarray:
    diligent, careless = [], []
    for row in rows:
        user = format_mcq(row["question"], row["options"])
        diligent.append(last_token_hidden(model, tokenizer, apply_chat(tokenizer, chat_messages(user, diligent=True)), layer))
        careless.append(last_token_hidden(model, tokenizer, apply_chat(tokenizer, chat_messages(user, diligent=False)), layer))
    v = np.mean(np.stack(diligent), axis=0) - np.mean(np.stack(careless), axis=0)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise RuntimeError("Degenerate task diligence vector")
    return v / norm


@torch.no_grad()
def generation_hidden_states(model, tokenizer, prompt: str, generation: str) -> tuple[int, tuple]:
    device = next(model.parameters()).device
    full = prompt + generation
    encoded = tokenizer(full, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    return prompt_len, out.hidden_states


def project_layer(hidden_states, layer: int, prompt_len: int, v: np.ndarray) -> np.ndarray:
    h = hidden_at_layer(hidden_states, layer)[0].float()
    h = torch.nn.functional.normalize(h, dim=-1)
    vec = torch.tensor(v, device=h.device, dtype=h.dtype)
    ndi = torch.matmul(h, vec).cpu().numpy()
    return ndi[max(prompt_len - 1, 0) :]


def _layer_report(phase2: np.ndarray, phase3: np.ndarray) -> dict[str, Any]:
    raw = paired_drift_tests(phase2, phase3)
    all_vals = np.concatenate([phase2, phase3])
    z_all = zscore(all_vals)
    z_phase2 = z_all[: len(phase2)]
    z_phase3 = z_all[len(phase2) :]
    z_stats = paired_drift_tests(z_phase2, z_phase3)
    raw["zscored"] = {
        k: z_stats[k]
        for k in ("phase2_mean", "phase3_mean", "delta_ndi_drift", "cohen_d", "p_ttest", "p_wilcoxon", "h1_supported")
    }
    return raw


def run_stage_a(cfg_path: str, overrides: dict | None = None) -> dict[str, Any]:
    cfg = load_yaml(cfg_path)
    overrides = overrides or {}
    if overrides.get("local_model_dir"):
        cfg["local_model_dir"] = overrides["local_model_dir"]
    if overrides.get("test_jsonl"):
        cfg["data"]["test_jsonl"] = overrides["test_jsonl"]
    if overrides.get("num_samples") is not None:
        cfg["data"]["num_probe_samples"] = int(overrides["num_samples"])
    if overrides.get("max_new_tokens") is not None:
        cfg["data"]["max_new_tokens"] = int(overrides["max_new_tokens"])

    model_name = resolve_model_name(cfg)
    rows = read_jsonl(cfg["data"]["test_jsonl"], limit=int(cfg["data"]["num_probe_samples"]))
    if not rows:
        raise FileNotFoundError(cfg["data"]["test_jsonl"])

    holdout_n = int(overrides.get("holdout_n") or cfg.get("probe", {}).get("holdout_n", 8))
    holdout, eval_rows = rows[:holdout_n], rows[holdout_n:]
    if not eval_rows:
        eval_rows = rows
        holdout = []

    layers = list(overrides.get("layers") or infer_layer_set(cfg))
    primary_layer = int(overrides.get("probe_layer") or cfg.get("probe_layer") or layers[len(layers) // 2])
    if primary_layer not in layers:
        layers.append(primary_layer)

    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(model_name)
    synthetic_v = {
        layer: build_diligence_vector(model, tokenizer, layer, DILIGENCE_PAIRS)
        for layer in layers
    }
    task_v = {}
    if holdout:
        for layer in layers:
            task_v[layer] = build_task_vector(model, tokenizer, holdout, layer)
    else:
        task_v = synthetic_v

    out_json = Path(overrides.get("output_json") or "reports/stage_a_h1.json")
    gen_path = Path(overrides.get("generations_jsonl") or Path("results/generations") / f"{out_json.stem}.jsonl")
    do_shuffle = bool(overrides.get("shuffle_control", True))
    max_new = int(cfg["data"]["max_new_tokens"])

    traces = []
    for i, row in enumerate(tqdm(eval_rows, desc="stage-a")):
        options = list(row["options"])
        prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], options), diligent=True))
        generation = generate_solution(model, tokenizer, prompt, max_new)
        span = phase3_token_start(tokenizer, generation, options=options, fallback_frac=float(cfg["data"].get("phase3_token_frac", 0.3)))
        behavior = analyze_behavior(generation, options, row.get("correct", ""))
        prompt_len, hidden = generation_hidden_states(model, tokenizer, prompt, generation)
        layer_ndi = {}
        for layer in layers:
            ndi = project_layer(hidden, layer, prompt_len, task_v[layer])
            p2, p3 = split_ndi(ndi, span["token_start"])
            rand = project_layer(hidden, layer, prompt_len, random_unit_vector(len(task_v[layer]), seed=1000 + layer))
            perm = project_layer(hidden, layer, prompt_len, permute_vector(task_v[layer], seed=2000 + layer))
            rp2, rp3 = split_ndi(rand, span["token_start"])
            pp2, pp3 = split_ndi(perm, span["token_start"])
            layer_ndi[str(layer)] = {
                "ndi": ndi.tolist(),
                "phase2": p2,
                "phase3": p3,
                "random_phase2": rp2,
                "random_phase3": rp3,
                "perm_phase2": pp2,
                "perm_phase3": pp3,
            }

        shuffled_trace = None
        if do_shuffle:
            sh_opts = shuffle_options(options, seed=42 + i)
            sh_prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], sh_opts), diligent=True))
            sh_gen = generate_solution(model, tokenizer, sh_prompt, max_new)
            sh_span = phase3_token_start(tokenizer, sh_gen, options=sh_opts)
            sh_len, sh_hidden = generation_hidden_states(model, tokenizer, sh_prompt, sh_gen)
            sh_ndi = project_layer(sh_hidden, primary_layer, sh_len, task_v[primary_layer])
            shp2, shp3 = split_ndi(sh_ndi, sh_span["token_start"])
            shuffled_trace = {"generation": sh_gen, "phase2": shp2, "phase3": shp3, "options": sh_opts}

        traces.append(
            {
                "question": row["question"],
                "options": options,
                "correct": row.get("correct"),
                "prompt": prompt,
                "generation": generation,
                "span": span,
                "behavior": behavior,
                "layers": layer_ndi,
                "shuffle": shuffled_trace,
            }
        )

    write_jsonl(gen_path, traces)

    def collect(getter) -> tuple[np.ndarray, np.ndarray]:
        a, b = [], []
        for tr in traces:
            x, y = getter(tr)
            if x == x and y == y:
                a.append(x)
                b.append(y)
        return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    layer_stats = {}
    for layer in layers:
        p2, p3 = collect(lambda tr, lyr=layer: (tr["layers"][str(lyr)]["phase2"], tr["layers"][str(lyr)]["phase3"]))
        r2, r3 = collect(lambda tr, lyr=layer: (tr["layers"][str(lyr)]["random_phase2"], tr["layers"][str(lyr)]["random_phase3"]))
        q2, q3 = collect(lambda tr, lyr=layer: (tr["layers"][str(lyr)]["perm_phase2"], tr["layers"][str(lyr)]["perm_phase3"]))
        layer_stats[str(layer)] = {
            "task_v": _layer_report(p2, p3),
            "random_v": _layer_report(r2, r3),
            "permuted_v": _layer_report(q2, q3),
        }

    sh2, sh3 = collect(lambda tr: (tr["shuffle"]["phase2"], tr["shuffle"]["phase3"]) if tr.get("shuffle") else (float("nan"), float("nan")))
    behavior_summary = summarize_behavior([tr["behavior"] for tr in traces])
    primary = layer_stats[str(primary_layer)]["task_v"]
    random_primary = layer_stats[str(primary_layer)]["random_v"]
    shuffle_stats = _layer_report(sh2, sh3) if len(sh2) else {}
    rand_d = float(random_primary.get("cohen_d") or 0)
    prim_d = float(primary.get("cohen_d") or 1) or 1.0
    sh_d = float(shuffle_stats.get("cohen_d") or 0)
    artifact = bool(
        (random_primary.get("h1_supported") and rand_d > 0.8 * prim_d)
        or (shuffle_stats.get("h1_supported") and sh_d > 0.8 * prim_d)
    )
    report = {
        "model": model_name,
        "primary_layer": primary_layer,
        "holdout_n": len(holdout),
        "n_eval": len(traces),
        "behavior": behavior_summary,
        "layers": layer_stats,
        "shuffle_options": shuffle_stats,
        "h1_supported": bool(primary.get("h1_supported")),
        "control_artifact": artifact,
        "generations": str(gen_path),
    }
    write_json(out_json, report)
    np.savez(
        Path(overrides.get("output_npz") or "results/stage_a_vectors.npz"),
        **{f"task_v_{layer}": task_v[layer] for layer in layers},
        **{f"synthetic_v_{layer}": synthetic_v[layer] for layer in layers},
        primary_layer=np.array([primary_layer]),
    )
    return report
