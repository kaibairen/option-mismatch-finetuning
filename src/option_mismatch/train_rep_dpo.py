"""LoRA SFT / DPO / Rep-DPO with Phase-III-masked representation hinge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from option_mismatch.io_utils import load_yaml, read_jsonl, write_json
from option_mismatch.model_runtime import load_causal_lm, load_tokenizer, resolve_model_name
from option_mismatch.span import phase3_token_start


class PreferenceDataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer, max_length: int):
        self.samples = []
        for row in pairs:
            prompt = row.get("prompt") or ""
            chosen = row.get("chosen") or ""
            rejected = row.get("rejected") or ""
            if not prompt or not chosen:
                continue
            if not rejected:
                rejected = chosen
            chosen_enc = truncate(tokenizer, prompt + chosen, max_length)
            rejected_enc = truncate(tokenizer, prompt + rejected, max_length)
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            # After left-pad truncation the prompt length used for masks is min(prompt, seq)
            c_start = int(row.get("chosen_phase3_token") or 0)
            if "chosen_phase3_token" not in row:
                c_start = phase3_token_start(tokenizer, chosen).get("token_start", 0)
            self.samples.append(
                {
                    "chosen": chosen_enc,
                    "rejected": rejected_enc,
                    "prompt_len": prompt_len,
                    "phase3_from_prompt": c_start,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def truncate(tokenizer, text: str, max_length: int) -> dict[str, torch.Tensor]:
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    return {k: v.squeeze(0) for k, v in enc.items()}


def collate(batch: list[dict], pad_id: int) -> dict:
    def pad_group(key: str) -> dict[str, torch.Tensor]:
        items = [row[key] for row in batch]
        max_len = max(x["input_ids"].size(0) for x in items)
        ids, mask = [], []
        for item in items:
            pad = max_len - item["input_ids"].size(0)
            ids.append(torch.nn.functional.pad(item["input_ids"], (pad, 0), value=pad_id))
            mask.append(torch.nn.functional.pad(item["attention_mask"], (pad, 0), value=0))
        return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(mask)}

    return {
        "chosen": pad_group("chosen"),
        "rejected": pad_group("rejected"),
        "prompt_len": torch.tensor([row["prompt_len"] for row in batch], dtype=torch.long),
        "phase3_from_prompt": torch.tensor([row["phase3_from_prompt"] for row in batch], dtype=torch.long),
    }


def logp(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, Any]:
    out = model(**batch, output_hidden_states=True)
    logits = out.logits[:, :-1]
    labels = batch["input_ids"][:, 1:]
    mask = batch["attention_mask"][:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask
    return token_logp.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1), out


def sft_nll(model, batch: dict[str, torch.Tensor], prompt_len: torch.Tensor) -> torch.Tensor:
    out = model(**batch)
    logits = out.logits[:, :-1]
    labels = batch["input_ids"][:, 1:]
    mask = batch["attention_mask"][:, 1:].clone()
    seq_len = labels.size(1)
    for i, plen in enumerate(prompt_len.tolist()):
        # left-padded: response tokens are the tail after pad + prompt
        pad = int((batch["attention_mask"][i] == 0).sum().item())
        cutoff = min(seq_len, max(pad + int(plen) - 1, 0))
        mask[i, :cutoff] = 0
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    nll = -(log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=-1)
    return (nll / mask.sum(dim=-1).clamp(min=1)).mean()


def phase3_ndi_masked(hidden_states, layer: int, prompt_len: torch.Tensor, phase3_from_prompt: torch.Tensor, v_dil: torch.Tensor) -> torch.Tensor:
    h = hidden_states[layer]
    h = torch.nn.functional.normalize(h.float(), dim=-1)
    scores = torch.matmul(h, v_dil.to(dtype=h.dtype, device=h.device))
    vals = []
    for i in range(scores.size(0)):
        start = int(prompt_len[i].item()) + int(phase3_from_prompt[i].item())
        start = min(max(start, 0), scores.size(1) - 1)
        gen = scores[i, start:]
        vals.append(gen.mean() if gen.numel() else scores[i, -1])
    return torch.stack(vals)


def attach_lora(base, train_cfg: dict):
    lora = LoraConfig(
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    return model


def train(cfg_path: str, *, method: str = "rep_dpo", pref_jsonl: str = "", output_dir: str = "", rep_lambda: float | None = None) -> dict:
    cfg = load_yaml(cfg_path)
    train_cfg = cfg["train"]
    model_name = resolve_model_name(cfg)
    layer = int(cfg["probe_layer"])
    tokenizer = load_tokenizer(model_name)
    base = load_causal_lm(model_name, for_training=True)
    model = attach_lora(base, train_cfg)

    vec_path = Path(cfg["probe"].get("output_npz") or "results/stage_a_vectors.npz")
    if vec_path.exists():
        blob = np.load(vec_path)
        key = f"task_v_{layer}"
        v_np = blob[key] if key in blob.files else blob[blob.files[0]]
    else:
        v_np = np.zeros((int(cfg["hidden_size"]),), dtype=np.float32)
        v_np[0] = 1.0
    v_dil = torch.nn.functional.normalize(torch.tensor(v_np, dtype=torch.float32), dim=-1)

    pref_path = pref_jsonl or str(train_cfg.get("pref_jsonl") or "data/processed/aqua_pref_100.jsonl")
    pairs = read_jsonl(pref_path)
    if not pairs:
        from option_mismatch.preferences import build_preferences

        pairs = build_preferences(cfg_path, {"output_jsonl": pref_path})
    dataset = PreferenceDataset(pairs, tokenizer, int(train_cfg["max_length"]))
    if len(dataset) == 0:
        raise RuntimeError(f"empty preference set: {pref_path}")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(train_cfg["learning_rate"]))
    beta = float(train_cfg["beta"])
    lam = float(train_cfg["rep_lambda"] if rep_lambda is None else rep_lambda)
    if method == "dpo":
        lam = 0.0
    target = float(train_cfg.get("phase3_target_ndi") or 0.054)
    accum = int(train_cfg["grad_accum"])
    history = []
    model.train()
    step = 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(int(train_cfg["num_epochs"])):
        for batch in tqdm(loader, desc=f"{method} epoch {epoch}"):
            device = next(model.parameters()).device
            v_dev = v_dil.to(device)
            chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
            prompt_len = batch["prompt_len"].to(device)
            phase3 = batch["phase3_from_prompt"].to(device)
            if method == "sft":
                loss = sft_nll(model, chosen, prompt_len)
                dpo = loss.detach()
                rep = torch.zeros((), device=device)
            else:
                chosen_logp, chosen_out = logp(model, chosen)
                rejected_logp, _ = logp(model, rejected)
                dpo = -torch.nn.functional.logsigmoid(beta * (chosen_logp - rejected_logp)).mean()
                ndi = phase3_ndi_masked(chosen_out.hidden_states, layer, prompt_len, phase3, v_dev)
                rep = torch.relu(target - ndi).mean()
                loss = dpo + lam * rep
            (loss / accum).backward()
            step += 1
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            history.append({"loss": float(loss.detach()), "dpo": float(dpo.detach()), "rep": float(rep.detach())})

    dest = Path(output_dir or train_cfg["output_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    write_json(dest / "train_history.json", {"method": method, "rep_lambda": lam, "steps": history[-50:]})
    return {"output_dir": str(dest), "method": method, "steps": len(history), "last": history[-1] if history else {}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_0.5b_2080.yaml")
    parser.add_argument("--method", default="rep_dpo", choices=["sft", "dpo", "rep_dpo"])
    parser.add_argument("--pref-jsonl", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--rep-lambda", type=float, default=None)
    args = parser.parse_args()
    print(train(args.config, method=args.method, pref_jsonl=args.pref_jsonl, output_dir=args.output_dir, rep_lambda=args.rep_lambda))


if __name__ == "__main__":
    main()
