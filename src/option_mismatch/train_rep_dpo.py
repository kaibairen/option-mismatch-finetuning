"""LoRA Rep-DPO: keep Phase III hidden states aligned with the diligence vector."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from option_mismatch.io_utils import load_yaml, read_jsonl, write_json
from option_mismatch.model_runtime import apply_chat, load_causal_lm, load_tokenizer, resolve_model_name
from option_mismatch.prompts import chat_messages, format_mcq


class PreferenceDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.samples = []
        for row in rows:
            prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], row["options"]), diligent=True))
            chosen = (row.get("rationale") or "").strip()
            if not chosen:
                continue
            if "Final answer" not in chosen:
                chosen = f"{chosen}\nFinal answer: {row.get('correct', 'A')}"
            rejected = (
                "I already have a rough number so I will just pick a letter.\n"
                f"Final answer: {guess_wrong(row.get('correct', 'A'))}"
            )
            self.samples.append(
                {
                    "chosen": truncate(tokenizer, prompt + chosen, max_length),
                    "rejected": truncate(tokenizer, prompt + rejected, max_length),
                    "prompt_len": len(tokenizer(prompt)["input_ids"]),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def guess_wrong(correct: str) -> str:
    letters = [c for c in "ABCDE" if c != (correct or "A")[:1]]
    return letters[0] if letters else "B"


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
    }


def logp(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**batch, output_hidden_states=True)
    logits = out.logits[:, :-1]
    labels = batch["input_ids"][:, 1:]
    mask = batch["attention_mask"][:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    return token_logp.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1), out


def phase3_ndi(hidden_states, layer: int, prompt_len: torch.Tensor, v_dil: torch.Tensor, phase3_frac: float) -> torch.Tensor:
    h = hidden_states[layer]
    h = torch.nn.functional.normalize(h.float(), dim=-1)
    scores = torch.matmul(h, v_dil.to(h.dtype))
    vals = []
    for i in range(scores.size(0)):
        start = int(prompt_len[i].item())
        gen = scores[i, start:]
        if gen.numel() == 0:
            vals.append(scores[i, -1])
            continue
        cut = max(1, int(gen.numel() * (1.0 - phase3_frac)))
        vals.append(gen[cut:].mean())
    return torch.stack(vals)


def train(cfg_path: str) -> dict:
    cfg = load_yaml(cfg_path)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_name = resolve_model_name(cfg)
    layer = int(cfg["probe_layer"])

    tokenizer = load_tokenizer(model_name)
    base = load_causal_lm(model_name, for_training=True)
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

    vec_path = Path(cfg["probe"]["output_npz"])
    if vec_path.exists():
        v_np = np.load(vec_path)["diligence_vector"]
    else:
        v_np = np.zeros((int(cfg["hidden_size"]),), dtype=np.float32)
        v_np[0] = 1.0
    v_dil = torch.tensor(v_np, device=next(model.parameters()).device)
    v_dil = torch.nn.functional.normalize(v_dil.float(), dim=-1)

    rows = read_jsonl(data_cfg.get("train_jsonl", data_cfg["test_jsonl"]), limit=int(train_cfg["max_train_samples"]))
    if not rows:
        rows = read_jsonl(data_cfg["test_jsonl"], limit=int(train_cfg["max_train_samples"]))
    dataset = PreferenceDataset(rows, tokenizer, int(train_cfg["max_length"]))
    if len(dataset) == 0:
        raise RuntimeError("Preference dataset is empty; prepare AQUA jsonl first")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(train_cfg["learning_rate"]))
    beta = float(train_cfg["beta"])
    rep_lambda = float(train_cfg["rep_lambda"])
    target = float(train_cfg["phase3_target_ndi"])
    phase3_frac = float(data_cfg["phase3_token_frac"])
    accum = int(train_cfg["grad_accum"])
    history = []
    model.train()
    step = 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(int(train_cfg["num_epochs"])):
        for batch in tqdm(loader, desc=f"rep-dpo epoch {epoch}"):
            device = next(model.parameters()).device
            chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
            chosen_logp, chosen_out = logp(model, chosen)
            rejected_logp, _ = logp(model, rejected)
            dpo = -torch.nn.functional.logsigmoid(beta * (chosen_logp - rejected_logp)).mean()
            ndi = phase3_ndi(chosen_out.hidden_states, layer, batch["prompt_len"].to(device), v_dil, phase3_frac)
            rep = torch.relu(target - ndi).mean()
            loss = dpo + rep_lambda * rep
            (loss / accum).backward()
            step += 1
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            history.append({"loss": float(loss.detach()), "dpo": float(dpo.detach()), "rep": float(rep.detach())})

    out_dir = Path(train_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    write_json(out_dir / "train_history.json", history[-50:])
    return {"output_dir": str(out_dir), "steps": len(history), "last": history[-1] if history else {}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_0.5b_2080.yaml")
    args = parser.parse_args()
    result = train(args.config)
    print(result)


if __name__ == "__main__":
    main()
