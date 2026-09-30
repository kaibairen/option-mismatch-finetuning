"""Uniform SFT, reference-model DPO, and DECORD-FT (answer-weighted SFT + letter margin)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from option_mismatch.behavior import analyze_behavior
from option_mismatch.decord import letter_token_ids
from option_mismatch.io_utils import load_yaml, read_jsonl, write_json, write_jsonl
from option_mismatch.losses import completion_token_mask, dpo_loss, letter_margin_loss, sequence_logprob
from option_mismatch.model_runtime import load_tokenizer, resolve_model_name
from option_mismatch.preferences import chosen_text
from option_mismatch.probe import generate_solution
from option_mismatch.prompts import chat_messages, format_mcq
from option_mismatch.model_runtime import apply_chat


def load_trainable(model_name: str, train_cfg: dict, init_adapter: str = ""):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.to("cuda")
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            r=int(train_cfg["lora_r"]),
            lora_alpha=int(train_cfg["lora_alpha"]),
            lora_dropout=float(train_cfg["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model


def pack(tokenizer, prompt: str, continuation: str, max_length: int) -> tuple[list[int], int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if len(cont_ids) > max_length - 8:
        cont_ids = cont_ids[-(max_length - 8) :]
    room = max_length - len(cont_ids)
    if room < 1:
        room = 1
        cont_ids = cont_ids[-(max_length - 1) :]
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]
    return prompt_ids + cont_ids, len(prompt_ids)


def answer_index(ids: list[int], prompt_len: int, gold_token: int) -> int:
    for index in range(len(ids) - 1, prompt_len - 1, -1):
        if ids[index] == gold_token:
            return index
    return -1


class SftDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def collate_sft(rows: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    ids, mask = [], []
    for row in rows:
        pad = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * pad)
        mask.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "prompt_len": torch.tensor([row["prompt_len"] for row in rows], dtype=torch.long),
        "answer_index": torch.tensor([row["answer_index"] for row in rows], dtype=torch.long),
        "gold_token": torch.tensor([row["gold_token"] for row in rows], dtype=torch.long),
    }


def collate_dpo(rows: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    def pad_side(key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(len(row[key]) for row in rows)
        ids, mask = [], []
        for row in rows:
            seq = row[key]
            pad = width - len(seq)
            ids.append(seq + [pad_id] * pad)
            mask.append([1] * len(seq) + [0] * pad)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor([row[key.replace("ids", "prompt_len")] for row in rows], dtype=torch.long),
        )

    chosen_ids, chosen_mask, chosen_prompt = pad_side("chosen_ids")
    rejected_ids, rejected_mask, rejected_prompt = pad_side("rejected_ids")
    return {
        "chosen_ids": chosen_ids,
        "chosen_mask": chosen_mask,
        "chosen_prompt_len": chosen_prompt,
        "rejected_ids": rejected_ids,
        "rejected_mask": rejected_mask,
        "rejected_prompt_len": rejected_prompt,
        "ref_chosen": torch.tensor([row["ref_chosen"] for row in rows], dtype=torch.float32),
        "ref_rejected": torch.tensor([row["ref_rejected"] for row in rows], dtype=torch.float32),
        "answer_index": torch.tensor([row["answer_index"] for row in rows], dtype=torch.long),
        "gold_token": torch.tensor([row["gold_token"] for row in rows], dtype=torch.long),
    }


def token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1), logits


def sft_objective(
    model,
    batch: dict[str, torch.Tensor],
    *,
    answer_weight: float,
    margin_weight: float,
    margin: float,
    letter_ids: dict[str, int],
) -> torch.Tensor:
    token_logp, logits = token_logprobs(model, batch["input_ids"], batch["attention_mask"])
    mask = completion_token_mask(batch["attention_mask"], batch["prompt_len"])
    weights = mask.clone()
    if answer_weight != 1.0:
        for row in range(weights.size(0)):
            idx = (mask[row] > 0).nonzero(as_tuple=False).flatten()
            if len(idx):
                weights[row, idx[-4:]] = weights[row, idx[-4:]] * answer_weight
    nll = -(token_logp * weights).sum() / weights.sum().clamp(min=1.0)
    if margin_weight <= 0:
        return nll
    return nll + margin_weight * margin_on_answer(logits, batch, letter_ids, margin)


def policy_logp(model, input_ids, attention_mask, prompt_len) -> torch.Tensor:
    token_logp, logits = token_logprobs(model, input_ids, attention_mask)
    mask = completion_token_mask(attention_mask, prompt_len)
    return sequence_logprob(token_logp, mask), logits


def margin_on_answer(logits, batch: dict[str, torch.Tensor], letter_ids: dict[str, int], margin: float) -> torch.Tensor:
    pieces = []
    all_ids = [letter_ids[letter] for letter in "ABCDE" if letter in letter_ids]
    for row, pos in enumerate(batch["answer_index"].tolist()):
        if pos <= 0 or pos - 1 >= logits.size(1):
            continue
        gold = int(batch["gold_token"][row].item())
        others = [token for token in all_ids if token != gold]
        if not others or gold < 0:
            continue
        pieces.append(
            letter_margin_loss(
                logits[row : row + 1, pos - 1, :],
                torch.tensor([gold], device=logits.device),
                torch.tensor(others, device=logits.device),
                margin,
            )
        )
    if not pieces:
        return logits.new_zeros(())
    return torch.stack(pieces).mean()


def prepare_sft_rows(tokenizer, rows: list[dict], max_length: int, letter_ids: dict[str, int]) -> list[dict]:
    packed = []
    for row in rows:
        prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], row["options"]), diligent=True))
        continuation = chosen_text(row)
        ids, prompt_len = pack(tokenizer, prompt, continuation, max_length)
        gold = str(row.get("correct") or "A").strip().upper()[:1]
        gold_token = letter_ids.get(gold, -1)
        packed.append(
            {
                "input_ids": ids,
                "prompt_len": prompt_len,
                "answer_index": answer_index(ids, prompt_len, gold_token),
                "gold_token": gold_token,
            }
        )
    return packed


def prepare_dpo_rows(tokenizer, pairs: list[dict], max_length: int, letter_ids: dict[str, int]) -> list[dict]:
    packed = []
    for row in pairs:
        chosen_ids, chosen_prompt = pack(tokenizer, row["prompt"], row["chosen"], max_length)
        rejected_ids, rejected_prompt = pack(tokenizer, row["prompt"], row["rejected"], max_length)
        gold = str(row.get("correct") or "A").strip().upper()[:1]
        gold_token = letter_ids.get(gold, -1)
        packed.append(
            {
                "chosen_ids": chosen_ids,
                "chosen_prompt_len": chosen_prompt,
                "rejected_ids": rejected_ids,
                "rejected_prompt_len": rejected_prompt,
                "answer_index": answer_index(chosen_ids, chosen_prompt, gold_token),
                "gold_token": gold_token,
                "ref_chosen": float(row.get("ref_chosen") or 0.0),
                "ref_rejected": float(row.get("ref_rejected") or 0.0),
            }
        )
    return packed


@torch.no_grad()
def fill_reference_logps(model, tokenizer, pairs: list[dict], max_length: int) -> list[dict]:
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    updated = []
    for row in tqdm(pairs, desc="ref-logp"):
        chosen_ids, chosen_prompt = pack(tokenizer, row["prompt"], row["chosen"], max_length)
        rejected_ids, rejected_prompt = pack(tokenizer, row["prompt"], row["rejected"], max_length)
        device = next(model.parameters()).device

        def one(ids, prompt_len) -> float:
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            mask = torch.ones_like(input_ids)
            value, _ = policy_logp(model, input_ids, mask, torch.tensor([prompt_len], device=device))
            return float(value.item())

        copied = dict(row)
        copied["ref_chosen"] = one(chosen_ids, chosen_prompt)
        copied["ref_rejected"] = one(rejected_ids, rejected_prompt)
        updated.append(copied)
    return updated


def build_preferences(model, tokenizer, rows: list[dict], max_new_tokens: int, output_jsonl: str) -> list[dict]:
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    pairs = []
    for row in tqdm(rows, desc="pref"):
        prompt = apply_chat(tokenizer, chat_messages(format_mcq(row["question"], row["options"]), diligent=True))
        chosen = chosen_text(row)
        rejected = generate_solution(model, tokenizer, prompt, max_new_tokens)
        behavior = analyze_behavior(rejected, row["options"], row.get("correct", ""))
        gold = str(row.get("correct") or "A").strip().upper()[:1]
        if behavior["letter_correct"] or not rejected.strip():
            wrong = next(letter for letter in "ABCDE" if letter != gold)
            rejected = f"{rejected.strip()}\nFinal answer: {wrong}".strip()
        pairs.append(
            {
                "question": row["question"],
                "options": list(row["options"]),
                "correct": gold,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    write_jsonl(output_jsonl, pairs)
    return pairs


def _run_epochs(model, loader, optimizer, epochs: int, accum: int, step_fn) -> list[dict]:
    scaler = torch.amp.GradScaler("cuda")
    history = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            batch = {key: value.cuda() for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = step_fn(batch)
            scaler.scale(loss / accum).backward()
            step += 1
            if step % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            history.append({"loss": float(loss.detach().float().cpu())})
            if step % 20 == 0:
                print({"step": step, "loss": history[-1]["loss"]}, flush=True)
    return history


def train_sft(
    cfg: dict,
    rows: list[dict],
    output_dir: str,
    *,
    model_name: str,
    epochs: int,
    answer_weight: float,
    margin_weight: float,
    margin: float,
    lr: float,
    init_adapter: str = "",
) -> dict:
    tokenizer = load_tokenizer(model_name)
    tokenizer.padding_side = "right"
    letter_ids = letter_token_ids(tokenizer)
    max_length = int(cfg["train"]["max_length"])
    packed = prepare_sft_rows(tokenizer, rows, max_length, letter_ids)
    model = load_trainable(model_name, cfg["train"], init_adapter)
    loader = DataLoader(
        SftDataset(packed),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        collate_fn=lambda batch: collate_sft(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()

    def step_fn(batch):
        return sft_objective(
            model,
            batch,
            answer_weight=answer_weight,
            margin_weight=margin_weight,
            margin=margin,
            letter_ids=letter_ids,
        )

    history = _run_epochs(model, loader, optimizer, epochs, int(cfg["train"]["grad_accum"]), step_fn)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    write_json(
        dest / "train_meta.json",
        {"answer_weight": answer_weight, "margin_weight": margin_weight, "margin": margin, "steps": len(history), "last": history[-1] if history else {}},
    )
    return {"output_dir": str(dest), "steps": len(history)}


def train_dpo(
    cfg: dict,
    pairs: list[dict],
    output_dir: str,
    *,
    model_name: str,
    init_adapter: str,
    epochs: int,
    lr: float,
    beta: float,
    margin_weight: float,
    margin: float,
) -> dict:
    tokenizer = load_tokenizer(model_name)
    tokenizer.padding_side = "right"
    letter_ids = letter_token_ids(tokenizer)
    max_length = int(cfg["train"]["max_length"])
    model = load_trainable(model_name, cfg["train"], init_adapter)
    pairs = fill_reference_logps(model, tokenizer, pairs, max_length)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    packed = prepare_dpo_rows(tokenizer, pairs, max_length, letter_ids)
    loader = DataLoader(
        SftDataset(packed),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        collate_fn=lambda batch: collate_dpo(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()

    def step_fn(batch):
        chosen, chosen_logits = policy_logp(model, batch["chosen_ids"], batch["chosen_mask"], batch["chosen_prompt_len"])
        rejected, _ = policy_logp(model, batch["rejected_ids"], batch["rejected_mask"], batch["rejected_prompt_len"])
        loss = dpo_loss(chosen, rejected, batch["ref_chosen"], batch["ref_rejected"], beta).mean()
        if margin_weight <= 0:
            return loss
        return loss + margin_weight * margin_on_answer(chosen_logits, batch, letter_ids, margin)

    history = _run_epochs(model, loader, optimizer, epochs, int(cfg["train"]["grad_accum"]), step_fn)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    write_json(dest / "train_meta.json", {"beta": beta, "margin_weight": margin_weight, "steps": len(history), "last": history[-1] if history else {}})
    return {"output_dir": str(dest), "steps": len(history)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_0.5b_2080.yaml")
    parser.add_argument("--method", required=True, choices=["sft", "decord_sft", "dpo", "decord_dpo", "prefs"])
    parser.add_argument("--local-model-dir", default="")
    parser.add_argument("--train-jsonl", default="")
    parser.add_argument("--pref-jsonl", default="")
    parser.add_argument("--init-adapter", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=0)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.local_model_dir:
        cfg["local_model_dir"] = args.local_model_dir
    if args.max_length:
        cfg["train"]["max_length"] = args.max_length
    model_name = resolve_model_name(cfg)
    rows = read_jsonl(args.train_jsonl) if args.train_jsonl else []
    if args.method == "prefs":
        tokenizer = load_tokenizer(model_name)
        model = load_trainable(model_name, cfg["train"], args.init_adapter)
        model.eval()
        build_preferences(model, tokenizer, rows, args.max_new_tokens, args.output_dir)
        return
    if args.method == "sft":
        print(train_sft(cfg, rows, args.output_dir, model_name=model_name, epochs=args.epochs, answer_weight=1.0, margin_weight=0.0, margin=1.0, lr=args.lr))
    elif args.method == "decord_sft":
        print(
            train_sft(
                cfg,
                rows,
                args.output_dir,
                model_name=model_name,
                epochs=args.epochs,
                answer_weight=4.0,
                margin_weight=0.5,
                margin=1.0,
                lr=args.lr,
            )
        )
    else:
        pairs = read_jsonl(args.pref_jsonl)
        margin_weight = 0.25 if args.method == "decord_dpo" else 0.0
        print(
            train_dpo(
                cfg,
                pairs,
                args.output_dir,
                model_name=model_name,
                init_adapter=args.init_adapter,
                epochs=args.epochs,
                lr=args.lr,
                beta=float(cfg["train"]["beta"]),
                margin_weight=margin_weight,
                margin=1.0,
            )
        )


if __name__ == "__main__":
    main()
