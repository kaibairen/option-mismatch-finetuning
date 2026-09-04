# Option-mismatch fine-tuning

Reproduce **H1** (Phase III diligence / cognitive collapse) on `Qwen2.5-0.5B-Instruct` and repair it with LoRA **Rep-DPO**. The intended GPU is an AutoDL **RTX 2080** (8 GB). This Cloud Agent VM has no GPU; it only prepares code, CPU deps, and AQUA-RAT splits.

## Why AutoDL needs a manual 2080

AutoDL’s official open API can create **Pro** SKUs only (`4090`, `3090`, `5090`, …). **RTX 2080 is a standard marketplace card** and is not in that catalog. After you add `AUTODL_TOKEN` and create a 2080 in the web console, this repo can check the account and run the probe/train stack on that box.

## Local / Cloud Agent setup

```bash
bash scripts/cloud_agent_install.sh
python scripts/autodl_client.py status   # needs AUTODL_TOKEN
```

## AutoDL 2080 workflow

1. Log in at [www.autodl.com](https://www.autodl.com), finish 实名认证.
2. Copy the developer token: 控制台 → 账号 → 设置 → 开发者 Token. Export `AUTODL_TOKEN`.
3. 算力市场 → filter **RTX 2080 / 2080 Super** → create 1 GPU, PyTorch + CUDA 11.8+ image → power on.
4. On the instance:

```bash
git clone https://github.com/kaibairen/option-mismatch-finetuning.git
cd option-mismatch-finetuning
bash scripts/autodl_bootstrap.sh
python -m option_mismatch.probe --config configs/qwen25_0.5b_2080.yaml
python -m option_mismatch.train_rep_dpo --config configs/qwen25_0.5b_2080.yaml
python -m option_mismatch.eval_mcq --config configs/qwen25_0.5b_2080.yaml --adapter results/rep_dpo_lora
```

H1 metrics are written to `results/h1_probe_report.json` (paired t-test, Wilcoxon, Cohen’s d). The original report claimed Phase II NDI `0.1126` vs Phase III `0.1035` (`p = 0.003304`).

## Layout

| Path | Role |
| --- | --- |
| `scripts/autodl_client.py` | Token login, wallet, instance list |
| `scripts/autodl_bootstrap.sh` | 2080 container deps |
| `scripts/prepare_data.py` | DeepMind AQUA-RAT → jsonl |
| `src/option_mismatch/probe.py` | Layer-16 NDI + H1 tests |
| `src/option_mismatch/train_rep_dpo.py` | LoRA Rep-DPO (DPO + Phase III NDI reward) |
| `configs/qwen25_0.5b_2080.yaml` | 0.5B / 2080 hyperparameters |
