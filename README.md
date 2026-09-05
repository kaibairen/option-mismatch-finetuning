# Option-mismatch fine-tuning

Reproduce **H1** (Phase III diligence collapse / option mismatch) on `Qwen2.5-0.5B-Instruct` and try to repair it with Phase-III-masked LoRA **Rep-DPO**. Primary GPU work ran on an AutoDL Tesla V100-32GB. This Cloud Agent VM has no GPU; it only prepares code, CPU deps, and AQUA-RAT splits.

The scientific goal is **not** “raise math score.” It is to reduce **Option Mismatch**: Phase II already has (or nearly has) the right intermediate result, but Phase III maps it to A–E and picks the wrong letter.

## Layout

```text
configs/                 # 0.5B / 1.5B experiment YAMLs
data/                    # raw + processed AQUA (gitignored except .gitkeep)
docs/                    # experiment design
reports/
  feasibility/           # ratio-cut 100-sample H1 probe
  stage_a/               # content-split H1 + controls
  stage_b/               # Base / SFT / DPO / Rep-DPO comparison
requirements/
  cpu.txt                # Cloud Agent / CI
  gpu.txt                # AutoDL CUDA image
  v100.txt               # slim V100 feasibility stack
scripts/
  autodl/                # instance client, bootstrap, ModelScope download
  ci/                    # Cloud Agent install
  data/                  # AQUA-RAT → jsonl
  experiments/           # V100 Stage A/B runners + report merges
src/option_mismatch/     # probe, protocol, Rep-DPO, eval, stats
tests/
```

Runtime artifacts (`results/`, adapters, caches, processed jsonl) stay local and are gitignored. Frozen metrics live under `reports/`.

## Local / Cloud Agent setup

```bash
bash scripts/ci/cloud_agent_install.sh
python scripts/autodl/client.py status   # needs AUTODL_TOKEN
```

## AutoDL workflow

1. Log in at [www.autodl.com](https://www.autodl.com), finish 实名认证.
2. Copy the developer token: 控制台 → 账号 → 设置 → 开发者 Token. Export `AUTODL_TOKEN`.
3. Create a GPU instance (V100 32GB was used for Stage A/B; RTX 2080 is the original 8 GB target).
4. On the instance:

```bash
git clone https://github.com/kaibairen/option-mismatch-finetuning.git
cd option-mismatch-finetuning
bash scripts/autodl/bootstrap.sh
# or the V100 path:
bash scripts/experiments/v100_feasibility.sh
bash scripts/experiments/v100_stage_a.sh
bash scripts/experiments/v100_stage_b.sh
```

Single-model entry points:

```bash
python -m option_mismatch.probe --protocol --config configs/qwen25_0.5b_2080.yaml
python -m option_mismatch.train_rep_dpo --config configs/qwen25_0.5b_2080.yaml
python -m option_mismatch.eval_mcq --config configs/qwen25_0.5b_2080.yaml --adapter results/adapters/rep_dpo_0.5b
```

Design notes: `docs/experiment_design.md`. Frozen numbers: `reports/stage_a/stage_a_h1.json`, `reports/stage_b/stage_b_repair.json`.
