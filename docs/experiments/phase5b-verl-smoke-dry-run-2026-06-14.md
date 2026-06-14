# Phase 5B Verl Smoke Launcher Dry Run

Date: 2026-06-14

## Goal

Add and validate a repo-local tiny verl / GRPO training smoke launcher without
starting real GPU training.

The dry run should:

- load `configs/experiments/phase5b_tiny_grpo_smoke.yaml`
- read Phase 4G GRPO rollouts
- generate tiny train/val files
- write parquet files for verl
- write an auditable launch command
- stop before launching `verl.trainer.main_ppo`

## Code And Environment

- Local repo: `D:\resume\Agent RL`
- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- GitHub branch: `main`
- Commit after Phase 5B fix: `1131b1e`
- Remote env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Remote Python packages previously verified:
  - `verl==0.8.0`
  - `vllm==0.12.0`
  - `torch==2.9.0`
  - `transformers==4.57.6`
  - `ray==2.55.1`
  - `yaml`, `pandas`, and `pyarrow` available

No tmux session was started. No real training was launched.

## Implementation Summary

Added:

- `src/lightningsearch_rl/verl_smoke.py`
- `configs/experiments/phase5b_tiny_grpo_smoke.yaml`
- `python -m lightningsearch_rl.cli train ...`
- tests for config loading, safe path checks, dry-run artifacts, CLI dry-run,
  command quoting, and execute-mode parquet safety

The launcher writes:

- `data/train.jsonl`
- `data/val.jsonl`
- `data/train.parquet`
- `data/val.parquet`
- `manifest.json`
- `launch_command.txt`
- `dry_run_summary.json`

## Local Verification

Feature branch and merged master both passed:

```text
66 passed
```

Manual config check passed with `PYTHONPATH=src`:

```text
config ok
```

## Remote Preflight

Preflight command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'pwd && mkdir -p /data/wzl/LightningSearch-RL/{repo,.conda-envs,data,indexes,logs,results,checkpoints,runs} && mkdir -p /home/user/wzl && ln -sfn /data/wzl/LightningSearch-RL /home/user/wzl/LightningSearch-RL && test -d /data/wzl/LightningSearch-RL && command -v tmux && source /home/user/anaconda3/etc/profile.d/conda.sh && conda --version && nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader && df -h /data/wzl/LightningSearch-RL'
```

Raw output:

```text
/home/user
/usr/bin/tmux
conda 26.1.1
0, 18459 MiB, 32607 MiB
1, 18403 MiB, 32607 MiB
2, 18469 MiB, 32607 MiB
3, 25985 MiB, 32607 MiB
4, 26101 MiB, 32607 MiB
5, 17685 MiB, 32607 MiB
6, 17333 MiB, 32607 MiB
7, 14192 MiB, 32607 MiB
/dev/sda1 7.3T 2.1T 4.8T 31% /data
```

Interpretation: remote environment and disk were usable. All GPUs were above
the project's `<4000 MiB used` free threshold at preflight time, so this phase
did not select a GPU for launch.

## Remote Sync And Test

Sync method:

```text
git archive HEAD -> scp tar -> extract under /data/wzl/LightningSearch-RL/repo
```

First remote pytest after the initial sync failed:

```text
1 failed, 65 passed
```

Root cause: command quoting was not stable across Windows and Linux. Linux paths
without spaces were not single-quoted by `shlex.quote`, while the test expected
auditable single-quoted Hydra overrides. Fixed by forcing single quotes around
all Hydra override arguments.

After fix and re-sync:

```text
66 passed in 0.58s
```

## Dry-Run Command

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train --config configs/experiments/phase5b_tiny_grpo_smoke.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke --dry-run --print-command'
```

Printed launch command:

```bash
PYTHONNOUSERSITE=1 python -m verl.trainer.main_ppo 'data.train_files=/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke/data/train.parquet' 'data.val_files=/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke/data/val.parquet' 'data.train_batch_size=4' 'data.max_prompt_length=512' 'data.max_response_length=256' 'actor_rollout_ref.model.path=Qwen/Qwen3-4B' 'actor_rollout_ref.rollout.name=hf' 'actor_rollout_ref.rollout.tensor_model_parallel_size=1' 'actor_rollout_ref.rollout.n=2' 'actor_rollout_ref.rollout.gpu_memory_utilization=0.25' 'actor_rollout_ref.actor.ppo_mini_batch_size=2' 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1' 'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048' 'algorithm.adv_estimator=grpo' 'trainer.project_name=lightningsearch-rl' 'trainer.experiment_name=phase5b-tiny-grpo-smoke' 'trainer.total_training_steps=1' 'trainer.total_epochs=1' 'trainer.n_gpus_per_node=1' 'trainer.nnodes=1' 'trainer.save_freq=1' 'trainer.test_freq=-1' 'trainer.logger=["console"]' 'trainer.default_local_dir=/data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke'
```

## Dry-Run Result

Summary:

```json
{
  "checkpoint_dir": "/data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke",
  "dry_run": true,
  "execute": false,
  "experiment_name": "phase5b-tiny-grpo-smoke",
  "output_dir": "/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke",
  "parquet_written": true,
  "project_name": "lightningsearch-rl",
  "train_rows": 16,
  "val_rows": 4
}
```

Generated files:

```text
/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke/data/train.jsonl
/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke/data/train.parquet
/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke/data/val.jsonl
/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke/data/val.parquet
/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke/manifest.json
/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke/launch_command.txt
/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke/dry_run_summary.json
```

File sizes:

```text
train.jsonl   16K
train.parquet 13K
val.jsonl    3.8K
val.parquet   10K
```

## Analysis

The Phase 5B launcher is ready for an approval-gated tiny training smoke. It
successfully converts existing GRPO rollouts into verl-readable parquet and
generates a minimal `verl.trainer.main_ppo` command configured with
`algorithm.adv_estimator=grpo`.

Important caveat: this dry run validates data preparation and command
generation, not that verl will complete a training step. The actual training
smoke may still expose model download, tokenizer, Ray, or verl config issues.
That is exactly why the next run should be one GPU, one step, and tmux-logged.

## Next Step

Wait for a GPU with used memory below `4000 MiB`, then provide the required
approval report before launching:

- repo / branch / commit
- conda env
- selected GPU and count
- exact tmux session
- exact launch command
- log path
- result and checkpoint paths
- success criteria

Do not launch until the user approves that exact report.
