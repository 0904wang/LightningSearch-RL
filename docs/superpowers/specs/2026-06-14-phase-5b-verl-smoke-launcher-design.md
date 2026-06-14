# Phase 5B Verl Smoke Launcher Design

## Goal

Add a repo-local, reproducible tiny verl / GRPO training smoke launcher for
LightningSearch-RL, without starting remote training as part of implementation.

The launcher should turn existing GRPO rollout artifacts into tiny verl-readable
training inputs, generate an auditable launch command, and support a dry-run mode
that validates paths and dependencies before any GPU job is started.

## Context

Current project state:

- The repo already exports deterministic GRPO artifacts:
  - `rollouts.jsonl`
  - `transitions.jsonl`
  - `reward_records.jsonl`
  - `summary.json`
- Phase 4G produced a strict 500-row synthetic dataset under:
  `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500`
- Phase 5A diagnostics showed the data is suitable for a pipeline smoke but not
  final training because duplicates remain high.
- The remote conda env contains `verl==0.8.0`, `vllm==0.12.0`, `torch==2.9.0`,
  `transformers==4.57.6`, and `ray==2.55.1`.
- The installed verl package exposes `verl.trainer.main_ppo`; GRPO is configured
  through `algorithm.adv_estimator=grpo`.
- The repo does not yet have a `train` CLI command, `configs/`, or a training
  launcher.

Remote experiment rules require:

- no real training launch without successful smoke checks and an approval report
- all outputs under `/data/wzl/LightningSearch-RL`
- explicit conda activation
- `PYTHONNOUSERSITE=1`
- one `tmux` session per real experiment
- no more than one GPU for this first smoke unless later approved

## Recommended Approach

Implement a small local training-prep layer rather than embedding shell commands
directly in docs or scripts.

The new layer should:

1. Load a repo-local YAML config.
2. Validate input, output, checkpoint, and log paths.
3. Read `grpo/rollouts.jsonl`.
4. Convert a tiny train/val slice into verl-style parquet files.
5. Write a manifest and launch command.
6. In dry-run mode, stop before invoking verl.
7. In run mode, invoke `python -m verl.trainer.main_ppo` with hydra overrides.

The implementation should keep GPU execution outside tests. Tests should verify
conversion, config parsing, path checks, and dry-run artifacts.

## CLI Design

Add:

```bash
python -m lightningsearch_rl.cli train \
  --config configs/experiments/phase5b_tiny_grpo_smoke.yaml \
  --output-dir /data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke \
  --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke \
  --dry-run
```

Arguments:

- `--config`: required path to a YAML config.
- `--output-dir`: required output directory for generated train data, manifest,
  logs, and dry-run summary.
- `--checkpoint-dir`: required checkpoint directory passed to verl.
- `--dry-run`: validate and generate artifacts but do not launch verl.
- `--print-command`: optional flag to print the generated command to stdout.

The command returns nonzero if:

- the config file is missing or malformed
- the input rollouts file is missing
- output/checkpoint/log paths are outside approved remote or local project roots
- required Python packages for the requested mode are missing
- there are not enough rollouts for the requested train/val sizes

## Config Design

Create:

```text
configs/experiments/phase5b_tiny_grpo_smoke.yaml
```

Initial fields:

```yaml
experiment_name: phase5b-tiny-grpo-smoke
project_name: lightningsearch-rl
rollouts_path: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo/rollouts.jsonl
train_samples: 16
val_samples: 4
seed: 20260614
model_path: Qwen/Qwen3-4B
max_prompt_length: 512
max_response_length: 256
train_batch_size: 4
ppo_mini_batch_size: 2
ppo_micro_batch_size_per_gpu: 1
n_gpus_per_node: 1
total_training_steps: 1
save_freq: 1
test_freq: -1
logger:
  - console
```

The config intentionally does not include secrets. Model path remains a plain
string so later runs can switch to a cached local model path if Hugging Face
download is slow.

## Data Format

The launcher writes:

```text
<output-dir>/
  data/
    train.parquet
    val.parquet
  manifest.json
  launch_command.txt
  dry_run_summary.json
```

Each parquet row should include at least:

- `prompt`: list of chat messages, using a user question from the rollout
- `data_source`: `lightningsearch_rl`
- `ability`: `search_agent`
- `reward_model`: dictionary with `style: rule` and the scalar reward
- `extra_info`: dictionary with rollout id, answer, retrieved ids, gold ids, and
  original response

This mirrors the fields commonly used by verl RLHF examples while preserving
our existing trace metadata for later custom reward work.

If parquet dependencies are missing, dry-run should fail with a clear message
that `pandas` and either `pyarrow` or `fastparquet` are required inside the
approved conda env. The implementation should not auto-install packages.

## Verl Command Generation

The generated command should target `verl.trainer.main_ppo` and include only
explicit overrides:

```bash
PYTHONNOUSERSITE=1 python -m verl.trainer.main_ppo \
  data.train_files=<output-dir>/data/train.parquet \
  data.val_files=<output-dir>/data/val.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B \
  algorithm.adv_estimator=grpo \
  trainer.project_name=lightningsearch-rl \
  trainer.experiment_name=phase5b-tiny-grpo-smoke \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.logger='["console"]' \
  trainer.default_local_dir=<checkpoint-dir>
```

Implementation can add required verl batch/rollout overrides after inspecting
the installed config, but should keep the smoke as small as possible.

## Safety And Remote Launch Boundary

Implementation is allowed to run:

- local pytest
- remote pytest
- remote `train --dry-run`

Implementation must not start the real verl training process. After dry-run
passes, the assistant must provide an approval report containing:

- repo and branch
- commit
- conda env
- selected GPU and GPU count
- exact tmux session name
- exact launch command
- exact log path
- expected input, output, checkpoint, and data paths
- expected metrics or success criteria

Only after the user approves that report should a real `tmux` launch happen.

## Testing

Add focused tests for:

- config loading and default validation
- rejecting unsafe output/checkpoint paths
- converting sample rollouts into parquet-compatible rows
- dry-run writing `manifest.json`, `launch_command.txt`, and
  `dry_run_summary.json`
- CLI `train --dry-run` calling the launcher and not invoking subprocess

Tests should use temporary local paths and tiny JSONL fixtures. They should not
import `verl`, require GPUs, or require remote-only paths.

## Non-Goals

This phase does not:

- implement the final custom online search environment
- implement model-generated tool-use rollouts
- run large-scale data generation
- run a full GRPO training job
- add external search APIs
- change remote conda packages automatically

## Success Criteria

Phase 5B is ready when:

1. The repo has a tested `train --dry-run` command.
2. The command generates tiny verl input data and an exact launch command.
3. Local pytest passes.
4. GitHub main is updated.
5. Remote code is synced.
6. Remote pytest passes.
7. Remote dry-run passes under the approved env.
8. A launch approval report is ready for the first 1-GPU tiny GRPO smoke.
