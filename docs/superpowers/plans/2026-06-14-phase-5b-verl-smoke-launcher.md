# Phase 5B Verl Smoke Launcher Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested `train --dry-run` CLI that prepares tiny verl / GRPO smoke artifacts and an auditable launch command.

**Architecture:** Add a focused `verl_smoke.py` module responsible for config loading, safe path validation, rollout conversion, artifact writing, and command generation. Extend `cli.py` with a `train` command and add a repo-local YAML config for the Phase 5B tiny smoke. Tests stay local and do not import verl or use GPUs.

**Tech Stack:** Python 3.10, pytest, standard-library JSON/YAML fallback handling, optional pandas/parquet dependency detection, verl 0.8.0 command generation.

---

## File Structure

- Create `src/lightningsearch_rl/verl_smoke.py`
  - Owns `load_train_config`, `prepare_verl_smoke`, rollout row conversion, path validation, dependency checks, and launch command generation.
- Modify `src/lightningsearch_rl/cli.py`
  - Adds `train` subcommand with `--config`, `--output-dir`, `--checkpoint-dir`, `--dry-run`, and `--print-command`.
- Create `configs/experiments/phase5b_tiny_grpo_smoke.yaml`
  - Repo-local config for the remote dry-run and later approved launch.
- Create `tests/test_verl_smoke.py`
  - Unit tests for config, path checks, rollout conversion, artifact writing, and command generation.
- Modify `tests/test_cli.py`
  - Adds CLI-level dry-run coverage.
- Create `docs/experiments/phase5b-verl-smoke-dry-run-2026-06-14.md`
  - Added only after remote dry-run completes.

## Chunk 1: Core Verl Smoke Module

### Task 1: Config Loading And Safe Path Checks

**Files:**
- Create: `src/lightningsearch_rl/verl_smoke.py`
- Test: `tests/test_verl_smoke.py`

- [ ] **Step 1: Write failing tests for config loading and unsafe paths**

Add to `tests/test_verl_smoke.py`:

```python
import json
from pathlib import Path

import pytest

from lightningsearch_rl.verl_smoke import load_train_config, prepare_verl_smoke


def test_load_train_config_reads_yaml_fields(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment_name: unit-smoke",
                "project_name: lightningsearch-rl",
                f"rollouts_path: {tmp_path / 'rollouts.jsonl'}",
                "train_samples: 2",
                "val_samples: 1",
                "seed: 7",
                "model_path: Qwen/Qwen3-4B",
                "max_prompt_length: 128",
                "max_response_length: 64",
                "train_batch_size: 2",
                "ppo_mini_batch_size: 1",
                "ppo_micro_batch_size_per_gpu: 1",
                "n_gpus_per_node: 1",
                "total_training_steps: 1",
                "save_freq: 1",
                "test_freq: -1",
                "logger:",
                "  - console",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_train_config(config)

    assert loaded["experiment_name"] == "unit-smoke"
    assert loaded["train_samples"] == 2
    assert loaded["logger"] == ["console"]


def test_prepare_verl_smoke_rejects_unsafe_remote_output_path(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
experiment_name: unit-smoke
project_name: lightningsearch-rl
rollouts_path: {rollouts}
train_samples: 1
val_samples: 0
seed: 7
model_path: Qwen/Qwen3-4B
max_prompt_length: 128
max_response_length: 64
train_batch_size: 1
ppo_mini_batch_size: 1
ppo_micro_batch_size_per_gpu: 1
n_gpus_per_node: 1
total_training_steps: 1
save_freq: 1
test_freq: -1
logger:
  - console
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside approved paths"):
        prepare_verl_smoke(
            config,
            Path("/tmp/not-approved/results"),
            tmp_path / "checkpoints",
            dry_run=True,
            execute=False,
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_verl_smoke.py -q
```

Expected: FAIL because `lightningsearch_rl.verl_smoke` does not exist.

- [ ] **Step 3: Implement minimal config loading and path validation**

Create `src/lightningsearch_rl/verl_smoke.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOCAL_ROOT_MARKER = "Agent RL"
REMOTE_ROOTS = ("/data/wzl/LightningSearch-RL", "/home/user/wzl/LightningSearch-RL")
REQUIRED_CONFIG_KEYS = {
    "experiment_name",
    "project_name",
    "rollouts_path",
    "train_samples",
    "val_samples",
    "seed",
    "model_path",
    "max_prompt_length",
    "max_response_length",
    "train_batch_size",
    "ppo_mini_batch_size",
    "ppo_micro_batch_size_per_gpu",
    "n_gpus_per_node",
    "total_training_steps",
    "save_freq",
    "test_freq",
    "logger",
}


def load_train_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read train configs") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("train config must be a mapping")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(payload))
    if missing:
        raise ValueError(f"train config missing required keys: {missing}")
    return payload


def prepare_verl_smoke(
    config_path: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    dry_run: bool,
    execute: bool = False,
    print_command: bool = False,
) -> dict[str, Any]:
    config = load_train_config(config_path)
    _ensure_approved_path(output_dir)
    _ensure_approved_path(checkpoint_dir)
    return {
        "dry_run": dry_run,
        "execute": execute,
        "config": config,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }


def _ensure_approved_path(path: Path) -> None:
    text = str(path)
    normalized = text.replace("\\", "/")
    if any(normalized.startswith(root) for root in REMOTE_ROOTS):
        return
    if LOCAL_ROOT_MARKER in normalized:
        return
    raise ValueError(f"path is outside approved paths: {path}")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_verl_smoke.py -q
```

Expected: PASS for these tests.

### Task 2: Rollout Conversion And Dry-Run Artifacts

**Files:**
- Modify: `src/lightningsearch_rl/verl_smoke.py`
- Test: `tests/test_verl_smoke.py`

- [ ] **Step 1: Write failing test for dry-run artifacts**

Add:

```python
def _write_rollouts(path, count=3):
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"r{index}",
                "prompt": f"Question {index}?",
                "response": f"<answer>Answer {index}</answer>",
                "reward": 0.5 + index,
                "metadata": {
                    "answer": f"Answer {index}",
                    "search_count": 1,
                    "gold_doc_ids": [f"gold-{index}"],
                    "retrieved_doc_ids": [f"gold-{index}", f"noise-{index}"],
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_prepare_verl_smoke_writes_dry_run_artifacts(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, count=3)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
experiment_name: unit-smoke
project_name: lightningsearch-rl
rollouts_path: {rollouts}
train_samples: 2
val_samples: 1
seed: 7
model_path: Qwen/Qwen3-4B
max_prompt_length: 128
max_response_length: 64
train_batch_size: 2
ppo_mini_batch_size: 1
ppo_micro_batch_size_per_gpu: 1
n_gpus_per_node: 1
total_training_steps: 1
save_freq: 1
test_freq: -1
logger:
  - console
""".strip(),
        encoding="utf-8",
    )

    summary = prepare_verl_smoke(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["train_rows"] == 2
    assert summary["val_rows"] == 1
    assert (tmp_path / "results" / "data" / "train.jsonl").exists()
    assert (tmp_path / "results" / "data" / "val.jsonl").exists()
    assert (tmp_path / "results" / "manifest.json").exists()
    assert (tmp_path / "results" / "launch_command.txt").exists()
    assert (tmp_path / "results" / "dry_run_summary.json").exists()
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "verl.trainer.main_ppo" in command
    assert "algorithm.adv_estimator=grpo" in command
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
python -m pytest tests/test_verl_smoke.py::test_prepare_verl_smoke_writes_dry_run_artifacts -q
```

Expected: FAIL because artifacts are not written.

- [ ] **Step 3: Implement rollout loading, split, rows, and artifacts**

Update `prepare_verl_smoke` to:

- create `<output-dir>/data`
- load rollout rows
- require `train_samples + val_samples <= len(rollouts)`
- create train and val records
- write `train.jsonl` and `val.jsonl` as a fallback format for tests
- attempt parquet writing only when dependencies are present
- write `manifest.json`
- write `launch_command.txt`
- write `dry_run_summary.json`

Implement helpers:

```python
def _load_jsonl(path: Path) -> list[dict[str, Any]]: ...
def _write_json(path: Path, payload: dict[str, Any]) -> None: ...
def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None: ...
def _to_verl_row(rollout: dict[str, Any], split: str, index: int) -> dict[str, Any]: ...
def _write_parquet_if_available(path: Path, rows: list[dict[str, Any]]) -> bool: ...
def _build_launch_command(config, train_file, val_file, checkpoint_dir) -> list[str]: ...
```

For local tests, parquet can be optional. If parquet dependencies are missing,
the generated command should still reference `train.jsonl` only in local dry-run
summary? No. To keep remote intent correct, command should reference `.parquet`
paths and `dry_run_summary.json` should include `parquet_written: false` when
not available. Remote dry-run will verify whether parquet was actually written.

- [ ] **Step 4: Run test and verify it passes**

Run:

```bash
python -m pytest tests/test_verl_smoke.py -q
```

Expected: PASS.

### Task 3: CLI Train Dry-Run

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Add to `tests/test_cli.py`:

```python
def test_train_cli_dry_run_writes_launch_artifacts(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        json.dumps(
            {
                "id": "r0",
                "prompt": "Question?",
                "response": "<answer>Answer</answer>",
                "reward": 1.0,
                "metadata": {
                    "answer": "Answer",
                    "search_count": 1,
                    "gold_doc_ids": ["gold"],
                    "retrieved_doc_ids": ["gold"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
experiment_name: cli-smoke
project_name: lightningsearch-rl
rollouts_path: {rollouts}
train_samples: 1
val_samples: 0
seed: 1
model_path: Qwen/Qwen3-4B
max_prompt_length: 128
max_response_length: 64
train_batch_size: 1
ppo_mini_batch_size: 1
ppo_micro_batch_size_per_gpu: 1
n_gpus_per_node: 1
total_training_steps: 1
save_freq: 1
test_freq: -1
logger:
  - console
""".strip(),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "train",
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path / "results"),
                "--checkpoint-dir",
                str(tmp_path / "checkpoints"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert (tmp_path / "results" / "dry_run_summary.json").exists()
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
python -m pytest tests/test_cli.py::test_train_cli_dry_run_writes_launch_artifacts -q
```

Expected: FAIL because `train` subcommand does not exist.

- [ ] **Step 3: Implement CLI parser and branch**

In `src/lightningsearch_rl/cli.py`:

- import `prepare_verl_smoke`
- add `train = subparsers.add_parser("train")`
- add args:
  - `--config`
  - `--output-dir`
  - `--checkpoint-dir`
  - `--dry-run`
  - `--print-command`
- branch:

```python
if args.command == "train":
    summary = prepare_verl_smoke(
        Path(args.config),
        Path(args.output_dir),
        Path(args.checkpoint_dir),
        dry_run=args.dry_run,
        execute=not args.dry_run,
        print_command=args.print_command,
    )
    if args.print_command:
        print(summary["launch_command"])
    return 0
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
python -m pytest tests/test_cli.py::test_train_cli_dry_run_writes_launch_artifacts -q
```

Expected: PASS.

## Chunk 2: Repo Config And Full Verification

### Task 4: Add Phase 5B Config

**Files:**
- Create: `configs/experiments/phase5b_tiny_grpo_smoke.yaml`

- [ ] **Step 1: Add config file**

Create exactly:

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

- [ ] **Step 2: Run config loading test manually**

Run:

```bash
python - <<'PY'
from pathlib import Path
from lightningsearch_rl.verl_smoke import load_train_config
cfg = load_train_config(Path("configs/experiments/phase5b_tiny_grpo_smoke.yaml"))
assert cfg["experiment_name"] == "phase5b-tiny-grpo-smoke"
assert cfg["train_samples"] == 16
print("config ok")
PY
```

Expected: `config ok`.

### Task 5: Full Local Verification And Commit

**Files:**
- All changed files.

- [ ] **Step 1: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect diff**

Run:

```bash
git diff --stat
git diff -- src/lightningsearch_rl/verl_smoke.py src/lightningsearch_rl/cli.py tests/test_verl_smoke.py tests/test_cli.py configs/experiments/phase5b_tiny_grpo_smoke.yaml
```

- [ ] **Step 3: Commit implementation**

Run:

```bash
git add src/lightningsearch_rl/verl_smoke.py src/lightningsearch_rl/cli.py tests/test_verl_smoke.py tests/test_cli.py configs/experiments/phase5b_tiny_grpo_smoke.yaml
git commit -m "feat: add tiny verl smoke launcher"
```

## Chunk 3: Merge, Sync, Remote Dry-Run, And Record

### Task 6: Merge Feature Branch And Push

**Files:**
- Git only.

- [ ] **Step 1: Merge to master**

From `D:\resume\Agent RL`:

```bash
git merge --no-ff codex/phase-5b-verl-smoke-launcher -m "merge: phase5b verl smoke launcher"
```

- [ ] **Step 2: Run full local tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Push GitHub main**

Run:

```bash
git push origin master:main
```

### Task 7: Remote Sync And Dry-Run

**Files:**
- Remote approved path only.

- [ ] **Step 1: Narrow sync by archive**

From local master:

```powershell
git archive --format=tar HEAD -o $env:TEMP\lightningsearch-rl-phase5b.tar
scp -P 29509 $env:TEMP\lightningsearch-rl-phase5b.tar user@ssh-22.e6.luyouxia.net:/data/wzl/LightningSearch-RL/repo/lightningsearch-rl-phase5b.tar
ssh user@ssh-22.e6.luyouxia.net -p 29509 'cd /data/wzl/LightningSearch-RL/repo && tar -xf lightningsearch-rl-phase5b.tar && rm lightningsearch-rl-phase5b.tar'
```

- [ ] **Step 2: Remote pytest**

Run:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m pytest -q'
```

Expected: all tests pass.

- [ ] **Step 3: Remote train dry-run**

Run:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train --config configs/experiments/phase5b_tiny_grpo_smoke.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke --dry-run --print-command'
```

Expected:

- command exits 0
- `dry_run_summary.json` exists
- `manifest.json` exists
- `launch_command.txt` exists
- train/val files exist

### Task 8: Record Dry-Run Experiment

**Files:**
- Create: `docs/experiments/phase5b-verl-smoke-dry-run-2026-06-14.md`

- [ ] **Step 1: Add experiment record**

Record:

- goal
- repo/branch/commit
- env
- exact dry-run command
- generated artifacts
- summary values
- whether parquet was written
- no tmux session and no GPU training was launched
- next approval report for real 1-GPU smoke

- [ ] **Step 2: Test and commit docs**

Run:

```bash
python -m pytest -q
git add docs/experiments/phase5b-verl-smoke-dry-run-2026-06-14.md
git commit -m "docs: record phase5b verl dry run"
git push origin master:main
```

### Task 9: Prepare Real Launch Approval Report

**Files:**
- No file changes required.

- [ ] **Step 1: Check GPUs**

Run:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader'
```

- [ ] **Step 2: Report launch details and wait for approval**

Report:

- selected repo and branch
- selected commit
- selected conda env
- selected GPU and count
- exact tmux session name
- exact launch command
- log path
- expected data and index paths
- checkpoint path
- results path
- expected success criteria

Do not launch until the user approves the exact report.
