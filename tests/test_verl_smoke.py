import json
from pathlib import Path

import pytest

from lightningsearch_rl.verl_smoke import load_train_config, prepare_verl_smoke


def _config_text(rollouts_path: Path, train_samples: int = 2, val_samples: int = 1) -> str:
    return f"""
experiment_name: unit-smoke
project_name: lightningsearch-rl
rollouts_path: {rollouts_path}
train_samples: {train_samples}
val_samples: {val_samples}
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
""".strip()


def test_load_train_config_reads_yaml_fields(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(_config_text(tmp_path / "rollouts.jsonl"), encoding="utf-8")

    loaded = load_train_config(config)

    assert loaded["experiment_name"] == "unit-smoke"
    assert loaded["train_samples"] == 2
    assert loaded["logger"] == ["console"]


def test_prepare_verl_smoke_rejects_unsafe_remote_output_path(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(_config_text(rollouts, train_samples=1, val_samples=0), encoding="utf-8")

    with pytest.raises(ValueError, match="outside approved paths"):
        prepare_verl_smoke(
            config,
            Path("/tmp/not-approved/results"),
            tmp_path / "checkpoints",
            dry_run=True,
            execute=False,
        )


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
    config.write_text(_config_text(rollouts), encoding="utf-8")

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
    assert command.startswith(
        "HF_HOME=/data/wzl/LightningSearch-RL/.cache/huggingface "
        "HF_ENDPOINT=https://hf-mirror.com "
        "PYTHONNOUSERSITE=1 python -m verl.trainer.main_ppo"
    )
    assert "verl.trainer.main_ppo" in command
    assert "HF_HOME=/data/wzl/LightningSearch-RL/.cache/huggingface" in command
    assert "HF_ENDPOINT=https://hf-mirror.com" in command
    assert "'data.train_files=" in command
    assert "algorithm.adv_estimator=grpo" in command
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=1" in command
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" in command
    assert "actor_rollout_ref.rollout.name=hf" in command
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in command
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" in command


def test_phase5b_tiny_grpo_smoke_4gpu_config_builds_4gpu_command(tmp_path):
    source_config = Path("configs/experiments/phase5b_tiny_grpo_smoke_4gpu.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, count=20)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo/rollouts.jsonl",
            str(rollouts),
        ),
        encoding="utf-8",
    )

    summary = prepare_verl_smoke(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["experiment_name"] == "phase5b-tiny-grpo-smoke-4gpu"
    assert summary["train_rows"] == 16
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "trainer.n_gpus_per_node=4" in command
    assert "data.train_batch_size=8" in command
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=4" in command


def test_prepare_verl_smoke_execute_requires_parquet(monkeypatch, tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, count=1)
    config = tmp_path / "config.yaml"
    config.write_text(_config_text(rollouts, train_samples=1, val_samples=0), encoding="utf-8")
    monkeypatch.setattr("lightningsearch_rl.verl_smoke._write_parquet_if_available", lambda path, rows: False)

    with pytest.raises(RuntimeError, match="parquet files were not written"):
        prepare_verl_smoke(
            config,
            tmp_path / "results",
            tmp_path / "checkpoints",
            dry_run=False,
            execute=True,
        )
