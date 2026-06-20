import json
from pathlib import Path

import pytest

from lightningsearch_rl.verl_sft_warmup import (
    load_sft_warmup_train_config,
    prepare_verl_sft_warmup,
)


def _write_sft_rows(path: Path, count: int = 4) -> None:
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"sft-{index}",
                "messages": [
                    {"role": "system", "content": "Use strict tags."},
                    {"role": "user", "content": f"Question {index}?"},
                    {
                        "role": "assistant",
                        "content": (
                            "<think>I should search.</think>\n"
                            f"<search>Question {index}?</search>\n"
                            "<observation>\n[1] Evidence.\n</observation>\n"
                            "<think>The evidence supports the answer.</think>\n"
                            f"<answer>Answer {index}</answer>"
                        ),
                    },
                ],
                "metadata": {
                    "answer": f"Answer {index}",
                    "gold_evidence_doc_ids": [f"gold-{index}"],
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _config_text(sft_path: Path, train_samples: int = 3, val_samples: int = 1) -> str:
    return f"""
experiment_name: unit-sft-warmup
project_name: lightningsearch-rl
sft_path: {sft_path}
train_samples: {train_samples}
val_samples: {val_samples}
seed: 11
model_path: /data/wzl/LightningSearch-RL/models/Qwen/Qwen3-4B
max_length: 1024
train_batch_size: 2
micro_batch_size_per_gpu: 1
max_token_len_per_gpu: 2048
learning_rate: 1.0e-5
cuda_visible_devices: "7"
n_gpus_per_node: 1
total_training_steps: 2
total_epochs: 1
save_freq: 1
test_freq: -1
logger:
  - console
""".strip()


def test_load_sft_warmup_train_config_reads_required_fields(tmp_path):
    sft_rows = tmp_path / "sft_warmup.jsonl"
    _write_sft_rows(sft_rows)
    config = tmp_path / "sft.yaml"
    config.write_text(_config_text(sft_rows), encoding="utf-8")

    loaded = load_sft_warmup_train_config(config)

    assert loaded["experiment_name"] == "unit-sft-warmup"
    assert loaded["sft_path"] == str(sft_rows)
    assert loaded["train_batch_size"] == 2


def test_prepare_verl_sft_warmup_writes_data_and_launch_command(tmp_path):
    sft_rows = tmp_path / "sft_warmup.jsonl"
    _write_sft_rows(sft_rows, count=4)
    config = tmp_path / "sft.yaml"
    config.write_text(_config_text(sft_rows), encoding="utf-8")

    summary = prepare_verl_sft_warmup(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["train_rows"] == 3
    assert summary["val_rows"] == 1
    assert (tmp_path / "results" / "data" / "train.jsonl").exists()
    assert (tmp_path / "results" / "data" / "val.jsonl").exists()
    assert (tmp_path / "results" / "manifest.json").exists()
    assert (tmp_path / "results" / "launch_command.txt").exists()
    train_row = json.loads((tmp_path / "results" / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert train_row["messages"][-1]["content"].endswith("<answer>Answer 0</answer>")
    assert train_row["enable_thinking"] is False

    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert command.startswith(
        "CUDA_VISIBLE_DEVICES=7 "
        "HF_HOME=/data/wzl/LightningSearch-RL/.cache/huggingface "
        "HF_ENDPOINT=https://hf-mirror.com "
        "HYDRA_FULL_ERROR=1 "
        "PYTHONNOUSERSITE=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 "
        "-m verl.trainer.sft_trainer"
    )
    assert "verl.trainer.sft_trainer" in command
    assert "'data.train_files=" in command
    assert "data.messages_key=messages" in command
    assert "data.enable_thinking_default=False" in command
    assert "data.ignore_input_ids_mismatch=True" in command
    assert "model.path=/data/wzl/LightningSearch-RL/models/Qwen/Qwen3-4B" in command
    assert "engine.param_offload=True" in command
    assert "engine.optimizer_offload=True" in command
    assert "engine.model_dtype=bfloat16" in command
    assert "engine.dtype=bfloat16" in command
    assert "optim.lr=1e-05" in command
    assert "trainer.default_local_dir=" in command


def test_prepare_verl_sft_warmup_execute_requires_parquet(monkeypatch, tmp_path):
    sft_rows = tmp_path / "sft_warmup.jsonl"
    _write_sft_rows(sft_rows, count=1)
    config = tmp_path / "sft.yaml"
    config.write_text(_config_text(sft_rows, train_samples=1, val_samples=0), encoding="utf-8")
    monkeypatch.setattr("lightningsearch_rl.verl_sft_warmup._write_parquet_if_available", lambda path, rows: False)

    with pytest.raises(RuntimeError, match="parquet files were not written"):
        prepare_verl_sft_warmup(
            config,
            tmp_path / "results",
            tmp_path / "checkpoints",
            dry_run=False,
            execute=True,
        )


def test_phase5c_sft_warmup_tiny_config_builds_verl_sft_command(tmp_path):
    source_config = Path("configs/experiments/phase5c_sft_warmup_tiny.yaml")
    sft_rows = tmp_path / "sft_warmup.jsonl"
    _write_sft_rows(sft_rows, count=500)
    config = tmp_path / "config.yaml"
    config.write_text(
        source_config.read_text(encoding="utf-8").replace(
            "/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-warmup-gold/sft_warmup.jsonl",
            str(sft_rows),
        ),
        encoding="utf-8",
    )

    summary = prepare_verl_sft_warmup(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["experiment_name"] == "phase5c-sft-warmup-tiny"
    assert summary["train_rows"] == 480
    assert summary["val_rows"] == 20
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.sft_trainer" in command
    assert command.startswith("CUDA_VISIBLE_DEVICES=7 ")
    assert "data.train_batch_size=1" in command
    assert "data.micro_batch_size_per_gpu=1" in command
    assert "trainer.total_training_steps=20" in command
    assert "trainer.save_freq=20" in command


def test_phase5c_sft_warmup_tiny_2gpu_config_builds_sharded_command(tmp_path):
    source_config = Path("configs/experiments/phase5c_sft_warmup_tiny_2gpu.yaml")
    sft_rows = tmp_path / "sft_warmup.jsonl"
    _write_sft_rows(sft_rows, count=500)
    config = tmp_path / "config.yaml"
    config.write_text(
        source_config.read_text(encoding="utf-8").replace(
            "/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-warmup-gold/sft_warmup.jsonl",
            str(sft_rows),
        ),
        encoding="utf-8",
    )

    summary = prepare_verl_sft_warmup(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["experiment_name"] == "phase5c-sft-warmup-tiny-2gpu"
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert command.startswith("CUDA_VISIBLE_DEVICES=6,7 ")
    assert "torchrun --standalone --nnodes=1 --nproc_per_node=2 -m verl.trainer.sft_trainer" in command
    assert "data.train_batch_size=2" in command
    assert "data.micro_batch_size_per_gpu=1" in command
    assert "trainer.total_training_steps=20" in command


def test_phase5c_sft_warmup_tiny_4gpu_config_builds_sharded_command(tmp_path):
    source_config = Path("configs/experiments/phase5c_sft_warmup_tiny_4gpu.yaml")
    sft_rows = tmp_path / "sft_warmup.jsonl"
    _write_sft_rows(sft_rows, count=500)
    config = tmp_path / "config.yaml"
    config.write_text(
        source_config.read_text(encoding="utf-8").replace(
            "/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-warmup-gold/sft_warmup.jsonl",
            str(sft_rows),
        ),
        encoding="utf-8",
    )

    summary = prepare_verl_sft_warmup(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["experiment_name"] == "phase5c-sft-warmup-tiny-4gpu"
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert command.startswith("CUDA_VISIBLE_DEVICES=0,1,2,7 ")
    assert "torchrun --standalone --nnodes=1 --nproc_per_node=4 -m verl.trainer.sft_trainer" in command
    assert "data.train_batch_size=4" in command
    assert "data.micro_batch_size_per_gpu=1" in command
    assert "trainer.total_training_steps=20" in command


def test_phase5d_sft_turns_4gpu_config_builds_turn_level_command(tmp_path):
    source_config = Path("configs/experiments/phase5d_sft_turns_4gpu.yaml")
    sft_rows = tmp_path / "sft_turns.jsonl"
    rows = []
    for index in range(500):
        rows.append(
            {
                "id": f"turn-{index}",
                "messages": [
                    {"role": "system", "content": "Output exactly one action."},
                    {"role": "user", "content": f"Question {index}?"},
                    {"role": "assistant", "content": f"<search>Question {index}?</search>"},
                    {"role": "user", "content": "<observation>\n[1] Evidence.\n</observation>"},
                    {"role": "assistant", "content": f"<answer>Answer {index}</answer>"},
                ],
                "metadata": {
                    "answer": f"Answer {index}",
                    "gold_evidence_doc_ids": [f"gold-{index}"],
                },
            }
        )
    sft_rows.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        source_config.read_text(encoding="utf-8").replace(
            "/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-turns-gold/sft_turns.jsonl",
            str(sft_rows),
        ),
        encoding="utf-8",
    )

    summary = prepare_verl_sft_warmup(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["experiment_name"] == "phase5d-sft-turns-4gpu"
    assert summary["train_rows"] == 480
    assert summary["val_rows"] == 20
    train_row = json.loads((tmp_path / "results" / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [message["role"] for message in train_row["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assistant_text = "\n".join(
        message["content"] for message in train_row["messages"] if message["role"] == "assistant"
    )
    assert "<observation>" not in assistant_text

    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert command.startswith("CUDA_VISIBLE_DEVICES=1,2,5,7 ")
    assert "torchrun --standalone --nnodes=1 --nproc_per_node=4 -m verl.trainer.sft_trainer" in command
    assert (
        "model.path=/data/wzl/LightningSearch-RL/checkpoints/"
        "phase5c-sft-warmup-tiny-4gpu/hf_merged_global_step_20"
    ) in command
    assert "data.train_batch_size=4" in command
    assert "data.micro_batch_size_per_gpu=1" in command
    assert "trainer.total_training_steps=40" in command
    assert "trainer.save_freq=40" in command


def test_phase5d_sft_turns_docidfix_4gpu_config_uses_clean_turn_data(tmp_path):
    source_config = Path("configs/experiments/phase5d_sft_turns_docidfix_4gpu.yaml")
    sft_rows = tmp_path / "sft_turns.jsonl"
    rows = []
    for index in range(500):
        rows.append(
            {
                "id": f"turn-{index}",
                "messages": [
                    {"role": "system", "content": "Output exactly one action."},
                    {"role": "user", "content": f"Question {index}?"},
                    {"role": "assistant", "content": f"<search>Question {index}?</search>"},
                    {
                        "role": "user",
                        "content": f"<observation>\n[1] Evidence contains Answer {index}.\n</observation>",
                    },
                    {"role": "assistant", "content": f"<answer>Answer {index}</answer>"},
                ],
                "metadata": {
                    "answer": f"Answer {index}",
                    "gold_evidence_doc_ids": [f"hotpot::turn-{index}::gold::0"],
                },
            }
        )
    sft_rows.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        source_config.read_text(encoding="utf-8").replace(
            "/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl",
            str(sft_rows),
        ),
        encoding="utf-8",
    )

    summary = prepare_verl_sft_warmup(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["experiment_name"] == "phase5d-sft-turns-docidfix-4gpu"
    assert summary["train_rows"] == 480
    assert summary["val_rows"] == 20
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert command.startswith("CUDA_VISIBLE_DEVICES=0,1,2,5 ")
    assert "torchrun --standalone --nnodes=1 --nproc_per_node=4 -m verl.trainer.sft_trainer" in command
    assert (
        "data.train_files="
        + str(tmp_path / "results" / "data" / "train.parquet")
    ) in command
    assert (
        "model.path=/data/wzl/LightningSearch-RL/checkpoints/"
        "phase5c-sft-warmup-tiny-4gpu/hf_merged_global_step_20"
    ) in command
    assert "trainer.total_training_steps=40" in command
    assert "trainer.save_freq=40" in command
