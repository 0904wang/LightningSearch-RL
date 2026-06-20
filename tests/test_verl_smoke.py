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
rollout_n: 1
rollout_gpu_memory_utilization: 0.18
actor_model_dtype: bfloat16
actor_param_offload: true
actor_optimizer_offload: true
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


def _write_sft_turns(path, count=2):
    rows = []
    for index in range(count):
        answer = f"Answer {index}"
        question = f"Question {index}?"
        rows.append(
            {
                "id": f"sft-{index}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a search agent. Output exactly one action per turn.",
                    },
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"<search>{question}</search>"},
                    {
                        "role": "user",
                        "content": f"<observation>\n[1] Evidence: The answer is {answer}.\n</observation>",
                    },
                    {"role": "assistant", "content": f"<answer>{answer}</answer>"},
                ],
                "metadata": {
                    "answer": answer,
                    "search_count": 1,
                    "gold_doc_ids": [f"gold-{index}"],
                    "gold_evidence_doc_ids": [f"gold-{index}"],
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_env_transitions(path, count=2):
    rows = []
    for index in range(count):
        answer = f"Answer {index}"
        question = f"Question {index}?"
        rows.extend(
            [
                {
                    "id": f"env-{index}",
                    "transition_id": f"env-{index}:0:search",
                    "step_index": 0,
                    "state_messages": [
                        {"role": "system", "content": "Output one action."},
                        {"role": "user", "content": question},
                    ],
                    "action": f"<search>{question}</search>",
                    "action_type": "search",
                    "valid_action": True,
                    "observation_doc_ids": [f"doc-answer-{index}"],
                    "gold_evidence_doc_ids": [f"doc-answer-{index}"],
                    "candidate_passages": [
                        {
                            "doc_id": f"doc-answer-{index}",
                            "title": f"Evidence {index}",
                            "text": f"The answer is {answer}.",
                        }
                    ],
                    "terminal": False,
                    "reward": 0.27,
                    "metadata": {
                        "question": question,
                        "gold_answer": answer,
                        "final_answer": answer,
                        "total_reward": 1.37,
                    },
                },
                {
                    "id": f"env-{index}",
                    "transition_id": f"env-{index}:1:answer",
                    "step_index": 1,
                    "state_messages": [
                        {"role": "system", "content": "Output one action."},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"<search>{question}</search>"},
                        {
                            "role": "user",
                            "content": f"<observation>\n[1] Evidence: The answer is {answer}.\n</observation>",
                        },
                    ],
                    "action": f"<answer>{answer}</answer>",
                    "action_type": "answer",
                    "valid_action": True,
                    "terminal": True,
                    "reward": 1.1,
                    "metadata": {
                        "question": question,
                        "gold_answer": answer,
                        "final_answer": answer,
                        "total_reward": 1.37,
                    },
                },
            ]
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
    assert "actor_rollout_ref.rollout.name=vllm" in command
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in command
    assert "actor_rollout_ref.rollout.n=1" in command
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.18" in command
    assert "actor_rollout_ref.rollout.max_model_len=768" in command
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=1024" in command
    assert "actor_rollout_ref.rollout.max_num_seqs=4" in command
    assert "actor_rollout_ref.rollout.enforce_eager=True" in command
    assert "actor_rollout_ref.rollout.enable_chunked_prefill=False" in command
    assert "actor_rollout_ref.rollout.enable_prefix_caching=False" in command
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" in command
    assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" in command
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=True" in command
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" in command
    assert "reward.custom_reward_function.path=src/lightningsearch_rl/verl_reward.py" in command
    assert "reward.custom_reward_function.name=compute_score" in command


def test_prepare_verl_smoke_can_prepend_agent_system_prompt(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, count=3)
    config = tmp_path / "config.yaml"
    config.write_text(
        _config_text(rollouts)
        + "\nagent_system_prompt: You are a search agent. Return exactly one tagged action.\n",
        encoding="utf-8",
    )

    prepare_verl_smoke(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    first_row = json.loads(
        (tmp_path / "results" / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_row["prompt"] == [
        {"role": "system", "content": "You are a search agent. Return exactly one tagged action."},
        {"role": "user", "content": "Question 0?"},
    ]


def test_prepare_verl_smoke_can_enable_reward_dump_env(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, count=3)
    config = tmp_path / "config.yaml"
    config.write_text(
        _config_text(rollouts)
        + "\nreward_dump_path: /data/wzl/LightningSearch-RL/results/unit/reward_dump.jsonl\n"
        + "reward_dump_max_chars: 256\n",
        encoding="utf-8",
    )

    summary = prepare_verl_smoke(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["reward_dump_path"] == "/data/wzl/LightningSearch-RL/results/unit/reward_dump.jsonl"
    assert summary["reward_dump_max_chars"] == 256
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "LIGHTNINGSEARCH_REWARD_DUMP_PATH=/data/wzl/LightningSearch-RL/results/unit/reward_dump.jsonl" in command
    assert "LIGHTNINGSEARCH_REWARD_DUMP_MAX_CHARS=256" in command
    assert command.index("LIGHTNINGSEARCH_REWARD_DUMP_PATH=") < command.index("python -m verl.trainer.main_ppo")


def test_prepare_verl_smoke_honors_sampling_and_reward_shape_overrides(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, count=3)
    config = tmp_path / "config.yaml"
    config.write_text(
        _config_text(rollouts)
        + "\nrollout_temperature: 1.2\n"
        + "rollout_top_p: 0.95\n"
        + "rollout_top_k: 50\n"
        + "answer_token_f1_threshold: 0.5\n",
        encoding="utf-8",
    )

    summary = prepare_verl_smoke(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    assert summary["answer_token_f1_threshold"] == 0.5
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "LIGHTNINGSEARCH_ANSWER_TOKEN_F1_THRESHOLD=0.5" in command
    assert "actor_rollout_ref.rollout.temperature=1.2" in command
    assert "actor_rollout_ref.rollout.top_p=0.95" in command
    assert "actor_rollout_ref.rollout.top_k=50" in command


def test_prepare_verl_smoke_honors_configured_total_epochs(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, count=4)
    config = tmp_path / "config.yaml"
    config.write_text(_config_text(rollouts) + "\ntotal_epochs: 2\n", encoding="utf-8")

    prepare_verl_smoke(
        config,
        tmp_path / "results",
        tmp_path / "checkpoints",
        dry_run=True,
        execute=False,
    )

    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "trainer.total_epochs=2" in command


def test_phase5b_tiny_grpo_smoke_4gpu_config_builds_4gpu_command(tmp_path):
    source_config = Path("configs/experiments/phase5b_tiny_grpo_smoke_4gpu.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, count=20)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo-gold-answer/rollouts.jsonl",
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
    assert summary["train_rows"] == 4
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "actor_rollout_ref.model.path=/data/wzl/LightningSearch-RL/models/Qwen/Qwen3-4B" in command
    assert "trainer.n_gpus_per_node=4" in command
    assert "data.train_batch_size=4" in command
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=4" in command
    assert "actor_rollout_ref.rollout.n=1" in command
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.25" in command
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in command
    assert "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512" in command
    assert "trainer.save_freq=-1" in command


def test_phase5e_tiny_grpo_docidfix_warmstart_uses_docidfix_checkpoint_and_system_prompt(tmp_path):
    source_config = Path("configs/experiments/phase5e_tiny_grpo_docidfix_warmstart_4gpu.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    rollouts = tmp_path / "rollouts.jsonl"
    _write_rollouts(rollouts, count=20)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/grpo-gold-answer/rollouts.jsonl",
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

    assert summary["experiment_name"] == "phase5e-tiny-grpo-docidfix-warmstart-4gpu"
    assert summary["train_rows"] == 4
    train_row = json.loads(
        (tmp_path / "results" / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert train_row["prompt"][0]["role"] == "system"
    assert "Do not write <observation>" in train_row["prompt"][0]["content"]
    assert train_row["prompt"][1] == {"role": "user", "content": "Question 0?"}
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "actor_rollout_ref.model.path=/data/wzl/LightningSearch-RL/checkpoints/"
        "phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40"
    ) in command
    assert "data.max_response_length=128" in command
    assert "trainer.n_gpus_per_node=4" in command
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in command


def test_phase5f_tiny_grpo_docidfix_two_stage_config_builds_stage_rows(tmp_path):
    source_config = Path("configs/experiments/phase5f_tiny_grpo_docidfix_two_stage_4gpu.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    sft_turns = tmp_path / "sft_turns.jsonl"
    _write_sft_turns(sft_turns, count=4)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl",
            str(sft_turns),
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

    assert summary["experiment_name"] == "phase5f-tiny-grpo-docidfix-two-stage-4gpu"
    assert summary["source_type"] == "sft_turns"
    assert summary["train_rows"] == 4
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "data" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["extra_info"]["reward_stage"] for row in train_rows] == [
        "search",
        "answer",
        "search",
        "answer",
    ]
    assert train_rows[0]["reward_model"]["ground_truth"] == ""
    assert train_rows[1]["reward_model"]["ground_truth"] == "Answer 0"
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "actor_rollout_ref.model.path=/data/wzl/LightningSearch-RL/checkpoints/"
        "phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40"
    ) in command
    assert "data.max_response_length=64" in command


def test_phase5m_env_transition_grpo_config_builds_from_transition_rows(tmp_path):
    source_config = Path("configs/experiments/phase5m_env_transition_grpo_4gpu.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=8)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5m-env-transition-grpo-4gpu"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 8
    assert summary["val_rows"] == 4
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "data" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["extra_info"]["reward_stage"] for row in train_rows[:2]] == ["search", "answer"]
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "actor_rollout_ref.model.path=/data/wzl/LightningSearch-RL/checkpoints/"
        "phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40"
    ) in command
    assert "data.max_prompt_length=1024" in command
    assert "actor_rollout_ref.rollout.max_model_len=1280" in command
    assert "trainer.n_gpus_per_node=4" in command
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in command


def test_phase5m_env_transition_grpo_lowlen_config_uses_short_vllm_context(tmp_path):
    source_config = Path("configs/experiments/phase5m_env_transition_grpo_4gpu_lowlen.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=8)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5m-env-transition-grpo-4gpu-lowlen"
    assert summary["train_rows"] == 8
    assert summary["val_rows"] == 4
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "data.max_prompt_length=384" in command
    assert "actor_rollout_ref.rollout.max_model_len=512" in command
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=768" in command
    assert "trainer.n_gpus_per_node=4" in command


def test_phase5n_env_transition_grpo_100x5_config_uses_full_transition_slice(tmp_path):
    source_config = Path("configs/experiments/phase5n_env_transition_grpo_4gpu_100x5.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=50)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5n-env-transition-grpo-4gpu-100x5"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 80
    assert summary["val_rows"] == 20
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "data" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert train_rows[0]["extra_info"]["reward_stage"] == "search"
    assert train_rows[1]["extra_info"]["reward_stage"] == "answer"
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "trainer.total_training_steps=5" in command
    assert "data.max_prompt_length=384" in command
    assert "actor_rollout_ref.rollout.max_model_len=512" in command
    assert "trainer.n_gpus_per_node=4" in command


def test_phase5o_env_transition_grpo_rewarddump_config_enables_dump_path(tmp_path):
    source_config = Path("configs/experiments/phase5o_env_transition_grpo_4gpu_100x5_rewarddump.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=50)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5o-env-transition-grpo-4gpu-100x5-rewarddump"
    assert summary["train_rows"] == 80
    assert summary["val_rows"] == 20
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/reward_dump.jsonl"
    ) in command
    assert "LIGHTNINGSEARCH_REWARD_DUMP_MAX_CHARS=1024" in command
    assert "trainer.total_training_steps=5" in command


def test_phase5p_soft_answer_grpo_config_uses_soft_transition_export(tmp_path):
    source_config = Path("configs/experiments/phase5p_env_transition_grpo_4gpu_100x5_softanswer.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=50)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5p-env-transitions-soft-answer-from-phase5k/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5p-env-transition-grpo-4gpu-100x5-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 80
    assert summary["val_rows"] == 20
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer/reward_dump.jsonl"
    ) in command
    assert "data.max_prompt_length=384" in command
    assert "actor_rollout_ref.rollout.max_model_len=512" in command
    assert "trainer.total_training_steps=5" in command


def test_phase5q_soft_answer_grpo_config_scales_transition_split(tmp_path):
    source_config = Path("configs/experiments/phase5q_env_transition_grpo_4gpu_400x20_softanswer.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=200)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5q-env-transition-grpo-4gpu-400x20-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 320
    assert summary["val_rows"] == 80
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=20" in command
    assert "data.train_batch_size=4" in command


def test_phase5q_filtered_soft_answer_grpo_config_uses_filtered_split(tmp_path):
    source_config = Path("configs/experiments/phase5q_filtered_env_transition_grpo_4gpu_390x20_softanswer.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=390)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200-filtered/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5q-filtered-env-transition-grpo-4gpu-390x20-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 312
    assert summary["val_rows"] == 78
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5q-filtered-env-transition-grpo-4gpu-390x20-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=20" in command
    assert "data.train_batch_size=4" in command


def test_phase5r_filtered_soft_answer_grpo_config_uses_scaled_split(tmp_path):
    source_config = Path("configs/experiments/phase5r_filtered_env_transition_grpo_4gpu_990x50_softanswer.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=990)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5r-filtered-env-transition-grpo-4gpu-990x50-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 792
    assert summary["val_rows"] == 198
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5r-filtered-env-transition-grpo-4gpu-990x50-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=50" in command
    assert "data.train_batch_size=4" in command


def test_phase5r_quality_manifest_marks_known_rollout_500_mismatches():
    manifest_path = Path("configs/data_quality/phase5r_500_known_mismatches.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    qa_type_mismatch_ids = {
        "syn-009012",
        "syn-009019",
        "syn-009432",
        "syn-009456",
        "syn-009536",
    }
    answer_none_ids = {
        "syn-009857",
        "syn-009947",
        "syn-010022",
        "syn-010102",
        "syn-010326",
        "syn-010401",
    }

    assert set(manifest) == qa_type_mismatch_ids | answer_none_ids
    assert all(manifest[row_id]["flags"] == ["qa_type_mismatch"] for row_id in qa_type_mismatch_ids)
    assert all(manifest[row_id]["flags"] == ["answer_none_low_reward"] for row_id in answer_none_ids)


def test_phase5r_filtered_v2_soft_answer_grpo_config_uses_cleaned_split(tmp_path):
    source_config = Path("configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x50_softanswer.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=978)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 782
    assert summary["val_rows"] == 196
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=50" in command
    assert "data.train_batch_size=4" in command


def test_phase5r_filtered_v2_soft_answer_grpo_200step_config_saves_checkpoints(tmp_path):
    source_config = Path("configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=978)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 782
    assert summary["val_rows"] == 196
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=200" in command
    assert "trainer.save_freq=100" in command
    assert "data.train_batch_size=4" in command


def test_phase5r_filtered_v2_soft_answer_grpo_200step_epoch2_config_reaches_200_steps(tmp_path):
    source_config = Path("configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer_epoch2.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=978)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2"
    assert summary["train_rows"] == 782
    assert summary["val_rows"] == 196
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=200" in command
    assert "trainer.total_epochs=2" in command
    assert "trainer.save_freq=100" in command
    assert "data.train_batch_size=4" in command


def test_phase5s_hard50_soft_answer_grpo_config_uses_hard50_transition_split(tmp_path):
    source_config = Path("configs/experiments/phase5s_hard50_env_transition_grpo_4gpu_978x50_softanswer.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=978)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 782
    assert summary["val_rows"] == 196
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=50" in command
    assert "trainer.save_freq=-1" in command
    assert "data.train_batch_size=4" in command


def test_phase5s_hard50_soft_answer_grpo_200step_config_saves_checkpoints(tmp_path):
    source_config = Path("configs/experiments/phase5s_hard50_env_transition_grpo_4gpu_978x200_softanswer.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=978)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5s-hard50-env-transition-grpo-4gpu-978x200-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 782
    assert summary["val_rows"] == 196
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x200-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=200" in command
    assert "trainer.save_freq=100" in command
    assert "data.train_batch_size=4" in command


def test_phase5s_hard50_soft_answer_grpo_200step_rollout4_config_uses_grpo_groups(tmp_path):
    source_config = Path("configs/experiments/phase5s_hard50_env_transition_grpo_4gpu_978x200_rollout4_softanswer.yaml")
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=978)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 782
    assert summary["val_rows"] == 196
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "actor_rollout_ref.rollout.n=4" in command
    assert "actor_rollout_ref.rollout.max_num_seqs=16" in command
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=200" in command
    assert "trainer.save_freq=100" in command
    assert "data.train_batch_size=4" in command


def test_phase5t_hard50_rollout4_diverse_smoke_config_enables_sampling_and_reward_shaping(tmp_path):
    source_config = Path(
        "configs/experiments/phase5t_hard50_env_transition_grpo_4gpu_978x50_rollout4_diverse_softanswer.yaml"
    )
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=978)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5t-hard50-env-transition-grpo-4gpu-978x50-rollout4-diverse-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 782
    assert summary["val_rows"] == 196
    assert summary["answer_token_f1_threshold"] == 0.5
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "LIGHTNINGSEARCH_ANSWER_TOKEN_F1_THRESHOLD=0.5" in command
    assert "actor_rollout_ref.rollout.n=4" in command
    assert "actor_rollout_ref.rollout.temperature=1.2" in command
    assert "actor_rollout_ref.rollout.top_p=0.95" in command
    assert "actor_rollout_ref.rollout.top_k=50" in command
    assert "actor_rollout_ref.rollout.max_num_seqs=16" in command
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5t-hard50-env-transition-grpo-4gpu-978x50-rollout4-diverse-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=50" in command
    assert "trainer.total_epochs=2" in command
    assert "trainer.save_freq=-1" in command


def test_phase5u_hard50_rollout4_diverse_checkpointed_config_runs_200_steps(tmp_path):
    source_config = Path(
        "configs/experiments/phase5u_hard50_env_transition_grpo_4gpu_978x200_rollout4_diverse_softanswer.yaml"
    )
    config_text = source_config.read_text(encoding="utf-8")
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=978)
    config = tmp_path / "config.yaml"
    config.write_text(
        config_text.replace(
            "/data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/transitions.jsonl",
            str(transitions),
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

    assert summary["experiment_name"] == "phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer"
    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 782
    assert summary["val_rows"] == 196
    assert summary["answer_token_f1_threshold"] == 0.5
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "LIGHTNINGSEARCH_ANSWER_TOKEN_F1_THRESHOLD=0.5" in command
    assert "actor_rollout_ref.rollout.n=4" in command
    assert "actor_rollout_ref.rollout.temperature=1.2" in command
    assert "actor_rollout_ref.rollout.top_p=0.95" in command
    assert "actor_rollout_ref.rollout.top_k=50" in command
    assert "actor_rollout_ref.rollout.max_num_seqs=16" in command
    assert (
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH="
        "/data/wzl/LightningSearch-RL/results/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer/reward_dump.jsonl"
    ) in command
    assert "trainer.total_training_steps=200" in command
    assert "trainer.total_epochs=2" in command
    assert "trainer.save_freq=100" in command


def test_prepare_verl_smoke_builds_two_stage_rows_from_sft_turns(tmp_path):
    sft_turns = tmp_path / "sft_turns.jsonl"
    _write_sft_turns(sft_turns, count=2)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
experiment_name: two-stage
project_name: lightningsearch-rl
sft_turns_path: {sft_turns}
prompt_stages:
  - search
  - answer
train_samples: 1
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

    train_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "data" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    val_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "data" / "val.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert summary["train_rows"] == 2
    assert summary["val_rows"] == 2
    assert [row["extra_info"]["reward_stage"] for row in train_rows] == ["search", "answer"]
    assert [message["role"] for message in train_rows[0]["prompt"]] == ["system", "user"]
    assert train_rows[0]["extra_info"]["expected_action"] == "<search>Question 0?</search>"
    assert train_rows[0]["reward_model"]["ground_truth"] == ""
    assert [message["role"] for message in train_rows[1]["prompt"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "<observation>" in train_rows[1]["prompt"][-1]["content"]
    assert train_rows[1]["reward_model"]["ground_truth"] == "Answer 0"
    assert val_rows[0]["extra_info"]["source_id"] == "sft-1"


def test_prepare_verl_smoke_builds_rows_from_env_transitions(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    _write_env_transitions(transitions, count=2)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
experiment_name: env-transition-grpo
project_name: lightningsearch-rl
transitions_path: {transitions}
train_samples: 2
val_samples: 2
seed: 7
model_path: Qwen/Qwen3-4B
max_prompt_length: 512
max_response_length: 64
train_batch_size: 2
ppo_mini_batch_size: 1
ppo_micro_batch_size_per_gpu: 1
n_gpus_per_node: 1
total_training_steps: 1
save_freq: -1
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

    train_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "data" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    val_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "data" / "val.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert summary["source_type"] == "transitions"
    assert summary["train_rows"] == 2
    assert summary["val_rows"] == 2
    assert [row["extra_info"]["reward_stage"] for row in train_rows] == ["search", "answer"]
    assert train_rows[0]["prompt"] == [
        {"role": "system", "content": "Output one action."},
        {"role": "user", "content": "Question 0?"},
    ]
    assert train_rows[0]["reward_model"]["ground_truth"] == ""
    assert train_rows[0]["reward_model"]["reward"] == 0.27
    assert train_rows[0]["extra_info"]["expected_action"] == "<search>Question 0?</search>"
    assert train_rows[0]["extra_info"]["source_id"] == "env-0"
    assert train_rows[0]["extra_info"]["gold_doc_ids"] == ["doc-answer-0"]
    assert train_rows[0]["extra_info"]["candidate_passages"] == [
        {
            "doc_id": "doc-answer-0",
            "title": "Evidence 0",
            "text": "The answer is Answer 0.",
        }
    ]
    assert train_rows[0]["extra_info"]["search_reward_top_k"] == 8
    assert train_rows[1]["prompt"][-1]["role"] == "user"
    assert "<observation>" in train_rows[1]["prompt"][-1]["content"]
    assert train_rows[1]["reward_model"]["ground_truth"] == "Answer 0"
    assert train_rows[1]["reward_model"]["reward"] == 1.1
    assert val_rows[0]["extra_info"]["source_id"] == "env-1"


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
