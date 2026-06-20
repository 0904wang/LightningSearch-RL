import json

from lightningsearch_rl.cli import main


def test_smoke_cli_writes_artifacts(tmp_path):
    out_dir = tmp_path / "smoke"

    exit_code = main(
        [
            "smoke",
            "--data",
            "tests/fixtures/tiny_multihop.jsonl",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "transitions.jsonl").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["answer_em"] == 1.0


def test_shared_corpus_retrieval_cli_pipeline_writes_metrics(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    metrics_path = tmp_path / "retrieval_metrics.json"

    assert (
        main(
            [
                "prepare-hotpot",
                "--raw",
                "tests/fixtures/hotpot_tiny_raw.json",
                "--corpus",
                str(corpus),
                "--examples",
                str(examples),
            ]
        )
        == 0
    )
    assert (
        main(["build-index", "--corpus", str(corpus), "--index", str(index)])
        == 0
    )
    assert (
        main(
            [
                "eval-retrieval",
                "--examples",
                str(examples),
                "--index",
                str(index),
                "--out",
                str(metrics_path),
                "--top-k",
                "2",
            ]
        )
        == 0
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["recall_at_2"] == 1.0


def test_retrieval_baseline_cli_writes_report_for_limited_mixed_input(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    report = tmp_path / "baseline_report.json"

    assert (
        main(
            [
                "prepare-hotpot",
                "--raw",
                "tests/fixtures/hotpot_mixed_raw.jsonl",
                "--corpus",
                str(corpus),
                "--examples",
                str(examples),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert main(["build-index", "--corpus", str(corpus), "--index", str(index)]) == 0
    assert (
        main(
            [
                "retrieval-baseline",
                "--dataset",
                "hotpot",
                "--examples",
                str(examples),
                "--index",
                str(index),
                "--report",
                str(report),
                "--top-k",
                "2",
            ]
        )
        == 0
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["example_count"] == 1
    assert payload["metrics"]["recall_at_2"] == 1.0


def test_export_sft_cli_writes_conversations_traces_and_summary(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "sft"

    assert (
        main(
            [
                "prepare-hotpot",
                "--raw",
                "tests/fixtures/hotpot_mixed_raw.jsonl",
                "--corpus",
                str(corpus),
                "--examples",
                str(examples),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert main(["build-index", "--corpus", str(corpus), "--index", str(index)]) == 0
    assert (
        main(
            [
                "export-sft",
                "--examples",
                str(examples),
                "--index",
                str(index),
                "--out-dir",
                str(out_dir),
                "--top-k",
                "2",
            ]
        )
        == 0
    )

    assert (out_dir / "sft.jsonl").exists()
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "summary.json").exists()


def test_export_sft_warmup_cli_writes_gold_evidence_conversations(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "sft-warmup"

    assert (
        main(
            [
                "prepare-hotpot",
                "--raw",
                "tests/fixtures/hotpot_mixed_raw.jsonl",
                "--corpus",
                str(corpus),
                "--examples",
                str(examples),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert main(["build-index", "--corpus", str(corpus), "--index", str(index)]) == 0
    assert (
        main(
            [
                "export-sft-warmup",
                "--examples",
                str(examples),
                "--index",
                str(index),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    row = json.loads((out_dir / "sft_warmup.jsonl").read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "<answer>Example City</answer>" in row["messages"][-1]["content"]
    assert summary["answer_tag_rate"] == 1.0
    assert (out_dir / "traces.jsonl").exists()


def test_export_sft_turns_cli_writes_turn_level_conversations(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "sft-turns"

    assert (
        main(
            [
                "prepare-hotpot",
                "--raw",
                "tests/fixtures/hotpot_mixed_raw.jsonl",
                "--corpus",
                str(corpus),
                "--examples",
                str(examples),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert main(["build-index", "--corpus", str(corpus), "--index", str(index)]) == 0
    assert (
        main(
            [
                "export-sft-turns",
                "--examples",
                str(examples),
                "--index",
                str(index),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    row = json.loads((out_dir / "sft_turns.jsonl").read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assistant_messages = [message["content"] for message in row["messages"] if message["role"] == "assistant"]
    assert summary["assistant_observation_rate"] == 0.0
    assert summary["assistant_single_action_rate"] == 1.0
    assert any(message["role"] == "user" and "<observation>" in message["content"] for message in row["messages"])
    assert all("<observation>" not in content for content in assistant_messages)
    assert assistant_messages[-1] == "<answer>Example City</answer>"
    assert (out_dir / "traces.jsonl").exists()


def test_export_grpo_cli_writes_rollouts_transitions_rewards_and_summary(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "grpo"

    assert (
        main(
            [
                "prepare-hotpot",
                "--raw",
                "tests/fixtures/hotpot_mixed_raw.jsonl",
                "--corpus",
                str(corpus),
                "--examples",
                str(examples),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert main(["build-index", "--corpus", str(corpus), "--index", str(index)]) == 0
    assert (
        main(
            [
                "export-grpo",
                "--examples",
                str(examples),
                "--index",
                str(index),
                "--out-dir",
                str(out_dir),
                "--top-k",
                "2",
            ]
        )
        == 0
    )

    assert (out_dir / "rollouts.jsonl").exists()
    assert (out_dir / "transitions.jsonl").exists()
    assert (out_dir / "reward_records.jsonl").exists()
    assert (out_dir / "summary.json").exists()


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


def test_train_sft_warmup_cli_dry_run_writes_launch_artifacts(tmp_path):
    sft_rows = tmp_path / "sft_warmup.jsonl"
    sft_rows.write_text(
        json.dumps(
            {
                "id": "s0",
                "messages": [
                    {"role": "system", "content": "Use strict tags."},
                    {"role": "user", "content": "Question?"},
                    {
                        "role": "assistant",
                        "content": (
                            "<think>I should search.</think>\n"
                            "<search>Question?</search>\n"
                            "<observation>\n[1] Evidence.\n</observation>\n"
                            "<think>The evidence supports the answer.</think>\n"
                            "<answer>Answer</answer>"
                        ),
                    },
                ],
                "metadata": {"answer": "Answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "sft.yaml"
    config.write_text(
        f"""
experiment_name: cli-sft-warmup
project_name: lightningsearch-rl
sft_path: {sft_rows}
train_samples: 1
val_samples: 0
seed: 1
model_path: /data/wzl/LightningSearch-RL/models/Qwen/Qwen3-4B
max_length: 1024
train_batch_size: 1
micro_batch_size_per_gpu: 1
max_token_len_per_gpu: 2048
learning_rate: 1.0e-5
n_gpus_per_node: 1
total_training_steps: 1
total_epochs: 1
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
                "train-sft-warmup",
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
    command = (tmp_path / "results" / "launch_command.txt").read_text(encoding="utf-8")
    assert "verl.trainer.sft_trainer" in command


def test_inspect_generation_cli_dry_run_writes_stage_prompts(tmp_path):
    sft_rows = tmp_path / "sft_turns.jsonl"
    sft_rows.write_text(
        json.dumps(
            {
                "id": "turn-0",
                "messages": [
                    {"role": "system", "content": "Output one action."},
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "<search>Question?</search>"},
                    {"role": "user", "content": "<observation>\n[1] Evidence.\n</observation>"},
                    {"role": "assistant", "content": "<answer>Answer</answer>"},
                ],
                "metadata": {"answer": "Answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "inspect-generation",
                "--sft",
                str(sft_rows),
                "--model",
                "unused-model",
                "--out-dir",
                str(tmp_path / "inspection"),
                "--offset",
                "0",
                "--limit",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )

    summary = json.loads((tmp_path / "inspection" / "dry_run_summary.json").read_text(encoding="utf-8"))
    assert summary["search_prompt_count"] == 1
    assert summary["answer_prompt_count"] == 1


def test_inspect_env_rollout_cli_dry_run_writes_prompts(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    index = tmp_path / "index.json"
    sft_rows = tmp_path / "sft_turns.jsonl"
    out_dir = tmp_path / "env-rollout"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "doc_voss",
                        "title": "Dr. Elena Voss",
                        "text": "Dr. Elena Voss founded the Global Health Initiative in 2012.",
                    }
                ),
                json.dumps(
                    {
                        "doc_id": "doc_ghi",
                        "title": "Global Health Initiative",
                        "text": "The Global Health Initiative won the Nobel Peace Prize in 2021.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sft_rows.write_text(
        json.dumps(
            {
                "id": "turn-0",
                "messages": [
                    {"role": "system", "content": "Output one action."},
                    {
                        "role": "user",
                        "content": "Which award did the organization founded by Dr. Elena Voss receive in 2021?",
                    },
                    {
                        "role": "assistant",
                        "content": "<search>Which award did the organization founded by Dr. Elena Voss receive in 2021?</search>",
                    },
                    {"role": "user", "content": "<observation>\n[1] Evidence.\n</observation>"},
                    {"role": "assistant", "content": "<answer>Nobel Peace Prize</answer>"},
                ],
                "metadata": {
                    "answer": "Nobel Peace Prize",
                    "gold_evidence_doc_ids": ["doc_voss", "doc_ghi"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["build-index", "--corpus", str(corpus), "--index", str(index)]) == 0
    assert (
        main(
            [
                "inspect-env-rollout",
                "--sft",
                str(sft_rows),
                "--index",
                str(index),
                "--model",
                "unused-model",
                "--out-dir",
                str(out_dir),
                "--offset",
                "0",
                "--limit",
                "1",
                "--top-k",
                "2",
                "--candidate-pool",
                "gold-distractors",
                "--distractor-count",
                "0",
                "--dry-run",
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "dry_run_summary.json").read_text(encoding="utf-8"))
    answer_prompt = json.loads((out_dir / "answer_prompts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["search_prompt_count"] == 1
    assert summary["answer_prompt_count"] == 1
    assert summary["candidate_pool"] == "gold-distractors"
    assert "Global Health Initiative won the Nobel Peace Prize in 2021" in answer_prompt["messages"][-1]["content"]


def test_inspect_env_rollout_cli_dry_run_records_sampling_options(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    index = tmp_path / "index.json"
    sft_rows = tmp_path / "sft_turns.jsonl"
    out_dir = tmp_path / "env-rollout-sampling"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "doc_voss",
                        "title": "Dr. Elena Voss",
                        "text": "Dr. Elena Voss founded the Global Health Initiative in 2012.",
                    }
                ),
                json.dumps(
                    {
                        "doc_id": "doc_ghi",
                        "title": "Global Health Initiative",
                        "text": "The Global Health Initiative won the Nobel Peace Prize in 2021.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sft_rows.write_text(
        json.dumps(
            {
                "id": "turn-0",
                "messages": [
                    {"role": "system", "content": "Output one action."},
                    {
                        "role": "user",
                        "content": "Which award did the organization founded by Dr. Elena Voss receive in 2021?",
                    },
                    {
                        "role": "assistant",
                        "content": "<search>Which award did the organization founded by Dr. Elena Voss receive in 2021?</search>",
                    },
                    {"role": "user", "content": "<observation>\n[1] Evidence.\n</observation>"},
                    {"role": "assistant", "content": "<answer>Nobel Peace Prize</answer>"},
                ],
                "metadata": {
                    "answer": "Nobel Peace Prize",
                    "gold_evidence_doc_ids": ["doc_voss", "doc_ghi"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["build-index", "--corpus", str(corpus), "--index", str(index)]) == 0
    assert (
        main(
            [
                "inspect-env-rollout",
                "--sft",
                str(sft_rows),
                "--index",
                str(index),
                "--model",
                "unused-model",
                "--out-dir",
                str(out_dir),
                "--offset",
                "0",
                "--limit",
                "1",
                "--do-sample",
                "--temperature",
                "0.7",
                "--top-p",
                "0.9",
                "--sample-top-k",
                "40",
                "--seed",
                "123",
                "--dry-run",
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "dry_run_summary.json").read_text(encoding="utf-8"))
    assert summary["do_sample"] is True
    assert summary["temperature"] == 0.7
    assert summary["top_p"] == 0.9
    assert summary["sample_top_k"] == 40
    assert summary["seed"] == 123


def test_export_env_transitions_cli_writes_transition_artifacts(tmp_path):
    rollouts = tmp_path / "env_rollouts.jsonl"
    out_dir = tmp_path / "env-transitions"
    rollouts.write_text(
        json.dumps(
            {
                "id": "env-1",
                "question": "Question?",
                "search_messages": [
                    {"role": "system", "content": "Output one action."},
                    {"role": "user", "content": "Question?"},
                ],
                "search_generated": "<search>Question evidence</search>",
                "search_action": {
                    "type": "search",
                    "valid": True,
                    "query": "Question evidence",
                    "answer": None,
                    "reason": None,
                },
                "observation": "<observation>\n[1] Evidence: Answer appears here.\n</observation>",
                "observation_doc_ids": ["doc_answer"],
                "candidate_doc_ids": ["doc_answer"],
                "candidate_pool": "gold-distractors",
                "gold_evidence_doc_ids": ["doc_answer"],
                "gold_evidence_recall": 1.0,
                "all_gold_evidence_retrieved": True,
                "answer_messages": [
                    {"role": "system", "content": "Output one action."},
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "<search>Question evidence</search>"},
                    {
                        "role": "user",
                        "content": "<observation>\n[1] Evidence: Answer appears here.\n</observation>",
                    },
                ],
                "answer_generated": "<answer>Answer</answer>",
                "answer_action": {
                    "type": "answer",
                    "valid": True,
                    "query": None,
                    "answer": "Answer",
                    "reason": None,
                },
                "final_answer": "Answer",
                "gold_answer": "Answer",
                "answer_exact_match": True,
                "answer_token_f1": 1.0,
                "answer_containment_match": True,
                "metadata": {"answer": "Answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "export-env-transitions",
                "--rollouts",
                str(rollouts),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["example_count"] == 1
    assert summary["transition_count"] == 2
    assert (out_dir / "transitions.jsonl").exists()
    assert (out_dir / "reward_records.jsonl").exists()
    assert (out_dir / "rollouts_for_grpo.jsonl").exists()


def test_export_env_transitions_cli_excludes_quality_manifest_rows(tmp_path):
    rollouts = tmp_path / "env_rollouts.jsonl"
    out_dir = tmp_path / "env-transitions"
    manifest = tmp_path / "quality_manifest.json"
    rollouts.write_text(
        json.dumps(
            {
                "id": "env-1",
                "question": "Question?",
                "search_messages": [
                    {"role": "system", "content": "Output one action."},
                    {"role": "user", "content": "Question?"},
                ],
                "search_generated": "<search>Question evidence</search>",
                "search_action": {
                    "type": "search",
                    "valid": True,
                    "query": "Question evidence",
                    "answer": None,
                    "reason": None,
                },
                "observation": "<observation>\n[1] Evidence: Answer appears here.\n</observation>",
                "observation_doc_ids": ["doc_answer"],
                "candidate_doc_ids": ["doc_answer"],
                "candidate_pool": "gold-distractors",
                "gold_evidence_doc_ids": ["doc_answer"],
                "gold_evidence_recall": 1.0,
                "all_gold_evidence_retrieved": True,
                "answer_messages": [
                    {"role": "system", "content": "Output one action."},
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "<search>Question evidence</search>"},
                    {
                        "role": "user",
                        "content": "<observation>\n[1] Evidence: Answer appears here.\n</observation>",
                    },
                ],
                "answer_generated": "<answer>Answer</answer>",
                "answer_action": {
                    "type": "answer",
                    "valid": True,
                    "query": None,
                    "answer": "Answer",
                    "reason": None,
                },
                "final_answer": "Answer",
                "gold_answer": "Answer",
                "answer_exact_match": True,
                "answer_token_f1": 1.0,
                "answer_containment_match": True,
                "metadata": {"answer": "Answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({"env-1": {"flags": ["qa_type_mismatch"], "notes": ["known bad row"]}}),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "export-env-transitions",
                "--rollouts",
                str(rollouts),
                "--out-dir",
                str(out_dir),
                "--quality-manifest",
                str(manifest),
                "--exclude-quality-flag",
                "qa_type_mismatch",
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_example_count"] == 1
    assert summary["example_count"] == 0
    assert summary["excluded_example_count"] == 1
    assert summary["excluded_example_ids"] == ["env-1"]
    assert summary["excluded_quality_flag_counts"] == {"qa_type_mismatch": 1}
    assert (out_dir / "transitions.jsonl").read_text(encoding="utf-8") == ""


def test_filter_transitions_by_reward_variance_cli_writes_filtered_artifacts(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    reward_dump = tmp_path / "reward_dump.jsonl"
    out_dir = tmp_path / "variance-filtered"
    transitions.write_text(
        "\n".join(
            [
                json.dumps({"id": "syn-a", "transition_id": "syn-a:0:search"}),
                json.dumps({"id": "syn-a", "transition_id": "syn-a:1:answer"}),
                json.dumps({"id": "syn-b", "transition_id": "syn-b:0:search"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reward_dump.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "reward_stage": "search",
                        "score": 0.97,
                        "extra_info": {"id": "syn-a:r0", "source_id": "syn-a"},
                    }
                ),
                json.dumps(
                    {
                        "reward_stage": "search",
                        "score": 0.07,
                        "extra_info": {"id": "syn-a:r1", "source_id": "syn-a"},
                    }
                ),
                json.dumps(
                    {
                        "reward_stage": "search",
                        "score": 0.97,
                        "extra_info": {"id": "syn-b:r0", "source_id": "syn-b"},
                    }
                ),
                json.dumps(
                    {
                        "reward_stage": "search",
                        "score": 0.97,
                        "extra_info": {"id": "syn-b:r1", "source_id": "syn-b"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "filter-transitions-by-reward-variance",
                "--transitions",
                str(transitions),
                "--reward-dump",
                str(reward_dump),
                "--out-dir",
                str(out_dir),
                "--stage",
                "search",
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_source_ids"] == ["syn-a"]
    assert len((out_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_build_preference_pairs_cli_writes_pair_artifacts(tmp_path):
    requests = tmp_path / "probe_requests.jsonl"
    generations = tmp_path / "generations.jsonl"
    reward_dump = tmp_path / "reward_dump.jsonl"
    out_dir = tmp_path / "pairs"
    requests.write_text(
        json.dumps(
            {
                "request_index": 0,
                "prompt": [{"role": "user", "content": "Question?"}],
                "ground_truth": "",
                "extra_info": {
                    "id": "syn-a:0:search",
                    "source_id": "syn-a",
                    "index": 0,
                    "reward_stage": "search",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generations.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_index": 0,
                        "sample_index": 0,
                        "id": "syn-a:0:search",
                        "source_id": "syn-a",
                        "reward_stage": "search",
                        "solution": "<search>strong query</search>",
                        "score": 0.97,
                    }
                ),
                json.dumps(
                    {
                        "request_index": 0,
                        "sample_index": 1,
                        "id": "syn-a:0:search",
                        "source_id": "syn-a",
                        "reward_stage": "search",
                        "solution": "<search>bad query</search>",
                        "score": 0.07,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reward_dump.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "reward_stage": "search",
                        "score": 0.97,
                        "extra_info": {
                            "id": "syn-a:0:search",
                            "source_id": "syn-a",
                            "index": 0,
                            "probe_sample_index": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "reward_stage": "search",
                        "score": 0.07,
                        "extra_info": {
                            "id": "syn-a:0:search",
                            "source_id": "syn-a",
                            "index": 0,
                            "probe_sample_index": 1,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "build-preference-pairs",
                "--probe-requests",
                str(requests),
                "--generations",
                str(generations),
                "--reward-dump",
                str(reward_dump),
                "--out-dir",
                str(out_dir),
                "--stage",
                "search",
                "--pair-category",
                "search_vs_search",
                "--min-score-gap",
                "0.5",
                "--val-fraction",
                "0",
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["pair_categories"] == ["search_vs_search"]
    assert summary["pair_count"] == 1
    assert summary["pair_category_counts"] == {"search_vs_search": 1}
    assert summary["stage_pair_counts"] == {"search": 1}
    assert len((out_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_build_synthetic_search_preferences_cli_writes_pair_artifacts(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    out_dir = tmp_path / "synthetic-pairs"
    passages = [
        {
            "doc_id": "doc_voss",
            "title": "Dr. Elena Voss",
            "text": "Dr. Elena Voss founded the Global Health Initiative in 2012.",
        },
        {
            "doc_id": "doc_ghi",
            "title": "Global Health Initiative",
            "text": "The Global Health Initiative won the Nobel Peace Prize in 2021.",
        },
        {
            "doc_id": "doc_archive",
            "title": "Ocean Archive",
            "text": "The Ocean Archive catalogues reef maps and harbor records.",
        },
    ]
    transitions.write_text(
        json.dumps(
            {
                "id": "syn-a",
                "transition_id": "syn-a:0:search",
                "action_type": "search",
                "action": "<search>Elena Voss Global Health Initiative Nobel Peace Prize</search>",
                "query": "Elena Voss Global Health Initiative Nobel Peace Prize",
                "state_messages": [
                    {"role": "system", "content": "Output one action."},
                    {
                        "role": "user",
                        "content": "Which award did the organization founded by Dr. Elena Voss receive in 2021?",
                    },
                ],
                "candidate_passages": passages,
                "gold_evidence_doc_ids": ["doc_voss", "doc_ghi"],
                "metadata": {
                    "gold_answer": "Nobel Peace Prize",
                    "candidate_doc_ids": ["doc_voss", "doc_ghi", "doc_archive"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "build-synthetic-search-preferences",
                "--transitions",
                str(transitions),
                "--out-dir",
                str(out_dir),
                "--search-reward-top-k",
                "2",
                "--min-chosen-score",
                "0.8",
                "--min-score-gap",
                "0.2",
                "--max-negatives-per-transition",
                "2",
                "--val-fraction",
                "0",
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["pair_count"] >= 1
    assert summary["pair_category_counts"] == {"search_vs_search": summary["pair_count"]}
    assert (out_dir / "pairs.jsonl").exists()
    assert (out_dir / "candidates.jsonl").exists()
    assert (out_dir / "reward_dump.jsonl").exists()


def test_probe_reward_variance_cli_dry_run_writes_requests(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    out_dir = tmp_path / "reward-probe"
    transitions.write_text(
        json.dumps(
            {
                "id": "syn-a",
                "transition_id": "syn-a:0:search",
                "action_type": "search",
                "state_messages": [
                    {"role": "system", "content": "Output one action."},
                    {"role": "user", "content": "Question?"},
                ],
                "candidate_passages": [],
                "metadata": {"gold_answer": "Answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "probe-reward-variance",
                "--transitions",
                str(transitions),
                "--model",
                "unused-model",
                "--out-dir",
                str(out_dir),
                "--limit",
                "1",
                "--samples-per-prompt",
                "2",
                "--stage",
                "search",
                "--search-diversity-prompt",
                "--dry-run",
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["stages"] == ["search"]
    assert summary["search_diversity_prompt"] is True
    assert summary["expected_reward_rows"] == 2
    assert (out_dir / "probe_requests.jsonl").exists()


def test_synthetic_cli_mock_generation_validation_and_prepare_pipeline(tmp_path):
    raw = tmp_path / "synthetic_raw.jsonl"
    synthesis_summary = tmp_path / "synthesis_summary.json"
    valid = tmp_path / "synthetic_valid.jsonl"
    rejects = tmp_path / "synthetic_rejects.jsonl"
    validation_summary = tmp_path / "validation_summary.json"
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"

    assert (
        main(
            [
                "synthesize-data",
                "--mock",
                "--out",
                str(raw),
                "--count",
                "2",
                "--topics",
                "awards,archives",
                "--concurrency",
                "50",
                "--seed",
                "3",
                "--summary",
                str(synthesis_summary),
            ]
        )
        == 0
    )
    synthesis_payload = json.loads(synthesis_summary.read_text(encoding="utf-8"))
    assert synthesis_payload["written"] == 2
    assert len(raw.read_text(encoding="utf-8").splitlines()) == 2

    assert (
        main(
            [
                "validate-synthetic",
                "--raw",
                str(raw),
                "--valid",
                str(valid),
                "--rejects",
                str(rejects),
                "--summary",
                str(validation_summary),
                "--require-chain-schema",
            ]
        )
        == 0
    )
    validation_payload = json.loads(validation_summary.read_text(encoding="utf-8"))
    assert validation_payload == {"raw_count": 2, "valid_count": 2, "reject_count": 0}

    assert (
        main(
            [
                "prepare-hotpot",
                "--raw",
                str(valid),
                "--corpus",
                str(corpus),
                "--examples",
                str(examples),
            ]
        )
        == 0
    )
    assert len(examples.read_text(encoding="utf-8").splitlines()) == 2


def test_synthetic_validated_cli_mock_pipeline_writes_target_valid_rows(tmp_path):
    raw = tmp_path / "synthetic_raw.jsonl"
    valid = tmp_path / "synthetic_valid.jsonl"
    rejects = tmp_path / "synthetic_rejects.jsonl"
    summary = tmp_path / "validated_summary.json"
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"

    assert (
        main(
            [
                "synthesize-validated-data",
                "--mock",
                "--raw",
                str(raw),
                "--valid",
                str(valid),
                "--rejects",
                str(rejects),
                "--target-valid",
                "2",
                "--topics",
                "awards,archives",
                "--concurrency",
                "50",
                "--batch-size",
                "2",
                "--max-attempts",
                "2",
                "--seed",
                "40",
                "--summary",
                str(summary),
                "--require-chain-schema",
                "--repair-chain-schema",
                "--few-shot-chain-schema",
            ]
        )
        == 0
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["valid_count"] == 2
    assert payload["reject_count"] == 0
    assert payload["stopped_reason"] == "target_valid_reached"
    assert payload["repair_chain_schema"] is True
    assert payload["repair_attempt_count"] == 0
    assert payload["repair_success_count"] == 0
    assert payload["few_shot_chain_schema"] is True

    assert (
        main(
            [
                "prepare-hotpot",
                "--raw",
                str(valid),
                "--corpus",
                str(corpus),
                "--examples",
                str(examples),
            ]
        )
        == 0
    )
    assert len(examples.read_text(encoding="utf-8").splitlines()) == 2
