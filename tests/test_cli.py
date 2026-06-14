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
