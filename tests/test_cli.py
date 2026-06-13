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
