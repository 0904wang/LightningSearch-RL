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
