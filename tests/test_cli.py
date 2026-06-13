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
