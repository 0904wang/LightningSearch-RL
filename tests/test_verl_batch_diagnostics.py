import json

from lightningsearch_rl.cli import main
from lightningsearch_rl.verl_batch_diagnostics import diagnose_verl_training_batches


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_diagnose_verl_training_batches_aligns_steps_with_contiguous_batches(tmp_path):
    train_jsonl = tmp_path / "train.jsonl"
    metrics_summary = tmp_path / "metrics_summary.json"
    _write_jsonl(
        train_jsonl,
        [
            _row("q0:0:search", "q0", "search", 0.27, 1.37),
            _row("q0:1:answer", "q0", "answer", 1.1, 1.37),
            _row("q1:0:search", "q1", "search", 0.27, 0.27),
            _row("q1:1:answer", "q1", "answer", 0.0, 0.27),
        ],
    )
    metrics_summary.write_text(
        json.dumps(
            {
                "steps": {
                    "1": {"training/global_step": 1, "critic/rewards/mean": 0.685},
                    "2": {"training/global_step": 2, "critic/rewards/mean": 0.31},
                }
            }
        ),
        encoding="utf-8",
    )

    report = diagnose_verl_training_batches(
        train_jsonl,
        metrics_summary_path=metrics_summary,
        train_batch_size=2,
    )

    assert report["train_rows"] == 4
    assert report["batch_count"] == 2
    assert report["overall"]["stage_counts"] == {"answer": 2, "search": 2}
    assert report["batches"][0]["stage_counts"] == {"answer": 1, "search": 1}
    assert report["batches"][0]["reward_model_reward"]["mean"] == 0.685
    assert report["batches"][1]["low_reward_row_count"] == 1
    assert report["step_alignment"] == [
        {
            "step": 1,
            "batch_index": 0,
            "row_start": 0,
            "row_end_exclusive": 2,
            "logged_reward_mean": 0.685,
            "precomputed_reward_mean": 0.685,
            "stage_counts": {"answer": 1, "search": 1},
            "low_reward_row_count": 0,
        },
        {
            "step": 2,
            "batch_index": 1,
            "row_start": 2,
            "row_end_exclusive": 4,
            "logged_reward_mean": 0.31,
            "precomputed_reward_mean": 0.135,
            "stage_counts": {"answer": 1, "search": 1},
            "low_reward_row_count": 1,
        },
    ]


def test_diagnose_verl_batches_cli_writes_report(tmp_path):
    train_jsonl = tmp_path / "train.jsonl"
    metrics_summary = tmp_path / "metrics_summary.json"
    out = tmp_path / "batch_diagnostics.json"
    _write_jsonl(
        train_jsonl,
        [
            _row("q0:0:search", "q0", "search", 0.27, 1.37),
            _row("q0:1:answer", "q0", "answer", 1.1, 1.37),
        ],
    )
    metrics_summary.write_text(
        json.dumps({"steps": {"1": {"training/global_step": 1, "critic/rewards/mean": 0.685}}}),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "diagnose-verl-batches",
                "--train-jsonl",
                str(train_jsonl),
                "--metrics-summary",
                str(metrics_summary),
                "--train-batch-size",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["batch_count"] == 1
    assert report["step_alignment"][0]["precomputed_reward_mean"] == 0.685


def _row(row_id, source_id, stage, reward, total_reward):
    return {
        "reward_model": {"reward": reward},
        "extra_info": {
            "id": row_id,
            "source_id": source_id,
            "reward_stage": stage,
            "precomputed_step_reward": reward,
            "precomputed_total_reward": total_reward,
            "expected_action": f"<{stage}>x</{stage}>",
        },
    }
