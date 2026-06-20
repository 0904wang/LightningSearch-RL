import json

from lightningsearch_rl.cli import main
from lightningsearch_rl.verl_log_parser import parse_verl_log


def _write_log(path):
    path.write_text(
        "\n".join(
            [
                "started_at=2026-06-16T08:24:51+00:00",
                "\x1b[36m(TaskRunner pid=1)\x1b[0m step:0 - val-core/lightningsearch_rl/reward/mean@1:0.8350000325590372 - val-aux/lightningsearch_rl/answer_reward/mean@1:0.25 - val-aux/lightningsearch_rl/search_reward/mean@1:0.5 - val-aux/lightningsearch_rl/format_reward/mean@1:1.0 - val-aux/lightningsearch_rl/search_cost/mean@1:0.015 - val-aux/num_turns/mean:2.0",
                "\x1b[36m(TaskRunner pid=1)\x1b[0m Training Progress: 100%|##########| 1/1 [00:17<00:00, 17.75s/it]",
                "\x1b[36m(TaskRunner pid=1)\x1b[0m step:1 - training/global_step:1 - critic/score/mean:1.0850000381469727 - critic/rewards/mean:1.0850000381469727 - actor/loss:-1.0796763896942139 - actor/grad_norm:0.0284423828125 - response_length/mean:15.5 - response_length/max:22.0 - response_length/clip_ratio:0.0 - response/aborted_ratio:0.0 - prompt_length/mean:191.25 - prompt_length/max:297.0 - prompt_length/clip_ratio:0.0 - num_turns/mean:2.0 - perf/time_per_step:17.620004686992615 - perf/throughput:11.733822077392881",
                "\x1b[36m(TaskRunner pid=1)\x1b[0m RuntimeError: DataLoader worker (pid 123) is killed by signal: Killed.",
                "\x1b[36m(vLLMHttpServer pid=2)\x1b[0m ERROR 06-16 08:26:32 [core_client.py:600] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.",
                "finished_at=2026-06-16T08:26:37+00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_verl_log_extracts_steps_completion_and_warning_classes(tmp_path):
    log = tmp_path / "train.log"
    _write_log(log)

    summary = parse_verl_log(log)

    assert summary["started_at"] == "2026-06-16T08:24:51+00:00"
    assert summary["finished_at"] == "2026-06-16T08:26:37+00:00"
    assert summary["completed"] is True
    assert summary["fatal_marker_count"] == 0
    assert summary["shutdown_warning_count"] == 2
    assert summary["training_progress_100_seen"] is True
    assert summary["final_step"] == 1
    assert summary["steps"]["0"]["val-core/lightningsearch_rl/reward/mean@1"] == 0.8350000325590372
    assert summary["steps"]["0"]["val-aux/lightningsearch_rl/answer_reward/mean@1"] == 0.25
    assert summary["initial_validation_metrics"] == summary["steps"]["0"]
    assert "0" not in summary["train_steps"]
    assert summary["steps"]["1"]["training/global_step"] == 1.0
    assert summary["steps"]["1"]["critic/score/mean"] == 1.0850000381469727
    assert summary["steps"]["1"]["response_length/clip_ratio"] == 0.0
    assert summary["latest_train_metrics"]["critic/rewards/mean"] == 1.0850000381469727
    assert summary["latest_train_metrics"]["prompt_length/max"] == 297.0
    assert summary["reward_curve"] == [
        {
            "step": 1,
            "critic/rewards/mean": 1.0850000381469727,
            "critic/score/mean": 1.0850000381469727,
        }
    ]
    assert len(summary["shutdown_warning_examples"]) == 2


def test_parse_verl_log_marks_fatal_run_when_command_error_seen(tmp_path):
    log = tmp_path / "failed.log"
    log.write_text(
        "\n".join(
            [
                "started_at=2026-06-16T08:17:33+00:00",
                "ValueError: To serve at least one request with the models's max seq len (1280)",
                "subprocess.CalledProcessError: Command returned non-zero exit status 1.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = parse_verl_log(log)

    assert summary["completed"] is False
    assert summary["fatal_marker_count"] == 2
    assert "kv_cache_or_model_len" in summary["fatal_markers"]
    assert "called_process_error" in summary["fatal_markers"]


def test_parse_verl_log_cli_writes_metrics_summary(tmp_path):
    log = tmp_path / "train.log"
    out = tmp_path / "metrics_summary.json"
    _write_log(log)

    assert main(["parse-verl-log", "--log", str(log), "--out", str(out)]) == 0

    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["completed"] is True
    assert summary["final_step"] == 1
    assert summary["latest_train_metrics"]["critic/score/mean"] == 1.0850000381469727
    assert summary["initial_validation_metrics"]["val-aux/lightningsearch_rl/search_reward/mean@1"] == 0.5


def test_parse_verl_log_flags_large_reward_drops(tmp_path):
    log = tmp_path / "drop.log"
    log.write_text(
        "\n".join(
            [
                "started_at=2026-06-16T08:40:11+00:00",
                "step:1 - training/global_step:1 - critic/rewards/mean:1.1 - critic/score/mean:1.1",
                "step:2 - training/global_step:2 - critic/rewards/mean:1.08 - critic/score/mean:1.08",
                "step:3 - training/global_step:3 - critic/rewards/mean:0.59 - critic/score/mean:0.59",
                "Training Progress: 100%|##########| 3/3",
                "finished_at=2026-06-16T08:42:19+00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = parse_verl_log(log)

    assert summary["completed"] is True
    assert summary["reward_drop_alerts"] == [
        {
            "step": 3,
            "previous_step": 2,
            "previous_reward_mean": 1.08,
            "reward_mean": 0.59,
            "delta": -0.49,
        }
    ]
