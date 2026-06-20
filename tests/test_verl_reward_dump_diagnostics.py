import json

from lightningsearch_rl.cli import main
from lightningsearch_rl.verl_reward_dump_diagnostics import diagnose_reward_dump


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_diagnose_reward_dump_summarizes_stage_components_and_low_scores(tmp_path):
    dump = tmp_path / "reward_dump.jsonl"
    _write_jsonl(
        dump,
        [
            _row("a0", "answer", 1.1, answer_reward=1.0, search_reward=0.0, valid=True),
            _row("a1", "answer", 0.1, answer_reward=0.0, search_reward=0.0, valid=True),
            _row("s0", "search", 1.07, answer_reward=0.0, search_reward=1.0, valid=True, search_cost=0.03),
            _row("s1", "search", -0.03, answer_reward=0.0, search_reward=0.0, valid=False, search_cost=0.03),
        ],
    )

    report = diagnose_reward_dump(dump, low_score_threshold=0.5)

    assert report["row_count"] == 4
    assert report["stage_counts"] == {"answer": 2, "search": 2}
    assert report["overall"]["score"]["mean"] == 0.56
    assert report["by_stage"]["answer"]["score"]["mean"] == 0.6
    assert report["by_stage"]["search"]["search_reward"]["mean"] == 0.5
    assert report["by_stage"]["search"]["invalid_action_count"] == 1
    assert report["by_stage"]["answer"]["low_score_count"] == 1
    assert report["by_stage"]["answer"]["answer_reward_type_counts"] == {"exact": 1, "none": 1}
    assert report["low_score_examples"][0]["id"] == "a1"


def test_diagnose_reward_dump_reads_stringified_extra_info(tmp_path):
    dump = tmp_path / "reward_dump.jsonl"
    _write_jsonl(
        dump,
        [
            {
                "reward_stage": "answer",
                "score": 0.1,
                "answer_reward": 0.0,
                "search_reward": 0.0,
                "format_reward": 1.0,
                "search_cost": 0.0,
                "solution_preview": "<answer>Golden Quill Award</answer>",
                "parsed_action": {"type": "answer", "valid": True},
                "extra_info": "{'id': 'syn-009020:1:answer', 'source_id': 'syn-009020', 'split': 'train'}",
            }
        ],
    )

    report = diagnose_reward_dump(dump, low_score_threshold=0.5)

    assert report["low_score_examples"][0]["id"] == "syn-009020:1:answer"
    assert report["low_score_examples"][0]["source_id"] == "syn-009020"


def test_diagnose_reward_dump_reports_group_score_variance(tmp_path):
    dump = tmp_path / "reward_dump.jsonl"
    _write_jsonl(
        dump,
        [
            _row_with_source("env-1:a", "env-1", "search", 0.97, search_reward=0.9),
            _row_with_source("env-1:b", "env-1", "search", 0.07, search_reward=0.0),
            _row_with_source("env-1:c", "env-1", "search", 0.97, search_reward=0.9),
            _row_with_source("env-1:d", "env-1", "search", 0.07, search_reward=0.0),
            _row_with_source("env-2:a", "env-2", "search", 0.97, search_reward=0.9),
            _row_with_source("env-2:b", "env-2", "search", 0.97, search_reward=0.9),
        ],
    )

    report = diagnose_reward_dump(dump)
    variance = report["by_stage"]["search"]["group_score_variance"]

    assert variance["group_count"] == 2
    assert variance["variable_group_count"] == 1
    assert variance["variable_group_rate"] == 0.5
    assert variance["avg_score_range"] == 0.45
    assert variance["top_variable_groups"][0] == {
        "source_id": "env-1",
        "count": 4,
        "score_min": 0.07,
        "score_max": 0.97,
        "score_range": 0.9,
    }


def test_diagnose_reward_dump_cli_writes_report(tmp_path):
    dump = tmp_path / "reward_dump.jsonl"
    out = tmp_path / "reward_dump_summary.json"
    _write_jsonl(dump, [_row("a0", "answer", 1.1, answer_reward=1.0, search_reward=0.0, valid=True)])

    assert main(["diagnose-reward-dump", "--dump", str(dump), "--out", str(out)]) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["row_count"] == 1
    assert report["by_stage"]["answer"]["answer_reward"]["mean"] == 1.0


def _row(row_id, stage, score, *, answer_reward, search_reward, valid, search_cost=0.0):
    return {
        "reward_stage": stage,
        "score": score,
        "answer_reward": answer_reward,
        "answer_reward_type": "exact" if answer_reward == 1.0 else "none",
        "search_reward": search_reward,
        "format_reward": 1.0 if valid else 0.0,
        "search_cost": search_cost,
        "solution_preview": f"<{stage}>preview</{stage}>",
        "parsed_action": {"type": stage, "valid": valid},
        "extra_info": {"id": row_id, "source_id": row_id.split("0")[0]},
    }


def _row_with_source(row_id, source_id, stage, score, *, search_reward):
    return {
        "reward_stage": stage,
        "score": score,
        "answer_reward": 0.0,
        "answer_reward_type": None,
        "search_reward": search_reward,
        "evidence_rank_reward": search_reward,
        "format_reward": 1.0,
        "search_cost": 0.03,
        "solution_preview": "<search>query</search>",
        "parsed_action": {"type": stage, "valid": True},
        "extra_info": {"id": row_id, "source_id": source_id},
    }
