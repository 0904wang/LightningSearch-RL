import json

from lightningsearch_rl.reward_variance_filter import filter_transitions_by_reward_variance


def test_filter_transitions_by_reward_variance_keeps_variable_source_groups(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    reward_dump = tmp_path / "reward_dump.jsonl"
    out_dir = tmp_path / "filtered"
    _write_jsonl(
        transitions,
        [
            _transition("syn-a", 0, "search"),
            _transition("syn-a", 1, "answer"),
            _transition("syn-b", 0, "search"),
            _transition("syn-b", 1, "answer"),
            _transition("syn-c", 0, "search"),
            _transition("syn-c", 1, "answer"),
        ],
    )
    _write_jsonl(
        reward_dump,
        [
            _reward_row("syn-a:r0", "syn-a", "search", 0.97),
            _reward_row("syn-a:r1", "syn-a", "search", 0.07),
            _reward_row("syn-b:r0", "syn-b", "search", 0.97),
            _reward_row("syn-b:r1", "syn-b", "search", 0.97),
            _reward_row("syn-c:r0", "syn-c", "answer", 1.1),
            _reward_row("syn-c:r1", "syn-c", "answer", 0.1),
            _reward_row("syn-d:r0", "syn-d", "search", 0.97),
        ],
    )

    summary = filter_transitions_by_reward_variance(
        transitions_path=transitions,
        reward_dump_path=reward_dump,
        out_dir=out_dir,
        stages=("search", "answer"),
    )

    filtered_rows = _read_jsonl(out_dir / "transitions.jsonl")
    assert [row["id"] for row in filtered_rows] == ["syn-a", "syn-a", "syn-c", "syn-c"]
    assert json.loads((out_dir / "selected_source_ids.json").read_text(encoding="utf-8")) == [
        "syn-a",
        "syn-c",
    ]
    assert summary["input_transition_count"] == 6
    assert summary["output_transition_count"] == 4
    assert summary["selected_source_count"] == 2
    assert summary["stage_variable_group_counts"] == {"answer": 1, "search": 1}
    assert summary["selected_source_ids"] == ["syn-a", "syn-c"]
    assert summary["top_variable_groups"][0]["source_id"] == "syn-c"
    assert summary["top_variable_groups"][0]["score_range"] == 1.0


def test_filter_transitions_by_reward_variance_limits_sources_by_score_range(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    reward_dump = tmp_path / "reward_dump.jsonl"
    out_dir = tmp_path / "filtered"
    _write_jsonl(
        transitions,
        [
            _transition("syn-a", 0, "search"),
            _transition("syn-b", 0, "search"),
            _transition("syn-c", 0, "search"),
        ],
    )
    _write_jsonl(
        reward_dump,
        [
            _reward_row("syn-a:r0", "syn-a", "search", 0.97),
            _reward_row("syn-a:r1", "syn-a", "search", 0.67),
            _reward_row("syn-b:r0", "syn-b", "search", 0.97),
            _reward_row("syn-b:r1", "syn-b", "search", 0.07),
            _reward_row("syn-c:r0", "syn-c", "search", 0.97),
            _reward_row("syn-c:r1", "syn-c", "search", 0.47),
        ],
    )

    summary = filter_transitions_by_reward_variance(
        transitions_path=transitions,
        reward_dump_path=reward_dump,
        out_dir=out_dir,
        stages=("search",),
        max_source_count=2,
    )

    assert summary["selected_source_ids"] == ["syn-b", "syn-c"]
    assert [row["id"] for row in _read_jsonl(out_dir / "transitions.jsonl")] == ["syn-b", "syn-c"]


def _transition(source_id, step_index, action_type):
    return {
        "id": source_id,
        "transition_id": f"{source_id}:{step_index}:{action_type}",
        "action_type": action_type,
        "state_messages": [{"role": "user", "content": "Question?"}],
        "action": f"<{action_type}>payload</{action_type}>",
        "reward": 1.0,
        "terminal": action_type == "answer",
    }


def _reward_row(row_id, source_id, stage, score):
    return {
        "reward_stage": stage,
        "score": score,
        "parsed_action": {"type": stage, "valid": True},
        "extra_info": {"id": row_id, "source_id": source_id},
    }


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
