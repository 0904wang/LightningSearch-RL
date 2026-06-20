import json

from lightningsearch_rl.preference_pairs import build_preference_pairs


def test_build_preference_pairs_selects_clear_reward_gap_and_deduplicates_actions(tmp_path):
    requests = tmp_path / "probe_requests.jsonl"
    generations = tmp_path / "generations.jsonl"
    reward_dump = tmp_path / "reward_dump.jsonl"
    out_dir = tmp_path / "pairs"
    _write_jsonl(
        requests,
        [
            _request(
                request_index=0,
                source_id="syn-a",
                transition_id="syn-a:0:search",
                stage="search",
            )
        ],
    )
    _write_jsonl(
        generations,
        [
            _generation(0, 0, "syn-a", "syn-a:0:search", "search", "<search>Elena Voss health prize</search>", 0.97),
            _generation(0, 1, "syn-a", "syn-a:0:search", "search", "<search>archive unrelated materials</search>", 0.07),
            _generation(0, 2, "syn-a", "syn-a:0:search", "search", "<search>  Elena   Voss health prize  </search>", 0.96),
        ],
    )
    _write_jsonl(
        reward_dump,
        [
            _reward_row(0, 0, "syn-a", "syn-a:0:search", "search", 0.97, evidence_rank_reward=1.0),
            _reward_row(0, 1, "syn-a", "syn-a:0:search", "search", 0.07, evidence_rank_reward=0.0),
            _reward_row(0, 2, "syn-a", "syn-a:0:search", "search", 0.96, evidence_rank_reward=1.0),
        ],
    )

    summary = build_preference_pairs(
        probe_requests_path=requests,
        generations_path=generations,
        reward_dump_path=reward_dump,
        out_dir=out_dir,
        stages=("search",),
        min_score_gap=0.5,
        max_pairs_per_group=2,
        val_fraction=0.0,
    )

    pairs = _read_jsonl(out_dir / "pairs.jsonl")
    assert summary["pair_count"] == 1
    assert summary["stage_pair_counts"] == {"search": 1}
    assert pairs[0]["pair_id"] == "syn-a:0:search:search:0:0>1"
    assert pairs[0]["prompt"][0]["role"] == "system"
    assert pairs[0]["chosen"] == "<search>Elena Voss health prize</search>"
    assert pairs[0]["rejected"] == "<search>archive unrelated materials</search>"
    assert pairs[0]["chosen_score"] == 0.97
    assert pairs[0]["rejected_score"] == 0.07
    assert pairs[0]["score_gap"] == 0.9
    assert pairs[0]["chosen_reward"]["evidence_rank_reward"] == 1.0
    assert pairs[0]["rejected_reward"]["evidence_rank_reward"] == 0.0
    assert len(_read_jsonl(out_dir / "train.jsonl")) == 1
    assert len(_read_jsonl(out_dir / "val.jsonl")) == 0


def test_build_preference_pairs_caps_answer_pairs_while_keeping_search_pairs(tmp_path):
    requests = tmp_path / "probe_requests.jsonl"
    generations = tmp_path / "generations.jsonl"
    reward_dump = tmp_path / "reward_dump.jsonl"
    out_dir = tmp_path / "pairs"
    _write_jsonl(
        requests,
        [
            _request(0, "syn-a", "syn-a:0:search", "search"),
            _request(1, "syn-b", "syn-b:1:answer", "answer"),
            _request(2, "syn-c", "syn-c:1:answer", "answer"),
        ],
    )
    _write_jsonl(
        generations,
        [
            _generation(0, 0, "syn-a", "syn-a:0:search", "search", "<search>strong query</search>", 0.97),
            _generation(0, 1, "syn-a", "syn-a:0:search", "search", "<search>bad query</search>", 0.07),
            _generation(1, 0, "syn-b", "syn-b:1:answer", "answer", "<answer>Nobel Peace Prize</answer>", 1.1),
            _generation(1, 1, "syn-b", "syn-b:1:answer", "answer", "no answer tag", 0.0),
            _generation(2, 0, "syn-c", "syn-c:1:answer", "answer", "<answer>Golden Quill Award</answer>", 1.1),
            _generation(2, 1, "syn-c", "syn-c:1:answer", "answer", "<answer>wrong</answer>", 0.1),
        ],
    )
    _write_jsonl(
        reward_dump,
        [
            _reward_row(0, 0, "syn-a", "syn-a:0:search", "search", 0.97),
            _reward_row(0, 1, "syn-a", "syn-a:0:search", "search", 0.07),
            _reward_row(1, 0, "syn-b", "syn-b:1:answer", "answer", 1.1),
            _reward_row(1, 1, "syn-b", "syn-b:1:answer", "answer", 0.0),
            _reward_row(2, 0, "syn-c", "syn-c:1:answer", "answer", 1.1),
            _reward_row(2, 1, "syn-c", "syn-c:1:answer", "answer", 0.1),
        ],
    )

    summary = build_preference_pairs(
        probe_requests_path=requests,
        generations_path=generations,
        reward_dump_path=reward_dump,
        out_dir=out_dir,
        min_score_gap=0.5,
        max_answer_pairs=1,
        val_fraction=0.5,
        seed=7,
    )

    pairs = _read_jsonl(out_dir / "pairs.jsonl")
    assert summary["candidate_pair_count"] == 3
    assert summary["pair_count"] == 2
    assert summary["stage_pair_counts"] == {"answer": 1, "search": 1}
    assert {row["reward_stage"] for row in pairs} == {"search", "answer"}
    assert len(_read_jsonl(out_dir / "train.jsonl")) == 1
    assert len(_read_jsonl(out_dir / "val.jsonl")) == 1


def _request(request_index, source_id, transition_id, stage):
    return {
        "request_index": request_index,
        "prompt": [
            {"role": "system", "content": "Output one action."},
            {"role": "user", "content": "Question?"},
        ],
        "ground_truth": "Nobel Peace Prize" if stage == "answer" else "",
        "extra_info": {
            "id": transition_id,
            "source_id": source_id,
            "index": request_index,
            "reward_stage": stage,
            "expected_action": f"<{stage}>gold</{stage}>",
            "gold_doc_ids": ["doc_bridge", "doc_answer"],
        },
    }


def _generation(request_index, sample_index, source_id, transition_id, stage, solution, score):
    return {
        "request_index": request_index,
        "sample_index": sample_index,
        "id": transition_id,
        "source_id": source_id,
        "reward_stage": stage,
        "solution": solution,
        "score": score,
        "reward": {"score": score},
    }


def _reward_row(
    request_index,
    sample_index,
    source_id,
    transition_id,
    stage,
    score,
    *,
    evidence_rank_reward=0.0,
):
    return {
        "reward_stage": stage,
        "score": score,
        "answer_reward": score if stage == "answer" else 0.0,
        "search_reward": score if stage == "search" else 0.0,
        "evidence_rank_reward": evidence_rank_reward,
        "format_reward": 1.0,
        "parsed_action": {"type": stage, "valid": True},
        "extra_info": {
            "id": transition_id,
            "source_id": source_id,
            "index": request_index,
            "probe_sample_index": sample_index,
            "reward_stage": stage,
        },
    }


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
