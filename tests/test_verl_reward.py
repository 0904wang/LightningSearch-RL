import json
from numbers import Real

from lightningsearch_rl.verl_reward import compute_score


def test_compute_score_rewards_exact_answer_in_answer_tag():
    result = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<think>done</think><answer>Example City</answer>",
        ground_truth="example city",
        extra_info={"search_count": 1},
    )

    assert result["score"] == 1.07
    assert result["answer_reward"] == 1.0
    assert result["format_reward"] == 1.0
    assert result["search_cost"] == 0.03


def test_compute_score_rewards_soft_answer_matches():
    award = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<answer>Golden Quill Award</answer>",
        ground_truth="Golden Quill",
        extra_info={"search_count": 0},
    )
    venue = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<answer>Vienna Conference Center</answer>",
        ground_truth="Vienna",
        extra_info={"search_count": 0},
    )

    assert award["score"] == 0.9
    assert award["answer_reward"] == 0.8
    assert award["answer_token_f1"] == 0.8
    assert "answer_reward_type" not in award
    assert venue["score"] == 0.6
    assert venue["answer_reward"] == 0.5
    assert "answer_reward_type" not in venue


def test_compute_score_can_lower_token_f1_threshold_from_env(monkeypatch):
    baseline = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<answer>Global Health Research</answer>",
        ground_truth="Global Health Institute",
        extra_info={"reward_stage": "answer", "search_count": 0},
    )
    monkeypatch.setenv("LIGHTNINGSEARCH_ANSWER_TOKEN_F1_THRESHOLD", "0.5")

    shaped = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<answer>Global Health Research</answer>",
        ground_truth="Global Health Institute",
        extra_info={"reward_stage": "answer", "search_count": 0},
    )

    assert baseline["answer_token_f1"] == 0.666667
    assert baseline["answer_reward"] == 0.0
    assert baseline["score"] == 0.1
    assert shaped["answer_token_f1"] == 0.666667
    assert shaped["answer_reward"] == 0.666667
    assert shaped["score"] == 0.766667


def test_compute_score_returns_consistent_numeric_verl_extra_info_keys():
    answer_result = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<answer>Golden Quill Award</answer>",
        ground_truth="Golden Quill",
        extra_info={"reward_stage": "answer", "search_count": 0},
    )
    search_result = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<search>Golden Quill award organization</search>",
        ground_truth="",
        extra_info={"reward_stage": "search", "search_count": 1},
    )

    assert set(answer_result) == set(search_result)
    assert "answer_reward_type" not in answer_result
    assert all(isinstance(value, Real) for value in answer_result.values())
    assert all(isinstance(value, Real) for value in search_result.values())
    assert search_result["answer_exact_match"] == 0.0
    assert search_result["answer_token_f1"] == 0.0
    assert search_result["answer_containment_match"] == 0.0


def test_compute_score_handles_unknown_data_source_as_zero():
    result = compute_score(
        data_source="other",
        solution_str="<answer>Example City</answer>",
        ground_truth="Example City",
        extra_info={},
    )

    assert result["score"] == 0.0


def test_compute_score_does_not_reward_empty_ground_truth():
    result = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<think>no answer tag</think>",
        ground_truth="",
        extra_info={"search_count": 1},
    )

    assert result["answer_reward"] == 0.0
    assert result["format_reward"] == 0.0
    assert result["score"] == -0.03


def test_compute_score_rewards_valid_search_stage_action():
    result = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<think>I should search.</think><search>Example City birthplace</search>",
        ground_truth="",
        extra_info={"reward_stage": "search", "search_count": 1},
    )

    assert result["score"] == 1.07
    assert result["search_reward"] == 1.0
    assert result["answer_reward"] == 0.0
    assert result["format_reward"] == 1.0
    assert result["search_cost"] == 0.03


def test_compute_score_scores_search_query_by_candidate_evidence_rank():
    extra_info = {
        "reward_stage": "search",
        "search_count": 1,
        "search_reward_top_k": 8,
        "gold_doc_ids": ["doc_bridge", "doc_answer"],
        "candidate_passages": [
            {
                "doc_id": "doc_answer",
                "title": "Global Health Initiative",
                "text": "The Global Health Initiative won the Nobel Peace Prize in 2021.",
            },
            {
                "doc_id": "doc_bridge",
                "title": "Dr. Elena Voss",
                "text": "Dr. Elena Voss founded the Global Health Initiative in 2012.",
            },
            {
                "doc_id": "doc_noise",
                "title": "Archive Note",
                "text": "The archive note discusses unrelated materials.",
            },
        ],
    }

    good = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<search>Global Health Initiative Nobel Peace Prize Elena Voss</search>",
        ground_truth="",
        extra_info=extra_info,
    )
    bad = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<search>archive unrelated materials</search>",
        ground_truth="",
        extra_info=extra_info,
    )

    assert good["evidence_rank_reward"] == 0.9
    assert good["search_reward"] == 0.9
    assert good["score"] == 0.97
    assert good["retrieved_gold_count"] == 2.0
    assert good["gold_top_rank"] == 1.0
    assert bad["evidence_rank_reward"] == 0.0
    assert bad["search_reward"] == 0.0
    assert bad["score"] == 0.07
    assert bad["retrieved_gold_count"] == 0.0


def test_compute_score_rejects_answer_when_search_stage_expected():
    result = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<answer>Example City</answer>",
        ground_truth="Example City",
        extra_info={"reward_stage": "search", "search_count": 1},
    )

    assert result["score"] == -0.03
    assert result["search_reward"] == 0.0
    assert result["answer_reward"] == 0.0
    assert result["format_reward"] == 0.0


def test_compute_score_optionally_dumps_reward_components(monkeypatch, tmp_path):
    dump_path = tmp_path / "reward_dump.jsonl"
    monkeypatch.setenv("LIGHTNINGSEARCH_REWARD_DUMP_PATH", str(dump_path))
    monkeypatch.setenv("LIGHTNINGSEARCH_REWARD_DUMP_MAX_CHARS", "32")

    answer_score = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<think>long reasoning text</think><answer>Example City</answer>",
        ground_truth="Example City",
        extra_info={
            "id": "env-0:1:answer",
            "source_id": "env-0",
            "reward_stage": "answer",
            "search_count": 0,
            "precomputed_step_reward": 1.1,
        },
    )
    search_score = compute_score(
        data_source="lightningsearch_rl",
        solution_str="<think>look it up</think><search>Example City relation</search>",
        ground_truth="",
        extra_info={
            "id": "env-1:0:search",
            "source_id": "env-1",
            "reward_stage": "search",
            "search_count": 1,
            "precomputed_step_reward": 0.27,
        },
    )

    rows = [json.loads(line) for line in dump_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["score"] == answer_score["score"]
    assert rows[0]["reward_stage"] == "answer"
    assert rows[0]["answer_reward"] == 1.0
    assert rows[0]["answer_reward_type"] == "exact"
    assert rows[0]["answer_token_f1"] == 1.0
    assert rows[0]["answer_exact_match"] == 1.0
    assert rows[0]["answer_containment_match"] == 1.0
    assert rows[0]["search_reward"] == 0.0
    assert rows[0]["format_reward"] == 1.0
    assert rows[0]["parsed_action"]["type"] == "answer"
    assert rows[0]["parsed_action"]["valid"] is True
    assert rows[0]["extra_info"]["id"] == "env-0:1:answer"
    assert rows[0]["extra_info"]["source_id"] == "env-0"
    assert len(rows[0]["solution_preview"]) <= 32
    assert rows[1]["score"] == search_score["score"]
    assert rows[1]["reward_stage"] == "search"
    assert rows[1]["search_reward"] == 1.0
    assert rows[1]["parsed_action"]["type"] == "search"
    assert rows[1]["search_cost"] == 0.03
