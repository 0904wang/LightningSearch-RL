import json

from lightningsearch_rl.reward_probe import run_reward_probe


def test_reward_probe_dry_run_writes_probe_requests(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    out_dir = tmp_path / "probe"
    _write_transitions(transitions)

    summary = run_reward_probe(
        transitions_path=transitions,
        out_dir=out_dir,
        model_path="unused-model",
        limit=1,
        samples_per_prompt=3,
        dry_run=True,
    )

    requests = _read_jsonl(out_dir / "probe_requests.jsonl")
    assert summary["dry_run"] is True
    assert summary["selected_transition_count"] == 1
    assert summary["expected_reward_rows"] == 3
    assert summary["stage_counts"] == {"search": 1}
    assert requests[0]["ground_truth"] == ""
    assert requests[0]["extra_info"]["source_id"] == "syn-a"
    assert requests[0]["extra_info"]["reward_stage"] == "search"
    assert requests[0]["extra_info"]["search_reward_top_k"] == 8


def test_reward_probe_scores_stubbed_samples_and_writes_reward_dump(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    out_dir = tmp_path / "probe"
    _write_transitions(transitions)

    def generator(requests, samples_per_prompt, max_new_tokens):
        assert samples_per_prompt == 2
        assert max_new_tokens == 64
        generated = []
        for request in requests:
            stage = request["extra_info"]["reward_stage"]
            if stage == "search":
                generated.append(
                    [
                        "<search>Global Health Initiative Nobel Peace Prize Elena Voss</search>",
                        "<search>archive unrelated materials</search>",
                    ]
                )
            else:
                generated.append(["<answer>Nobel Peace Prize</answer>", "no answer tag"])
        return generated

    summary = run_reward_probe(
        transitions_path=transitions,
        out_dir=out_dir,
        model_path="unused-model",
        samples_per_prompt=2,
        generator=generator,
    )

    generations = _read_jsonl(out_dir / "generations.jsonl")
    reward_rows = _read_jsonl(out_dir / "reward_dump.jsonl")
    assert summary["dry_run"] is False
    assert summary["selected_transition_count"] == 2
    assert summary["generated_sample_count"] == 4
    assert summary["reward_dump_count"] == 4
    assert len(generations) == 4
    assert len(reward_rows) == 4

    search_scores = [
        row["score"] for row in reward_rows if row["reward_stage"] == "search"
    ]
    answer_scores = [
        row["score"] for row in reward_rows if row["reward_stage"] == "answer"
    ]
    assert search_scores == [0.97, 0.07]
    assert answer_scores == [1.1, 0.0]
    assert reward_rows[0]["extra_info"]["probe_sample_index"] == 0
    assert reward_rows[1]["extra_info"]["probe_sample_index"] == 1


def _write_transitions(path):
    rows = [
        {
            "id": "syn-a",
            "transition_id": "syn-a:0:search",
            "action_type": "search",
            "state_messages": [
                {"role": "system", "content": "Output one action."},
                {
                    "role": "user",
                    "content": "Which award did the organization founded by Dr. Elena Voss receive in 2021?",
                },
            ],
            "action": "<search>Global Health Initiative award 2021</search>",
            "reward": 0.27,
            "observation_doc_ids": ["doc_answer", "doc_bridge"],
            "gold_evidence_doc_ids": ["doc_bridge", "doc_answer"],
            "candidate_passages": _candidate_passages(),
            "metadata": {"gold_answer": "Nobel Peace Prize", "total_reward": 1.37},
        },
        {
            "id": "syn-a",
            "transition_id": "syn-a:1:answer",
            "action_type": "answer",
            "state_messages": [
                {"role": "system", "content": "Output one action."},
                {
                    "role": "user",
                    "content": "Which award did the organization founded by Dr. Elena Voss receive in 2021?",
                },
                {
                    "role": "assistant",
                    "content": "<search>Global Health Initiative award 2021</search>",
                },
                {
                    "role": "user",
                    "content": (
                        "<observation>\n"
                        "[1] Global Health Initiative: The Global Health Initiative won the Nobel Peace Prize in 2021.\n"
                        "[2] Dr. Elena Voss: Dr. Elena Voss founded the Global Health Initiative in 2012.\n"
                        "</observation>"
                    ),
                },
            ],
            "action": "<answer>Nobel Peace Prize</answer>",
            "reward": 1.1,
            "observation_doc_ids": ["doc_answer", "doc_bridge"],
            "gold_evidence_doc_ids": ["doc_bridge", "doc_answer"],
            "candidate_passages": _candidate_passages(),
            "metadata": {"gold_answer": "Nobel Peace Prize", "total_reward": 1.37},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _candidate_passages():
    return [
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
    ]


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
