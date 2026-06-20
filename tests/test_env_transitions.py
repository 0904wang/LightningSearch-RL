import json

from lightningsearch_rl.data import Passage
from lightningsearch_rl.env_transitions import export_env_rollout_transitions
from lightningsearch_rl.index_store import save_lexical_index


def _write_env_rollouts(path):
    rows = [
        {
            "id": "env-1",
            "question": "Which award did the organization founded by Dr. Elena Voss receive in 2021?",
            "search_messages": [
                {"role": "system", "content": "Output one action."},
                {
                    "role": "user",
                    "content": "Which award did the organization founded by Dr. Elena Voss receive in 2021?",
                },
            ],
            "search_generated": "<search>Global Health Initiative award 2021</search>",
            "search_action": {
                "type": "search",
                "valid": True,
                "query": "Global Health Initiative award 2021",
                "answer": None,
                "reason": None,
            },
            "observation": (
                "<observation>\n"
                "[1] Global Health Initiative: The Global Health Initiative won the Nobel Peace Prize in 2021.\n"
                "[2] Dr. Elena Voss: Dr. Elena Voss founded the Global Health Initiative in 2012.\n"
                "</observation>"
            ),
            "observation_doc_ids": ["doc_ghi", "doc_voss"],
            "candidate_doc_ids": ["doc_ghi", "doc_voss", "doc_noise"],
            "candidate_passages": [
                {
                    "doc_id": "doc_ghi",
                    "title": "Global Health Initiative",
                    "text": "The Global Health Initiative won the Nobel Peace Prize in 2021.",
                },
                {
                    "doc_id": "doc_voss",
                    "title": "Dr. Elena Voss",
                    "text": "Dr. Elena Voss founded the Global Health Initiative in 2012.",
                },
                {
                    "doc_id": "doc_noise",
                    "title": "Archive Note",
                    "text": "The archive note discusses unrelated materials.",
                },
            ],
            "candidate_pool": "gold-distractors",
            "gold_evidence_doc_ids": ["doc_voss", "doc_ghi"],
            "gold_evidence_recall": 1.0,
            "all_gold_evidence_retrieved": True,
            "answer_messages": [
                {"role": "system", "content": "Output one action."},
                {
                    "role": "user",
                    "content": "Which award did the organization founded by Dr. Elena Voss receive in 2021?",
                },
                {"role": "assistant", "content": "<search>Global Health Initiative award 2021</search>"},
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
            "answer_generated": "<answer>Nobel Peace Prize</answer>",
            "answer_action": {
                "type": "answer",
                "valid": True,
                "query": None,
                "answer": "Nobel Peace Prize",
                "reason": None,
            },
            "final_answer": "Nobel Peace Prize",
            "gold_answer": "Nobel Peace Prize",
            "answer_exact_match": True,
            "answer_token_f1": 1.0,
            "answer_containment_match": True,
            "metadata": {"answer": "Nobel Peace Prize"},
        },
        {
            "id": "env-soft",
            "question": "Which award was established by the organization founded by Dr. Alice Chen?",
            "search_messages": [
                {"role": "system", "content": "Output one action."},
                {
                    "role": "user",
                    "content": "Which award was established by the organization founded by Dr. Alice Chen?",
                },
            ],
            "search_generated": "<search>Dr. Alice Chen founded organization award</search>",
            "search_action": {
                "type": "search",
                "valid": True,
                "query": "Dr. Alice Chen founded organization award",
                "answer": None,
                "reason": None,
            },
            "observation": (
                "<observation>\n"
                "[1] Dr. Alice Chen: Dr. Alice Chen founded the Chen Foundation in 2005.\n"
                "[2] Chen Foundation: The Chen Foundation established the Golden Quill Award in 2010.\n"
                "</observation>"
            ),
            "observation_doc_ids": ["doc_chen", "doc_award"],
            "candidate_doc_ids": ["doc_chen", "doc_award", "doc_noise"],
            "candidate_pool": "gold-distractors",
            "gold_evidence_doc_ids": ["doc_chen", "doc_award"],
            "gold_evidence_recall": 1.0,
            "all_gold_evidence_retrieved": True,
            "answer_messages": [
                {"role": "system", "content": "Output one action."},
                {
                    "role": "user",
                    "content": "Which award was established by the organization founded by Dr. Alice Chen?",
                },
                {"role": "assistant", "content": "<search>Dr. Alice Chen founded organization award</search>"},
                {
                    "role": "user",
                    "content": (
                        "<observation>\n"
                        "[1] Dr. Alice Chen: Dr. Alice Chen founded the Chen Foundation in 2005.\n"
                        "[2] Chen Foundation: The Chen Foundation established the Golden Quill Award in 2010.\n"
                        "</observation>"
                    ),
                },
            ],
            "answer_generated": "<answer>Golden Quill Award</answer>",
            "answer_action": {
                "type": "answer",
                "valid": True,
                "query": None,
                "answer": "Golden Quill Award",
                "reason": None,
            },
            "final_answer": "Golden Quill Award",
            "gold_answer": "Golden Quill",
            "answer_exact_match": False,
            "answer_token_f1": 0.8,
            "answer_containment_match": True,
            "metadata": {"answer": "Golden Quill"},
        },
        {
            "id": "env-invalid",
            "question": "Question with malformed model output?",
            "search_messages": [
                {"role": "system", "content": "Output one action."},
                {"role": "user", "content": "Question with malformed model output?"},
            ],
            "search_generated": "I will answer directly.",
            "search_action": {
                "type": "invalid",
                "valid": False,
                "query": None,
                "answer": None,
                "reason": "no_valid_action",
            },
            "observation": "<observation>\n</observation>",
            "observation_doc_ids": [],
            "candidate_doc_ids": ["doc_noise"],
            "candidate_pool": "gold-distractors",
            "gold_evidence_doc_ids": ["doc_gold"],
            "gold_evidence_recall": 0.0,
            "all_gold_evidence_retrieved": False,
            "answer_messages": [],
            "answer_generated": "",
            "answer_action": {
                "type": "invalid",
                "valid": False,
                "query": None,
                "answer": None,
                "reason": "no_valid_action",
            },
            "final_answer": "",
            "gold_answer": "Example Answer",
            "answer_exact_match": False,
            "answer_token_f1": 0.0,
            "answer_containment_match": False,
            "metadata": {"answer": "Example Answer"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_export_env_rollout_transitions_writes_step_credit_records(tmp_path):
    rollouts = tmp_path / "env_rollouts.jsonl"
    out_dir = tmp_path / "env-transitions"
    _write_env_rollouts(rollouts)

    summary = export_env_rollout_transitions(rollouts, out_dir)

    transitions = [
        json.loads(line)
        for line in (out_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rewards = [
        json.loads(line)
        for line in (out_dir / "reward_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rollouts_for_grpo = [
        json.loads(line)
        for line in (out_dir / "rollouts_for_grpo.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["example_count"] == 3
    assert summary["transition_count"] == 5
    assert summary["avg_total_reward"] == 0.846667
    assert summary["avg_search_credit"] == 0.18
    assert summary["avg_answer_credit"] == 0.666667
    assert summary["valid_search_action_rate"] == 0.666667
    assert summary["valid_answer_action_rate"] == 0.666667

    assert rewards[0]["total"] == 1.37
    assert rewards[0]["search_credit"] == 0.27
    assert rewards[0]["answer_credit"] == 1.1
    assert rewards[0]["search_cost"] == 0.03
    assert rewards[0]["answer_reward"] == 1.0
    assert rewards[0]["evidence_reward"] == 1.0
    assert rewards[0]["format_reward"] == 1.0
    assert rewards[0]["tool_validity_reward"] == 1.0

    assert transitions[0]["id"] == "env-1"
    assert transitions[0]["transition_id"] == "env-1:0:search"
    assert transitions[0]["action_type"] == "search"
    assert transitions[0]["terminal"] is False
    assert transitions[0]["reward"] == 0.27
    assert transitions[0]["state_messages"][0]["role"] == "system"
    assert transitions[0]["observation_doc_ids"] == ["doc_ghi", "doc_voss"]
    assert [passage["doc_id"] for passage in transitions[0]["candidate_passages"]] == [
        "doc_ghi",
        "doc_voss",
        "doc_noise",
    ]
    assert transitions[0]["reward_components"]["evidence_reward"] == 1.0

    assert transitions[1]["transition_id"] == "env-1:1:answer"
    assert transitions[1]["action_type"] == "answer"
    assert transitions[1]["terminal"] is True
    assert transitions[1]["reward"] == 1.1
    assert transitions[1]["state_messages"][-1]["role"] == "user"
    assert transitions[1]["reward_components"]["answer_reward"] == 1.0
    assert transitions[3]["id"] == "env-soft"
    assert transitions[3]["reward"] == 0.9
    assert transitions[3]["reward_components"]["answer_reward"] == 0.8
    assert transitions[3]["reward_components"]["answer_reward_type"] == "containment"

    assert transitions[4]["id"] == "env-invalid"
    assert transitions[4]["action_type"] == "invalid"
    assert transitions[4]["reward"] == 0.0
    assert rewards[1]["total"] == 1.17
    assert rewards[1]["answer_reward"] == 0.8
    assert rewards[1]["answer_reward_type"] == "containment"
    assert rewards[2]["total"] == 0.0

    assert rollouts_for_grpo[0]["prompt"].startswith("Which award")
    assert "<search>Global Health Initiative award 2021</search>" in rollouts_for_grpo[0]["response"]
    assert "<observation>" in rollouts_for_grpo[0]["response"]
    assert "<answer>Nobel Peace Prize</answer>" in rollouts_for_grpo[0]["response"]
    assert rollouts_for_grpo[0]["reward"] == 1.37
    assert [passage["doc_id"] for passage in rollouts_for_grpo[0]["metadata"]["candidate_passages"]] == [
        "doc_ghi",
        "doc_voss",
        "doc_noise",
    ]
    assert (out_dir / "summary.json").exists()


def test_export_env_rollout_transitions_tags_quality_manifest_rows(tmp_path):
    rollouts = tmp_path / "env_rollouts.jsonl"
    out_dir = tmp_path / "env-transitions"
    manifest = tmp_path / "quality_manifest.json"
    _write_env_rollouts(rollouts)
    manifest.write_text(
        json.dumps(
            {
                "env-soft": {
                    "flags": ["qa_type_mismatch"],
                    "notes": ["gold answer is a partial alias for the generated answer"],
                }
            }
        ),
        encoding="utf-8",
    )

    summary = export_env_rollout_transitions(
        rollouts,
        out_dir,
        quality_manifest_path=manifest,
    )

    transitions = [
        json.loads(line)
        for line in (out_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rewards = [
        json.loads(line)
        for line in (out_dir / "reward_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rollouts_for_grpo = [
        json.loads(line)
        for line in (out_dir / "rollouts_for_grpo.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    env_soft_transition = next(row for row in transitions if row["id"] == "env-soft")
    env_soft_reward = next(row for row in rewards if row["id"] == "env-soft")
    env_soft_rollout = next(row for row in rollouts_for_grpo if row["id"] == "env-soft")
    assert env_soft_transition["metadata"]["quality_flags"] == ["qa_type_mismatch"]
    assert env_soft_transition["metadata"]["quality_notes"] == [
        "gold answer is a partial alias for the generated answer"
    ]
    assert env_soft_reward["quality_flags"] == ["qa_type_mismatch"]
    assert env_soft_rollout["metadata"]["quality_flags"] == ["qa_type_mismatch"]
    assert summary["quality_flag_counts"] == {"qa_type_mismatch": 1}
    assert summary["excluded_example_count"] == 0


def test_export_env_rollout_transitions_excludes_manifest_flagged_rows(tmp_path):
    rollouts = tmp_path / "env_rollouts.jsonl"
    out_dir = tmp_path / "env-transitions"
    manifest = tmp_path / "quality_manifest.json"
    _write_env_rollouts(rollouts)
    manifest.write_text(
        json.dumps(
            {
                "env-soft": {
                    "flags": ["qa_type_mismatch"],
                    "notes": ["question asks for award but gold answer is abbreviated"],
                }
            }
        ),
        encoding="utf-8",
    )

    summary = export_env_rollout_transitions(
        rollouts,
        out_dir,
        quality_manifest_path=manifest,
        exclude_quality_flags={"qa_type_mismatch"},
    )

    transitions = [
        json.loads(line)
        for line in (out_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rewards = [
        json.loads(line)
        for line in (out_dir / "reward_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["input_example_count"] == 3
    assert summary["example_count"] == 2
    assert summary["excluded_example_count"] == 1
    assert summary["excluded_example_ids"] == ["env-soft"]
    assert summary["excluded_quality_flag_counts"] == {"qa_type_mismatch": 1}
    assert {row["id"] for row in transitions} == {"env-1", "env-invalid"}
    assert {row["id"] for row in rewards} == {"env-1", "env-invalid"}


def test_export_env_rollout_transitions_backfills_candidate_passages_from_index(tmp_path):
    rollouts = tmp_path / "env_rollouts.jsonl"
    out_dir = tmp_path / "env-transitions"
    index = tmp_path / "index.json"
    save_lexical_index(
        index,
        [
            Passage(
                doc_id="doc_answer",
                title="Answer Evidence",
                text="The answer evidence contains the final answer.",
            ),
            Passage(doc_id="doc_bridge", title="Bridge Evidence", text="The bridge evidence names the entity."),
        ],
    )
    rollouts.write_text(
        json.dumps(
            {
                "id": "env-backfill",
                "question": "Question?",
                "search_messages": [
                    {"role": "system", "content": "Output one action."},
                    {"role": "user", "content": "Question?"},
                ],
                "search_generated": "<search>answer evidence bridge</search>",
                "search_action": {
                    "type": "search",
                    "valid": True,
                    "query": "answer evidence bridge",
                    "answer": None,
                    "reason": None,
                },
                "observation": "<observation>\n[1] Answer Evidence: final answer\n</observation>",
                "observation_doc_ids": ["doc_answer"],
                "candidate_doc_ids": ["doc_bridge", "doc_answer"],
                "candidate_pool": "gold-distractors",
                "gold_evidence_doc_ids": ["doc_answer"],
                "gold_evidence_recall": 1.0,
                "all_gold_evidence_retrieved": True,
                "answer_messages": [],
                "answer_generated": "",
                "answer_action": {"type": "invalid", "valid": False, "query": None, "answer": None, "reason": None},
                "final_answer": "",
                "gold_answer": "Answer",
                "answer_exact_match": False,
                "answer_token_f1": 0.0,
                "answer_containment_match": False,
                "metadata": {"answer": "Answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = export_env_rollout_transitions(rollouts, out_dir, index_path=index)
    transition = json.loads((out_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert summary["avg_candidate_passage_count"] == 2.0
    assert [passage["doc_id"] for passage in transition["candidate_passages"]] == [
        "doc_bridge",
        "doc_answer",
    ]
    assert transition["candidate_passages"][1]["title"] == "Answer Evidence"
