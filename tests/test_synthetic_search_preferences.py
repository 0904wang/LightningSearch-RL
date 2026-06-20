import json

from lightningsearch_rl.synthetic_search_preferences import build_synthetic_search_preferences


def test_build_synthetic_search_preferences_creates_ranked_search_pairs(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    out_dir = tmp_path / "phase6c"
    _write_jsonl(
        transitions,
        [
            _search_transition(
                row_id="syn-a",
                query="Elena Voss Global Health Initiative Nobel Peace Prize",
                gold_answer="Nobel Peace Prize",
            ),
            _search_transition(
                row_id="syn-b",
                query="Mira Chen Horizon Archive Silver Compass Award",
                gold_answer="Silver Compass Award",
            ),
        ],
    )

    summary = build_synthetic_search_preferences(
        transitions_path=transitions,
        out_dir=out_dir,
        search_reward_top_k=2,
        min_chosen_score=0.8,
        min_score_gap=0.2,
        max_negatives_per_transition=3,
        val_fraction=0.25,
        seed=7,
    )

    pairs = _read_jsonl(out_dir / "pairs.jsonl")
    train = _read_jsonl(out_dir / "train.jsonl")
    val = _read_jsonl(out_dir / "val.jsonl")
    candidates = _read_jsonl(out_dir / "candidates.jsonl")
    reward_rows = _read_jsonl(out_dir / "reward_dump.jsonl")

    assert summary["input_transition_count"] == 2
    assert summary["selected_transition_count"] == 2
    assert summary["pair_count"] >= 2
    assert summary["pair_category_counts"] == {"search_vs_search": summary["pair_count"]}
    assert summary["stage_pair_counts"] == {"search": summary["pair_count"]}
    assert len(train) + len(val) == len(pairs)
    assert len(candidates) > len(pairs)
    assert len(reward_rows) == len(candidates)
    assert {row["reward_stage"] for row in pairs} == {"search"}
    assert {row["pair_category"] for row in pairs} == {"search_vs_search"}
    assert all(row["chosen_score"] >= 0.8 for row in pairs)
    assert all(row["score_gap"] >= 0.2 for row in pairs)
    assert all(row["chosen_action_type"] == "search" for row in pairs)
    assert all(row["rejected_action_type"] == "search" for row in pairs)
    assert all(row["rejected_corruption_type"] for row in pairs)
    assert any(row["rejected_corruption_type"] == "generic" for row in pairs)
    assert (out_dir / "summary.json").exists()


def test_build_synthetic_search_preferences_skips_low_scoring_chosen_queries(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    out_dir = tmp_path / "phase6c-low"
    _write_jsonl(
        transitions,
        [
            {
                **_search_transition(
                    row_id="syn-low",
                    query="unmatched query terms",
                    gold_answer="Nobel Peace Prize",
                ),
                "candidate_passages": [
                    {
                        "doc_id": "doc_voss",
                        "title": "Dr. Elena Voss",
                        "text": "Dr. Elena Voss founded the Global Health Initiative in 2012.",
                    }
                ],
                "gold_evidence_doc_ids": ["doc_voss", "doc_missing"],
            }
        ],
    )

    summary = build_synthetic_search_preferences(
        transitions_path=transitions,
        out_dir=out_dir,
        search_reward_top_k=2,
        min_chosen_score=0.8,
        val_fraction=0.0,
    )

    assert summary["pair_count"] == 0
    assert summary["skip_reason_counts"]["chosen_score_below_min"] == 1
    assert _read_jsonl(out_dir / "pairs.jsonl") == []


def _search_transition(row_id, query, gold_answer):
    if row_id == "syn-a":
        gold_docs = [
            {
                "doc_id": "doc_voss",
                "title": "Dr. Elena Voss",
                "text": "Dr. Elena Voss founded the Global Health Initiative in 2012.",
            },
            {
                "doc_id": "doc_ghi",
                "title": "Global Health Initiative",
                "text": "The Global Health Initiative won the Nobel Peace Prize in 2021.",
            },
        ]
        distractor = {
            "doc_id": "doc_archive",
            "title": "Ocean Archive",
            "text": "The Ocean Archive catalogues reef maps and harbor records.",
        }
        gold_ids = ["doc_voss", "doc_ghi"]
        question = "Which award did the organization founded by Dr. Elena Voss receive in 2021?"
    else:
        gold_docs = [
            {
                "doc_id": "doc_chen",
                "title": "Mira Chen",
                "text": "Mira Chen founded the Horizon Archive in 2016.",
            },
            {
                "doc_id": "doc_horizon",
                "title": "Horizon Archive",
                "text": "The Horizon Archive received the Silver Compass Award in 2020.",
            },
        ]
        distractor = {
            "doc_id": "doc_orbit",
            "title": "Orbit Museum",
            "text": "The Orbit Museum hosts astronomy exhibits and meteorite displays.",
        }
        gold_ids = ["doc_chen", "doc_horizon"]
        question = "Which award did the archive founded by Mira Chen receive in 2020?"
    return {
        "id": row_id,
        "transition_id": f"{row_id}:0:search",
        "step_index": 0,
        "state_messages": [
            {"role": "system", "content": "Output exactly one action."},
            {"role": "user", "content": question},
        ],
        "action": f"<search>{query}</search>",
        "action_type": "search",
        "valid_action": True,
        "query": query,
        "observation_doc_ids": gold_ids,
        "gold_evidence_doc_ids": gold_ids,
        "candidate_passages": [*gold_docs, distractor],
        "metadata": {
            "question": question,
            "gold_answer": gold_answer,
            "candidate_passages": [*gold_docs, distractor],
            "candidate_doc_ids": [doc["doc_id"] for doc in [*gold_docs, distractor]],
        },
    }


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
