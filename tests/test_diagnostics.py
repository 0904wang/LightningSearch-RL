import json

from lightningsearch_rl.diagnostics import diagnose_dataset
from lightningsearch_rl.cli import main


def _row(row_id, question, answer, context, supporting_facts, answer_type="city"):
    return {
        "id": row_id,
        "question": question,
        "answer": answer,
        "context": context,
        "supporting_facts": supporting_facts,
        "chain_schema": {
            "hop1_title": supporting_facts[0][0],
            "hop1_sentence_index": supporting_facts[0][1],
            "intermediate_entity": supporting_facts[1][0],
            "hop2_title": supporting_facts[1][0],
            "hop2_sentence_index": supporting_facts[1][1],
            "answer_type": answer_type,
            "final_answer": answer,
        },
    }


def test_diagnose_dataset_reports_quality_distribution_and_duplicates(tmp_path):
    valid = tmp_path / "valid.jsonl"
    grpo_dir = tmp_path / "grpo"
    grpo_dir.mkdir()
    row1 = _row(
        "a",
        "Which city hosts the archive associated with Ada?",
        "Bluehaven",
        [
            ["Ada", ["Ada founded the Center for Applied Optics in 2015."]],
            ["Center for Applied Optics", ["The Center for Applied Optics is located in Bluehaven."]],
        ],
        [["Ada", 0], ["Center for Applied Optics", 0]],
    )
    row2 = _row(
        "b",
        "Which award honored the institute linked to Ben?",
        "North Prize",
        [
            ["Ben", ["Ben advised the Archive Lab in 2018."]],
            ["Archive Lab", ["The Archive Lab received the North Prize."]],
        ],
        [["Ben", 0], ["Archive Lab", 0]],
        answer_type="award",
    )
    valid.write_text("\n".join(json.dumps(row) for row in [row1, row2]) + "\n", encoding="utf-8")
    (grpo_dir / "reward_records.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"question_id": "a", "reward": {"total": 1.0}}),
                json.dumps({"question_id": "b", "reward": {"total": 0.5}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (grpo_dir / "summary.json").write_text(
        json.dumps({"example_count": 2, "transition_count": 4, "avg_reward": 0.75}),
        encoding="utf-8",
    )

    report = diagnose_dataset(valid, grpo_dir)

    assert report["row_count"] == 2
    assert report["answer_type_counts"] == {"city": 1, "award": 1}
    assert report["quality"]["answer_equals_context_title"] == 0
    assert report["quality"]["answer_in_question"] == 0
    assert report["quality"]["answer_support_sentence_hits"] == {"1": 2}
    assert report["duplicates"]["duplicate_questions"] == 0
    assert report["reward"]["count"] == 2
    assert report["reward"]["avg"] == 0.75
    assert report["grpo_summary"]["transition_count"] == 4


def test_diagnose_dataset_cli_writes_report(tmp_path):
    valid = tmp_path / "valid.jsonl"
    grpo_dir = tmp_path / "grpo"
    out = tmp_path / "diagnostics.json"
    grpo_dir.mkdir()
    row = _row(
        "a",
        "Which city hosts the archive associated with Ada?",
        "Bluehaven",
        [
            ["Ada", ["Ada founded the Center for Applied Optics in 2015."]],
            ["Center for Applied Optics", ["The Center for Applied Optics is located in Bluehaven."]],
        ],
        [["Ada", 0], ["Center for Applied Optics", 0]],
    )
    valid.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (grpo_dir / "reward_records.jsonl").write_text(
        json.dumps({"question_id": "a", "reward": {"total": 1.0}}) + "\n",
        encoding="utf-8",
    )
    (grpo_dir / "summary.json").write_text(json.dumps({"example_count": 1}), encoding="utf-8")

    assert main(["diagnose-data", "--valid", str(valid), "--grpo-dir", str(grpo_dir), "--out", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["row_count"] == 1
    assert payload["reward"]["avg"] == 1.0
