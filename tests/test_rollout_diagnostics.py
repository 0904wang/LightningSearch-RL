import json

from lightningsearch_rl.cli import main
from lightningsearch_rl.rollout_diagnostics import diagnose_rollout_answers


def test_diagnose_rollout_answers_reports_f1_and_suspicious_rows(tmp_path):
    rollouts = tmp_path / "env_rollouts.jsonl"
    rows = [
        {
            "id": "suffix",
            "question": "Which award was established by the organization?",
            "final_answer": "Golden Quill Award",
            "gold_answer": "Golden Quill",
            "answer_exact_match": False,
            "observation": "<observation>\n[1] Chen Foundation: The Chen Foundation established the Golden Quill Award.\n</observation>",
        },
        {
            "id": "bad_gold",
            "question": "Which research institute founded in 2021 is led by Dr. Elena Vasquez?",
            "final_answer": "Global Health Research Institute",
            "gold_answer": "Barcelona",
            "answer_exact_match": False,
            "observation": (
                "<observation>\n"
                "[1] Dr. Elena Vasquez: Dr. Elena Vasquez leads the Global Health Research Institute.\n"
                "[2] Global Health Research Institute: The Global Health Research Institute was founded in Barcelona in 2021.\n"
                "</observation>"
            ),
        },
        {
            "id": "exact",
            "question": "Which city is home to the institute?",
            "final_answer": "Oakridge",
            "gold_answer": "Oakridge",
            "answer_exact_match": True,
            "observation": "<observation>\n[1] Institute: The institute is located in Oakridge.\n</observation>",
        },
    ]
    rollouts.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = diagnose_rollout_answers(rollouts)

    assert report["example_count"] == 3
    assert report["answer_exact_match_rate"] == 0.333333
    assert report["answer_containment_match_rate"] == 0.666667
    assert round(report["answer_token_f1"], 6) == 0.6
    assert report["suspicious_count"] == 1
    assert report["suspicious_adjusted_example_count"] == 2
    assert report["suspicious_adjusted_exact_match_rate"] == 0.5
    assert report["suspicious_rows"][0]["id"] == "bad_gold"
    assert "prediction_matches_observation_title" in report["suspicious_rows"][0]["reasons"]
    assert "question_gold_type_mismatch" in report["suspicious_rows"][0]["reasons"]


def test_diagnose_rollout_answers_cli_writes_report(tmp_path):
    rollouts = tmp_path / "env_rollouts.jsonl"
    out = tmp_path / "diagnostics.json"
    rollouts.write_text(
        json.dumps(
            {
                "id": "r1",
                "question": "Which award was established by the organization?",
                "final_answer": "Golden Quill Award",
                "gold_answer": "Golden Quill",
                "answer_exact_match": False,
                "observation": "<observation>\n[1] Foundation: It established the Golden Quill Award.\n</observation>",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["diagnose-rollout-answers", "--rollouts", str(rollouts), "--out", str(out)]) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["example_count"] == 1
    assert report["answer_containment_match_rate"] == 1.0
