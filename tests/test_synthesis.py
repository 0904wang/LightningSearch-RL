import json

from lightningsearch_rl.synthesis import (
    synthesize_file,
    validate_synthetic_file,
    validate_synthetic_row,
)


def _valid_row(row_id: str = "syn-000001") -> dict:
    return {
        "id": row_id,
        "question": "Which city hosts the archive that stores the journal founded by Ada Example?",
        "answer": "Example City",
        "context": [
            [
                "Ada Example",
                ["Ada Example founded the Journal of Synthetic Methods in 2010."],
            ],
            [
                "Journal of Synthetic Methods",
                ["The journal is stored in the Northern Archive in Example City."],
            ],
            ["Distractor", ["A separate archive is located in Other City."]],
        ],
        "supporting_facts": [
            ["Ada Example", 0],
            ["Journal of Synthetic Methods", 0],
        ],
    }


def test_validate_synthetic_row_accepts_multihop_answer_in_gold_evidence():
    result = validate_synthetic_row(_valid_row())

    assert result.valid is True
    assert result.reason is None


def test_validate_synthetic_row_rejects_missing_supporting_fact():
    row = _valid_row()
    row["supporting_facts"] = [["Missing Article", 0]]

    result = validate_synthetic_row(row)

    assert result.valid is False
    assert "supporting" in result.reason


def test_validate_synthetic_row_rejects_malformed_supporting_fact():
    row = _valid_row()
    row["supporting_facts"] = ["not-a-pair"]

    result = validate_synthetic_row(row)

    assert result.valid is False
    assert "supporting" in result.reason


def test_validate_synthetic_file_splits_valid_and_rejected_rows(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    rejects_path = tmp_path / "rejects.jsonl"
    invalid = _valid_row("syn-bad")
    invalid["answer"] = "Not In Evidence"
    raw_path.write_text(
        "\n".join(
            [
                json.dumps(_valid_row(), ensure_ascii=False),
                json.dumps(invalid, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_synthetic_file(raw_path, valid_path, rejects_path)

    assert summary == {"raw_count": 2, "valid_count": 1, "reject_count": 1}
    assert len(valid_path.read_text(encoding="utf-8").splitlines()) == 1
    reject = json.loads(rejects_path.read_text(encoding="utf-8"))
    assert reject["id"] == "syn-bad"
    assert "answer" in reject["reason"]


def test_synthesize_file_uses_client_and_writes_jsonl_without_api_key(tmp_path, monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def complete_json(self, messages, temperature, max_tokens):
            self.calls += 1
            assert "json" in messages[-1]["content"].lower()
            assert temperature == 0.7
            assert max_tokens == 512
            return _valid_row()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value-that-must-not-leak")
    out_path = tmp_path / "synthetic_raw.jsonl"
    client = FakeClient()

    summary = synthesize_file(
        out_path,
        count=2,
        topics=["research awards"],
        client=client,
        concurrency=2,
        seed=7,
        temperature=0.7,
        max_tokens=512,
    )

    assert summary["requested"] == 2
    assert summary["written"] == 2
    assert summary["failed"] == 0
    assert client.calls == 2
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["syn-000007", "syn-000008"]
    assert "secret-value-that-must-not-leak" not in out_path.read_text(encoding="utf-8")
