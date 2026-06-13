import json

from lightningsearch_rl.synthesis import (
    DeepSeekClient,
    build_synthesis_prompt,
    synthesize_file,
    synthesize_validated_file,
    validate_synthetic_file,
    validate_synthetic_row,
)


def _valid_row(row_id: str = "syn-000001") -> dict:
    return {
        "id": row_id,
        "question": "Which city hosts the archive associated with the journal founded by Ada Example?",
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


def _strict_valid_row(row_id: str = "syn-000001") -> dict:
    row = _valid_row(row_id)
    row["context"][1][1][0] = (
        "Journal of Synthetic Methods is stored in the Northern Archive in Example City."
    )
    row["chain_schema"] = {
        "hop1_title": "Ada Example",
        "hop1_sentence_index": 0,
        "intermediate_entity": "Journal of Synthetic Methods",
        "hop2_title": "Journal of Synthetic Methods",
        "hop2_sentence_index": 0,
        "answer_type": "city",
        "final_answer": "Example City",
    }
    return row


def test_validate_synthetic_row_accepts_multihop_answer_in_gold_evidence():
    result = validate_synthetic_row(_valid_row())

    assert result.valid is True
    assert result.reason is None


def test_validate_synthetic_row_accepts_chain_schema_when_required():
    result = validate_synthetic_row(_strict_valid_row(), require_chain_schema=True)

    assert result.valid is True
    assert result.reason is None


def test_validate_synthetic_row_rejects_missing_chain_schema_when_required():
    result = validate_synthetic_row(_valid_row(), require_chain_schema=True)

    assert result.valid is False
    assert result.reason == "missing chain_schema"


def test_validate_synthetic_row_rejects_final_answer_leak_in_hop1():
    row = _strict_valid_row()
    row["context"][0][1][0] = (
        "Ada Example founded the Journal of Synthetic Methods in Example City."
    )

    result = validate_synthetic_row(row, require_chain_schema=True)

    assert result.valid is False
    assert result.reason == "final answer leaks in hop1"


def test_validate_synthetic_row_rejects_missing_intermediate_in_hop2():
    row = _strict_valid_row()
    row["context"][1][1][0] = "The Northern Archive is located in Example City."

    result = validate_synthetic_row(row, require_chain_schema=True)

    assert result.valid is False
    assert result.reason == "intermediate entity missing from hop2"


def test_validate_synthetic_row_rejects_chain_schema_supporting_fact_mismatch():
    row = _strict_valid_row()
    row["supporting_facts"] = [["Ada Example", 0], ["Distractor", 0]]

    result = validate_synthetic_row(row, require_chain_schema=True)

    assert result.valid is False
    assert result.reason == "chain_schema does not match supporting_facts"


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


def test_validate_synthetic_row_rejects_answer_in_question():
    row = _valid_row()
    row["question"] = "Which archive is in Example City?"

    result = validate_synthetic_row(row)

    assert result.valid is False
    assert result.reason == "answer appears in question"


def test_validate_synthetic_row_rejects_answer_equal_to_context_title():
    row = _valid_row()
    row["context"].append(["Example City", ["Example City is a historical city."]])

    result = validate_synthetic_row(row)

    assert result.valid is False
    assert result.reason == "answer equals context title"


def test_validate_synthetic_row_rejects_answer_in_multiple_supporting_sentences():
    row = _valid_row()
    row["context"][0][1][0] = (
        "Ada Example founded the Journal of Synthetic Methods while living in Example City."
    )

    result = validate_synthetic_row(row)

    assert result.valid is False
    assert result.reason == "answer appears in multiple supporting sentences"


def test_validate_synthetic_row_rejects_non_ascii_text():
    row = _valid_row()
    row["answer"] = "Chlo谷 Zhao"

    result = validate_synthetic_row(row)

    assert result.valid is False
    assert result.reason == "non-ascii text detected"


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


def test_deepseek_client_strips_key_newlines():
    client = DeepSeekClient(api_key="dummy-key\r\n")

    assert client.api_key == "dummy-key"


def test_synthesize_file_redacts_api_key_from_failure_summary(tmp_path):
    class FailingClient:
        def complete_json(self, messages, temperature, max_tokens):
            raise RuntimeError("Invalid header value b'Bearer secret-value-that-must-not-leak\\r'")

    out_path = tmp_path / "synthetic_raw.jsonl"

    summary = synthesize_file(
        out_path,
        count=1,
        topics=["research"],
        client=FailingClient(),
        concurrency=1,
        retries=1,
    )

    assert summary["failed"] == 1
    assert not out_path.exists()
    reason = summary["failures"][0]["reason"]
    assert "secret-value-that-must-not-leak" not in reason
    assert "[REDACTED]" in reason


def test_build_synthesis_prompt_emphasizes_distinct_titles_and_verbatim_answer():
    user_prompt = build_synthesis_prompt("syn-000001", "research")[-1]["content"].lower()

    assert "exactly two supporting_facts" in user_prompt
    assert "two different titles" in user_prompt
    assert "chain_schema" in user_prompt
    assert "intermediate_entity" in user_prompt
    assert "answer must appear verbatim" in user_prompt
    assert "answer must not appear in the question" in user_prompt
    assert "answer must not equal any context title" in user_prompt
    assert "ascii-only english" in user_prompt


def test_validate_synthetic_file_can_require_chain_schema(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    rejects_path = tmp_path / "rejects.jsonl"
    raw_path.write_text(
        "\n".join(
            [
                json.dumps(_strict_valid_row("syn-good"), ensure_ascii=False),
                json.dumps(_valid_row("syn-bad"), ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_synthetic_file(
        raw_path,
        valid_path,
        rejects_path,
        require_chain_schema=True,
    )

    assert summary == {"raw_count": 2, "valid_count": 1, "reject_count": 1}
    reject = json.loads(rejects_path.read_text(encoding="utf-8"))
    assert reject["id"] == "syn-bad"
    assert reject["reason"] == "missing chain_schema"


def test_synthesize_validated_file_retries_until_target_valid(tmp_path):
    invalid_same_title = _valid_row("ignored")
    invalid_same_title["supporting_facts"] = [["Ada Example", 0], ["Ada Example", 0]]
    invalid_missing_answer = _valid_row("ignored")
    invalid_missing_answer["answer"] = "Missing Answer"

    class MixedClient:
        def __init__(self):
            self.rows = [
                invalid_same_title,
                _valid_row("ignored"),
                invalid_missing_answer,
                _valid_row("ignored"),
            ]

        def complete_json(self, messages, temperature, max_tokens):
            return self.rows.pop(0)

    raw = tmp_path / "raw.jsonl"
    valid = tmp_path / "valid.jsonl"
    rejects = tmp_path / "rejects.jsonl"

    summary = synthesize_validated_file(
        raw,
        valid,
        rejects,
        target_valid=2,
        topics=["research"],
        client=MixedClient(),
        concurrency=1,
        seed=20,
        batch_size=1,
        max_attempts=5,
    )

    assert summary["requested"] == 4
    assert summary["generated"] == 4
    assert summary["valid_count"] == 2
    assert summary["reject_count"] == 2
    assert summary["api_failed"] == 0
    assert summary["stopped_reason"] == "target_valid_reached"
    assert [json.loads(line)["id"] for line in raw.read_text(encoding="utf-8").splitlines()] == [
        "syn-000020",
        "syn-000021",
        "syn-000022",
        "syn-000023",
    ]
    assert len(valid.read_text(encoding="utf-8").splitlines()) == 2
    assert len(rejects.read_text(encoding="utf-8").splitlines()) == 2


def test_synthesize_validated_file_can_require_chain_schema(tmp_path):
    class StrictMixedClient:
        def __init__(self):
            self.rows = [_valid_row("ignored"), _strict_valid_row("ignored")]

        def complete_json(self, messages, temperature, max_tokens):
            return self.rows.pop(0)

    raw = tmp_path / "raw.jsonl"
    valid = tmp_path / "valid.jsonl"
    rejects = tmp_path / "rejects.jsonl"

    summary = synthesize_validated_file(
        raw,
        valid,
        rejects,
        target_valid=1,
        topics=["research"],
        client=StrictMixedClient(),
        concurrency=1,
        seed=60,
        batch_size=1,
        max_attempts=2,
        require_chain_schema=True,
    )

    assert summary["requested"] == 2
    assert summary["valid_count"] == 1
    assert summary["reject_count"] == 1
    reject = json.loads(rejects.read_text(encoding="utf-8"))
    assert reject["reason"] == "missing chain_schema"


def test_synthesize_validated_file_stops_at_max_attempts(tmp_path):
    invalid = _valid_row("ignored")
    invalid["answer"] = "Missing Answer"

    class InvalidClient:
        def complete_json(self, messages, temperature, max_tokens):
            return invalid

    summary = synthesize_validated_file(
        tmp_path / "raw.jsonl",
        tmp_path / "valid.jsonl",
        tmp_path / "rejects.jsonl",
        target_valid=2,
        topics=["research"],
        client=InvalidClient(),
        concurrency=1,
        seed=30,
        batch_size=2,
        max_attempts=3,
    )

    assert summary["requested"] == 3
    assert summary["valid_count"] == 0
    assert summary["reject_count"] == 3
    assert summary["stopped_reason"] == "max_attempts_reached"
