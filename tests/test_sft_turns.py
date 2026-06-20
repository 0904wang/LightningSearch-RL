import json

import pytest

from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.index_store import save_lexical_index
from lightningsearch_rl.sft_turns import export_sft_turns


def test_export_sft_turns_writes_runtime_observation_conversations(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "sft-turns"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "doc_answer",
                        "title": "Research Center",
                        "text": "The research center founded by Dr. Elena Voss was funded by the National Science Foundation.",
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    examples.write_text(
        json.dumps(
            {
                "id": "turns-1",
                "question": "Which organization funded the research center founded by Dr. Elena Voss?",
                "answers": ["National Science Foundation"],
                "gold_doc_ids": ["doc_answer"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    save_lexical_index(index, load_corpus_jsonl(corpus))

    summary = export_sft_turns(examples, index, out_dir)

    row = json.loads((out_dir / "sft_turns.jsonl").read_text(encoding="utf-8").splitlines()[0])
    messages = row["messages"]
    assistant_messages = [message["content"] for message in messages if message["role"] == "assistant"]
    user_messages = [message["content"] for message in messages if message["role"] == "user"]

    assert summary["sft_rows"] == 1
    assert summary["assistant_observation_rate"] == 0.0
    assert summary["assistant_single_action_rate"] == 1.0
    assert summary["answer_tag_rate"] == 1.0
    assert messages[0]["role"] == "system"
    assert "Never output <observation>" in messages[0]["content"]
    assert assistant_messages == [
        "<search>Which organization funded the research center founded by Dr. Elena Voss?</search>",
        "<answer>National Science Foundation</answer>",
    ]
    assert any("<observation>" in content for content in user_messages)
    assert "Research Center: The research center founded by Dr. Elena Voss was funded" in "\n".join(user_messages)
    assert all("<observation>" not in content for content in assistant_messages)
    assert row["metadata"]["answer"] == "National Science Foundation"
    assert row["metadata"]["gold_evidence_doc_ids"] == ["doc_answer"]
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "summary.json").exists()


def test_export_sft_turns_rejects_gold_evidence_answer_mismatch(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "sft-turns"
    corpus.write_text(
        json.dumps(
            {
                "doc_id": "doc_answer",
                "title": "Institute for Quantum Materials",
                "text": "The institute received a grant from the Global Science Foundation.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples.write_text(
        json.dumps(
            {
                "id": "turns-bad",
                "question": "Which organization awarded the grant?",
                "answers": ["National Science Foundation"],
                "gold_doc_ids": ["doc_answer"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    save_lexical_index(index, load_corpus_jsonl(corpus))

    with pytest.raises(ValueError, match="answer not found in gold evidence"):
        export_sft_turns(examples, index, out_dir)
