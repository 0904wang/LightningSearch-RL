import json

import pytest

from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.index_store import save_lexical_index
from lightningsearch_rl.sft_warmup import export_sft_warmup


def test_export_sft_warmup_uses_gold_evidence_and_answer(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "sft-warmup"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "doc_noise",
                        "title": "Noise",
                        "text": "This unrelated passage would rank highly for retrieval.",
                    }
                ),
                json.dumps(
                    {
                        "doc_id": "doc_answer",
                        "title": "Alice Smith",
                        "text": "Alice Smith was born in Example City.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    examples.write_text(
        json.dumps(
            {
                "id": "warmup-1",
                "question": "Where was Alice Smith born?",
                "answers": ["Example City"],
                "gold_doc_ids": ["doc_answer"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    save_lexical_index(index, load_corpus_jsonl(corpus))

    summary = export_sft_warmup(examples, index, out_dir)

    row = json.loads((out_dir / "sft_warmup.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assistant = row["messages"][-1]["content"]
    assert summary["sft_rows"] == 1
    assert summary["answer_tag_rate"] == 1.0
    assert summary["non_empty_answer_rate"] == 1.0
    assert summary["gold_evidence_coverage"] == 1.0
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["content"] == "Where was Alice Smith born?"
    assert "<observation>" in assistant
    assert "Alice Smith was born in Example City." in assistant
    assert "This unrelated passage" not in assistant
    assert "<answer>Example City</answer>" in assistant
    assert row["metadata"]["answer"] == "Example City"
    assert row["metadata"]["gold_doc_ids"] == ["doc_answer"]
    assert row["metadata"]["gold_evidence_doc_ids"] == ["doc_answer"]
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "summary.json").exists()


def test_export_sft_warmup_rejects_gold_evidence_answer_mismatch(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "sft-warmup"
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
                "id": "warmup-bad",
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
        export_sft_warmup(examples, index, out_dir)
