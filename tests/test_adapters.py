from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.data import load_jsonl_examples


def test_convert_hotpot_file_writes_shared_corpus_and_examples(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"

    convert_hotpot_file(
        Path("tests/fixtures/hotpot_tiny_raw.json"),
        corpus_path,
        examples_path,
    )

    passages = load_corpus_jsonl(corpus_path)
    examples = load_jsonl_examples(examples_path)

    assert [p.doc_id for p in passages] == [
        "hotpot::Example Book::0",
        "hotpot::Alice Smith::0",
        "hotpot::Noise::0",
    ]
    assert examples[0].gold_doc_ids == ["hotpot::Alice Smith::0"]
    assert examples[0].corpus_doc_ids == [p.doc_id for p in passages]
