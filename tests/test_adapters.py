from pathlib import Path

from lightningsearch_rl.adapters import convert_2wiki_file, convert_hotpot_file
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


def test_convert_2wiki_file_writes_shared_corpus_and_examples(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"

    convert_2wiki_file(
        Path("tests/fixtures/2wiki_tiny_raw.json"),
        corpus_path,
        examples_path,
    )

    passages = load_corpus_jsonl(corpus_path)
    examples = load_jsonl_examples(examples_path)

    assert passages[1].doc_id == "2wiki::Alice Smith::0"
    assert examples[0].gold_doc_ids == ["2wiki::Alice Smith::0"]


def test_convert_hotpot_file_supports_jsonl_mixed_shapes_and_limit(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"

    convert_hotpot_file(
        Path("tests/fixtures/hotpot_mixed_raw.jsonl"),
        corpus_path,
        examples_path,
        limit=1,
    )

    passages = load_corpus_jsonl(corpus_path)
    examples = load_jsonl_examples(examples_path)

    assert len(examples) == 1
    assert examples[0].id == "hp_mixed_1"
    assert examples[0].answers == ["Example City"]
    assert examples[0].gold_doc_ids == ["hotpot::Alice Smith::0"]
    assert [p.doc_id for p in passages] == [
        "hotpot::Example Book::0",
        "hotpot::Alice Smith::0",
    ]


def test_convert_2wiki_file_supports_jsonl_mixed_shapes_and_limit(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"

    convert_2wiki_file(
        Path("tests/fixtures/2wiki_mixed_raw.jsonl"),
        corpus_path,
        examples_path,
        limit=1,
    )

    examples = load_jsonl_examples(examples_path)

    assert len(examples) == 1
    assert examples[0].id == "tw_mixed_1"
    assert examples[0].gold_doc_ids == ["2wiki::Alice Smith::0"]
