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
        "hotpot::hp1::Example Book::0",
        "hotpot::hp1::Alice Smith::0",
        "hotpot::hp1::Noise::0",
    ]
    assert examples[0].gold_doc_ids == ["hotpot::hp1::Alice Smith::0"]
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

    assert passages[1].doc_id == "2wiki::tw1::Alice Smith::0"
    assert examples[0].gold_doc_ids == ["2wiki::tw1::Alice Smith::0"]


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
    assert examples[0].gold_doc_ids == ["hotpot::hp_mixed_1::Alice Smith::0"]
    assert [p.doc_id for p in passages] == [
        "hotpot::hp_mixed_1::Example Book::0",
        "hotpot::hp_mixed_1::Alice Smith::0",
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
    assert examples[0].gold_doc_ids == ["2wiki::tw_mixed_1::Alice Smith::0"]


def test_convert_hotpot_file_uses_row_scoped_doc_ids_for_repeated_titles(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"
    raw_path.write_text(
        "\n".join(
            [
                '{"id":"syn-a","question":"Q1?","answer":"Answer A","context":[["Shared Title",["Shared Title points to Answer A."]]],"supporting_facts":[["Shared Title",0]]}',
                '{"id":"syn-b","question":"Q2?","answer":"Answer B","context":[["Shared Title",["Shared Title points to Answer B."]]],"supporting_facts":[["Shared Title",0]]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    convert_hotpot_file(raw_path, corpus_path, examples_path)

    passages = load_corpus_jsonl(corpus_path)
    examples = load_jsonl_examples(examples_path)
    assert [passage.doc_id for passage in passages] == [
        "hotpot::syn-a::Shared Title::0",
        "hotpot::syn-b::Shared Title::0",
    ]
    assert len({passage.doc_id for passage in passages}) == 2
    assert examples[0].gold_doc_ids == ["hotpot::syn-a::Shared Title::0"]
    assert examples[1].gold_doc_ids == ["hotpot::syn-b::Shared Title::0"]
