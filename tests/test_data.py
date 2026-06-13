from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples


def test_load_jsonl_examples_parses_fixture():
    examples = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))

    assert len(examples) == 1
    example = examples[0]
    assert example.id == "ex1"
    assert example.answers == ["Example City"]
    assert example.gold_doc_ids == ["doc_author"]
    assert example.corpus[0].doc_id == "doc_book"


def test_load_jsonl_examples_supports_shared_corpus_doc_ids(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        '{"id":"ex2","question":"Q?","answers":["A"],"gold_doc_ids":["doc1"],"corpus_doc_ids":["doc1","doc2"]}\n',
        encoding="utf-8",
    )

    example = load_jsonl_examples(path)[0]

    assert example.corpus == []
    assert example.corpus_doc_ids == ["doc1", "doc2"]
