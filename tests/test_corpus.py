from lightningsearch_rl.corpus import load_corpus_jsonl, write_corpus_jsonl
from lightningsearch_rl.data import Passage


def test_write_and_load_corpus_deduplicates_doc_ids(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_corpus_jsonl(
        path,
        [
            Passage("doc1", "Title", "First text."),
            Passage("doc1", "Title Duplicate", "Second text."),
            Passage("doc2", "Other", "Other text."),
        ],
    )

    passages = load_corpus_jsonl(path)

    assert [p.doc_id for p in passages] == ["doc1", "doc2"]
    assert passages[0].title == "Title"
