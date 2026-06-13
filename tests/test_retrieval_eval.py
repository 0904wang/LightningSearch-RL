from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.retrieval import LexicalRetriever
from lightningsearch_rl.retrieval_eval import evaluate_retrieval


def test_evaluate_retrieval_reports_recall_at_k(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"
    convert_hotpot_file(Path("tests/fixtures/hotpot_tiny_raw.json"), corpus_path, examples_path)

    examples = load_jsonl_examples(examples_path)
    retriever = LexicalRetriever(load_corpus_jsonl(corpus_path))

    metrics = evaluate_retrieval(examples, retriever, top_k=2)

    assert metrics["example_count"] == 1
    assert metrics["recall_at_2"] == 1.0
    assert metrics["evidence_recall_at_2"] == 1.0
    assert metrics["avg_retrieved_docs"] == 2.0
