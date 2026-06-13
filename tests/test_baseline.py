import json
from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.baseline import run_retrieval_baseline
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.index_store import save_lexical_index


def test_run_retrieval_baseline_writes_report(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    report = tmp_path / "report.json"
    convert_hotpot_file(Path("tests/fixtures/hotpot_mixed_raw.jsonl"), corpus, examples, limit=1)
    save_lexical_index(index, load_corpus_jsonl(corpus))

    run_retrieval_baseline("hotpot", examples, index, report, top_k=2)

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["dataset"] == "hotpot"
    assert payload["top_k"] == 2
    assert payload["metrics"]["recall_at_2"] == 1.0
    assert payload["artifacts"]["examples"].endswith("examples.jsonl")
