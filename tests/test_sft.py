import json
from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.index_store import save_lexical_index
from lightningsearch_rl.sft import export_sft


def test_export_sft_writes_conversations_traces_and_summary(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "sft"
    convert_hotpot_file(Path("tests/fixtures/hotpot_mixed_raw.jsonl"), corpus, examples, limit=1)
    save_lexical_index(index, load_corpus_jsonl(corpus))

    summary = export_sft(examples, index, out_dir, top_k=2)

    row = json.loads((out_dir / "sft.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["sft_rows"] == 1
    assert row["messages"][0]["role"] == "system"
    assert "<answer>Example City</answer>" in row["messages"][-1]["content"]
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "summary.json").exists()
