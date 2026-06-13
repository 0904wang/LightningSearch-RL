import json
from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.grpo import export_grpo
from lightningsearch_rl.index_store import save_lexical_index


def test_export_grpo_writes_rollouts_transitions_rewards_and_summary(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "grpo"
    convert_hotpot_file(Path("tests/fixtures/hotpot_mixed_raw.jsonl"), corpus, examples, limit=1)
    save_lexical_index(index, load_corpus_jsonl(corpus))

    summary = export_grpo(examples, index, out_dir, top_k=2)

    rollout = json.loads((out_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    reward = json.loads(
        (out_dir / "reward_records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    transition = json.loads(
        (out_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )

    assert summary["rollout_count"] == 1
    assert summary["transition_count"] == 2
    assert summary["avg_reward"] == 1.37
    assert rollout["prompt"] == "Which city is the birthplace of the author of Example Book?"
    assert "<answer>Example City</answer>" in rollout["response"]
    assert rollout["reward"] == 1.37
    assert rollout["metadata"]["answer"] == "Example City"
    assert rollout["metadata"]["search_count"] == 1
    assert rollout["metadata"]["gold_doc_ids"] == ["hotpot::Alice Smith::0"]
    assert rollout["metadata"]["retrieved_doc_ids"] == [
        "hotpot::Alice Smith::0",
        "hotpot::Example Book::0",
    ]
    assert reward["total"] == 1.37
    assert reward["answer_reward"] == 1.0
    assert reward["evidence_reward"] == 1.0
    assert reward["format_reward"] == 1.0
    assert reward["tool_validity_reward"] == 1.0
    assert reward["search_count"] == 1
    assert reward["search_cost"] == 0.03
    assert transition["terminal"] is True
    assert transition["reward"] == 1.37
    assert (out_dir / "summary.json").exists()
