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
    assert rollout["metadata"]["gold_doc_ids"] == ["hotpot::hp_mixed_1::Alice Smith::0"]
    assert rollout["metadata"]["retrieved_doc_ids"] == [
        "hotpot::hp_mixed_1::Alice Smith::0",
        "hotpot::hp_mixed_1::Example Book::0",
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


def test_export_grpo_keeps_gold_answer_when_retrieval_misses(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "grpo"
    examples.write_text(
        json.dumps(
            {
                "id": "miss-1",
                "question": "Where was Alice Smith born?",
                "answers": ["Example City"],
                "gold_doc_ids": ["doc_answer"],
                "corpus": [
                    {
                        "doc_id": "doc_answer",
                        "title": "Alice Smith",
                        "text": "Alice Smith was born in Example City.",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus.write_text(
        json.dumps(
            {
                "doc_id": "doc_noise",
                "title": "Noise",
                "text": "This passage does not contain the answer.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    save_lexical_index(index, load_corpus_jsonl(corpus))

    export_grpo(examples, index, out_dir, top_k=1)

    rollout = json.loads((out_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "<answer></answer>" in rollout["response"]
    assert rollout["metadata"]["answer"] == "Example City"
