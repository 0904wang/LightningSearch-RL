import json

from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.environment_rollout import run_environment_rollout
from lightningsearch_rl.index_store import save_lexical_index


def _write_env_rollout_fixture(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    index = tmp_path / "index.json"
    sft = tmp_path / "sft_turns.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "doc_voss",
                        "title": "Dr. Elena Voss",
                        "text": "Dr. Elena Voss founded the Global Health Initiative in 2012.",
                    }
                ),
                json.dumps(
                    {
                        "doc_id": "doc_ghi",
                        "title": "Global Health Initiative",
                        "text": "The Global Health Initiative won the Nobel Peace Prize in 2021.",
                    }
                ),
                json.dumps(
                    {
                        "doc_id": "doc_distractor",
                        "title": "Archive Note",
                        "text": "The archive note discusses unrelated materials.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    save_lexical_index(index, load_corpus_jsonl(corpus))
    sft.write_text(
        json.dumps(
            {
                "id": "env-1",
                "messages": [
                    {
                        "role": "system",
                        "content": "Output exactly one action per turn.",
                    },
                    {
                        "role": "user",
                        "content": "Which award did the organization founded by Dr. Elena Voss receive in 2021?",
                    },
                    {
                        "role": "assistant",
                        "content": "<search>Which award did the organization founded by Dr. Elena Voss receive in 2021?</search>",
                    },
                    {
                        "role": "user",
                        "content": "<observation>\n[1] gold evidence\n</observation>",
                    },
                    {"role": "assistant", "content": "<answer>Nobel Peace Prize</answer>"},
                ],
                "metadata": {
                    "answer": "Nobel Peace Prize",
                    "gold_evidence_doc_ids": ["doc_ghi", "doc_voss"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sft, index


def test_environment_rollout_executes_generated_search_and_inserts_observation(tmp_path):
    sft, index = _write_env_rollout_fixture(tmp_path)
    out_dir = tmp_path / "env-rollout"
    seen_messages = []

    def fake_generate(messages, max_new_tokens):
        seen_messages.append(messages)
        if len(messages) == 2:
            return "<search>Global Health Initiative award 2021</search>"
        assert messages[-1]["role"] == "user"
        assert "Global Health Initiative won the Nobel Peace Prize in 2021" in messages[-1]["content"]
        return "<answer>Nobel Peace Prize</answer>"

    summary = run_environment_rollout(
        sft_path=sft,
        index_path=index,
        model_path="unused-model",
        out_dir=out_dir,
        generator=fake_generate,
        offset=0,
        limit=1,
        top_k=2,
        max_new_tokens=64,
    )

    rows = [json.loads(line) for line in (out_dir / "env_rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert len(seen_messages) == 2
    assert summary["example_count"] == 1
    assert summary["valid_search_action_rate"] == 1.0
    assert summary["valid_answer_action_rate"] == 1.0
    assert summary["answer_exact_match_rate"] == 1.0
    assert summary["answer_token_f1"] == 1.0
    assert summary["answer_containment_match_rate"] == 1.0
    assert summary["gold_evidence_recall"] == 1.0
    assert summary["all_gold_evidence_retrieved_rate"] == 1.0
    assert rows[0]["search_generated"] == "<search>Global Health Initiative award 2021</search>"
    assert rows[0]["answer_generated"] == "<answer>Nobel Peace Prize</answer>"
    assert rows[0]["observation_doc_ids"] == ["doc_ghi", "doc_voss"]
    assert [passage["doc_id"] for passage in rows[0]["candidate_passages"]] == [
        "doc_voss",
        "doc_ghi",
        "doc_distractor",
    ]
    assert rows[0]["candidate_passages"][0]["title"] == "Dr. Elena Voss"
    assert rows[0]["final_answer"] == "Nobel Peace Prize"
    assert rows[0]["answer_token_f1"] == 1.0
    assert rows[0]["answer_containment_match"] is True
    assert rows[0]["answer_messages"][-1]["content"].startswith("<observation>")


def test_environment_rollout_passes_sampling_options_to_transformers_generator(tmp_path, monkeypatch):
    import lightningsearch_rl.environment_rollout as environment_rollout

    sft, index = _write_env_rollout_fixture(tmp_path)
    created = {}

    class FakeGenerator:
        def __init__(self, model_path, *, do_sample, temperature, top_p, top_k, seed):
            created.update(
                {
                    "model_path": model_path,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "seed": seed,
                }
            )

        def __call__(self, messages, max_new_tokens):
            if len(messages) == 2:
                return "<search>Global Health Initiative award 2021</search>"
            return "<answer>Nobel Peace Prize</answer>"

    monkeypatch.setattr(environment_rollout, "TransformersChatGenerator", FakeGenerator)

    summary = run_environment_rollout(
        sft_path=sft,
        index_path=index,
        model_path="sampling-model",
        out_dir=tmp_path / "env-rollout-sampled",
        offset=0,
        limit=1,
        top_k=2,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        sample_top_k=40,
        seed=123,
    )

    assert created == {
        "model_path": "sampling-model",
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "seed": 123,
    }
    assert summary["do_sample"] is True
    assert summary["temperature"] == 0.7
    assert summary["top_p"] == 0.9
    assert summary["sample_top_k"] == 40
    assert summary["seed"] == 123


def test_environment_rollout_dry_run_writes_search_and_retrieved_answer_prompts(tmp_path):
    sft, index = _write_env_rollout_fixture(tmp_path)
    out_dir = tmp_path / "env-rollout-dry-run"

    summary = run_environment_rollout(
        sft_path=sft,
        index_path=index,
        model_path="unused-model",
        out_dir=out_dir,
        offset=0,
        limit=1,
        top_k=2,
        max_new_tokens=64,
        dry_run=True,
    )

    search_prompt = json.loads((out_dir / "search_prompts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    answer_prompt = json.loads((out_dir / "answer_prompts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["dry_run"] is True
    assert summary["search_prompt_count"] == 1
    assert summary["answer_prompt_count"] == 1
    assert summary["gold_evidence_recall"] == 1.0
    assert summary["all_gold_evidence_retrieved_rate"] == 1.0
    assert [message["role"] for message in search_prompt["messages"]] == ["system", "user"]
    assert [message["role"] for message in answer_prompt["messages"]] == ["system", "user", "assistant", "user"]
    assert "Global Health Initiative won the Nobel Peace Prize in 2021" in answer_prompt["messages"][-1]["content"]


def test_environment_rollout_gold_distractor_pool_restricts_candidates(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    index = tmp_path / "index.json"
    sft = tmp_path / "sft_turns.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "doc_id": "gold_bridge",
                    "title": "The Quantum Horizon",
                    "text": "The Quantum Horizon is a science fiction novel by author Elena Voss.",
                },
                {
                    "doc_id": "gold_answer",
                    "title": "Stellar Award",
                    "text": "The Stellar Award is presented annually by the Global Science Foundation.",
                },
                {
                    "doc_id": "distractor_1",
                    "title": "Knight Award",
                    "text": "The Knight Award is presented by the National Engineering Society.",
                },
                {
                    "doc_id": "distractor_2",
                    "title": "Archive",
                    "text": "The archive contains unrelated letters.",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    save_lexical_index(index, load_corpus_jsonl(corpus))
    sft.write_text(
        json.dumps(
            {
                "id": "pool-1",
                "messages": [
                    {"role": "system", "content": "Output one action."},
                    {
                        "role": "user",
                        "content": "Which organization presented the award won by the author of The Quantum Horizon?",
                    },
                    {
                        "role": "assistant",
                        "content": "<search>Which organization presented the award won by the author of The Quantum Horizon?</search>",
                    },
                    {"role": "user", "content": "<observation>\n[1] gold\n</observation>"},
                    {"role": "assistant", "content": "<answer>Global Science Foundation</answer>"},
                ],
                "metadata": {
                    "answer": "Global Science Foundation",
                    "gold_evidence_doc_ids": ["gold_bridge", "gold_answer"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_environment_rollout(
        sft_path=sft,
        index_path=index,
        model_path="unused-model",
        out_dir=tmp_path / "candidate-pool",
        offset=0,
        limit=1,
        top_k=3,
        max_new_tokens=64,
        dry_run=True,
        candidate_pool="gold-distractors",
        distractor_count=1,
    )

    answer_prompt = json.loads(
        ((tmp_path / "candidate-pool" / "answer_prompts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    )
    assert summary["candidate_pool"] == "gold-distractors"
    assert summary["distractor_count"] == 1
    assert summary["avg_candidate_doc_count"] == 3.0
    assert answer_prompt["candidate_doc_ids"] == ["gold_bridge", "gold_answer", "distractor_1"]
    assert [passage["doc_id"] for passage in answer_prompt["candidate_passages"]] == [
        "gold_bridge",
        "gold_answer",
        "distractor_1",
    ]
    assert set(answer_prompt["observation_doc_ids"]) == {"gold_bridge", "gold_answer", "distractor_1"}
    assert answer_prompt["gold_evidence_recall"] == 1.0
