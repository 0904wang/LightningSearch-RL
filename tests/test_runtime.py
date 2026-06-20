from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.retrieval import LexicalRetriever
from lightningsearch_rl.runtime import run_retrieval_episode, run_rule_based_episode


def test_rule_based_episode_collects_search_and_answer_trace():
    example = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))[0]

    trace = run_rule_based_episode(example, top_k=2)

    assert trace.question_id == "ex1"
    assert trace.final_answer == "Example City"
    assert [step.action_type for step in trace.steps] == ["search", "answer"]
    assert trace.steps[0].observation[0].doc_id == "doc_author"
    assert trace.metadata["search_count"] == 1


def test_retrieval_episode_uses_injected_shared_retriever(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    convert_hotpot_file(Path("tests/fixtures/hotpot_mixed_raw.jsonl"), corpus, examples, limit=1)
    example = load_jsonl_examples(examples)[0]
    retriever = LexicalRetriever(load_corpus_jsonl(corpus))

    trace = run_retrieval_episode(example, retriever, top_k=2)

    assert trace.final_answer == "Example City"
    assert trace.steps[0].observation[0].doc_id == "hotpot::hp_mixed_1::Alice Smith::0"
