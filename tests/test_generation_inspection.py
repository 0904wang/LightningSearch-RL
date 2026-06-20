import json

from lightningsearch_rl.generation_inspection import (
    prepare_generation_inspection,
    summarize_generation_records,
)


def _write_turn_rows(path, count=2):
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"turn-{index}",
                "messages": [
                    {"role": "system", "content": "Output exactly one action."},
                    {"role": "user", "content": f"Question {index}?"},
                    {"role": "assistant", "content": f"<search>Question {index}?</search>"},
                    {"role": "user", "content": f"<observation>\n[1] Evidence {index}.\n</observation>"},
                    {"role": "assistant", "content": f"<answer>Answer {index}</answer>"},
                ],
                "metadata": {"answer": f"Answer {index}"},
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_prepare_generation_inspection_dry_run_writes_stage_prompts(tmp_path):
    sft_turns = tmp_path / "sft_turns.jsonl"
    _write_turn_rows(sft_turns, count=2)

    summary = prepare_generation_inspection(
        sft_path=sft_turns,
        model_path="unused-model",
        out_dir=tmp_path / "inspection",
        offset=0,
        limit=2,
        max_new_tokens=32,
        dry_run=True,
        modes=["search", "answer"],
    )

    assert summary["dry_run"] is True
    assert summary["search_prompt_count"] == 2
    assert summary["answer_prompt_count"] == 2
    search_row = json.loads((tmp_path / "inspection" / "search_prompts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    answer_row = json.loads((tmp_path / "inspection" / "answer_prompts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [message["role"] for message in search_row["messages"]] == ["system", "user"]
    assert [message["role"] for message in answer_row["messages"]] == ["system", "user", "assistant", "user"]
    assert "<observation>" not in search_row["messages"][-1]["content"]
    assert "<answer>" not in "\n".join(message["content"] for message in answer_row["messages"])


def test_summarize_generation_records_counts_agent_action_quality():
    records = [
        {
            "mode": "search",
            "generated": "<search>clean query</search>",
            "gold_answer": "Paris",
            "new_tokens": 4,
            "eos": True,
        },
        {
            "mode": "search",
            "generated": "<search>query</search><answer>Paris</answer>",
            "gold_answer": "Paris",
            "new_tokens": 8,
            "eos": False,
        },
        {
            "mode": "answer",
            "generated": "<answer>Paris</answer>",
            "gold_answer": "Paris",
            "new_tokens": 3,
            "eos": True,
        },
        {
            "mode": "answer",
            "generated": "<observation>fabricated</observation><answer>London</answer>",
            "gold_answer": "Paris",
            "new_tokens": 9,
            "eos": True,
        },
    ]

    summary = summarize_generation_records(records)

    assert summary["overall"]["example_count"] == 4
    assert summary["overall"]["observation_tag_rate"] == 0.25
    assert summary["by_mode"]["search"]["single_action_rate"] == 0.5
    assert summary["by_mode"]["search"]["search_tag_rate"] == 1.0
    assert summary["by_mode"]["answer"]["answer_tag_rate"] == 1.0
    assert summary["by_mode"]["answer"]["gold_answer_mention_rate"] == 0.5
    assert summary["by_mode"]["answer"]["observation_tag_rate"] == 0.5
