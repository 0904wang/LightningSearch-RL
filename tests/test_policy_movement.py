import json

import pytest

from lightningsearch_rl.policy_movement import (
    build_stage_prompts,
    compare_safetensor_dirs,
    prepare_policy_movement_dry_run,
)
from lightningsearch_rl.cli import main


safetensors_torch = pytest.importorskip("safetensors.torch")


def _write_sft_rows(path):
    path.write_text(
        json.dumps(
            {
                "id": "turn-1",
                "messages": [
                    {"role": "system", "content": "Output one action."},
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "<search>Question evidence</search>"},
                    {"role": "user", "content": "<observation>\n[1] Evidence.\n</observation>"},
                    {"role": "assistant", "content": "<answer>Answer</answer>"},
                ],
                "metadata": {"answer": "Answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_compare_safetensor_dirs_reports_changed_tensor(tmp_path):
    import torch

    base_dir = tmp_path / "base"
    candidate_dir = tmp_path / "candidate"
    base_dir.mkdir()
    candidate_dir.mkdir()
    safetensors_torch.save_file(
        {
            "same.weight": torch.tensor([1.0, 2.0]),
            "changed.weight": torch.tensor([1.0, 1.0]),
        },
        base_dir / "model.safetensors",
    )
    safetensors_torch.save_file(
        {
            "same.weight": torch.tensor([1.0, 2.0]),
            "changed.weight": torch.tensor([2.0, 1.0]),
        },
        candidate_dir / "model.safetensors",
    )

    report = compare_safetensor_dirs(base_dir, candidate_dir, top_k=2)

    assert report["compared_tensors"] == 2
    assert report["changed_tensors"] == 1
    assert report["unchanged_tensors"] == 1
    assert report["total_elements"] == 4
    assert report["relative_l2_diff"] > 0.0
    assert report["top_tensor_changes"][0]["name"] == "changed.weight"


def test_build_stage_prompts_extracts_search_and_answer_targets(tmp_path):
    sft_path = tmp_path / "sft_turns.jsonl"
    _write_sft_rows(sft_path)

    prompts = build_stage_prompts(sft_path, offset=0, limit=1)

    assert [prompt["stage"] for prompt in prompts] == ["search", "answer"]
    assert prompts[0]["target"] == "<search>Question evidence</search>"
    assert prompts[1]["target"] == "<answer>Answer</answer>"
    assert prompts[1]["messages"][-1]["role"] == "user"


def test_prepare_policy_movement_dry_run_writes_prompt_manifest(tmp_path):
    sft_path = tmp_path / "sft_turns.jsonl"
    _write_sft_rows(sft_path)
    base_model = tmp_path / "base-model"
    candidate_model = tmp_path / "candidate-model"
    base_model.mkdir()
    candidate_model.mkdir()
    out_dir = tmp_path / "policy-movement"

    summary = prepare_policy_movement_dry_run(
        base_model=base_model,
        candidate_model=candidate_model,
        sft_path=sft_path,
        out_dir=out_dir,
        offset=0,
        limit=1,
    )

    assert summary["dry_run"] is True
    assert summary["prompt_count"] == 2
    assert summary["stage_counts"] == {"answer": 1, "search": 1}
    assert (out_dir / "dry_run_prompts.jsonl").exists()
    assert json.loads((out_dir / "dry_run_summary.json").read_text(encoding="utf-8")) == summary


def test_diagnose_policy_movement_cli_dry_run_writes_summary(tmp_path):
    sft_path = tmp_path / "sft_turns.jsonl"
    _write_sft_rows(sft_path)
    base_model = tmp_path / "base-model"
    candidate_model = tmp_path / "candidate-model"
    out_dir = tmp_path / "policy-movement-cli"
    base_model.mkdir()
    candidate_model.mkdir()

    assert (
        main(
            [
                "diagnose-policy-movement",
                "--base-model",
                str(base_model),
                "--candidate-model",
                str(candidate_model),
                "--sft",
                str(sft_path),
                "--out-dir",
                str(out_dir),
                "--offset",
                "0",
                "--limit",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )

    summary = json.loads((out_dir / "dry_run_summary.json").read_text(encoding="utf-8"))
    assert summary["prompt_count"] == 2
    assert summary["stage_counts"] == {"answer": 1, "search": 1}
