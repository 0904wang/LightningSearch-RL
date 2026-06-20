from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightningsearch_rl.agent_loop import parse_agent_action


LOCAL_ROOT_MARKER = "Agent RL"
REMOTE_ROOTS = ("/data/wzl/LightningSearch-RL", "/home/user/wzl/LightningSearch-RL")
DEFAULT_MODES = ("search", "answer")


def prepare_generation_inspection(
    *,
    sft_path: Path,
    model_path: str,
    out_dir: Path,
    offset: int = 0,
    limit: int = 5,
    max_new_tokens: int = 64,
    dry_run: bool = False,
    modes: list[str] | tuple[str, ...] = DEFAULT_MODES,
) -> dict[str, Any]:
    _ensure_approved_path(out_dir)
    rows = _load_jsonl(sft_path)
    selected = rows[offset : offset + limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_modes = _normalize_modes(modes)
    prompts_by_mode = {
        mode: _build_prompts(selected, mode)
        for mode in normalized_modes
    }
    for mode, prompts in prompts_by_mode.items():
        _write_jsonl(out_dir / f"{mode}_prompts.jsonl", prompts)

    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "model_path": model_path,
        "sft_path": str(sft_path),
        "out_dir": str(out_dir),
        "offset": offset,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "modes": list(normalized_modes),
    }
    for mode, prompts in prompts_by_mode.items():
        summary[f"{mode}_prompt_count"] = len(prompts)

    if dry_run:
        _write_json(out_dir / "dry_run_summary.json", summary)
        return summary

    records: list[dict[str, Any]] = []
    for mode, prompts in prompts_by_mode.items():
        records.extend(_generate_records(model_path, prompts, mode, max_new_tokens=max_new_tokens))
    _write_jsonl(out_dir / "generations.jsonl", records)
    generation_summary = summarize_generation_records(records)
    generation_summary.update(summary)
    generation_summary["dry_run"] = False
    _write_json(out_dir / "summary.json", generation_summary)
    return generation_summary


def summarize_generation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({str(record.get("mode", "")) for record in records if record.get("mode")})
    return {
        "overall": _summarize_subset(records),
        "by_mode": {
            mode: _summarize_subset([record for record in records if record.get("mode") == mode])
            for mode in modes
        },
    }


def _build_prompts(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    prompts = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 5:
            raise ValueError(f"SFT row {row.get('id')} must contain at least five messages")
        if mode == "search":
            prompt_messages = messages[:2]
        elif mode == "answer":
            prompt_messages = messages[:4]
        else:  # pragma: no cover - guarded by _normalize_modes
            raise ValueError(f"unsupported mode: {mode}")
        metadata = row.get("metadata", {})
        prompts.append(
            {
                "id": row.get("id"),
                "mode": mode,
                "messages": prompt_messages,
                "gold_answer": metadata.get("answer", ""),
                "metadata": metadata,
            }
        )
    return prompts


def _generate_records(
    model_path: str,
    prompts: list[dict[str, Any]],
    mode: str,
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    device = next(model.parameters()).device

    records = []
    with torch.no_grad():
        for prompt in prompts:
            prompt_text = _apply_chat_template(tokenizer, prompt["messages"])
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_length = inputs["input_ids"].shape[-1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = outputs[0, input_length:]
            generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            eos = bool(len(new_ids) and tokenizer.eos_token_id is not None and new_ids[-1].item() == tokenizer.eos_token_id)
            records.append(
                {
                    "id": prompt["id"],
                    "mode": mode,
                    "messages": prompt["messages"],
                    "generated": generated,
                    "gold_answer": prompt.get("gold_answer", ""),
                    "new_tokens": int(new_ids.numel()),
                    "eos": eos,
                    "metadata": prompt.get("metadata", {}),
                }
            )
    return records


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _summarize_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [str(record.get("generated", "")) for record in records]
    actions = [parse_agent_action(text) for text in generated]
    return {
        "example_count": len(records),
        "search_tag_rate": _rate("<search>" in text and "</search>" in text for text in generated),
        "answer_tag_rate": _rate("<answer>" in text and "</answer>" in text for text in generated),
        "observation_tag_rate": _rate("<observation>" in text or "</observation>" in text for text in generated),
        "single_action_rate": _rate(action.valid for action in actions),
        "valid_search_action_rate": _rate(action.valid and action.type == "search" for action in actions),
        "valid_answer_action_rate": _rate(action.valid and action.type == "answer" for action in actions),
        "gold_answer_mention_rate": _rate(
            _contains_gold_answer(record.get("generated", ""), record.get("gold_answer", ""))
            for record in records
        ),
        "eos_rate": _rate(bool(record.get("eos")) for record in records),
        "avg_new_tokens": _average(int(record.get("new_tokens", 0)) for record in records),
    }


def _contains_gold_answer(generated: Any, gold_answer: Any) -> bool:
    gold = str(gold_answer).strip().lower()
    if not gold:
        return False
    return gold in str(generated).lower()


def _normalize_modes(modes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(mode.strip() for mode in modes if mode.strip())
    unsupported = sorted(set(normalized) - set(DEFAULT_MODES))
    if unsupported:
        raise ValueError(f"unsupported inspection modes: {unsupported}")
    return normalized or DEFAULT_MODES


def _ensure_approved_path(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if any(normalized.startswith(root) for root in REMOTE_ROOTS):
        return
    if LOCAL_ROOT_MARKER in normalized:
        return
    raise ValueError(f"path is outside approved paths: {path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rate(values: Any) -> float:
    items = list(values)
    return round(sum(1 for value in items if value) / len(items), 6) if items else 0.0


def _average(values: Any) -> float:
    items = list(values)
    return round(sum(float(value) for value in items) / len(items), 6) if items else 0.0
