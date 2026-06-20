from __future__ import annotations

from collections import Counter
import gc
import json
import math
from pathlib import Path
from typing import Any


def prepare_policy_movement_dry_run(
    *,
    base_model: Path,
    candidate_model: Path,
    sft_path: Path,
    out_dir: Path,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    prompts = build_stage_prompts(sft_path, offset=offset, limit=limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "dry_run_prompts.jsonl", prompts)
    summary = {
        "dry_run": True,
        "base_model": str(base_model),
        "candidate_model": str(candidate_model),
        "sft_path": str(sft_path),
        "out_dir": str(out_dir),
        "offset": offset,
        "limit": limit,
        "prompt_count": len(prompts),
        "stage_counts": dict(sorted(Counter(prompt["stage"] for prompt in prompts).items())),
    }
    _write_json(out_dir / "dry_run_summary.json", summary)
    return summary


def diagnose_policy_movement(
    *,
    base_model: Path,
    candidate_model: Path,
    sft_path: Path,
    out_dir: Path,
    offset: int,
    limit: int,
    device: str = "cuda",
    dtype: str = "bfloat16",
    top_k_tensors: int = 20,
    skip_logprobs: bool = False,
    skip_params: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return prepare_policy_movement_dry_run(
            base_model=base_model,
            candidate_model=candidate_model,
            sft_path=sft_path,
            out_dir=out_dir,
            offset=offset,
            limit=limit,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_stage_prompts(sft_path, offset=offset, limit=limit)
    _write_jsonl(out_dir / "stage_prompts.jsonl", prompts)

    parameter_diff = None
    if not skip_params:
        parameter_diff = compare_safetensor_dirs(base_model, candidate_model, top_k=top_k_tensors)
        _write_json(out_dir / "parameter_diff.json", parameter_diff)

    logprob_comparison = None
    if not skip_logprobs:
        base_scores = score_stage_logprobs(
            model_path=base_model,
            prompts=prompts,
            device=device,
            dtype=dtype,
        )
        _write_json(out_dir / "base_logprobs.json", base_scores)
        _cleanup_torch()
        candidate_scores = score_stage_logprobs(
            model_path=candidate_model,
            prompts=prompts,
            device=device,
            dtype=dtype,
        )
        _write_json(out_dir / "candidate_logprobs.json", candidate_scores)
        _cleanup_torch()
        logprob_comparison = compare_logprob_reports(base_scores, candidate_scores)
        _write_json(out_dir / "logprob_comparison.json", logprob_comparison)

    summary = {
        "dry_run": False,
        "base_model": str(base_model),
        "candidate_model": str(candidate_model),
        "sft_path": str(sft_path),
        "out_dir": str(out_dir),
        "offset": offset,
        "limit": limit,
        "prompt_count": len(prompts),
        "stage_counts": dict(sorted(Counter(prompt["stage"] for prompt in prompts).items())),
        "parameter_diff": parameter_diff,
        "logprob_comparison": logprob_comparison,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def compare_safetensor_dirs(base_dir: Path, candidate_dir: Path, *, top_k: int = 20) -> dict[str, Any]:
    from safetensors import safe_open

    base_tensors = _safetensor_weight_map(base_dir)
    candidate_tensors = _safetensor_weight_map(candidate_dir)
    common_names = sorted(set(base_tensors) & set(candidate_tensors))
    missing_in_candidate = sorted(set(base_tensors) - set(candidate_tensors))
    extra_in_candidate = sorted(set(candidate_tensors) - set(base_tensors))

    total_elements = 0
    changed_tensors = 0
    unchanged_tensors = 0
    diff_l2_sq = 0.0
    base_l2_sq = 0.0
    sum_abs_diff = 0.0
    max_abs_diff = 0.0
    tensor_changes = []

    for name in common_names:
        base_file = base_tensors[name]
        candidate_file = candidate_tensors[name]
        with safe_open(base_file, framework="pt", device="cpu") as base_handle:
            base_tensor = base_handle.get_tensor(name)
        with safe_open(candidate_file, framework="pt", device="cpu") as candidate_handle:
            candidate_tensor = candidate_handle.get_tensor(name)
        if base_tensor.shape != candidate_tensor.shape:
            raise ValueError(f"tensor shape mismatch for {name}: {base_tensor.shape} vs {candidate_tensor.shape}")

        base_float = base_tensor.float()
        candidate_float = candidate_tensor.float()
        diff = candidate_float - base_float
        tensor_numel = diff.numel()
        tensor_diff_l2_sq = float(diff.pow(2).sum().item())
        tensor_base_l2_sq = float(base_float.pow(2).sum().item())
        tensor_max_abs = float(diff.abs().max().item()) if tensor_numel else 0.0
        tensor_sum_abs = float(diff.abs().sum().item())

        total_elements += tensor_numel
        diff_l2_sq += tensor_diff_l2_sq
        base_l2_sq += tensor_base_l2_sq
        sum_abs_diff += tensor_sum_abs
        max_abs_diff = max(max_abs_diff, tensor_max_abs)
        if tensor_max_abs > 0.0:
            changed_tensors += 1
        else:
            unchanged_tensors += 1

        tensor_changes.append(
            {
                "name": name,
                "numel": tensor_numel,
                "max_abs_diff": tensor_max_abs,
                "mean_abs_diff": round(tensor_sum_abs / tensor_numel, 12) if tensor_numel else 0.0,
                "relative_l2_diff": _safe_ratio(math.sqrt(tensor_diff_l2_sq), math.sqrt(tensor_base_l2_sq)),
            }
        )

    tensor_changes.sort(key=lambda item: (item["relative_l2_diff"], item["max_abs_diff"]), reverse=True)
    return {
        "base_dir": str(base_dir),
        "candidate_dir": str(candidate_dir),
        "base_tensor_count": len(base_tensors),
        "candidate_tensor_count": len(candidate_tensors),
        "compared_tensors": len(common_names),
        "changed_tensors": changed_tensors,
        "unchanged_tensors": unchanged_tensors,
        "missing_in_candidate_count": len(missing_in_candidate),
        "extra_in_candidate_count": len(extra_in_candidate),
        "missing_in_candidate_preview": missing_in_candidate[:20],
        "extra_in_candidate_preview": extra_in_candidate[:20],
        "total_elements": total_elements,
        "l2_diff": math.sqrt(diff_l2_sq),
        "base_l2": math.sqrt(base_l2_sq),
        "relative_l2_diff": _safe_ratio(math.sqrt(diff_l2_sq), math.sqrt(base_l2_sq)),
        "mean_abs_diff": round(sum_abs_diff / total_elements, 12) if total_elements else 0.0,
        "max_abs_diff": max_abs_diff,
        "top_tensor_changes": tensor_changes[:top_k],
    }


def build_stage_prompts(sft_path: Path, *, offset: int, limit: int) -> list[dict[str, Any]]:
    rows = _load_jsonl(sft_path)[offset : offset + limit]
    prompts: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 5:
            raise ValueError(f"SFT row {row.get('id')} must contain at least 5 messages")
        common = {
            "id": row.get("id"),
            "gold_answer": row.get("metadata", {}).get("answer", ""),
            "metadata": row.get("metadata", {}),
        }
        prompts.append(
            {
                **common,
                "stage": "search",
                "messages": messages[:2],
                "target": messages[2]["content"],
            }
        )
        prompts.append(
            {
                **common,
                "stage": "answer",
                "messages": messages[:4],
                "target": messages[4]["content"],
            }
        )
    return prompts


def score_stage_logprobs(
    *,
    model_path: Path,
    prompts: list[dict[str, Any]],
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"unsupported dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=False,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    records = []
    stage_totals: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for prompt in prompts:
            prompt_text = _apply_chat_template(tokenizer, prompt["messages"])
            target_text = str(prompt["target"])
            full_text = prompt_text + target_text
            prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
            full_inputs = tokenizer(full_text, return_tensors="pt")
            input_ids = full_inputs["input_ids"].to(model.device)
            attention_mask = full_inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            prompt_len = int(prompt_inputs["input_ids"].shape[-1])
            if input_ids.shape[-1] <= prompt_len:
                token_logprobs = torch.empty(0, device=input_ids.device)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :].float()
                labels = input_ids[:, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                token_logprobs = gathered[:, prompt_len - 1 :].squeeze(0)
            token_count = int(token_logprobs.numel())
            sum_logprob = float(token_logprobs.sum().item()) if token_count else 0.0
            mean_logprob = sum_logprob / token_count if token_count else 0.0
            records.append(
                {
                    "id": prompt.get("id"),
                    "stage": prompt["stage"],
                    "target": target_text,
                    "target_token_count": token_count,
                    "sum_logprob": sum_logprob,
                    "mean_logprob": mean_logprob,
                }
            )
            totals = stage_totals.setdefault(prompt["stage"], {"sum_logprob": 0.0, "token_count": 0.0, "row_count": 0.0})
            totals["sum_logprob"] += sum_logprob
            totals["token_count"] += token_count
            totals["row_count"] += 1

    by_stage = {
        stage: {
            "row_count": int(values["row_count"]),
            "token_count": int(values["token_count"]),
            "mean_logprob": values["sum_logprob"] / values["token_count"] if values["token_count"] else 0.0,
        }
        for stage, values in sorted(stage_totals.items())
    }
    total_logprob = sum(values["sum_logprob"] for values in stage_totals.values())
    total_tokens = sum(values["token_count"] for values in stage_totals.values())
    return {
        "model_path": str(model_path),
        "prompt_count": len(prompts),
        "token_count": int(total_tokens),
        "mean_logprob": total_logprob / total_tokens if total_tokens else 0.0,
        "by_stage": by_stage,
        "records": records,
    }


def compare_logprob_reports(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_by_key = {(record["id"], record["stage"]): record for record in base.get("records", [])}
    candidate_by_key = {(record["id"], record["stage"]): record for record in candidate.get("records", [])}
    records = []
    for key in sorted(set(base_by_key) & set(candidate_by_key)):
        base_record = base_by_key[key]
        candidate_record = candidate_by_key[key]
        records.append(
            {
                "id": key[0],
                "stage": key[1],
                "base_mean_logprob": base_record["mean_logprob"],
                "candidate_mean_logprob": candidate_record["mean_logprob"],
                "delta_mean_logprob": candidate_record["mean_logprob"] - base_record["mean_logprob"],
                "target_token_count": candidate_record["target_token_count"],
            }
        )
    by_stage = {}
    for stage in sorted({record["stage"] for record in records}):
        stage_records = [record for record in records if record["stage"] == stage]
        by_stage[stage] = {
            "row_count": len(stage_records),
            "mean_delta_logprob": _average(record["delta_mean_logprob"] for record in stage_records),
            "improved_count": sum(1 for record in stage_records if record["delta_mean_logprob"] > 0),
            "regressed_count": sum(1 for record in stage_records if record["delta_mean_logprob"] < 0),
            "unchanged_count": sum(1 for record in stage_records if record["delta_mean_logprob"] == 0),
        }
    return {
        "compared_records": len(records),
        "base_mean_logprob": base.get("mean_logprob"),
        "candidate_mean_logprob": candidate.get("mean_logprob"),
        "delta_mean_logprob": (
            candidate.get("mean_logprob", 0.0) - base.get("mean_logprob", 0.0)
            if base.get("mean_logprob") is not None and candidate.get("mean_logprob") is not None
            else None
        ),
        "by_stage": by_stage,
        "records": records,
    }


def _safetensor_weight_map(model_dir: Path) -> dict[str, Path]:
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return {
            name: model_dir / filename
            for name, filename in index.get("weight_map", {}).items()
        }
    weight_map = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = path
    return weight_map


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


def _cleanup_torch() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _average(values: Any) -> float:
    items = list(values)
    return sum(float(item) for item in items) / len(items) if items else 0.0


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
