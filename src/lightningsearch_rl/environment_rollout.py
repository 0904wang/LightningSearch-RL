from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Callable

from lightningsearch_rl.agent_loop import parse_agent_action
from lightningsearch_rl.answer_metrics import score_answer
from lightningsearch_rl.formatting import format_observation
from lightningsearch_rl.index_store import load_lexical_index
from lightningsearch_rl.retrieval import LexicalRetriever


LOCAL_ROOT_MARKER = "Agent RL"
REMOTE_ROOTS = ("/data/wzl/LightningSearch-RL", "/home/user/wzl/LightningSearch-RL")
Generator = Callable[[list[dict[str, str]], int], str]


def run_environment_rollout(
    *,
    sft_path: Path,
    index_path: Path,
    model_path: str,
    out_dir: Path,
    offset: int = 0,
    limit: int = 5,
    top_k: int = 2,
    max_new_tokens: int = 64,
    dry_run: bool = False,
    generator: Generator | None = None,
    candidate_pool: str = "global",
    distractor_count: int = 0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    sample_top_k: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    _ensure_approved_path(out_dir)
    rows = _load_jsonl(sft_path)[offset : offset + limit]
    retriever = load_lexical_index(index_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        summary = _write_dry_run(
            rows,
            retriever,
            out_dir,
            model_path,
            offset,
            limit,
            top_k,
            max_new_tokens,
            candidate_pool=candidate_pool,
            distractor_count=distractor_count,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            sample_top_k=sample_top_k,
            seed=seed,
        )
        return summary

    active_generator = generator or TransformersChatGenerator(
        model_path,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=sample_top_k,
        seed=seed,
    )
    records = [
        _run_one_row(
            row,
            retriever,
            active_generator,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            candidate_pool=candidate_pool,
            distractor_count=distractor_count,
        )
        for row in rows
    ]
    _write_jsonl(out_dir / "env_rollouts.jsonl", records)
    summary = summarize_environment_rollouts(records)
    summary.update(
        {
            "dry_run": False,
            "model_path": model_path,
            "sft_path": str(sft_path),
            "index_path": str(index_path),
            "out_dir": str(out_dir),
            "offset": offset,
            "limit": limit,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "candidate_pool": candidate_pool,
            "distractor_count": distractor_count,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "sample_top_k": sample_top_k,
            "seed": seed,
        }
    )
    _write_json(out_dir / "summary.json", summary)
    return summary


def summarize_environment_rollouts(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "example_count": len(records),
        "valid_search_action_rate": _rate(record.get("search_action", {}).get("valid") for record in records),
        "valid_answer_action_rate": _rate(record.get("answer_action", {}).get("valid") for record in records),
        "answer_exact_match_rate": _rate(record.get("answer_exact_match") for record in records),
        "answer_token_f1": _average(record.get("answer_token_f1", 0.0) for record in records),
        "answer_containment_match_rate": _rate(record.get("answer_containment_match") for record in records),
        "gold_evidence_recall": _average(record.get("gold_evidence_recall", 0.0) for record in records),
        "all_gold_evidence_retrieved_rate": _rate(record.get("all_gold_evidence_retrieved") for record in records),
        "assistant_observation_rate": _rate(
            "<observation>" in str(record.get("search_generated", ""))
            or "<observation>" in str(record.get("answer_generated", ""))
            for record in records
        ),
        "avg_observation_doc_count": _average(len(record.get("observation_doc_ids", [])) for record in records),
    }


class TransformersChatGenerator:
    def __init__(
        self,
        model_path: str,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        if seed is not None:
            set_seed(seed)
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def __call__(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        import torch

        prompt_text = _apply_chat_template(self.tokenizer, messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                generate_kwargs["top_k"] = self.top_k
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        new_ids = outputs[0, input_length:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _run_one_row(
    row: dict[str, Any],
    retriever: Any,
    generator: Generator,
    *,
    top_k: int,
    max_new_tokens: int,
    candidate_pool: str,
    distractor_count: int,
) -> dict[str, Any]:
    messages = _messages(row)
    search_messages = messages[:2]
    search_generated = generator(search_messages, max_new_tokens)
    search_action = parse_agent_action(search_generated)
    row_retriever, candidate_doc_ids = _build_row_retriever(
        row,
        retriever,
        candidate_pool=candidate_pool,
        distractor_count=distractor_count,
    )
    passages = row_retriever.search(search_action.query or "", top_k=top_k) if search_action.valid else []
    gold_doc_ids = _gold_evidence_doc_ids(row)
    evidence = _evidence_metrics(gold_doc_ids, [passage.doc_id for passage in passages])
    observation = format_observation(passages)
    answer_messages = [
        *search_messages,
        {"role": "assistant", "content": search_generated},
        {"role": "user", "content": observation},
    ]
    answer_generated = generator(answer_messages, max_new_tokens) if search_action.valid else ""
    answer_action = parse_agent_action(answer_generated) if answer_generated else parse_agent_action("")
    gold_answer = _gold_answer(row)
    final_answer = answer_action.answer or ""
    answer_score = score_answer(final_answer, gold_answer)
    return {
        "id": row.get("id"),
        "question": search_messages[-1]["content"] if search_messages else "",
        "search_messages": search_messages,
        "search_generated": search_generated,
        "search_action": asdict(search_action),
        "observation": observation,
        "observation_doc_ids": [passage.doc_id for passage in passages],
        "candidate_doc_ids": candidate_doc_ids,
        "candidate_passages": _serialize_passages(row_retriever.passages),
        "candidate_pool": candidate_pool,
        "gold_evidence_doc_ids": gold_doc_ids,
        "gold_evidence_recall": evidence["gold_evidence_recall"],
        "all_gold_evidence_retrieved": evidence["all_gold_evidence_retrieved"],
        "answer_messages": answer_messages,
        "answer_generated": answer_generated,
        "answer_action": asdict(answer_action),
        "final_answer": final_answer,
        "gold_answer": gold_answer,
        "answer_exact_match": answer_score["exact_match"],
        "answer_token_f1": answer_score["token_f1"],
        "answer_containment_match": answer_score["containment_match"],
        "metadata": row.get("metadata", {}),
    }


def _write_dry_run(
    rows: list[dict[str, Any]],
    retriever: Any,
    out_dir: Path,
    model_path: str,
    offset: int,
    limit: int,
    top_k: int,
    max_new_tokens: int,
    *,
    candidate_pool: str,
    distractor_count: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    sample_top_k: int | None,
    seed: int | None,
) -> dict[str, Any]:
    search_prompts = []
    answer_prompts = []
    for row in rows:
        messages = _messages(row)
        search_messages = messages[:2]
        gold_search = messages[2]["content"] if len(messages) > 2 else ""
        search_action = parse_agent_action(gold_search)
        row_retriever, candidate_doc_ids = _build_row_retriever(
            row,
            retriever,
            candidate_pool=candidate_pool,
            distractor_count=distractor_count,
        )
        passages = row_retriever.search(search_action.query or "", top_k=top_k) if search_action.valid else []
        gold_doc_ids = _gold_evidence_doc_ids(row)
        evidence = _evidence_metrics(gold_doc_ids, [passage.doc_id for passage in passages])
        answer_messages = [
            *search_messages,
            {"role": "assistant", "content": gold_search},
            {"role": "user", "content": format_observation(passages)},
        ]
        common = {
            "id": row.get("id"),
            "gold_answer": _gold_answer(row),
            "metadata": row.get("metadata", {}),
        }
        search_prompts.append({**common, "mode": "search", "messages": search_messages})
        answer_prompts.append(
            {
                **common,
                "mode": "answer",
                "messages": answer_messages,
                "observation_doc_ids": [passage.doc_id for passage in passages],
                "candidate_doc_ids": candidate_doc_ids,
                "candidate_passages": _serialize_passages(row_retriever.passages),
                "gold_evidence_doc_ids": gold_doc_ids,
                "gold_evidence_recall": evidence["gold_evidence_recall"],
                "all_gold_evidence_retrieved": evidence["all_gold_evidence_retrieved"],
            }
        )

    _write_jsonl(out_dir / "search_prompts.jsonl", search_prompts)
    _write_jsonl(out_dir / "answer_prompts.jsonl", answer_prompts)
    summary = {
        "dry_run": True,
        "model_path": model_path,
        "out_dir": str(out_dir),
        "offset": offset,
        "limit": limit,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "candidate_pool": candidate_pool,
        "distractor_count": distractor_count,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "sample_top_k": sample_top_k,
        "seed": seed,
        "search_prompt_count": len(search_prompts),
        "answer_prompt_count": len(answer_prompts),
        "avg_candidate_doc_count": _average(len(prompt["candidate_doc_ids"]) for prompt in answer_prompts),
        "gold_evidence_recall": _average(prompt["gold_evidence_recall"] for prompt in answer_prompts),
        "all_gold_evidence_retrieved_rate": _rate(
            prompt["all_gold_evidence_retrieved"] for prompt in answer_prompts
        ),
    }
    _write_json(out_dir / "dry_run_summary.json", summary)
    return summary


def _messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"SFT row {row.get('id')} must contain at least system and user messages")
    return messages


def _gold_answer(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    return str(metadata.get("answer", "")).strip()


def _gold_evidence_doc_ids(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {})
    doc_ids = metadata.get("gold_evidence_doc_ids") or metadata.get("gold_doc_ids") or []
    return [str(doc_id) for doc_id in doc_ids]


def _build_row_retriever(
    row: dict[str, Any],
    retriever: Any,
    *,
    candidate_pool: str,
    distractor_count: int,
) -> tuple[Any, list[str]]:
    if candidate_pool == "global":
        return retriever, [passage.doc_id for passage in retriever.passages]
    if candidate_pool != "gold-distractors":
        raise ValueError(f"unsupported candidate_pool: {candidate_pool}")

    passages_by_id = {passage.doc_id: passage for passage in retriever.passages}
    gold_doc_ids = _gold_evidence_doc_ids(row)
    gold_passages = [passages_by_id[doc_id] for doc_id in gold_doc_ids if doc_id in passages_by_id]
    gold_set = {passage.doc_id for passage in gold_passages}
    distractors = [passage for passage in retriever.passages if passage.doc_id not in gold_set]
    selected_distractors = _stable_distractors(distractors, str(row.get("id", "")), distractor_count)
    passages = [*gold_passages, *selected_distractors]
    return LexicalRetriever(passages), [passage.doc_id for passage in passages]


def _serialize_passages(passages: list[Any]) -> list[dict[str, str]]:
    return [
        {
            "doc_id": str(passage.doc_id),
            "title": str(passage.title),
            "text": str(passage.text),
        }
        for passage in passages
    ]


def _stable_distractors(passages: list[Any], row_id: str, count: int) -> list[Any]:
    if count <= 0 or not passages:
        return []
    start = sum(ord(char) for char in row_id) % len(passages)
    rotated = [*passages[start:], *passages[:start]]
    return rotated[:count]


def _evidence_metrics(gold_doc_ids: list[str], observed_doc_ids: list[str]) -> dict[str, float | bool]:
    if not gold_doc_ids:
        return {"gold_evidence_recall": 0.0, "all_gold_evidence_retrieved": False}
    hits = len(set(gold_doc_ids) & set(observed_doc_ids))
    return {
        "gold_evidence_recall": round(hits / len(set(gold_doc_ids)), 6),
        "all_gold_evidence_retrieved": hits == len(set(gold_doc_ids)),
    }


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


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


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
