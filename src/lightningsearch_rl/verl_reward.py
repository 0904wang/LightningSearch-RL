from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from lightningsearch_rl.agent_loop import parse_agent_action
from lightningsearch_rl.answer_metrics import soft_answer_reward
from lightningsearch_rl.data import Passage
from lightningsearch_rl.retrieval import LexicalRetriever


SEARCH_COST = 0.03
ZERO_REWARD_RESULT: dict[str, float] = {
    "score": 0.0,
    "answer_reward": 0.0,
    "answer_exact_match": 0.0,
    "answer_token_f1": 0.0,
    "answer_containment_match": 0.0,
    "search_reward": 0.0,
    "evidence_rank_reward": 0.0,
    "gold_top_rank": 0.0,
    "retrieved_gold_count": 0.0,
    "format_reward": 0.0,
    "search_cost": 0.0,
}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    if data_source != "lightningsearch_rl":
        return dict(ZERO_REWARD_RESULT)

    extra = extra_info or {}
    reward_stage = str(extra.get("reward_stage") or extra.get("stage") or "answer").strip().lower()
    if reward_stage == "search":
        result = _compute_search_stage_score(solution_str, extra)
    else:
        result = _compute_answer_stage_score(solution_str, ground_truth, extra)
    _maybe_dump_reward_record(data_source, solution_str, ground_truth, extra, reward_stage, result)
    return result


def _compute_answer_stage_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, float]:
    answer = _extract_answer(solution_str)
    answer_score = soft_answer_reward(
        answer or "",
        ground_truth,
        token_f1_threshold=_answer_token_f1_threshold(),
    )
    answer_reward = float(answer_score["answer_reward"]) if answer is not None else 0.0
    format_reward = 1.0 if answer is not None else 0.0
    search_count = int(extra_info.get("search_count") or 0)
    search_cost = SEARCH_COST * search_count
    score = answer_reward + 0.1 * format_reward - search_cost
    return {
        "score": round(score, 6),
        "answer_reward": answer_reward,
        "answer_exact_match": float(bool(answer_score["exact_match"])),
        "answer_token_f1": float(answer_score["token_f1"]),
        "answer_containment_match": float(bool(answer_score["containment_match"])),
        "search_reward": 0.0,
        "evidence_rank_reward": 0.0,
        "gold_top_rank": 0.0,
        "retrieved_gold_count": 0.0,
        "format_reward": format_reward,
        "search_cost": round(search_cost, 6),
    }


def _compute_search_stage_score(solution_str: str, extra_info: dict[str, Any]) -> dict[str, float]:
    action = parse_agent_action(solution_str)
    valid_search = action.valid and action.type == "search"
    rank = _score_search_query_by_evidence_rank(action.query or "", extra_info) if valid_search else None
    if rank is None:
        search_reward = 1.0 if valid_search else 0.0
        evidence_rank_reward = 0.0
        gold_top_rank = 0.0
        retrieved_gold_count = 0.0
    else:
        evidence_rank_reward = rank["evidence_rank_reward"]
        search_reward = evidence_rank_reward
        gold_top_rank = rank["gold_top_rank"]
        retrieved_gold_count = rank["retrieved_gold_count"]
    format_reward = 1.0 if valid_search else 0.0
    search_count = int(extra_info.get("search_count") or 0)
    search_cost = SEARCH_COST * search_count
    score = search_reward + 0.1 * format_reward - search_cost
    return {
        "score": round(score, 6),
        "answer_reward": 0.0,
        "answer_exact_match": 0.0,
        "answer_token_f1": 0.0,
        "answer_containment_match": 0.0,
        "search_reward": search_reward,
        "evidence_rank_reward": evidence_rank_reward,
        "gold_top_rank": gold_top_rank,
        "retrieved_gold_count": retrieved_gold_count,
        "format_reward": format_reward,
        "search_cost": round(search_cost, 6),
    }


def _score_search_query_by_evidence_rank(query: str, extra_info: dict[str, Any]) -> dict[str, float] | None:
    passages = _candidate_passages(extra_info)
    gold_doc_ids = _strings(extra_info.get("gold_doc_ids") or extra_info.get("gold_evidence_doc_ids"))
    if not passages or not gold_doc_ids:
        return None
    retrieved = LexicalRetriever(passages).search(query, top_k=_search_reward_top_k(extra_info))
    ranks = {passage.doc_id: float(index) for index, passage in enumerate(retrieved, start=1)}
    unique_gold_doc_ids = list(dict.fromkeys(gold_doc_ids))
    gold_ranks = [ranks[doc_id] for doc_id in unique_gold_doc_ids if doc_id in ranks]
    if not gold_ranks:
        return {
            "evidence_rank_reward": 0.0,
            "gold_top_rank": 0.0,
            "retrieved_gold_count": 0.0,
        }
    scores = [_rank_reward(rank) for rank in gold_ranks]
    scores.extend([0.0] * (len(unique_gold_doc_ids) - len(gold_ranks)))
    return {
        "evidence_rank_reward": round(sum(scores) / len(scores), 6),
        "gold_top_rank": min(gold_ranks),
        "retrieved_gold_count": float(len(gold_ranks)),
    }


def _candidate_passages(extra_info: dict[str, Any]) -> list[Passage]:
    rows = extra_info.get("candidate_passages")
    if not isinstance(rows, list):
        return []
    passages = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        passages.append(
            Passage(
                doc_id=doc_id,
                title=str(row.get("title") or ""),
                text=str(row.get("text") or ""),
            )
        )
    return passages


def _search_reward_top_k(extra_info: dict[str, Any]) -> int:
    try:
        return max(1, int(extra_info.get("search_reward_top_k") or 8))
    except (TypeError, ValueError):
        return 8


def _rank_reward(rank: float) -> float:
    if rank <= 1:
        return 1.0
    if rank <= 3:
        return 0.8
    if rank <= 8:
        return 0.5
    return 0.0


def _extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _maybe_dump_reward_record(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    reward_stage: str,
    result: dict[str, float],
) -> None:
    dump_path = os.environ.get("LIGHTNINGSEARCH_REWARD_DUMP_PATH", "").strip()
    if not dump_path:
        return
    max_chars = _dump_max_chars()
    action = parse_agent_action(solution_str)
    row = {
        "data_source": data_source,
        "reward_stage": reward_stage,
        "score": result["score"],
        "answer_reward": result["answer_reward"],
        "search_reward": result["search_reward"],
        "evidence_rank_reward": result["evidence_rank_reward"],
        "gold_top_rank": result["gold_top_rank"],
        "retrieved_gold_count": result["retrieved_gold_count"],
        "format_reward": result["format_reward"],
        "search_cost": result["search_cost"],
        "answer_reward_type": _answer_reward_type_for_dump(reward_stage, solution_str, ground_truth),
        "answer_exact_match": result.get("answer_exact_match"),
        "answer_token_f1": result.get("answer_token_f1"),
        "answer_containment_match": result.get("answer_containment_match"),
        "ground_truth": _truncate(ground_truth, max_chars),
        "solution_preview": _truncate(solution_str, max_chars),
        "parsed_action": {
            "type": action.type,
            "valid": action.valid,
        },
        "extra_info": _jsonable(extra_info),
        "pid": os.getpid(),
    }
    path = Path(dump_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _answer_reward_type_for_dump(reward_stage: str, solution_str: str, ground_truth: str) -> str | None:
    if reward_stage != "answer":
        return None
    answer = _extract_answer(solution_str)
    if answer is None:
        return "none"
    return str(
        soft_answer_reward(
            answer,
            ground_truth,
            token_f1_threshold=_answer_token_f1_threshold(),
        )["answer_reward_type"]
    )


def _answer_token_f1_threshold() -> float:
    raw = os.environ.get("LIGHTNINGSEARCH_ANSWER_TOKEN_F1_THRESHOLD", "").strip()
    if not raw:
        return 0.75
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return 0.75


def _dump_max_chars() -> int:
    try:
        return max(1, int(os.environ.get("LIGHTNINGSEARCH_REWARD_DUMP_MAX_CHARS", "2048")))
    except ValueError:
        return 2048


def _truncate(text: str, max_chars: int) -> str:
    return text[:max_chars]


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)
