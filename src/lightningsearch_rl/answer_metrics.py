from __future__ import annotations

from collections import Counter
import re


_ARTICLES = {"a", "an", "the"}


def score_answer(prediction: str, gold: str) -> dict[str, float | bool]:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    return {
        "exact_match": bool(pred_norm and pred_norm == gold_norm),
        "token_f1": token_f1(pred_norm, gold_norm),
        "containment_match": containment_match(pred_norm, gold_norm),
    }


def soft_answer_reward(
    prediction: str,
    gold: str,
    *,
    token_f1_threshold: float = 0.75,
) -> dict[str, float | bool | str]:
    score = score_answer(prediction, gold)
    token_score = float(score["token_f1"])
    if score["exact_match"]:
        reward = 1.0
        reward_type = "exact"
    elif score["containment_match"]:
        reward = max(0.5, token_score)
        reward_type = "containment"
    elif token_score >= token_f1_threshold:
        reward = token_score
        reward_type = "token_f1"
    else:
        reward = 0.0
        reward_type = "none"
    return {
        **score,
        "answer_reward": round(reward, 6),
        "answer_reward_type": reward_type,
    }


def normalize_answer(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", " ", value.casefold())
    tokens = [token for token in text.split() if token not in _ARTICLES]
    return " ".join(tokens)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = prediction.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    hit_count = sum(overlap.values())
    if hit_count == 0:
        return 0.0
    precision = hit_count / len(pred_tokens)
    recall = hit_count / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 6)


def containment_match(prediction: str, gold: str) -> bool:
    if not prediction or not gold:
        return False
    return prediction in gold or gold in prediction
