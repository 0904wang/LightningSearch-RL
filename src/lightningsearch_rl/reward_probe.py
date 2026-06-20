from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from lightningsearch_rl.verl_reward import compute_score


LOCAL_ROOT_MARKER = "Agent RL"
REMOTE_ROOTS = ("/data/wzl/LightningSearch-RL", "/home/user/wzl/LightningSearch-RL")
ProbeGenerator = Callable[[list[dict[str, Any]], int, int], list[list[str]]]


def run_reward_probe(
    *,
    transitions_path: Path,
    out_dir: Path,
    model_path: str,
    offset: int = 0,
    limit: int | None = None,
    samples_per_prompt: int = 4,
    max_new_tokens: int = 64,
    search_reward_top_k: int = 8,
    answer_token_f1_threshold: float | None = None,
    stages: tuple[str, ...] = (),
    search_diversity_prompt: bool = False,
    dry_run: bool = False,
    generator: ProbeGenerator | None = None,
    backend: str = "vllm",
    batch_size: int = 64,
    temperature: float = 1.2,
    top_p: float = 0.95,
    top_k: int | None = 50,
    seed: int | None = None,
    gpu_memory_utilization: float = 0.45,
    max_model_len: int = 768,
    tensor_parallel_size: int = 1,
) -> dict[str, Any]:
    _ensure_approved_path(out_dir)
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    transition_rows = _load_jsonl(transitions_path)
    selected_stages = tuple(str(stage).strip().lower() for stage in stages if str(stage).strip())
    filtered_rows = _filter_rows_by_stage(transition_rows, selected_stages)
    selected_rows = filtered_rows[offset:] if limit is None else filtered_rows[offset : offset + limit]
    requests = [
        _build_probe_request(
            row,
            index,
            search_reward_top_k=search_reward_top_k,
            search_diversity_prompt=search_diversity_prompt,
        )
        for index, row in enumerate(selected_rows, start=offset)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "probe_requests.jsonl", requests)
    base_summary = _summary(
        transitions_path=transitions_path,
        out_dir=out_dir,
        model_path=model_path,
        offset=offset,
        limit=limit,
        samples_per_prompt=samples_per_prompt,
        max_new_tokens=max_new_tokens,
        search_reward_top_k=search_reward_top_k,
        answer_token_f1_threshold=answer_token_f1_threshold,
        stages=selected_stages,
        search_diversity_prompt=search_diversity_prompt,
        backend=backend,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        input_transition_count=len(transition_rows),
        filtered_transition_count=len(filtered_rows),
        requests=requests,
        dry_run=dry_run,
    )
    if dry_run:
        _write_json(out_dir / "summary.json", base_summary)
        return base_summary

    active_generator = generator or _build_generator(
        backend=backend,
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )
    reward_dump_path = out_dir / "reward_dump.jsonl"
    reward_dump_path.write_text("", encoding="utf-8")
    generations: list[dict[str, Any]] = []
    with _temporary_reward_env(reward_dump_path, answer_token_f1_threshold):
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            generated = active_generator(chunk, samples_per_prompt, max_new_tokens)
            _validate_generated(chunk, generated, samples_per_prompt)
            for request, outputs in zip(chunk, generated):
                for sample_index, solution in enumerate(outputs):
                    extra_info = {
                        **request["extra_info"],
                        "probe_sample_index": sample_index,
                    }
                    result = compute_score(
                        data_source="lightningsearch_rl",
                        solution_str=solution,
                        ground_truth=request["ground_truth"],
                        extra_info=extra_info,
                    )
                    generations.append(
                        {
                            "request_index": request["request_index"],
                            "sample_index": sample_index,
                            "id": extra_info["id"],
                            "source_id": extra_info["source_id"],
                            "reward_stage": extra_info["reward_stage"],
                            "solution": solution,
                            "score": result["score"],
                            "reward": result,
                        }
                    )

    _write_jsonl(out_dir / "generations.jsonl", generations)
    final_summary = {
        **base_summary,
        "generated_sample_count": len(generations),
        "reward_dump_count": _jsonl_count(reward_dump_path),
    }
    _write_json(out_dir / "summary.json", final_summary)
    return final_summary


def _build_probe_request(
    row: dict[str, Any],
    index: int,
    *,
    search_reward_top_k: int,
    search_diversity_prompt: bool = False,
) -> dict[str, Any]:
    prompt = row.get("state_messages")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"transition row {row.get('transition_id', index)} must contain state_messages")
    action_type = str(row.get("action_type", "")).strip().lower() or "answer"
    if action_type not in {"search", "answer"}:
        action_type = "answer"
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    transition_id = str(row.get("transition_id") or f"{row.get('id', 'transition')}:{index}:{action_type}")
    source_id = str(row.get("id") or transition_id.split(":", 1)[0])
    ground_truth = "" if action_type == "search" else str(metadata.get("gold_answer", ""))
    request_prompt = _search_diversity_prompt(prompt) if search_diversity_prompt and action_type == "search" else prompt
    return {
        "request_index": index,
        "prompt": request_prompt,
        "ground_truth": ground_truth,
        "extra_info": {
            "id": transition_id,
            "source_id": source_id,
            "index": index,
            "answer": metadata.get("gold_answer", ""),
            "search_count": 1 if action_type == "search" else 0,
            "gold_doc_ids": _list(row.get("gold_evidence_doc_ids")),
            "retrieved_doc_ids": _list(row.get("observation_doc_ids")),
            "candidate_passages": _list(row.get("candidate_passages")),
            "search_reward_top_k": search_reward_top_k,
            "search_diversity_prompt": bool(search_diversity_prompt and action_type == "search"),
            "reward_stage": action_type,
            "expected_action": str(row.get("action", "")),
            "precomputed_step_reward": _float(row.get("reward")),
            "precomputed_total_reward": _float(metadata.get("total_reward")),
        },
    }


def _build_generator(
    *,
    backend: str,
    model_path: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int,
) -> ProbeGenerator:
    if backend != "vllm":
        raise ValueError(f"unsupported probe backend: {backend}")
    return VllmProbeGenerator(
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )


class VllmProbeGenerator:
    def __init__(
        self,
        *,
        model_path: str,
        temperature: float,
        top_p: float,
        top_k: int | None,
        seed: int | None,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.llm = LLM(
            model=model_path,
            trust_remote_code=False,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
        )

    def __call__(
        self,
        requests: list[dict[str, Any]],
        samples_per_prompt: int,
        max_new_tokens: int,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        prompts = [_apply_chat_template(self.tokenizer, request["prompt"]) for request in requests]
        params_kwargs: dict[str, Any] = {
            "n": samples_per_prompt,
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            params_kwargs["top_k"] = self.top_k
        if self.seed is not None:
            params_kwargs["seed"] = self.seed
        try:
            params = SamplingParams(**params_kwargs)
        except TypeError:
            params_kwargs.pop("seed", None)
            params = SamplingParams(**params_kwargs)
        results = self.llm.generate(prompts, params, use_tqdm=True)
        return [[output.text.strip() for output in result.outputs] for result in results]


def _summary(
    *,
    transitions_path: Path,
    out_dir: Path,
    model_path: str,
    offset: int,
    limit: int | None,
    samples_per_prompt: int,
    max_new_tokens: int,
    search_reward_top_k: int,
    answer_token_f1_threshold: float | None,
    stages: tuple[str, ...],
    search_diversity_prompt: bool,
    backend: str,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int,
    input_transition_count: int,
    filtered_transition_count: int,
    requests: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "dry_run": dry_run,
        "transitions_path": str(transitions_path),
        "out_dir": str(out_dir),
        "model_path": model_path,
        "offset": offset,
        "limit": limit,
        "samples_per_prompt": samples_per_prompt,
        "max_new_tokens": max_new_tokens,
        "search_reward_top_k": search_reward_top_k,
        "answer_token_f1_threshold": answer_token_f1_threshold,
        "stages": list(stages),
        "search_diversity_prompt": search_diversity_prompt,
        "backend": backend,
        "batch_size": batch_size,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "input_transition_count": input_transition_count,
        "filtered_transition_count": filtered_transition_count,
        "selected_transition_count": len(requests),
        "source_count": len({request["extra_info"]["source_id"] for request in requests}),
        "stage_counts": _stage_counts(requests),
        "expected_reward_rows": len(requests) * samples_per_prompt,
        "artifacts": {
            "probe_requests": str(out_dir / "probe_requests.jsonl"),
            "generations": str(out_dir / "generations.jsonl"),
            "reward_dump": str(out_dir / "reward_dump.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }


def _validate_generated(
    requests: list[dict[str, Any]],
    generated: list[list[str]],
    samples_per_prompt: int,
) -> None:
    if len(generated) != len(requests):
        raise ValueError(f"generator returned {len(generated)} prompt groups for {len(requests)} requests")
    for index, outputs in enumerate(generated):
        if len(outputs) != samples_per_prompt:
            raise ValueError(
                f"generator returned {len(outputs)} samples for request {index}; expected {samples_per_prompt}"
            )


@contextmanager
def _temporary_reward_env(
    reward_dump_path: Path,
    answer_token_f1_threshold: float | None,
) -> Iterator[None]:
    updates = {
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH": str(reward_dump_path),
        "LIGHTNINGSEARCH_REWARD_DUMP_MAX_CHARS": "1024",
    }
    if answer_token_f1_threshold is not None:
        updates["LIGHTNINGSEARCH_ANSWER_TOKEN_F1_THRESHOLD"] = f"{answer_token_f1_threshold:g}"
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


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


def _stage_counts(requests: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for request in requests:
        stage = str(request["extra_info"]["reward_stage"])
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _filter_rows_by_stage(rows: list[dict[str, Any]], stages: tuple[str, ...]) -> list[dict[str, Any]]:
    if not stages:
        return rows
    stage_set = set(stages)
    return [row for row in rows if _action_type(row) in stage_set]


def _action_type(row: dict[str, Any]) -> str:
    action_type = str(row.get("action_type", "")).strip().lower() or "answer"
    return action_type if action_type in {"search", "answer"} else "answer"


def _search_diversity_prompt(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    instruction = (
        "Search-probe mode: output exactly one <search>...</search> action. "
        "Do not output <answer>. Use a concise evidence-seeking query with useful "
        "entities, relation words, or constraints from the question. Avoid copying "
        "a previous obvious query verbatim when another plausible retrieval query is possible."
    )
    copied = [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in messages
    ]
    if copied and copied[0]["role"] == "system":
        copied[0] = {
            **copied[0],
            "content": copied[0]["content"].rstrip() + "\n\n" + instruction,
        }
        return copied
    return [{"role": "system", "content": instruction}, *copied]


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
