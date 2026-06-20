from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


LOCAL_ROOT_MARKER = "Agent RL"
REMOTE_ROOTS = ("/data/wzl/LightningSearch-RL", "/home/user/wzl/LightningSearch-RL")
REQUIRED_CONFIG_KEYS = {
    "experiment_name",
    "project_name",
    "train_samples",
    "val_samples",
    "seed",
    "model_path",
    "max_prompt_length",
    "max_response_length",
    "train_batch_size",
    "ppo_mini_batch_size",
    "ppo_micro_batch_size_per_gpu",
    "n_gpus_per_node",
    "total_training_steps",
    "save_freq",
    "test_freq",
    "logger",
}


def load_train_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("PyYAML is required to read train configs") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("train config must be a mapping")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(payload))
    if missing:
        raise ValueError(f"train config missing required keys: {missing}")
    if "rollouts_path" not in payload and "sft_turns_path" not in payload and "transitions_path" not in payload:
        raise ValueError("train config must define rollouts_path, sft_turns_path, or transitions_path")
    return payload


def prepare_verl_smoke(
    config_path: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    dry_run: bool,
    execute: bool = False,
    print_command: bool = False,
) -> dict[str, Any]:
    config = load_train_config(config_path)
    _ensure_approved_path(output_dir)
    _ensure_approved_path(checkpoint_dir)

    source_type = _source_type(config)
    source_path = _source_path(config, source_type)
    source_rows = _load_jsonl(source_path)
    train_samples = int(config["train_samples"])
    val_samples = int(config["val_samples"])
    requested = train_samples + val_samples
    if requested > len(source_rows):
        raise ValueError(f"requested {requested} samples but only found {len(source_rows)} {source_type} rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = str(config.get("agent_system_prompt", "")).strip()
    train_rows = _build_verl_rows(source_rows[:train_samples], "train", config, system_prompt=system_prompt)
    val_rows = _build_verl_rows(
        source_rows[train_samples : train_samples + val_samples],
        "val",
        config,
        system_prompt=system_prompt,
    )
    train_jsonl = data_dir / "train.jsonl"
    val_jsonl = data_dir / "val.jsonl"
    train_parquet = data_dir / "train.parquet"
    val_parquet = data_dir / "val.parquet"
    _write_jsonl(train_jsonl, train_rows)
    _write_jsonl(val_jsonl, val_rows)
    train_parquet_written = _write_parquet_if_available(train_parquet, train_rows)
    val_parquet_written = _write_parquet_if_available(val_parquet, val_rows)

    launch_parts = _build_launch_command(config, train_parquet, val_parquet, checkpoint_dir)
    launch_command = _format_shell_command(launch_parts)
    manifest = {
        "config_path": str(config_path),
        "source_type": source_type,
        "source_path": str(source_path),
        "rollouts_path": str(config["rollouts_path"]) if config.get("rollouts_path") else None,
        "sft_turns_path": str(config["sft_turns_path"]) if config.get("sft_turns_path") else None,
        "transitions_path": str(config["transitions_path"]) if config.get("transitions_path") else None,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "train_parquet": str(train_parquet),
        "val_parquet": str(val_parquet),
        "train_parquet_written": train_parquet_written,
        "val_parquet_written": val_parquet_written,
        "reward_dump_path": str(config.get("reward_dump_path", "")) or None,
        "reward_dump_max_chars": int(config.get("reward_dump_max_chars", 2048)),
        "answer_token_f1_threshold": _optional_float(config.get("answer_token_f1_threshold")),
    }
    summary = {
        "dry_run": dry_run,
        "execute": execute,
        "print_command": print_command,
        "experiment_name": config["experiment_name"],
        "project_name": config["project_name"],
        "source_type": source_type,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "parquet_written": train_parquet_written and val_parquet_written,
        "manifest": str(output_dir / "manifest.json"),
        "launch_command_path": str(output_dir / "launch_command.txt"),
        "launch_command": launch_command,
        "runner_command": "bash -lc " + json.dumps(launch_command),
        "reward_dump_path": str(config.get("reward_dump_path", "")) or None,
        "reward_dump_max_chars": int(config.get("reward_dump_max_chars", 2048)),
        "answer_token_f1_threshold": _optional_float(config.get("answer_token_f1_threshold")),
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "launch_command.txt").write_text(launch_command + "\n", encoding="utf-8")
    _write_json(output_dir / "dry_run_summary.json", summary)

    if execute and (not train_parquet_written or (val_rows and not val_parquet_written)):
        raise RuntimeError("parquet files were not written; cannot launch verl training")
    if execute:
        subprocess.run(["bash", "-lc", launch_command], check=True)
    return summary


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


def _build_verl_rows(
    rows: list[dict[str, Any]],
    split: str,
    config: dict[str, Any],
    *,
    system_prompt: str,
) -> list[dict[str, Any]]:
    if config.get("sft_turns_path"):
        stages = _normalize_prompt_stages(config.get("prompt_stages", ["search", "answer"]))
        expanded = []
        for index, row in enumerate(rows):
            expanded.extend(_to_sft_turn_verl_rows(row, split, index, stages=stages))
        return expanded
    if config.get("transitions_path"):
        return [
            _to_transition_verl_row(
                row,
                split,
                index,
                search_reward_top_k=int(config.get("search_reward_top_k", 8) or 8),
            )
            for index, row in enumerate(rows)
        ]
    return [
        _to_rollout_verl_row(
            row,
            split,
            index,
            system_prompt=system_prompt,
            reward_stage=str(config.get("reward_stage", "answer")),
        )
        for index, row in enumerate(rows)
    ]


def _to_rollout_verl_row(
    rollout: dict[str, Any],
    split: str,
    index: int,
    *,
    system_prompt: str = "",
    reward_stage: str = "answer",
) -> dict[str, Any]:
    metadata = rollout.get("metadata", {})
    prompt = str(rollout.get("prompt", ""))
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return {
        "prompt": messages,
        "data_source": "lightningsearch_rl",
        "ability": "search_agent",
        "reward_model": {
            "style": "rule",
            "ground_truth": metadata.get("answer", ""),
            "reward": float(rollout.get("reward", 0.0)),
        },
        "extra_info": {
            "id": rollout.get("id", f"{split}-{index}"),
            "split": split,
            "index": index,
            "answer": metadata.get("answer", ""),
            "search_count": metadata.get("search_count", 0),
            "gold_doc_ids": metadata.get("gold_doc_ids", []),
            "retrieved_doc_ids": metadata.get("retrieved_doc_ids", []),
            "response": rollout.get("response", ""),
            "reward_stage": reward_stage,
        },
    }


def _to_sft_turn_verl_rows(
    row: dict[str, Any],
    split: str,
    index: int,
    *,
    stages: tuple[str, ...],
) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 5:
        raise ValueError(f"SFT-turn row {row.get('id', index)} must contain at least five messages")
    metadata = row.get("metadata", {})
    source_id = str(row.get("id", f"{split}-{index}"))
    answer = str(metadata.get("answer", ""))
    stage_rows = []
    for stage in stages:
        if stage == "search":
            prompt = messages[:2]
            expected_action = str(messages[2].get("content", ""))
            ground_truth = ""
            search_count = int(metadata.get("search_count", 1) or 1)
        elif stage == "answer":
            prompt = messages[:4]
            expected_action = str(messages[4].get("content", ""))
            ground_truth = answer
            search_count = 0
        else:  # pragma: no cover - guarded by _normalize_prompt_stages
            raise ValueError(f"unsupported prompt stage: {stage}")
        stage_rows.append(
            {
                "prompt": prompt,
                "data_source": "lightningsearch_rl",
                "ability": "search_agent",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                    "reward": 0.0,
                },
                "extra_info": {
                    "id": f"{source_id}::{stage}",
                    "source_id": source_id,
                    "split": split,
                    "index": index,
                    "answer": answer,
                    "search_count": search_count,
                    "gold_doc_ids": metadata.get("gold_doc_ids", []),
                    "gold_evidence_doc_ids": metadata.get("gold_evidence_doc_ids", []),
                    "reward_stage": stage,
                    "expected_action": expected_action,
                },
            }
        )
    return stage_rows


def _to_transition_verl_row(
    row: dict[str, Any],
    split: str,
    index: int,
    *,
    search_reward_top_k: int = 8,
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
    source_id = str(row.get("id", row.get("transition_id", f"{split}-{index}")))
    transition_id = str(row.get("transition_id", f"{source_id}:{index}"))
    ground_truth = "" if action_type == "search" else str(metadata.get("gold_answer", ""))
    search_count = 1 if action_type == "search" else 0
    return {
        "prompt": prompt,
        "data_source": "lightningsearch_rl",
        "ability": "search_agent",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
            "reward": float(row.get("reward", 0.0) or 0.0),
        },
        "extra_info": {
            "id": transition_id,
            "source_id": source_id,
            "split": split,
            "index": index,
            "answer": metadata.get("gold_answer", ""),
            "search_count": search_count,
            "gold_doc_ids": row.get("gold_evidence_doc_ids", []),
            "retrieved_doc_ids": row.get("observation_doc_ids", []),
            "candidate_passages": row.get("candidate_passages", []),
            "search_reward_top_k": search_reward_top_k,
            "reward_stage": action_type,
            "expected_action": str(row.get("action", "")),
            "precomputed_step_reward": float(row.get("reward", 0.0) or 0.0),
            "precomputed_total_reward": float(metadata.get("total_reward", 0.0) or 0.0),
        },
    }


def _source_type(config: dict[str, Any]) -> str:
    if config.get("sft_turns_path"):
        return "sft_turns"
    if config.get("transitions_path"):
        return "transitions"
    return "rollouts"


def _source_path(config: dict[str, Any], source_type: str) -> Path:
    if source_type == "sft_turns":
        return Path(str(config["sft_turns_path"]))
    if source_type == "transitions":
        return Path(str(config["transitions_path"]))
    return Path(str(config["rollouts_path"]))


def _normalize_prompt_stages(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        stages = [stage.strip() for stage in value.split(",")]
    else:
        stages = [str(stage).strip() for stage in value]
    normalized = tuple(stage for stage in stages if stage)
    unsupported = sorted(set(normalized) - {"search", "answer"})
    if unsupported:
        raise ValueError(f"unsupported prompt stages: {unsupported}")
    if not normalized:
        raise ValueError("prompt_stages must include at least one stage")
    return normalized


def _write_parquet_if_available(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    try:
        import pandas as pd
    except ImportError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
    except (ImportError, OSError, ValueError):
        return False
    return True


def _build_launch_command(config: dict[str, Any], train_file: Path, val_file: Path, checkpoint_dir: Path) -> list[str]:
    logger = json.dumps(config["logger"])
    env_parts = [
        "HF_HOME=/data/wzl/LightningSearch-RL/.cache/huggingface",
        "HF_ENDPOINT=https://hf-mirror.com",
        "PYTHONNOUSERSITE=1",
    ]
    reward_dump_path = str(config.get("reward_dump_path", "")).strip()
    if reward_dump_path:
        env_parts.append(f"LIGHTNINGSEARCH_REWARD_DUMP_PATH={reward_dump_path}")
        env_parts.append(f"LIGHTNINGSEARCH_REWARD_DUMP_MAX_CHARS={int(config.get('reward_dump_max_chars', 2048))}")
    answer_token_f1_threshold = _optional_float(config.get("answer_token_f1_threshold"))
    if answer_token_f1_threshold is not None:
        env_parts.append(f"LIGHTNINGSEARCH_ANSWER_TOKEN_F1_THRESHOLD={answer_token_f1_threshold:g}")
    command = [
        *env_parts,
        "python",
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={config['train_batch_size']}",
        f"data.max_prompt_length={config['max_prompt_length']}",
        f"data.max_response_length={config['max_response_length']}",
        f"actor_rollout_ref.model.path={config['model_path']}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.n={config.get('rollout_n', 1)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.get('rollout_gpu_memory_utilization', 0.18)}",
        f"actor_rollout_ref.rollout.max_model_len={config.get('rollout_max_model_len', 768)}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={config.get('rollout_max_num_batched_tokens', 1024)}",
        f"actor_rollout_ref.rollout.max_num_seqs={config.get('rollout_max_num_seqs', 4)}",
        f"actor_rollout_ref.rollout.enforce_eager={_bool_override(config.get('rollout_enforce_eager', True))}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={_bool_override(config.get('rollout_enable_chunked_prefill', False))}",
        f"actor_rollout_ref.rollout.enable_prefix_caching={_bool_override(config.get('rollout_enable_prefix_caching', False))}",
        f"actor_rollout_ref.rollout.agent.num_workers={config.get('rollout_agent_num_workers', 4)}",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="
        f"{config.get('rollout_update_weights_bucket_megabytes', 512)}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config['ppo_micro_batch_size_per_gpu']}",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={config.get('actor_model_dtype', 'bfloat16')}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={_bool_override(config.get('actor_param_offload', True))}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={_bool_override(config.get('actor_optimizer_offload', True))}",
        f"algorithm.adv_estimator={config.get('adv_estimator', 'grpo')}",
        "reward.custom_reward_function.path=src/lightningsearch_rl/verl_reward.py",
        "reward.custom_reward_function.name=compute_score",
        f"trainer.project_name={config['project_name']}",
        f"trainer.experiment_name={config['experiment_name']}",
        f"trainer.total_training_steps={config['total_training_steps']}",
        f"trainer.total_epochs={config.get('total_epochs', 1)}",
        f"trainer.n_gpus_per_node={config['n_gpus_per_node']}",
        "trainer.nnodes=1",
        f"trainer.save_freq={config['save_freq']}",
        f"trainer.test_freq={config['test_freq']}",
        f"trainer.logger={logger}",
        f"trainer.default_local_dir={checkpoint_dir}",
    ]
    if config.get("gdpo_reward_keys"):
        command.append(f"+algorithm.gdpo_reward_keys={json.dumps(config['gdpo_reward_keys'])}")
    if config.get("reward_manager_name"):
        command.append(f"reward.reward_manager.name={config['reward_manager_name']}")
    _append_optional_rollout_override(command, config, "rollout_temperature", "temperature")
    _append_optional_rollout_override(command, config, "rollout_top_p", "top_p")
    _append_optional_rollout_override(command, config, "rollout_top_k", "top_k")
    return command


def _format_shell_command(parts: list[str]) -> str:
    python_index = parts.index("python")
    prefix = parts[: python_index + 3]
    overrides = parts[python_index + 3 :]
    return " ".join(prefix + [_single_quote(part) for part in overrides])


def _bool_override(value: Any) -> str:
    return "True" if bool(value) else "False"


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _append_optional_rollout_override(command: list[str], config: dict[str, Any], config_key: str, verl_key: str) -> None:
    if config_key not in config or str(config[config_key]).strip() == "":
        return
    command.append(f"actor_rollout_ref.rollout.{verl_key}={config[config_key]}")


def _single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
