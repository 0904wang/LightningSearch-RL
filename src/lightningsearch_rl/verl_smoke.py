from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


LOCAL_ROOT_MARKER = "Agent RL"
REMOTE_ROOTS = ("/data/wzl/LightningSearch-RL", "/home/user/wzl/LightningSearch-RL")
REQUIRED_CONFIG_KEYS = {
    "experiment_name",
    "project_name",
    "rollouts_path",
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

    rollouts_path = Path(str(config["rollouts_path"]))
    rollouts = _load_jsonl(rollouts_path)
    train_samples = int(config["train_samples"])
    val_samples = int(config["val_samples"])
    requested = train_samples + val_samples
    if requested > len(rollouts):
        raise ValueError(f"requested {requested} samples but only found {len(rollouts)} rollouts")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [_to_verl_row(row, "train", index) for index, row in enumerate(rollouts[:train_samples])]
    val_rows = [
        _to_verl_row(row, "val", index)
        for index, row in enumerate(rollouts[train_samples : train_samples + val_samples])
    ]
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
        "rollouts_path": str(rollouts_path),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "train_parquet": str(train_parquet),
        "val_parquet": str(val_parquet),
        "train_parquet_written": train_parquet_written,
        "val_parquet_written": val_parquet_written,
    }
    summary = {
        "dry_run": dry_run,
        "execute": execute,
        "print_command": print_command,
        "experiment_name": config["experiment_name"],
        "project_name": config["project_name"],
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "parquet_written": train_parquet_written and val_parquet_written,
        "manifest": str(output_dir / "manifest.json"),
        "launch_command_path": str(output_dir / "launch_command.txt"),
        "launch_command": launch_command,
        "runner_command": "bash -lc " + json.dumps(launch_command),
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


def _to_verl_row(rollout: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    metadata = rollout.get("metadata", {})
    prompt = str(rollout.get("prompt", ""))
    return {
        "prompt": [{"role": "user", "content": prompt}],
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
        },
    }


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
    return [
        "PYTHONNOUSERSITE=1",
        "python",
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={config['train_batch_size']}",
        f"data.max_prompt_length={config['max_prompt_length']}",
        f"data.max_response_length={config['max_response_length']}",
        f"actor_rollout_ref.model.path={config['model_path']}",
        "actor_rollout_ref.rollout.name=hf",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.25",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config['ppo_micro_batch_size_per_gpu']}",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048",
        "algorithm.adv_estimator=grpo",
        f"trainer.project_name={config['project_name']}",
        f"trainer.experiment_name={config['experiment_name']}",
        f"trainer.total_training_steps={config['total_training_steps']}",
        "trainer.total_epochs=1",
        f"trainer.n_gpus_per_node={config['n_gpus_per_node']}",
        "trainer.nnodes=1",
        f"trainer.save_freq={config['save_freq']}",
        f"trainer.test_freq={config['test_freq']}",
        f"trainer.logger={logger}",
        f"trainer.default_local_dir={checkpoint_dir}",
    ]


def _format_shell_command(parts: list[str]) -> str:
    prefix = parts[:4]
    overrides = parts[4:]
    return " ".join(prefix + [shlex.quote(part) for part in overrides])
