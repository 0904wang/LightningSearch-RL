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
    "sft_path",
    "train_samples",
    "val_samples",
    "seed",
    "model_path",
    "max_length",
    "train_batch_size",
    "micro_batch_size_per_gpu",
    "max_token_len_per_gpu",
    "learning_rate",
    "n_gpus_per_node",
    "total_training_steps",
    "total_epochs",
    "save_freq",
    "test_freq",
    "logger",
}


def load_sft_warmup_train_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("PyYAML is required to read SFT warmup configs") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SFT warmup config must be a mapping")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(payload))
    if missing:
        raise ValueError(f"SFT warmup config missing required keys: {missing}")
    return payload


def prepare_verl_sft_warmup(
    config_path: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    dry_run: bool,
    execute: bool = False,
    print_command: bool = False,
) -> dict[str, Any]:
    config = load_sft_warmup_train_config(config_path)
    _ensure_approved_path(output_dir)
    _ensure_approved_path(checkpoint_dir)

    sft_path = Path(str(config["sft_path"]))
    rows = _load_jsonl(sft_path)
    train_samples = int(config["train_samples"])
    val_samples = int(config["val_samples"])
    requested = train_samples + val_samples
    if requested > len(rows):
        raise ValueError(f"requested {requested} samples but only found {len(rows)} SFT rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [_to_sft_row(row, "train", index) for index, row in enumerate(rows[:train_samples])]
    val_rows = [
        _to_sft_row(row, "val", index)
        for index, row in enumerate(rows[train_samples : train_samples + val_samples])
    ]
    train_jsonl = data_dir / "train.jsonl"
    val_jsonl = data_dir / "val.jsonl"
    train_parquet = data_dir / "train.parquet"
    val_parquet = data_dir / "val.parquet"
    _write_jsonl(train_jsonl, train_rows)
    _write_jsonl(val_jsonl, val_rows)
    train_parquet_written = _write_parquet_if_available(train_parquet, train_rows)
    val_parquet_written = _write_parquet_if_available(val_parquet, val_rows) if val_rows else True

    launch_parts = _build_launch_command(config, train_parquet, val_parquet, checkpoint_dir, has_val=bool(val_rows))
    launch_command = _format_shell_command(launch_parts)
    manifest = {
        "config_path": str(config_path),
        "sft_path": str(sft_path),
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

    if execute and not (train_parquet_written and val_parquet_written):
        raise RuntimeError("parquet files were not written; cannot launch verl SFT training")
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


def _to_sft_row(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"SFT row {index} is missing messages")
    assistant_content = str(messages[-1].get("content", "")) if isinstance(messages[-1], dict) else ""
    if "<answer>" not in assistant_content:
        raise ValueError(f"SFT row {index} is missing an <answer> tag")
    return {
        "id": row.get("id", f"{split}-{index}"),
        "messages": messages,
        "enable_thinking": bool(row.get("enable_thinking", False)),
        "data_source": "lightningsearch_rl_sft_warmup",
        "extra_info": {
            "split": split,
            "index": index,
            "metadata": row.get("metadata", {}),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _build_launch_command(
    config: dict[str, Any],
    train_file: Path,
    val_file: Path,
    checkpoint_dir: Path,
    *,
    has_val: bool,
) -> list[str]:
    logger = json.dumps(config["logger"])
    parts = [
        "HF_HOME=/data/wzl/LightningSearch-RL/.cache/huggingface",
        "HF_ENDPOINT=https://hf-mirror.com",
        "HYDRA_FULL_ERROR=1",
        "PYTHONNOUSERSITE=1",
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={config['n_gpus_per_node']}",
        "-m",
        "verl.trainer.sft_trainer",
        f"data.train_files={train_file}",
        f"data.val_files={val_file if has_val else 'null'}",
        f"data.train_batch_size={config['train_batch_size']}",
        f"data.micro_batch_size_per_gpu={config['micro_batch_size_per_gpu']}",
        f"data.max_token_len_per_gpu={config['max_token_len_per_gpu']}",
        f"data.max_length={config['max_length']}",
        "data.messages_key=messages",
        "data.enable_thinking_key=enable_thinking",
        "data.enable_thinking_default=False",
        "data.pad_mode=no_padding",
        "data.truncation=error",
        "data.ignore_input_ids_mismatch=True",
        f"data.num_workers={config.get('num_workers', 0)}",
        f"model.path={config['model_path']}",
        f"model.trust_remote_code={_bool_override(config.get('trust_remote_code', False))}",
        "model.enable_gradient_checkpointing=True",
        "model.use_remove_padding=True",
        f"engine.param_offload={_bool_override(config.get('param_offload', True))}",
        f"engine.optimizer_offload={_bool_override(config.get('optimizer_offload', True))}",
        f"engine.model_dtype={config.get('model_dtype', 'bfloat16')}",
        f"engine.dtype={config.get('engine_dtype', 'bfloat16')}",
        f"optim.lr={config['learning_rate']}",
        f"optim.lr_warmup_steps={config.get('lr_warmup_steps', 0)}",
        f"trainer.project_name={config['project_name']}",
        f"trainer.experiment_name={config['experiment_name']}",
        f"trainer.total_training_steps={config['total_training_steps']}",
        f"trainer.total_epochs={config['total_epochs']}",
        f"trainer.n_gpus_per_node={config['n_gpus_per_node']}",
        "trainer.nnodes=1",
        f"trainer.save_freq={config['save_freq']}",
        f"trainer.test_freq={config['test_freq']}",
        f"trainer.logger={logger}",
        f"trainer.default_local_dir={checkpoint_dir}",
        "trainer.resume_mode=disable",
    ]
    cuda_visible_devices = config.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        parts.insert(0, f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    return parts


def _format_shell_command(parts: list[str]) -> str:
    prefix_end = parts.index("verl.trainer.sft_trainer") + 1
    prefix = parts[:prefix_end]
    overrides = parts[prefix_end:]
    return " ".join(prefix + [_single_quote(part) for part in overrides])


def _bool_override(value: Any) -> str:
    return "True" if bool(value) else "False"


def _single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
