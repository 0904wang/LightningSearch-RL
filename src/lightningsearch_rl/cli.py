from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lightningsearch_rl.adapters import convert_2wiki_file, convert_hotpot_file
from lightningsearch_rl.baseline import run_retrieval_baseline
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.eval import evaluate_traces
from lightningsearch_rl.environment_rollout import run_environment_rollout
from lightningsearch_rl.generation_inspection import prepare_generation_inspection
from lightningsearch_rl.diagnostics import diagnose_dataset
from lightningsearch_rl.env_transitions import export_env_rollout_transitions
from lightningsearch_rl.grpo import export_grpo
from lightningsearch_rl.index_store import load_lexical_index, save_lexical_index
from lightningsearch_rl.policy_movement import diagnose_policy_movement
from lightningsearch_rl.preference_pairs import build_preference_pairs
from lightningsearch_rl.reward_probe import run_reward_probe
from lightningsearch_rl.rewards import compute_reward
from lightningsearch_rl.retrieval_eval import evaluate_retrieval
from lightningsearch_rl.rollout_diagnostics import diagnose_rollout_answers
from lightningsearch_rl.runtime import run_rule_based_episode
from lightningsearch_rl.sft import export_sft
from lightningsearch_rl.sft_turns import export_sft_turns
from lightningsearch_rl.sft_warmup import export_sft_warmup
from lightningsearch_rl.reward_variance_filter import filter_transitions_by_reward_variance
from lightningsearch_rl.synthesis import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekClient,
    mock_synthetic_row,
    synthesize_file,
    synthesize_validated_file,
    validate_synthetic_file,
)
from lightningsearch_rl.transitions import build_transitions
from lightningsearch_rl.verl_sft_warmup import prepare_verl_sft_warmup
from lightningsearch_rl.verl_log_parser import write_verl_log_summary
from lightningsearch_rl.verl_batch_diagnostics import write_verl_batch_diagnostics
from lightningsearch_rl.verl_reward_dump_diagnostics import write_reward_dump_diagnostics
from lightningsearch_rl.verl_smoke import prepare_verl_smoke


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lightningsearch-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--data", required=True)
    smoke.add_argument("--out-dir", required=True)
    smoke.add_argument("--top-k", type=int, default=2)
    prepare_hotpot = subparsers.add_parser("prepare-hotpot")
    prepare_hotpot.add_argument("--raw", required=True)
    prepare_hotpot.add_argument("--corpus", required=True)
    prepare_hotpot.add_argument("--examples", required=True)
    prepare_hotpot.add_argument("--limit", type=int, default=None)
    prepare_2wiki = subparsers.add_parser("prepare-2wiki")
    prepare_2wiki.add_argument("--raw", required=True)
    prepare_2wiki.add_argument("--corpus", required=True)
    prepare_2wiki.add_argument("--examples", required=True)
    prepare_2wiki.add_argument("--limit", type=int, default=None)
    build_index = subparsers.add_parser("build-index")
    build_index.add_argument("--corpus", required=True)
    build_index.add_argument("--index", required=True)
    eval_retrieval = subparsers.add_parser("eval-retrieval")
    eval_retrieval.add_argument("--examples", required=True)
    eval_retrieval.add_argument("--index", required=True)
    eval_retrieval.add_argument("--out", required=True)
    eval_retrieval.add_argument("--top-k", type=int, default=5)
    retrieval_baseline = subparsers.add_parser("retrieval-baseline")
    retrieval_baseline.add_argument("--dataset", required=True)
    retrieval_baseline.add_argument("--examples", required=True)
    retrieval_baseline.add_argument("--index", required=True)
    retrieval_baseline.add_argument("--report", required=True)
    retrieval_baseline.add_argument("--top-k", type=int, default=5)
    export_sft_parser = subparsers.add_parser("export-sft")
    export_sft_parser.add_argument("--examples", required=True)
    export_sft_parser.add_argument("--index", required=True)
    export_sft_parser.add_argument("--out-dir", required=True)
    export_sft_parser.add_argument("--top-k", type=int, default=5)
    export_sft_warmup_parser = subparsers.add_parser("export-sft-warmup")
    export_sft_warmup_parser.add_argument("--examples", required=True)
    export_sft_warmup_parser.add_argument("--index", required=True)
    export_sft_warmup_parser.add_argument("--out-dir", required=True)
    export_sft_turns_parser = subparsers.add_parser("export-sft-turns")
    export_sft_turns_parser.add_argument("--examples", required=True)
    export_sft_turns_parser.add_argument("--index", required=True)
    export_sft_turns_parser.add_argument("--out-dir", required=True)
    export_grpo_parser = subparsers.add_parser("export-grpo")
    export_grpo_parser.add_argument("--examples", required=True)
    export_grpo_parser.add_argument("--index", required=True)
    export_grpo_parser.add_argument("--out-dir", required=True)
    export_grpo_parser.add_argument("--top-k", type=int, default=5)
    diagnose_data = subparsers.add_parser("diagnose-data")
    diagnose_data.add_argument("--valid", required=True)
    diagnose_data.add_argument("--grpo-dir", required=True)
    diagnose_data.add_argument("--out", required=True)
    diagnose_rollout = subparsers.add_parser("diagnose-rollout-answers")
    diagnose_rollout.add_argument("--rollouts", required=True)
    diagnose_rollout.add_argument("--out", required=True)
    export_env_transitions = subparsers.add_parser("export-env-transitions")
    export_env_transitions.add_argument("--rollouts", required=True)
    export_env_transitions.add_argument("--out-dir", required=True)
    export_env_transitions.add_argument("--index", default=None)
    export_env_transitions.add_argument("--quality-manifest", default=None)
    export_env_transitions.add_argument("--exclude-quality-flag", action="append", default=[])
    parse_verl_log = subparsers.add_parser("parse-verl-log")
    parse_verl_log.add_argument("--log", required=True)
    parse_verl_log.add_argument("--out", required=True)
    diagnose_verl_batches = subparsers.add_parser("diagnose-verl-batches")
    diagnose_verl_batches.add_argument("--train-jsonl", required=True)
    diagnose_verl_batches.add_argument("--metrics-summary", required=True)
    diagnose_verl_batches.add_argument("--train-batch-size", type=int, required=True)
    diagnose_verl_batches.add_argument("--out", required=True)
    diagnose_reward_dump = subparsers.add_parser("diagnose-reward-dump")
    diagnose_reward_dump.add_argument("--dump", required=True)
    diagnose_reward_dump.add_argument("--out", required=True)
    diagnose_reward_dump.add_argument("--low-score-threshold", type=float, default=0.5)
    filter_variance = subparsers.add_parser("filter-transitions-by-reward-variance")
    filter_variance.add_argument("--transitions", required=True)
    filter_variance.add_argument("--reward-dump", required=True)
    filter_variance.add_argument("--out-dir", required=True)
    filter_variance.add_argument("--stage", action="append", default=[])
    filter_variance.add_argument("--min-score-range", type=float, default=1e-9)
    filter_variance.add_argument("--min-samples", type=int, default=2)
    filter_variance.add_argument("--max-source-count", type=int, default=None)
    preference_pairs = subparsers.add_parser("build-preference-pairs")
    preference_pairs.add_argument("--probe-requests", required=True)
    preference_pairs.add_argument("--generations", required=True)
    preference_pairs.add_argument("--reward-dump", required=True)
    preference_pairs.add_argument("--out-dir", required=True)
    preference_pairs.add_argument("--stage", action="append", default=[])
    preference_pairs.add_argument("--min-score-gap", type=float, default=0.25)
    preference_pairs.add_argument("--min-samples", type=int, default=2)
    preference_pairs.add_argument("--max-pairs-per-group", type=int, default=1)
    preference_pairs.add_argument("--max-search-pairs", type=int, default=None)
    preference_pairs.add_argument("--max-answer-pairs", type=int, default=None)
    preference_pairs.add_argument("--val-fraction", type=float, default=0.05)
    preference_pairs.add_argument("--seed", type=int, default=0)
    reward_probe = subparsers.add_parser("probe-reward-variance")
    reward_probe.add_argument("--transitions", required=True)
    reward_probe.add_argument("--model", required=True)
    reward_probe.add_argument("--out-dir", required=True)
    reward_probe.add_argument("--offset", type=int, default=0)
    reward_probe.add_argument("--limit", type=int, default=None)
    reward_probe.add_argument("--samples-per-prompt", type=int, default=4)
    reward_probe.add_argument("--max-new-tokens", type=int, default=64)
    reward_probe.add_argument("--search-reward-top-k", type=int, default=8)
    reward_probe.add_argument("--answer-token-f1-threshold", type=float, default=None)
    reward_probe.add_argument("--backend", choices=["vllm"], default="vllm")
    reward_probe.add_argument("--batch-size", type=int, default=64)
    reward_probe.add_argument("--temperature", type=float, default=1.2)
    reward_probe.add_argument("--top-p", type=float, default=0.95)
    reward_probe.add_argument("--top-k", type=int, default=50)
    reward_probe.add_argument("--seed", type=int, default=None)
    reward_probe.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    reward_probe.add_argument("--max-model-len", type=int, default=768)
    reward_probe.add_argument("--tensor-parallel-size", type=int, default=1)
    reward_probe.add_argument("--dry-run", action="store_true")
    diagnose_policy = subparsers.add_parser("diagnose-policy-movement")
    diagnose_policy.add_argument("--base-model", required=True)
    diagnose_policy.add_argument("--candidate-model", required=True)
    diagnose_policy.add_argument("--sft", required=True)
    diagnose_policy.add_argument("--out-dir", required=True)
    diagnose_policy.add_argument("--offset", type=int, default=0)
    diagnose_policy.add_argument("--limit", type=int, default=5)
    diagnose_policy.add_argument("--device", default="cuda")
    diagnose_policy.add_argument("--dtype", default="bfloat16")
    diagnose_policy.add_argument("--top-k-tensors", type=int, default=20)
    diagnose_policy.add_argument("--skip-logprobs", action="store_true")
    diagnose_policy.add_argument("--skip-params", action="store_true")
    diagnose_policy.add_argument("--dry-run", action="store_true")
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--checkpoint-dir", required=True)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--print-command", action="store_true")
    train_sft_warmup = subparsers.add_parser("train-sft-warmup")
    train_sft_warmup.add_argument("--config", required=True)
    train_sft_warmup.add_argument("--output-dir", required=True)
    train_sft_warmup.add_argument("--checkpoint-dir", required=True)
    train_sft_warmup.add_argument("--dry-run", action="store_true")
    train_sft_warmup.add_argument("--print-command", action="store_true")
    inspect_generation = subparsers.add_parser("inspect-generation")
    inspect_generation.add_argument("--sft", required=True)
    inspect_generation.add_argument("--model", required=True)
    inspect_generation.add_argument("--out-dir", required=True)
    inspect_generation.add_argument("--offset", type=int, default=480)
    inspect_generation.add_argument("--limit", type=int, default=5)
    inspect_generation.add_argument("--max-new-tokens", type=int, default=64)
    inspect_generation.add_argument("--modes", default="search,answer")
    inspect_generation.add_argument("--dry-run", action="store_true")
    inspect_env_rollout = subparsers.add_parser("inspect-env-rollout")
    inspect_env_rollout.add_argument("--sft", required=True)
    inspect_env_rollout.add_argument("--index", required=True)
    inspect_env_rollout.add_argument("--model", required=True)
    inspect_env_rollout.add_argument("--out-dir", required=True)
    inspect_env_rollout.add_argument("--offset", type=int, default=0)
    inspect_env_rollout.add_argument("--limit", type=int, default=5)
    inspect_env_rollout.add_argument("--top-k", type=int, default=2)
    inspect_env_rollout.add_argument("--max-new-tokens", type=int, default=64)
    inspect_env_rollout.add_argument("--candidate-pool", choices=["global", "gold-distractors"], default="global")
    inspect_env_rollout.add_argument("--distractor-count", type=int, default=0)
    inspect_env_rollout.add_argument("--do-sample", action="store_true")
    inspect_env_rollout.add_argument("--temperature", type=float, default=1.0)
    inspect_env_rollout.add_argument("--top-p", type=float, default=1.0)
    inspect_env_rollout.add_argument("--sample-top-k", type=int, default=None)
    inspect_env_rollout.add_argument("--seed", type=int, default=None)
    inspect_env_rollout.add_argument("--dry-run", action="store_true")
    synthesize_data = subparsers.add_parser("synthesize-data")
    synthesize_data.add_argument("--out", required=True)
    synthesize_data.add_argument("--count", type=int, required=True)
    synthesize_data.add_argument("--topics", required=True)
    synthesize_data.add_argument("--concurrency", type=int, default=50)
    synthesize_data.add_argument("--seed", type=int, default=0)
    synthesize_data.add_argument("--summary", required=True)
    synthesize_data.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL)
    synthesize_data.add_argument("--base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    synthesize_data.add_argument("--temperature", type=float, default=0.8)
    synthesize_data.add_argument("--max-tokens", type=int, default=1200)
    synthesize_data.add_argument("--retries", type=int, default=3)
    synthesize_data.add_argument("--mock", action="store_true")
    validate_synthetic = subparsers.add_parser("validate-synthetic")
    validate_synthetic.add_argument("--raw", required=True)
    validate_synthetic.add_argument("--valid", required=True)
    validate_synthetic.add_argument("--rejects", required=True)
    validate_synthetic.add_argument("--summary", required=True)
    validate_synthetic.add_argument("--require-chain-schema", action="store_true")
    synthesize_validated = subparsers.add_parser("synthesize-validated-data")
    synthesize_validated.add_argument("--raw", required=True)
    synthesize_validated.add_argument("--valid", required=True)
    synthesize_validated.add_argument("--rejects", required=True)
    synthesize_validated.add_argument("--target-valid", type=int, required=True)
    synthesize_validated.add_argument("--topics", required=True)
    synthesize_validated.add_argument("--concurrency", type=int, default=50)
    synthesize_validated.add_argument("--batch-size", type=int, default=None)
    synthesize_validated.add_argument("--max-attempts", type=int, default=None)
    synthesize_validated.add_argument("--seed", type=int, default=0)
    synthesize_validated.add_argument("--summary", required=True)
    synthesize_validated.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL)
    synthesize_validated.add_argument("--base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    synthesize_validated.add_argument("--temperature", type=float, default=0.8)
    synthesize_validated.add_argument("--max-tokens", type=int, default=1200)
    synthesize_validated.add_argument("--retries", type=int, default=3)
    synthesize_validated.add_argument("--mock", action="store_true")
    synthesize_validated.add_argument("--require-chain-schema", action="store_true")
    synthesize_validated.add_argument("--repair-chain-schema", action="store_true")
    synthesize_validated.add_argument("--few-shot-chain-schema", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        return _run_smoke(Path(args.data), Path(args.out_dir), args.top_k)
    if args.command == "prepare-hotpot":
        convert_hotpot_file(
            Path(args.raw),
            Path(args.corpus),
            Path(args.examples),
            limit=args.limit,
        )
        return 0
    if args.command == "prepare-2wiki":
        convert_2wiki_file(
            Path(args.raw),
            Path(args.corpus),
            Path(args.examples),
            limit=args.limit,
        )
        return 0
    if args.command == "build-index":
        save_lexical_index(Path(args.index), load_corpus_jsonl(Path(args.corpus)))
        return 0
    if args.command == "eval-retrieval":
        metrics = evaluate_retrieval(
            load_jsonl_examples(Path(args.examples)),
            load_lexical_index(Path(args.index)),
            args.top_k,
        )
        _write_json(Path(args.out), metrics)
        return 0
    if args.command == "retrieval-baseline":
        run_retrieval_baseline(
            args.dataset,
            Path(args.examples),
            Path(args.index),
            Path(args.report),
            args.top_k,
        )
        return 0
    if args.command == "export-sft":
        export_sft(
            Path(args.examples),
            Path(args.index),
            Path(args.out_dir),
            top_k=args.top_k,
        )
        return 0
    if args.command == "export-sft-warmup":
        export_sft_warmup(
            Path(args.examples),
            Path(args.index),
            Path(args.out_dir),
        )
        return 0
    if args.command == "export-sft-turns":
        export_sft_turns(
            Path(args.examples),
            Path(args.index),
            Path(args.out_dir),
        )
        return 0
    if args.command == "export-grpo":
        export_grpo(
            Path(args.examples),
            Path(args.index),
            Path(args.out_dir),
            top_k=args.top_k,
        )
        return 0
    if args.command == "diagnose-data":
        report = diagnose_dataset(Path(args.valid), Path(args.grpo_dir))
        _write_json(Path(args.out), report)
        return 0
    if args.command == "diagnose-rollout-answers":
        report = diagnose_rollout_answers(Path(args.rollouts))
        _write_json(Path(args.out), report)
        return 0
    if args.command == "export-env-transitions":
        export_env_rollout_transitions(
            Path(args.rollouts),
            Path(args.out_dir),
            quality_manifest_path=Path(args.quality_manifest) if args.quality_manifest else None,
            exclude_quality_flags=set(args.exclude_quality_flag),
            index_path=Path(args.index) if args.index else None,
        )
        return 0
    if args.command == "parse-verl-log":
        write_verl_log_summary(Path(args.log), Path(args.out))
        return 0
    if args.command == "diagnose-verl-batches":
        write_verl_batch_diagnostics(
            Path(args.train_jsonl),
            Path(args.out),
            metrics_summary_path=Path(args.metrics_summary),
            train_batch_size=args.train_batch_size,
        )
        return 0
    if args.command == "diagnose-reward-dump":
        write_reward_dump_diagnostics(
            Path(args.dump),
            Path(args.out),
            low_score_threshold=args.low_score_threshold,
        )
        return 0
    if args.command == "filter-transitions-by-reward-variance":
        filter_transitions_by_reward_variance(
            transitions_path=Path(args.transitions),
            reward_dump_path=Path(args.reward_dump),
            out_dir=Path(args.out_dir),
            stages=tuple(args.stage) if args.stage else (),
            min_score_range=args.min_score_range,
            min_samples=args.min_samples,
            max_source_count=args.max_source_count,
        )
        return 0
    if args.command == "build-preference-pairs":
        build_preference_pairs(
            probe_requests_path=Path(args.probe_requests),
            generations_path=Path(args.generations),
            reward_dump_path=Path(args.reward_dump),
            out_dir=Path(args.out_dir),
            stages=tuple(args.stage) if args.stage else (),
            min_score_gap=args.min_score_gap,
            min_samples=args.min_samples,
            max_pairs_per_group=args.max_pairs_per_group,
            max_search_pairs=args.max_search_pairs,
            max_answer_pairs=args.max_answer_pairs,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        return 0
    if args.command == "probe-reward-variance":
        run_reward_probe(
            transitions_path=Path(args.transitions),
            model_path=args.model,
            out_dir=Path(args.out_dir),
            offset=args.offset,
            limit=args.limit,
            samples_per_prompt=args.samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            search_reward_top_k=args.search_reward_top_k,
            answer_token_f1_threshold=args.answer_token_f1_threshold,
            backend=args.backend,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            dry_run=args.dry_run,
        )
        return 0
    if args.command == "diagnose-policy-movement":
        diagnose_policy_movement(
            base_model=Path(args.base_model),
            candidate_model=Path(args.candidate_model),
            sft_path=Path(args.sft),
            out_dir=Path(args.out_dir),
            offset=args.offset,
            limit=args.limit,
            device=args.device,
            dtype=args.dtype,
            top_k_tensors=args.top_k_tensors,
            skip_logprobs=args.skip_logprobs,
            skip_params=args.skip_params,
            dry_run=args.dry_run,
        )
        return 0
    if args.command == "train":
        summary = prepare_verl_smoke(
            Path(args.config),
            Path(args.output_dir),
            Path(args.checkpoint_dir),
            dry_run=args.dry_run,
            execute=not args.dry_run,
            print_command=args.print_command,
        )
        if args.print_command:
            print(summary["launch_command"])
        return 0
    if args.command == "train-sft-warmup":
        summary = prepare_verl_sft_warmup(
            Path(args.config),
            Path(args.output_dir),
            Path(args.checkpoint_dir),
            dry_run=args.dry_run,
            execute=not args.dry_run,
            print_command=args.print_command,
        )
        if args.print_command:
            print(summary["launch_command"])
        return 0
    if args.command == "inspect-generation":
        prepare_generation_inspection(
            sft_path=Path(args.sft),
            model_path=args.model,
            out_dir=Path(args.out_dir),
            offset=args.offset,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            dry_run=args.dry_run,
            modes=[mode.strip() for mode in args.modes.split(",")],
        )
        return 0
    if args.command == "inspect-env-rollout":
        run_environment_rollout(
            sft_path=Path(args.sft),
            index_path=Path(args.index),
            model_path=args.model,
            out_dir=Path(args.out_dir),
            offset=args.offset,
            limit=args.limit,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            candidate_pool=args.candidate_pool,
            distractor_count=args.distractor_count,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            sample_top_k=args.sample_top_k,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        return 0
    if args.command == "synthesize-data":
        topics = _parse_topics(args.topics)
        client = (
            _MockSynthesisClient()
            if args.mock
            else DeepSeekClient(base_url=args.base_url, model=args.model)
        )
        summary = synthesize_file(
            Path(args.out),
            count=args.count,
            topics=topics,
            client=client,
            concurrency=args.concurrency,
            seed=args.seed,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
        _write_json(Path(args.summary), summary)
        return 0
    if args.command == "validate-synthetic":
        summary = validate_synthetic_file(
            Path(args.raw),
            Path(args.valid),
            Path(args.rejects),
            require_chain_schema=args.require_chain_schema,
        )
        _write_json(Path(args.summary), summary)
        return 0
    if args.command == "synthesize-validated-data":
        topics = _parse_topics(args.topics)
        client = (
            _MockSynthesisClient()
            if args.mock
            else DeepSeekClient(base_url=args.base_url, model=args.model)
        )
        summary = synthesize_validated_file(
            Path(args.raw),
            Path(args.valid),
            Path(args.rejects),
            target_valid=args.target_valid,
            topics=topics,
            client=client,
            concurrency=args.concurrency,
            seed=args.seed,
            batch_size=args.batch_size,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            require_chain_schema=args.require_chain_schema,
            repair_chain_schema=args.repair_chain_schema,
            few_shot_chain_schema=args.few_shot_chain_schema,
        )
        _write_json(Path(args.summary), summary)
        return 0
    raise ValueError(f"unknown command: {args.command}")


def _run_smoke(data_path: Path, out_dir: Path, top_k: int) -> int:
    examples = load_jsonl_examples(data_path)
    traces = []
    transitions = []
    for example in examples:
        trace = run_rule_based_episode(example, top_k=top_k)
        reward = compute_reward(example, trace)
        trace = trace.__class__(
            question_id=trace.question_id,
            question=trace.question,
            steps=trace.steps,
            final_answer=trace.final_answer,
            reward=reward.total,
            metadata={**trace.metadata, "reward": reward.total},
        )
        traces.append(trace)
        transitions.extend(build_transitions(trace))
    metrics = evaluate_traces(examples, traces)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "traces.jsonl", [trace.to_dict() for trace in traces])
    _write_jsonl(
        out_dir / "transitions.jsonl",
        [transition.to_dict() for transition in transitions],
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_topics(value: str) -> list[str]:
    topics = [topic.strip() for topic in value.split(",") if topic.strip()]
    if not topics:
        raise ValueError("--topics must contain at least one non-empty topic")
    return topics


class _MockSynthesisClient:
    def complete_json(self, messages, temperature, max_tokens):
        content = messages[-1]["content"]
        request_id = _extract_prompt_field(content, "Use id ", ".")
        topic = _extract_prompt_field(content, "Topic: ", ". The json object")
        return mock_synthetic_row(request_id, topic)


def _extract_prompt_field(content: str, prefix: str, suffix: str) -> str:
    start = content.index(prefix) + len(prefix)
    end = content.index(suffix, start)
    return content[start:end].strip()


if __name__ == "__main__":
    raise SystemExit(main())
