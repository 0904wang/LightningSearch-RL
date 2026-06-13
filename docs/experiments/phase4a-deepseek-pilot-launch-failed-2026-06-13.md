# Phase 4A DeepSeek Pilot Launch Failed

Date: 2026-06-13

## Goal

Start the first real DeepSeek synthetic data pilot for 50 HotpotQA-like
multi-hop QA rows with concurrency 50, then validate and export GRPO artifacts.

## Code and Environment

- Local commit: `39e65d3`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- tmux session: `lightningsearch-20260613-phase4a-deepseek-pilot`
- Log path: `/data/wzl/LightningSearch-RL/logs/phase4a-deepseek-pilot.log`
- Results path: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot`
- GPU: not used

## Launch Command Shape

The launched tmux command activated the approved conda environment and attempted
to run:

```bash
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-data \
  --out /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthetic_raw.jsonl \
  --count 50 \
  --topics awards,archives,research \
  --concurrency 50 \
  --seed 1000 \
  --summary /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthesis_summary.json \
  --model deepseek-chat \
  --base-url https://api.deepseek.com
```

## Result

The tmux session exited immediately before any API request or data generation.

Raw log excerpt:

```text
bash: line 1: DEEPSEEK_API_KEY: DEEPSEEK_API_KEY is not set
```

Follow-up checks showed:

```text
LEN=0
ENV_UNSET
```

No synthetic rows, validation summary, corpus, index, or GRPO artifacts were
created under the results directory.

## Analysis

The DeepSeek API key is not present in the remote shell environment available to
the launch command. A previous SET/UNSET preflight check was invalid because of
quote escaping; the corrected check shows the remote variable length is zero.

## Next Step

Set `DEEPSEEK_API_KEY` securely in the remote tmux environment without printing
the key or writing it to files, then rerun the same pilot command.
