# Phase 5D Turn-Level SFT Export

Date: 2026-06-15

## Goal

Fix the mismatch found in Phase 5C: the model should not generate
`<observation>` text. Observations are inserted by the runtime after a valid
`<search>` action. This phase adds turn-level SFT data where assistant messages
contain only one action at a time.

## Code Changes

- Added `src/lightningsearch_rl/agent_loop.py`.
  - `parse_agent_action`
  - `SearchEnvironment.search_observation`
- Added `src/lightningsearch_rl/sft_turns.py`.
  - `export_sft_turns`
  - strict single-action system prompt
- Added CLI command `export-sft-turns`.
- Added tests:
  - `tests/test_agent_loop.py`
  - `tests/test_sft_turns.py`
  - `tests/test_cli.py::test_export_sft_turns_cli_writes_turn_level_conversations`

## Remote Export Command

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-sft-turns \
  --examples /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/examples.jsonl \
  --index /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/index.json \
  --out-dir /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-turns-gold
```

## Artifacts

```text
/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-turns-gold/sft_turns.jsonl
/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-turns-gold/traces.jsonl
/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-turns-gold/summary.json
```

## Validation

Local tests:

```text
36 passed in 3.56s
```

Remote tests:

```text
36 passed in 0.55s
```

Remote export summary:

```json
{
  "answer_tag_rate": 1.0,
  "assistant_observation_rate": 0.0,
  "assistant_single_action_rate": 1.0,
  "avg_gold_evidence_count": 2.0,
  "example_count": 500,
  "gold_evidence_coverage": 1.0,
  "non_empty_answer_rate": 1.0,
  "sft_rows": 500
}
```

Sample role shape:

```text
('system', 'user', 'assistant', 'user', 'assistant'): 500
assistant_observation_rows: 0
```

Sample row:

```text
system:
You are a search agent for multi-hop QA. Output exactly one action per turn:
<search>...</search> or <answer>...</answer>. Never output <observation>;
observations are provided by the environment.

user:
Which award did the organization founded by Dr. Elena Voss receive in 2021?

assistant:
<search>Which award did the organization founded by Dr. Elena Voss receive in 2021?</search>

user:
<observation>
[1] Dr. Elena Voss: Dr. Elena Voss founded the Global Health Initiative in 2012.
[2] Global Health Initiative: The Global Health Initiative won the Nobel Peace Prize in 2021.
</observation>

assistant:
<answer>Nobel Peace Prize</answer>
```

## Analysis

This directly addresses the Phase 5C failure mode. The previous full-trace SFT
taught the model to sometimes hallucinate `<observation>` blocks. The new data
shape teaches the actual runtime contract:

- model emits one action;
- runtime inserts observation;
- model emits the next action.

The next step should be a 4-GPU SFT warmup on `sft_turns.jsonl`, then generation
inspection with two prompts:

1. question-only prompt should produce a single `<search>`.
2. question plus observation prompt should produce a single `<answer>`.
