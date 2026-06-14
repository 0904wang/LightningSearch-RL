# Phase 4E DeepSeek Repair Pilot 50

## Goal

Run a real DeepSeek pilot with Phase 4E targeted repair enabled to test whether deterministic chain-schema normalization can recover some of the strict Phase 4D rejects.

## Code And Environment

- Local branch: `master`
- GitHub main commit before launch: `437f390`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Remote sync: narrow `git archive` sync
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Python mode: `PYTHONNOUSERSITE=1`
- GPU: none used
- Session: `lightningsearch-20260614-phase4e-deepseek-repair-50`
- Log: `/data/wzl/LightningSearch-RL/logs/phase4e-deepseek-repair-50.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase4e-deepseek-repair-50`

## Launch

The API key was supplied to the tmux session through silent stdin and exported only inside the session environment. It was not written to command-line arguments, files, logs, or experiment records.

```bash
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-validated-data \
  --raw /data/wzl/LightningSearch-RL/results/phase4e-deepseek-repair-50/raw.jsonl \
  --valid /data/wzl/LightningSearch-RL/results/phase4e-deepseek-repair-50/valid.jsonl \
  --rejects /data/wzl/LightningSearch-RL/results/phase4e-deepseek-repair-50/rejects.jsonl \
  --target-valid 50 \
  --topics awards,archives,research\ institutes,scientific\ discoveries,academic\ journals \
  --concurrency 50 \
  --batch-size 50 \
  --max-attempts 500 \
  --seed 6000 \
  --summary /data/wzl/LightningSearch-RL/results/phase4e-deepseek-repair-50/synthesis_summary.json \
  --require-chain-schema \
  --repair-chain-schema
```

## Raw Result

```json
{
  "api_failed": 0,
  "batch_size": 50,
  "concurrency": 50,
  "generated": 500,
  "max_attempts": 500,
  "requested": 500,
  "require_chain_schema": true,
  "repair_chain_schema": true,
  "repair_attempt_count": 467,
  "repair_success_count": 0,
  "stopped_reason": "max_attempts_reached",
  "target_valid": 50,
  "valid_count": 33,
  "reject_count": 467
}
```

## Reject Reasons

```text
intermediate entity missing from hop2: 168
intermediate entity missing from hop1: 95
non-ascii text detected: 58
supporting_facts must cover at least two titles: 50
final answer leaks in hop1: 41
final answer missing from hop2: 26
chain_schema does not match supporting_facts: 14
answer equals context title: 13
```

## Valid-Set Checks

```json
{
  "valid_count": 33,
  "valid_answer_title": 0,
  "valid_answer_question": 0,
  "valid_answer_support_sentence_hits": {
    "1": 33
  }
}
```

## Analysis

The repair pilot improved throughput versus Phase 4D, but not because repair actually rescued rows: `repair_success_count` stayed at 0. The valid set remains clean, yet the dominant rejects are still missing-intermediate cases that our current one-step normalization cannot fix.

This means the next iteration should change the repair strategy, not the validator:

- add a stricter few-shot example that repeats `intermediate_entity` verbatim in both hop sentences
- repair from the supporting title pair rather than a single title guess
- if the model’s declared `intermediate_entity` is paraphrased, rewrite it to the exact hop2 title only when the title itself is safe and exact

The current output is useful as a safe upper bound on strict quality, but not yet enough to reach 50 valid rows in 500 attempts.
