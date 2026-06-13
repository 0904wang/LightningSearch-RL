# Phase 4D DeepSeek Schema Pilot 50

## Goal

Run the first real DeepSeek pilot with Phase 4D strict `chain_schema` validation to test whether schema-first generation removes Phase 4C shallow answer artifacts.

## Code And Environment

- Local branch: `master`
- GitHub main commit before launch: `0ae4483`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Remote sync: narrow `git archive` sync
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Python mode: `PYTHONNOUSERSITE=1`
- GPU: none used
- Session: `lightningsearch-20260614-phase4d-deepseek-schema-50`
- Log: `/data/wzl/LightningSearch-RL/logs/phase4d-deepseek-schema-50.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase4d-deepseek-schema-50`

## Launch

The API key was provided to the tmux session through silent stdin and exported only inside the session environment. It was not written to command-line arguments, files, logs, or experiment records.

```bash
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-validated-data \
  --raw /data/wzl/LightningSearch-RL/results/phase4d-deepseek-schema-50/raw.jsonl \
  --valid /data/wzl/LightningSearch-RL/results/phase4d-deepseek-schema-50/valid.jsonl \
  --rejects /data/wzl/LightningSearch-RL/results/phase4d-deepseek-schema-50/rejects.jsonl \
  --target-valid 50 \
  --topics awards,archives,research\ institutes,scientific\ discoveries,academic\ journals \
  --concurrency 50 \
  --batch-size 50 \
  --max-attempts 250 \
  --seed 5000 \
  --summary /data/wzl/LightningSearch-RL/results/phase4d-deepseek-schema-50/synthesis_summary.json \
  --require-chain-schema
```

## Raw Result

```json
{
  "api_failed": 0,
  "batch_size": 50,
  "concurrency": 50,
  "generated": 250,
  "max_attempts": 250,
  "requested": 250,
  "require_chain_schema": true,
  "stopped_reason": "max_attempts_reached",
  "target_valid": 50,
  "valid_count": 10,
  "reject_count": 240
}
```

## Reject Reasons

```text
intermediate entity missing from hop2: 92
intermediate entity missing from hop1: 48
non-ascii text detected: 35
supporting_facts must cover at least two titles: 22
final answer leaks in hop1: 12
chain_schema does not match supporting_facts: 11
final answer missing from hop2: 11
answer equals context title: 7
supporting fact missing from context: 2
```

## Valid-Set Checks

```json
{
  "valid_count": 10,
  "valid_answer_title": 0,
  "valid_answer_question": 0,
  "valid_answer_support_sentence_hits": {
    "1": 10
  }
}
```

## Analysis

The strict validator achieved the intended quality filter: accepted rows have no answer-as-title, no answer leakage in the question, and exactly one supporting sentence containing the final answer. The valid rate was only 4.0% because the model often declared an `intermediate_entity` in `chain_schema` without explicitly including that entity string in both hop evidence sentences.

Compared with Phase 4C, this is a precision improvement but not yet a usable high-throughput synthesis setup. The dominant failures are now repairable formatting and evidence-binding issues rather than shallow final-answer artifacts.

## Next Step

Add a Phase 4E targeted repair or retry layer before rejection:

- If hop1/hop2 misses the intermediate but the declared title contains it, normalize the schema to use the hop title as the intermediate.
- If the evidence sentence paraphrases the intermediate, request a one-row repair instead of consuming a full new generation.
- Add a stricter prompt example that shows hop1 and hop2 both repeating the exact `intermediate_entity` string.
- Keep strict final validation unchanged.

Then rerun a 50-valid pilot with a higher cap, for example `max_attempts=500`, and compare valid rate and reject distribution.
