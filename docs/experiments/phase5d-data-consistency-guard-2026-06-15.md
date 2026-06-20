# Phase 5D Data Consistency Guard

Date: 2026-06-15

## Goal

Fix the evidence inconsistency found during Phase 5D generation inspection. The
bad held-out row was not caused by the raw synthetic sample. It was introduced
when SFT export selected gold passages by a non-unique `doc_id`.

## Root Cause

`prepare-hotpot` and `prepare-2wiki` previously generated passage ids as:

```text
<dataset>::<title>::<sentence_index>
```

Synthetic rows frequently reuse titles such as `Dr. Elena Voss` with different
sentences. Shared corpus export deduplicated by `doc_id`, and SFT exporters then
looked up gold passages by the collided id. This allowed a valid row answer to be
paired with evidence from another row.

Observed example:

```text
raw/valid syn-010502 answer: National Science Foundation
raw/valid evidence: grant from the National Science Foundation
sft-turns evidence after collision: grant from the Global Science Foundation
```

## Change

Passage ids are now row-scoped:

```text
<dataset>::<row_id>::<title>::<sentence_index>
```

This prevents same-title synthetic rows from overwriting each other in the
shared corpus and index.

Both gold SFT exporters now fail fast when the selected gold evidence does not
contain the gold answer:

```text
answer not found in gold evidence for example <id>: <answer>
```

Updated exporters:

- `src/lightningsearch_rl/adapters.py`
- `src/lightningsearch_rl/sft_warmup.py`
- `src/lightningsearch_rl/sft_turns.py`

## Verification

Local:

```text
python -m pytest -q
95 passed in 2.13s
```

Remote:

```text
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m pytest -q
95 passed in 0.65s
```

Additional manual remote smoke was attempted with a tiny generated JSONL, but
PowerShell/SSH here-doc quoting corrupted the JSON before the project code read
it. It is not counted as verification evidence. The same repeated-title behavior
is covered by `tests/test_adapters.py`.

## Impact

Existing prepared artifacts under
`/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500` still use
the old doc_id format and should be regenerated before the next SFT or GRPO run.
The raw and valid synthetic files do not need regeneration for this fix; the
prepared corpus, examples, index, SFT exports, and GRPO exports do.
