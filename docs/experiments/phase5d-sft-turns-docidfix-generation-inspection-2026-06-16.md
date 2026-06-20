# Phase 5D DocID-Fixed Turn-Level SFT Generation Inspection

Date: 2026-06-16

## Goal

Inspect whether the doc-id-fixed Phase 5D turn-level SFT checkpoint preserves
the desired agent-loop behavior:

- search stage: `system + question -> <search>...</search>`
- answer stage: `system + question + assistant search + runtime observation -> <answer>...</answer>`
- no assistant-generated `<observation>` blocks

This inspection uses the corrected docidfix SFT-turns data and the merged
checkpoint from the docidfix 4-GPU SFT warmup.

## Merge Artifact

The FSDP checkpoint was merged before inspection:

```text
source checkpoint: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/global_step_40
merged HF checkpoint: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
merge log: /data/wzl/LightningSearch-RL/logs/phase5d-sft-turns-docidfix-4gpu-merge.log
merged size: 7.6G
merge fatal log scan: 0
```

Merge command:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/global_step_40 \
  --target_dir /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40 \
  --use_cpu_initialization
```

## Launch

The first direct tmux launch failed because nested PowerShell, SSH, bash, and
tmux quoting broke before Python started. The second launcher-script attempt
failed because `set -u` conflicted with conda's activation hook:

```text
/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl/etc/conda/activate.d/activate-gcc_linux-64.sh: line 107: SYS_SYSROOT: unbound variable
```

The final launcher disables nounset only around conda activation, then restores
it before running Python.

Local launcher:

```text
D:\resume\Agent RL\scripts\remote\phase5d_docidfix_inspect.sh
```

Remote launcher:

```text
/data/wzl/LightningSearch-RL/runs/phase5d_docidfix_inspect.sh
```

Final launch command:

```bash
tmux new-session -d -s lightningsearch-20260616-sft-turns-docidfix-inspect \
  "bash /data/wzl/LightningSearch-RL/runs/phase5d_docidfix_inspect.sh"
```

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
repo sync: not-a-git-repo narrow sync from local workspace
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=6
sft rows: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
offset: 480
limit: 5
max_new_tokens: 64
started_at: 2026-06-15T16:16:31+00:00
finished_at: 2026-06-15T16:16:43+00:00
```

The UTC timestamps correspond to 2026-06-16 in Asia/Shanghai.

## Artifacts

```text
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu-generation-inspection/search_prompts.jsonl
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu-generation-inspection/answer_prompts.jsonl
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu-generation-inspection/generations.jsonl
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu-generation-inspection/summary.json
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu-generation-inspection/EXPERIMENT_RECORD.md
/data/wzl/LightningSearch-RL/logs/phase5d-sft-turns-docidfix-4gpu-generation-inspection.log
```

File counts:

```text
generations.jsonl: 10 rows
search_prompts.jsonl: 5 rows
answer_prompts.jsonl: 5 rows
```

## Metrics

```json
{
  "search": {
    "example_count": 5,
    "search_tag_rate": 1.0,
    "answer_tag_rate": 0.0,
    "observation_tag_rate": 0.0,
    "single_action_rate": 1.0,
    "valid_search_action_rate": 1.0,
    "valid_answer_action_rate": 0.0,
    "gold_answer_mention_rate": 0.0,
    "eos_rate": 1.0,
    "avg_new_tokens": 22.0
  },
  "answer": {
    "example_count": 5,
    "search_tag_rate": 0.0,
    "answer_tag_rate": 1.0,
    "observation_tag_rate": 0.0,
    "single_action_rate": 1.0,
    "valid_search_action_rate": 0.0,
    "valid_answer_action_rate": 1.0,
    "gold_answer_mention_rate": 1.0,
    "eos_rate": 1.0,
    "avg_new_tokens": 10.4
  },
  "overall": {
    "example_count": 10,
    "single_action_rate": 1.0,
    "observation_tag_rate": 0.0,
    "eos_rate": 1.0
  }
}
```

## Representative Outputs

Search-stage outputs:

```text
syn-010484 -> <search>Which organization publishes the journal that Dr. Elena Marchetti edits?</search>
syn-010489 -> <search>Which university published the journal that featured the work of Dr. Elena Voss?</search>
syn-010490 -> <search>Which award did the author of 'The Quantum Labyrinth' win?</search>
syn-010499 -> <search>Which organization awards the prize that Dr. Elena Voss received in 2020?</search>
syn-010502 -> <search>Which organization awarded the grant that funded the research center founded by Dr. Elena Voss?</search>
```

Answer-stage outputs:

```text
syn-010484 gold=Springer Nature generated=<answer>Springer Nature</answer>
syn-010489 gold=University of Riverton generated=<answer>University of Riverton</answer>
syn-010490 gold=Nobel Prize in Physics generated=<answer>Nobel Prize in Physics</answer>
syn-010499 gold=Global Science Foundation generated=<answer>Global Science Foundation</answer>
syn-010502 gold=National Science Foundation generated=<answer>National Science Foundation</answer>
```

## Validation

```text
remote generation inspection tests: 3 passed
dry-run prompt check: search=5, answer=5
search prompt roles: system,user
answer prompt roles: system,user,assistant,user
fatal log scan: 0 matches for Traceback, ChildFailedError, OutOfMemory, CUDA error, RuntimeError, Exception
tmux session: no active session after completion
GPU 6 after completion: 3505 MiB / 32607 MiB
```

Warnings were non-fatal:

```text
tokenizer incorrect regex pattern warning
torch_dtype deprecation warning
generation flags temperature/top_p/top_k ignored under deterministic decoding
```

## Analysis

The docidfix turn-level checkpoint passes the small held-out inspection slice.
It preserves the main behavior needed before GRPO: the model emits a single
valid search action for question-only prompts, emits a single valid answer
action after runtime observation insertion, and never emits assistant-side
observation tags.

The corrected docidfix data also fixes the previous `syn-010502` mismatch seen
in the old Phase 5D inspection. The answer-stage generation now matches the
stored gold answer for all five inspected examples.

This checkpoint is now a reasonable initial policy candidate for a tiny GRPO
smoke, but the next GRPO run should remain small and should use the docidfix
dataset artifacts only.
