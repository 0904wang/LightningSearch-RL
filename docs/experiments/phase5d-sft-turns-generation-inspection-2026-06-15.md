# Phase 5D Turn-Level SFT Generation Inspection

Date: 2026-06-15

## Goal

Inspect whether the Phase 5D turn-level SFT checkpoint fixes the Phase 5C
behavior where the model generated fabricated `<observation>` blocks. The test
separates the agent loop into two stages:

- search stage: `system + question -> <search>...</search>`
- answer stage: `system + question + assistant search + runtime observation -> <answer>...</answer>`

## Checkpoint Merge

Merged FSDP checkpoint:

```text
/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-4gpu/global_step_40
```

Merge command:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-4gpu/global_step_40 \
  --target_dir /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-4gpu/hf_merged_global_step_40 \
  --use_cpu_initialization
```

Merged model:

```text
/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-4gpu/hf_merged_global_step_40
size: 7.6G
```

## Inspection Command

```bash
tmux new-session -d -s lightningsearch-20260615-sft-turns-inspect "bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli inspect-generation --sft /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-turns-gold/sft_turns.jsonl --model /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-4gpu/hf_merged_global_step_40 --out-dir /data/wzl/LightningSearch-RL/results/phase5d-sft-turns-4gpu-generation-inspection --offset 480 --limit 5 --max-new-tokens 64 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5d-sft-turns-4gpu-generation-inspection.log'"
```

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=7
sft rows: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-turns-gold/sft_turns.jsonl
offset: 480
limit: 5
max_new_tokens: 64
```

## Artifacts

```text
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-4gpu-generation-inspection/search_prompts.jsonl
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-4gpu-generation-inspection/answer_prompts.jsonl
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-4gpu-generation-inspection/generations.jsonl
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-4gpu-generation-inspection/summary.json
/data/wzl/LightningSearch-RL/logs/phase5d-sft-turns-4gpu-generation-inspection.log
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
    "gold_answer_mention_rate": 0.8,
    "eos_rate": 1.0,
    "avg_new_tokens": 10.4
  },
  "overall": {
    "example_count": 10,
    "single_action_rate": 1.0,
    "observation_tag_rate": 0.0
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
syn-010502 gold=National Science Foundation generated=<answer>Global Science Foundation</answer>
```

## Validation

```text
tmux: no active session after completion
gpu 7 after completion: 18 MiB / 32607 MiB
fatal log scan: OutOfMemory / Traceback / ChildFailedError / CUDA error / RuntimeError = 0
```

Warnings were non-fatal:

```text
tokenizer incorrect regex pattern warning
torch_dtype deprecation warning
generation flags temperature/top_p/top_k ignored under deterministic decoding
```

## Analysis

Phase 5D fixed the main agent-loop mismatch observed in Phase 5C. The model no
longer emits `<observation>` in either stage, and every sampled generation is a
single valid agent action. Search-stage behavior is clean: all five prompts
produce exactly one `<search>` action and stop. Answer-stage behavior is also
clean format-wise: all five prompts produce exactly one `<answer>` action.

The answer accuracy is 4/5 on this held-out slice. The one mismatch,
`syn-010502`, appears to be a data issue rather than an instruction-following
failure: the runtime observation says the grant came from the Global Science
Foundation, while the stored gold answer is National Science Foundation. The
model followed the observation.

## Conclusion

The turn-level SFT warmup is a successful format and agent-loop bootstrap. It is
now reasonable to use
`/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-4gpu/hf_merged_global_step_40`
as the initial policy for the next tiny GRPO smoke, but the synthetic validation
pipeline should add a consistency check that the final answer is entailed by the
gold evidence text before scaling data further.
