# Phase 5B Generation Inspection: Gold-Answer Smoke

Date: 2026-06-15

## Goal

Inspect raw Qwen3-4B generations for the fixed gold-answer tiny GRPO smoke prompts to understand why `response_length/clip_ratio=1.0`, `format_reward=0.0`, and `answer_reward=0.0`.

## Command

Generated 5 samples from the fixed smoke parquet rows using GPU 7 and the cached local Qwen3-4B model:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 python <inline generation inspection script>
```

Output:

```text
/data/wzl/LightningSearch-RL/results/phase5b-generation-inspection-gold-answer/generations.jsonl
```

The script generated two variants for the same prompts:

- `default`: tokenizer chat template defaults.
- `no_thinking`: tokenizer chat template with `enable_thinking=False`.

Both variants used:

- `max_new_tokens=256`
- `do_sample=False`
- local model `/data/wzl/LightningSearch-RL/models/Qwen/Qwen3-4B`

## Findings

Default Qwen3 behavior:

- 5 / 5 samples used all 256 new tokens.
- 5 / 5 samples did not finish with EOS.
- 5 / 5 samples did not contain `<answer>...</answer>`.
- Outputs start with long `<think>` chains and never reach the required answer tag.

No-thinking behavior:

- 5 / 5 samples ended with EOS.
- Output lengths were 36 to 118 tokens.
- 5 / 5 samples still did not contain `<answer>...</answer>`.
- Several answers were hallucinated because the prompt only contained the question, not retrieved evidence or a strict output schema.

## Raw Samples

Default sample, `syn-009000`, gold answer `Nobel Peace Prize`:

```text
<think>
Okay, so I need to figure out which award the organization founded by Dr. Elena Voss received in 2021. First, I should start by identifying who Dr. Elena Voss is and what organization she founded. I'm not immediately familiar with her, so I'll have to do some research.
...
```

No-thinking sample, `syn-009000`:

```text
The organization founded by Dr. Elena Voss is the **Elena Voss Foundation**. In 2021, this foundation received the **Global Health Council's Award for Excellence in Public Health Advocacy**. This award recognizes outstanding contributions to public health advocacy and education.<|im_end|>
```

No-thinking sample, `syn-009002`, gold answer `Riverstone`:

```text
Dr. Elena Voss serves as director of the **Max Planck Institute for Human Development** in **Berlin, Germany**. Therefore, the city is **Berlin**.<|im_end|>
```

No-thinking sample, `syn-009009`, gold answer `Edinburgh`:

```text
As of the most recent information available (as of 2024), the editor-in-chief of the *Journal of Quantum Computing* is **Seth Lloyd**. He is a professor of quantum computation at MIT.

Seth Lloyd was born in **Cambridge, Massachusetts, USA**.
```

## Interpretation

The previous `response_length/clip_ratio=1.0` is mainly caused by Qwen3 thinking mode. The model spends the whole 256-token budget reasoning in `<think>` and never emits the required `<answer>` tag.

Disabling thinking solves the length cap for these samples, but not the tool-use format. Without a system prompt, retrieved observations, or SFT warmup, Qwen3 answers in generic chat style and may hallucinate.

## Next Steps

- Add a strict system prompt / chat template for GRPO rows instead of sending only the user question.
- Disable Qwen3 thinking in smoke runs or explicitly budget it.
- Build a small SFT warmup set with gold evidence and non-empty `<answer>` tags.
- Re-run a tiny generation inspection after prompt changes before another GRPO smoke.
