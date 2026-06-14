# Phase 5A Title-Fix 500 Dataset Diagnostics

Date: 2026-06-14

## Goal

Run a pre-training diagnostic pass on the Phase 4G 500-row strict synthetic
dataset before using it for verl / GRPO smoke work.

This pass checks:

- answer type distribution
- answer length distribution
- answer leakage into titles or questions
- whether answers appear in supporting evidence
- duplicate questions, answers, context titles, and supporting title pairs
- reward distribution from exported GRPO reward records

## Code And Environment

- Local repo: `D:\resume\Agent RL`
- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- GitHub branch: `main`
- Commit after diagnostics reward fix: `9ec6b1f`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- GPU selection: none; this was a CPU-only diagnostics CLI run
- tmux session: none; direct SSH command
- Diagnostics output: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/diagnostics.json`

Input artifacts:

- Valid data: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/valid.jsonl`
- GRPO export dir: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo`
- Reward records: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo/reward_records.jsonl`
- GRPO summary: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo/summary.json`

## Preflight

Remote preflight command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'pwd && mkdir -p /data/wzl/LightningSearch-RL/{repo,.conda-envs,data,indexes,logs,results,checkpoints,runs} && mkdir -p /home/user/wzl && ln -sfn /data/wzl/LightningSearch-RL /home/user/wzl/LightningSearch-RL && test -d /data/wzl/LightningSearch-RL && command -v tmux && source /home/user/anaconda3/etc/profile.d/conda.sh && conda --version && nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader && df -h /data/wzl/LightningSearch-RL'
```

Raw preflight output:

```text
/home/user
/usr/bin/tmux
conda 26.1.1
0, 4403 MiB, 32607 MiB
1, 4407 MiB, 32607 MiB
2, 4405 MiB, 32607 MiB
3, 25985 MiB, 32607 MiB
4, 26101 MiB, 32607 MiB
5, 4395 MiB, 32607 MiB
6, 3505 MiB, 32607 MiB
7, 18 MiB, 32607 MiB
Filesystem size used avail use% mount
/dev/sda1 7.3T 2.1T 4.8T 31% /data
```

## Diagnostics Command

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && out=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500 && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-data --valid "$out/valid.jsonl" --grpo-dir "$out/grpo" --out "$out/diagnostics.json" && cat "$out/diagnostics.json"'
```

## Tooling Fix During Run

The first diagnostics attempt reported:

```json
{
  "reward": {
    "avg": null,
    "count": 0,
    "max": null,
    "min": null
  }
}
```

Root cause: the real `reward_records.jsonl` exported top-level `total` fields,
for example:

```json
{"answer_reward": 0.0, "evidence_reward": 0.5, "format_reward": 1.0, "tool_validity_reward": 1.0, "search_count": 1, "total": 0.27, "id": "syn-009000", "search_cost": 0.03}
```

The diagnostics parser only handled nested `reward.total` and `total_reward`.
I added a regression test for top-level `total` and updated the parser. Local
and remote pytest then passed:

```text
61 passed
```

## Raw Diagnostics Result

```json
{
  "answer_length": {
    "avg": 2.146,
    "max": 5,
    "min": 1
  },
  "answer_type_counts": {
    "academic journal": 1,
    "academic publisher": 1,
    "award": 53,
    "city": 133,
    "conference": 3,
    "country": 1,
    "institution": 26,
    "museum": 1,
    "organization": 107,
    "person": 3,
    "prize": 2,
    "publisher": 3,
    "research facility": 1,
    "university": 112,
    "university press": 1,
    "venue": 2,
    "year": 50
  },
  "duplicates": {
    "duplicate_answers": 280,
    "duplicate_context_titles": 881,
    "duplicate_questions": 31,
    "duplicate_supporting_title_pairs": 64
  },
  "quality": {
    "answer_equals_context_title": 0,
    "answer_in_question": 0,
    "answer_support_sentence_hits": {
      "1": 500
    }
  },
  "reward": {
    "avg": 0.325,
    "count": 500,
    "max": 1.37,
    "min": 0.17
  },
  "row_count": 500
}
```

GRPO export summary:

```json
{
  "avg_reward": 0.325,
  "avg_search_count": 1.0,
  "example_count": 500,
  "rollout_count": 500,
  "top_k": 2,
  "transition_count": 1000
}
```

## Analysis

The Phase 4G prompt fixes the main grounding problems that blocked earlier
scaling. Across 500 valid rows:

- no final answer equals a context title
- no final answer appears in the question
- every row has exactly one supporting sentence hit for the answer
- the GRPO export has 500 rollouts and 1000 transitions

This is good enough for a small pipeline-level training smoke or reward-modeling
debug pass.

The main remaining risk is diversity. The duplicate counts are high:

- 280 duplicate answers
- 881 duplicate context title repeats
- 31 duplicate questions
- 64 duplicate supporting title-pair repeats

This does not block a smoke run, but it does mean the dataset is not yet strong
enough for a resume-facing final training result. A model could overfit repeated
answer and title patterns instead of learning robust search policy.

The answer type distribution is also skewed toward `city`, `university`,
`organization`, `award`, and `year`. This is acceptable for smoke testing but
should be diversified before larger runs.

## Conclusion

Use this 500-row set for the next controlled smoke step, not as the final
training corpus.

Recommended next step:

1. Run a minimal verl / GRPO environment smoke with the existing exported GRPO
   artifacts and one free GPU.
2. Keep the run short and focus on whether the training stack reads data,
   initializes Qwen, logs metrics, and writes checkpoints correctly.
3. In parallel, plan Phase 5B data improvements: diversity constraints,
   duplicate filtering, and answer-type balancing before generating a larger
   2k-5k corpus.
