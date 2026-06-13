# Phase 4B Validated 200-Row Data Analysis

Date: 2026-06-14

## Scope

Analyze the 200 valid rows and 54 rejected rows from:

```text
/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200
```

This analysis focuses on whether the synthetic data is useful for the
LightningSearch-RL resume project, not on model training yet.

## Aggregate Findings

Validated synthesis reached the target sample count:

```json
{
  "requested": 254,
  "generated": 254,
  "valid_count": 200,
  "reject_count": 54,
  "api_failed": 0,
  "valid_rate": 0.787
}
```

Reject reasons:

```json
{
  "supporting_facts must cover at least two titles": 27,
  "answer not found in supporting evidence": 27
}
```

Valid row structure:

```json
{
  "supporting_facts": {"2": 200},
  "supporting_titles": {"2": 200},
  "context_titles": {"2": 1, "3": 145, "4": 52, "5": 2}
}
```

Answer shape:

```json
{
  "answer_in_question_count": 4,
  "answer_is_context_title_count": 51,
  "answer_in_both_support_sentences_count": 81
}
```

## Quality Notes

The validator is doing useful work. It removed rows where supporting facts stayed
inside one page even though the prompt asked for two titles, and rows where the
answer could be inferred from a page title but did not appear in the supporting
sentence text. These two failure modes are equally common in the current run.

The 200 valid rows are structurally clean: every valid row has exactly two
supporting facts from two distinct titles, and most examples include 3-4 context
titles. This is good enough for smoke-scale SFT/GRPO export and query/reward
debugging.

The remaining weakness is multi-hop depth. Many valid rows are format-valid but
not strongly multi-hop. In 51 rows, the answer is itself a context title. In 81
rows, both supporting sentences contain the answer string, which makes the task
closer to evidence lookup than compositional reasoning. A rule-based lexical
agent can satisfy those rows without learning sophisticated tool-use policy.

There are also occasional encoding/noise artifacts in generated text, for
example `Chlo谷 Zhao` and `每 Drama`. These do not currently dominate the data,
but they should be filtered before larger runs.

## Example Patterns

Useful valid pattern:

- The question asks for an award won by a film director.
- One supporting title identifies the film/director relation.
- Another supporting title states the director/award relation.
- The answer appears in supporting evidence.

Weak valid pattern:

- The answer appears in both supporting sentences.
- The second hop confirms rather than derives the answer.
- The example is valid for parsing and reward export, but weak for multi-hop
  reasoning.

Rejected same-title pattern:

- Both supporting facts point to sentences in the same page.
- The row often contains enough information, but not a real cross-document
  chain.

Rejected answer-grounding pattern:

- The answer appears as a page title or is inferable from title text.
- The answer string does not appear verbatim in the selected supporting
  evidence sentence.

## Recommendation

Do not scale this prompt directly to thousands of rows yet. First add one of
these two controls:

1. Repair pass for rejected rows:
   - same-title failure: ask the model to rewrite `supporting_facts` so they
     point to two different titles, or rewrite context if needed
   - answer-grounding failure: ask the model to insert the exact answer string
     into a supporting evidence sentence

2. Stricter generation prompt:
   - require hop 1 evidence to mention an intermediate entity but not the final
     answer
   - require hop 2 evidence to mention the intermediate entity and final answer
   - forbid the final answer from being a context title
   - forbid the final answer from appearing in the question
   - require ASCII-only English output for now

The better next engineering step is option 2 plus validator checks for:

- final answer not in question
- final answer not equal to any context title
- final answer appears in exactly one supporting sentence
- all text is ASCII, or at least reject obvious mojibake/non-English artifacts

After that, run another 200-valid-row pilot and compare:

- valid rate
- reject reason distribution
- average reward
- answer-as-title count
- answer-in-both-supporting-sentences count

This gives a clean story for the resume project: data synthesis was not just
bulk generation; it used validation, error analysis, and iterative controls to
improve multi-hop quality before RL training.
