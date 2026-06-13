# Phase 4C Synthetic Data Quality Validator Design

## Objective

Improve synthetic multi-hop QA quality after the 200-row analysis found many
format-valid but shallow rows.

## Scope

- Strengthen prompt constraints:
  - answer must not appear in the question
  - answer must not equal any context title
  - answer must appear in exactly one supporting sentence
  - hop 1 introduces an intermediate entity without revealing the final answer
  - hop 2 connects the intermediate entity to the final answer
  - output should be ASCII-only English
- Add validator checks matching those constraints.
- Keep the existing `synthesize-validated-data` command shape unchanged.

## Expected Tradeoff

The stricter validator may reduce valid rate. That is acceptable because the
target-valid loop can request replacements, and the goal is higher-quality data
for SFT/GRPO rather than maximum raw throughput.
