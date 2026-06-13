# Phase 4C Synthetic Data Quality Validator Plan

## Steps

1. Add failing tests for shallow-data reject cases:
   - answer appears in question
   - answer equals context title
   - answer appears in multiple supporting sentences
   - non-ASCII text is present
2. Update prompt and validator checks.
3. Update mock rows so dry-run examples remain valid under the stricter rules.
4. Verify with pytest and local/remote mock smoke.
5. Report a real 200-valid-row pilot command before launching.

## Pilot Comparison

Compare against Phase 4B:

- valid rate
- reject reason distribution
- answer-in-question count
- answer-as-context-title count
- answer-in-multiple-supporting-sentences count
- GRPO average reward
