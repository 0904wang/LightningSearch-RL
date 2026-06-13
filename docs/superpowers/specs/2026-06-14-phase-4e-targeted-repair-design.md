# Phase 4E Targeted Repair Design

## Goal

Improve strict schema synthesis throughput without weakening Phase 4D validation. The Phase 4D pilot accepted only 10 of 250 rows because many generated rows declared an `intermediate_entity` that did not appear verbatim in both hop evidence sentences.

## Design

Add an optional deterministic repair pass before rejection in validated synthesis. The repair pass only edits `chain_schema.intermediate_entity` when the row is otherwise structurally valid and one of the declared hop titles is a better exact intermediate candidate. After repair, the row must pass the unchanged strict validator.

The first repair rule is conservative:

- If `hop1_title` and `hop2_title` differ, set `intermediate_entity` to `hop2_title`.
- This matches the common pattern where hop 1 introduces a document/entity title and hop 2 is that same title's evidence sentence.
- Do not repair final answer leakage, answer-as-title, missing support facts, non-ASCII text, or unsupported context references.

Expose this through `synthesize_validated_file(..., repair_chain_schema=False)` and CLI flag `--repair-chain-schema`. Preserve the original row in `raw.jsonl`, write repaired accepted rows to `valid.jsonl`, and include repair counters in the summary.

## Testing

Use TDD for the repair layer:

- Unit test that a row rejected for missing intermediate in hop2 can be repaired by normalizing intermediate to `hop2_title`.
- Unit test that unrecoverable rows remain rejected with original strict reasons.
- Validated synthesis test that raw rows remain original while valid rows contain the repaired schema.
- CLI test that `--repair-chain-schema` is accepted and summary includes repair counts.
