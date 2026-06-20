# Env Transition Quality Filter Design

## Goal

Prevent known synthetic QA type-mismatch rows from contaminating larger
environment-transition GRPO runs while preserving reproducibility and diagnostics.

## Design

Use a deterministic quality manifest instead of early natural-language
heuristics. The manifest maps source/example IDs to one or more quality flags
and notes. During `export-env-transitions`, every rollout row is checked against
the manifest:

- without exclusion, rows are kept and exported metadata records
  `quality_flags` and `quality_notes`;
- with `--exclude-quality-flag <flag>`, matching rollout rows are skipped before
  transitions, reward records, and GRPO rollouts are written;
- the export summary reports excluded row count, IDs, and flag counts.

This keeps existing artifacts reproducible, makes the next 500-rollout /
1000-transition export cleaner, and leaves enough metadata to explain which
samples were filtered.

## Scope

The first manifest records confirmed Phase 5Q low-quality IDs:

- `syn-009012`
- `syn-009019`
- `syn-009432`
- `syn-009456`
- `syn-009536`

All are tagged as `qa_type_mismatch`. Borderline containment rows are not
filtered by default.

## Testing

Add tests that first fail against current behavior:

1. manifest-loaded rows are tagged in exported transition and reward metadata;
2. rows with an excluded flag are omitted and summarized;
3. the CLI accepts manifest and exclusion arguments.
