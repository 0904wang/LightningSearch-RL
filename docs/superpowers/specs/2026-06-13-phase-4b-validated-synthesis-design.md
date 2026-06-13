# Phase 4B Validated Synthetic Data Design

## Objective

Improve the real synthetic data path after the first 50-row DeepSeek pilot showed
13 validation rejects. The next path should generate until a target number of
valid HotpotQA-like rows is reached, while preserving rejected rows for analysis.

## Scope

- Strengthen the prompt around the two main reject causes:
  - supporting facts must use exactly two different titles
  - answer text must appear verbatim in supporting evidence
- Add a `synthesize-validated-data` CLI that:
  - generates rows in batches
  - validates each row immediately
  - writes raw, valid, and reject JSONL files
  - stops at `target_valid` or `max_attempts`
  - records generated, valid, reject, and API failure counts

## Out of Scope

- No changes to reward, retrieval, GRPO export, or training code.
- No automatic deletion of previous experiment directories.
- No external search APIs.

## Safety

API credentials remain environment-only. Failure summaries use existing secret
redaction before writing errors to result artifacts.
