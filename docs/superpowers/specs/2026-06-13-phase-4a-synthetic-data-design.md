# Phase 4A Synthetic Data Factory Design

## Objective

Add a controlled synthetic data path for HotpotQA-style multi-hop QA rows so the
project can bootstrap larger offline retrieval experiments before downloading or
curating full public datasets.

## Scope

- Generate raw HotpotQA-like JSONL rows with an OpenAI-compatible DeepSeek chat
  endpoint.
- Keep the API key out of source files, configs, command arguments, and logs.
- Validate rows before feeding them into existing `prepare-hotpot`,
  `build-index`, `export-sft`, and `export-grpo` workflows.
- Provide a deterministic mock mode for local and remote smoke tests without API
  usage.

## Interfaces

- `synthesize-data`
  - Writes raw synthetic JSONL.
  - Reads DeepSeek credentials only from `DEEPSEEK_API_KEY` in real mode.
  - Supports `--mock` for no-network smoke tests.
- `validate-synthetic`
  - Splits raw rows into valid and rejected JSONL files.
  - Requires at least two supporting facts, at least two evidence titles, valid
    references into `context`, and an answer string present in supporting
    evidence.

## Safety

The generated raw rows are not trusted. Every real or mock run must pass
`validate-synthetic` before entering the shared corpus/index pipeline.
