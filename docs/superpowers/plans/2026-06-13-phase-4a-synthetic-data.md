# Phase 4A Synthetic Data Factory Plan

## Steps

1. Add failing tests for synthetic row validation, JSONL validation, and no-key
   leakage in generation output.
2. Implement `lightningsearch_rl.synthesis` with:
   - OpenAI-compatible DeepSeek client
   - prompt builder
   - concurrent raw generation
   - resumable output IDs
   - row/file validation
   - deterministic mock row generation
3. Add CLI commands:
   - `synthesize-data`
   - `validate-synthetic`
4. Verify the mock pipeline:
   - synthesize raw JSONL
   - validate rows
   - prepare Hotpot-style corpus/examples
   - build lexical index
   - export GRPO artifacts
5. Prepare remote pilot commands under the `AGENTS.md` remote rules.

## Remote Pilot Gate

Run the mock path remotely first. Run the real DeepSeek path only after
`DEEPSEEK_API_KEY` is present in the remote shell environment and the command is
approved without exposing the key in arguments or logs.
