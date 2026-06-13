# Phase 4D Chain Schema Design

## Goal

Improve synthetic multi-hop QA quality after the Phase 4C pilot showed many rejects caused by shallow title answers. Phase 4D makes generation schema-first: the model must declare the two-hop reasoning chain before the row is accepted for strict synthesis pilots.

## Design

Keep the existing synthesis pipeline and add an optional strict validator mode. Rows may include a `chain_schema` object with `hop1_title`, `hop1_sentence_index`, `intermediate_entity`, `hop2_title`, `hop2_sentence_index`, `answer_type`, and `final_answer`. When `require_chain_schema` is enabled, validation checks that supporting facts match the declared hops, hop 1 mentions the intermediate without leaking the final answer, and hop 2 mentions both the intermediate and final answer.

The prompt will ask DeepSeek to generate rows with this schema, choose attribute-style final answers instead of context titles, and keep the answer in exactly one supporting sentence. Existing validation commands remain backward compatible unless `--require-chain-schema` is passed.

## Testing

Add focused unit tests for prompt requirements, strict acceptance, hop leakage, missing intermediate, schema/support mismatch, mock synthesis compatibility, and CLI propagation. Verify with the full local pytest suite before remote sync.
