# LightningSearch-RL

Local MVP for a Lightning-style retrieval tool-use Agent RL framework.

Phase 1 validates the local contracts for data loading, offline retrieval, action parsing, trace collection, transition building, shaped rewards, and smoke evaluation. It does not run remote training or call external search APIs.

## Smoke Target

```bash
python -m pytest
python -m lightningsearch_rl.cli smoke --data tests/fixtures/tiny_multihop.jsonl --out-dir results/smoke
```
