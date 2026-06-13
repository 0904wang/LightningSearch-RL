# Phase 4B Validated Synthetic Data Plan

## Steps

1. Add tests for stronger prompt constraints and a target-valid generation loop.
2. Implement `synthesize_validated_file` in `synthesis.py`.
3. Add `synthesize-validated-data` to the CLI.
4. Verify locally with pytest and a mock target-valid smoke.
5. Sync to remote, run remote pytest, and run a mock target-valid smoke.
6. Report the real 200-row pilot command before launching.

## Pilot Command Shape

```bash
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-validated-data \
  --raw /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_raw.jsonl \
  --valid /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_valid.jsonl \
  --rejects /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_rejects.jsonl \
  --target-valid 200 \
  --topics awards,archives,research \
  --concurrency 50 \
  --batch-size 50 \
  --max-attempts 320 \
  --seed 2000 \
  --summary /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/validated_summary.json \
  --model deepseek-chat \
  --base-url https://api.deepseek.com
```
