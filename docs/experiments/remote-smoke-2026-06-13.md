# Remote Smoke Experiment - 2026-06-13

## Goal

Prepare the approved remote RTX 5090 environment for LightningSearch-RL and run a minimal offline retrieval / GRPO export smoke test without starting model training.

## Remote Environment

- Server: `user@ssh-22.e6.luyouxia.net -p 29509`
- Workspace: `/data/wzl/LightningSearch-RL`
- Repo path: `/data/wzl/LightningSearch-RL/repo`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Python: `3.10.20`
- GPU smoke device: `CUDA_VISIBLE_DEVICES=7`
- GPU: `NVIDIA GeForce RTX 5090`
- Driver / CUDA from `nvidia-smi`: driver `580.159.03`, CUDA `13.0`

## Environment Packages

Installed inside the approved conda env only:

- `verl==0.8.0`
- `vllm==0.12.0`
- `torch==2.9.0+cu128`
- `transformers==4.57.6`
- `ray==2.55.1`
- `pytest==9.0.3`

Environment smoke output:

```text
python ok
torch=2.9.0+cu128
cuda_available=True
cuda_runtime=12.8
device_count=1
device_name=NVIDIA GeForce RTX 5090
verl=0.8.0
vllm=0.12.0
transformers=4.57.6
ray=2.55.1
```

## Code Sync

Primary GitHub repo:

```text
git@github.com:0904wang/LightningSearch-RL.git
```

Local code was pushed to GitHub `main` from local commit:

```text
149a8dc chore: add remote env smoke check
```

Remote GitHub HTTPS clone/fetch failed twice with TLS / connection reset errors, so the approved fallback was used:

```text
git archive HEAD -> scp tar to /data/wzl/LightningSearch-RL/runs/repo-sync-149a8dc.tar -> tar extract into /data/wzl/LightningSearch-RL/repo
```

No broad deletion or system-level changes were used.

## Commands

Create env:

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda create -y -p /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl python=3.10
```

Install project/runtime packages inside approved env:

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m pip install verl[vllm]==0.8.0
PYTHONNOUSERSITE=1 python -m pip install -e /data/wzl/LightningSearch-RL/repo
PYTHONNOUSERSITE=1 python -m pip install pytest
```

Run remote tests:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m pytest
```

Remote pytest result:

```text
28 passed in 0.12s
```

Run offline GRPO export smoke:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli prepare-hotpot \
  --raw tests/fixtures/hotpot_mixed_raw.jsonl \
  --corpus /data/wzl/LightningSearch-RL/data/remote-smoke/corpus.jsonl \
  --examples /data/wzl/LightningSearch-RL/data/remote-smoke/examples.jsonl \
  --limit 1
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli build-index \
  --corpus /data/wzl/LightningSearch-RL/data/remote-smoke/corpus.jsonl \
  --index /data/wzl/LightningSearch-RL/indexes/remote-smoke/index.json
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-grpo \
  --examples /data/wzl/LightningSearch-RL/data/remote-smoke/examples.jsonl \
  --index /data/wzl/LightningSearch-RL/indexes/remote-smoke/index.json \
  --out-dir /data/wzl/LightningSearch-RL/results/remote-smoke/grpo \
  --top-k 2
```

## Results

Result summary:

```json
{
  "avg_reward": 1.37,
  "avg_search_count": 1.0,
  "example_count": 1,
  "rollout_count": 1,
  "top_k": 2,
  "transition_count": 2
}
```

Remote artifacts:

- `/data/wzl/LightningSearch-RL/data/remote-smoke/corpus.jsonl`
- `/data/wzl/LightningSearch-RL/data/remote-smoke/examples.jsonl`
- `/data/wzl/LightningSearch-RL/indexes/remote-smoke/index.json`
- `/data/wzl/LightningSearch-RL/results/remote-smoke/grpo/rollouts.jsonl`
- `/data/wzl/LightningSearch-RL/results/remote-smoke/grpo/transitions.jsonl`
- `/data/wzl/LightningSearch-RL/results/remote-smoke/grpo/reward_records.jsonl`
- `/data/wzl/LightningSearch-RL/results/remote-smoke/grpo/summary.json`

## Status

Completed successfully.

No long-running training job was launched. No `tmux` session was started. The machine is ready for a user-approved 1-GPU training or rollout smoke once the training entrypoint/config is implemented.

## Next Steps

1. Add a repo-local remote training launcher/config for a tiny SFT or GRPO smoke.
2. Push future code updates to GitHub `main`; retry remote git clone/pull, falling back to narrow sync only if GitHub TLS still fails on the server.
3. Before any real run, report selected GPU, exact `tmux` session, command, log path, results path, and checkpoint path for approval.
