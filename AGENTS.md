# LightningSearch-RL Remote Experiment Rules

This project is allowed to run LightningSearch-RL experiments on a shared remote GPU server, but only within the constraints in this file. These rules are hard constraints, not suggestions.

## Goal

Set up, run, monitor, and record remote experiments for a retrieval tool-use Agent RL project focused on:

- multi-hop QA search agents
- offline BM25 / FAISS retrieval environments
- `think/search/observe/answer` trajectory collection
- trace-to-transition adaptation for RL training
- evidence-aware reward shaping and credit assignment
- SFT warmup, rollout, GRPO training, and evaluation with `verl`

Target work includes:

- syncing this project to the approved remote workspace
- creating and using an isolated project-local conda environment
- preparing HotpotQA, 2WikiMultiHopQA, or approved small retrieval corpora
- running smoke tests, dry runs, and approved SFT / rollout / GRPO jobs
- monitoring logs, checkpoints, metrics, and result artifacts
- recording experiment metadata and conclusions for resume-ready analysis

## Required Remote Config

This block is the source of truth for the `safe-remote-experiments` workflow.

```yaml
backend: ssh
ssh_alias: "user@ssh-22.e6.luyouxia.net -p 29509"
ssh_entrypoint: "ssh user@ssh-22.e6.luyouxia.net -p 29509"
work_dir: "/data/wzl/LightningSearch-RL/repo"
allowed_paths:
  - "/data/wzl/LightningSearch-RL"
  - "/home/user/wzl/LightningSearch-RL"
  - "D:\\resume\\Agent RL"
activate_env: "source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl"
scheduler: "tmux"
code_sync: "explicit git pull --ff-only or narrow file sync approved by the user"
branch: "main"
dry_run_command: "cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m pytest && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli smoke --config configs/smoke/offline_bm25_hotpotqa.yaml --out-dir /data/wzl/LightningSearch-RL/results/dry-run"
launch_command: "tmux new-session -d -s lightningsearch-YYYYMMDD-task-name \"bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train --config configs/experiments/task-name.yaml --output-dir /data/wzl/LightningSearch-RL/results/task-name --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/task-name 2>&1 | tee /data/wzl/LightningSearch-RL/logs/task-name.log'\""
log_path: "/data/wzl/LightningSearch-RL/logs/task-name.log"
results_dir: "/data/wzl/LightningSearch-RL/results/task-name"
monitor_commands:
  - "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"
  - "tmux list-sessions"
  - "tail -n 100 /data/wzl/LightningSearch-RL/logs/task-name.log"
  - "ls -lah /data/wzl/LightningSearch-RL/results/task-name"
  - "ls -lah /data/wzl/LightningSearch-RL/checkpoints/task-name"
stop_command: "tmux list-sessions && echo 'Stop requires user approval for the exact lightningsearch-* session before running tmux kill-session.'"
forbidden_commands:
  - "sudo"
  - "su"
  - "rm -rf"
  - "reboot"
  - "shutdown"
  - "systemctl"
  - "service"
  - "apt"
  - "yum"
  - "dnf"
  - "pip install outside the approved conda env"
  - "conda install outside the approved conda env"
  - "editing ~/.ssh/config"
  - "editing ~/.bashrc"
  - "cron edits"
  - "user-management commands"
approved_setup_command: "Only create directories under /data/wzl/LightningSearch-RL and create / activate the approved conda environment when explicitly needed."
```

If any required field is missing, ambiguous, or contradicted by another file, stop and ask before any remote command.

## Hard Safety Constraints

- Only operate under `/data/wzl/LightningSearch-RL`, `/home/user/wzl/LightningSearch-RL`, and the local project workspace `D:\resume\Agent RL`.
- Treat `/data/wzl/LightningSearch-RL` as the real remote workspace.
- Treat `/home/user/wzl/LightningSearch-RL` only as a symlink entrypoint to the real workspace.
- Treat `D:\resume\Agent RL` as an approved local project workspace for documentation, planning files, helper scripts, and repo-local artifacts.
- Never write project files outside the approved remote `wzl` paths or the approved local project workspace.
- Never use `sudo`, `su`, system package managers, service managers, firewall changes, cron edits, or user-management commands.
- Never modify `.bashrc`, `~/.ssh/config`, system CUDA drivers, or machine-wide settings.
- Never install anything into the `base` conda environment.
- Never run global `pip install` or install into system Python.
- Never run destructive cleanup commands such as `rm -rf` without explicit user approval for the exact path.
- Never delete `data`, `indexes`, `logs`, `checkpoints`, `results`, or `runs` automatically.
- Never start a real data preparation, SFT, rollout, GRPO, or evaluation run before reporting the plan and receiving user approval.

System-level CUDA driver changes are forbidden. Environment-local PyTorch and CUDA runtime packages are allowed only inside the approved conda environment.

## Remote Access

Use this exact SSH entrypoint:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509
```

Do not create SSH aliases by editing shell init files or SSH config. Do not rewrite SSH settings.

## Allowed Paths

Real workspace:

```bash
/data/wzl/LightningSearch-RL
```

Symlink entrypoint:

```bash
/home/user/wzl/LightningSearch-RL -> /data/wzl/LightningSearch-RL
```

Allowed project subpaths:

```bash
/data/wzl/LightningSearch-RL/repo
/data/wzl/LightningSearch-RL/.conda-envs
/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
/data/wzl/LightningSearch-RL/data
/data/wzl/LightningSearch-RL/indexes
/data/wzl/LightningSearch-RL/logs
/data/wzl/LightningSearch-RL/results
/data/wzl/LightningSearch-RL/checkpoints
/data/wzl/LightningSearch-RL/runs
```

If a required directory is missing, it may be created with `mkdir -p` only under `/data/wzl/LightningSearch-RL`.

If the symlink is missing, it may be created as:

```bash
mkdir -p /home/user/wzl
ln -sfn /data/wzl/LightningSearch-RL /home/user/wzl/LightningSearch-RL
```

## Repository Layout

Expected remote layout:

```text
/data/wzl/LightningSearch-RL/
  repo/
  .conda-envs/
    lightningsearch-rl/
  data/
  indexes/
  logs/
  results/
  checkpoints/
  runs/
```

Expected local layout:

```text
D:\resume\Agent RL\
  AGENTS.md
  docs\
  scripts\
  configs\
  src\
```

The local project may start with only planning files. Do not infer that missing code should be created or synced remotely without user approval.

## Code Sync Policy

Use only one of these sync methods:

- `git pull --ff-only` from a known branch inside `/data/wzl/LightningSearch-RL/repo`
- a narrow file sync explicitly approved by the user
- a user-provided sync command that writes only under approved paths

If `/data/wzl/LightningSearch-RL/repo` is not a git repository, stop and ask for the approved clone or sync command.

Do not use `git add -A`, broad copy commands, destructive cleanup, or force-pushes as deployment shortcuts. Do not rewrite history unless the user explicitly asks.

## Conda and Environment Policy

Always initialize conda explicitly with the `conda.sh` profile script. This form is the approved default over the configured SSH entrypoint:

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
```

Do not rely on `.bashrc`.

Approved environment:

```bash
/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
```

Python target:

```text
Python 3.10
```

If the environment does not exist, it may be created only after confirming the active target path:

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda create -y -p /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl python=3.10
```

Activation example:

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
```

Allowed installation scope:

- active env is `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- `pip`, `uv`, and `conda` installs that clearly target the active approved env
- PyTorch CUDA runtime packages inside the approved env
- project dependencies for `transformers`, `datasets`, `vllm`, `verl`, BM25 / FAISS, and evaluation tools inside the approved env
- commands that run Python in the approved env must set `PYTHONNOUSERSITE=1` to avoid loading packages from `/home/user/.local`

Forbidden installation scope:

- `base` environment
- system Python
- global `pip`
- system CUDA driver
- OS package managers

If dependencies are missing and no approved setup command covers them, stop and ask instead of installing ad hoc.

## GPU Policy

This is a shared 8-GPU machine.

Hard limits:

- never use more than 4 GPUs
- always check free memory before selecting GPUs
- a GPU counts as free only if used memory is below `4000 MiB`
- prefer 1 GPU for smoke tests, retrieval indexing checks, and short rollout tests
- prefer 1 to 2 GPUs by default for first real runs
- do not use 4 GPUs without explicit user approval for that exact run

Preferred check:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

If GPU usage is ambiguous or too crowded, stop and report instead of guessing.

## Required Directory Conventions

All experiment inputs and outputs must stay under the approved workspace.

Data:

```bash
/data/wzl/LightningSearch-RL/data
```

Retrieval indexes:

```bash
/data/wzl/LightningSearch-RL/indexes
```

Logs:

```bash
/data/wzl/LightningSearch-RL/logs
```

Results:

```bash
/data/wzl/LightningSearch-RL/results
```

Checkpoints:

```bash
/data/wzl/LightningSearch-RL/checkpoints
```

Runs:

```bash
/data/wzl/LightningSearch-RL/runs
```

## Experiment Scope Policy

Use this project for controlled offline retrieval experiments first.

Allowed default datasets:

- HotpotQA
- 2WikiMultiHopQA
- Natural Questions subsets
- user-approved small corpora built from gold documents plus distractors

Default model scope:

- `Qwen3-4B` for the main GRPO line
- `Qwen3-8B` only for user-approved LoRA / QLoRA or evaluation comparisons

Default baselines:

- No Search
- Static RAG
- SFT Agent
- RL sparse reward
- RL shaped reward

Default metrics:

- Answer EM / F1
- Evidence Recall
- Invalid Tool Call Rate
- Average Search Calls
- Average Tokens / Cost
- Rollout Success Rate

Do not connect real external search APIs unless the user explicitly approves the API, budget, credential handling, and logging policy.

## Execution Workflow

Follow this sequence strictly.

### 1. Preflight

Before any setup or run:

- verify SSH access
- verify current working directory
- verify allowed paths exist or create only the approved missing paths
- verify symlink exists or create it
- verify `tmux` exists
- verify conda hook works
- inspect GPU memory usage
- inspect CPU and memory status when relevant
- inspect disk usage for `/data/wzl/LightningSearch-RL`
- verify `log_path`, `results_dir`, checkpoint path, and index path are inside approved paths

Example preflight:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 '\
  pwd && \
  mkdir -p /data/wzl/LightningSearch-RL/{repo,.conda-envs,data,indexes,logs,results,checkpoints,runs} && \
  mkdir -p /home/user/wzl && \
  ln -sfn /data/wzl/LightningSearch-RL /home/user/wzl/LightningSearch-RL && \
  test -d /data/wzl/LightningSearch-RL && \
  command -v tmux && \
  source /home/user/anaconda3/etc/profile.d/conda.sh && \
  conda --version && \
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader && \
  df -h /data/wzl/LightningSearch-RL'
```

### 2. Code Sync

Sync code using the configured method only. Prefer a narrow, explicit update.

If `/data/wzl/LightningSearch-RL/repo` is a git repository:

```bash
cd /data/wzl/LightningSearch-RL/repo
git fetch origin --prune
git checkout main
git pull --ff-only origin main
```

If it is not a git repository, stop and ask for the approved clone or sync command.

### 3. Environment Setup

If the approved env is missing:

- create only `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- activate that env
- install only dependencies needed for this project and task

Never install into `base`.

### 4. Smoke Test and Dry Run

Before any real run:

- run `python -m pytest` if tests exist
- run a parser / trace-schema smoke test if available
- run a tiny offline retrieval dry run
- run the smallest possible rollout with 1 GPU if CUDA is required
- use a small config, small dataset slice, short response length, and strict time budget

Default dry run:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 '\
  cd /data/wzl/LightningSearch-RL/repo && \
  source /home/user/anaconda3/etc/profile.d/conda.sh && \
  conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && \
  PYTHONNOUSERSITE=1 python -m pytest && \
  PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli smoke --config configs/smoke/offline_bm25_hotpotqa.yaml --out-dir /data/wzl/LightningSearch-RL/results/dry-run'
```

If the smoke test fails, stop and report after at most one automatic retry.

### 5. Report Before Launch

After the smoke test succeeds, report all of the following before launching:

- selected repo and branch
- selected commit
- selected conda env
- selected GPUs
- number of GPUs
- exact launch command
- exact `tmux` session name
- exact log path
- expected data and index paths
- expected checkpoint path if training is involved
- expected results path
- expected evaluation metrics

Wait for user approval before the real launch.

### 6. Launch

Only after user approval:

- start one experiment per `tmux` session
- redirect stdout and stderr to a log file under `/data/wzl/LightningSearch-RL/logs`
- keep the launch command explicit and reproducible
- include `CUDA_VISIBLE_DEVICES=...` in GPU jobs
- include config path, output path, checkpoint path, and seed in the logged command when supported

Example launch pattern after user approval:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "\
  tmux new-session -d -s lightningsearch-YYYYMMDD-task-name \
  \"bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train --config configs/experiments/task-name.yaml --output-dir /data/wzl/LightningSearch-RL/results/task-name --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/task-name 2>&1 | tee /data/wzl/LightningSearch-RL/logs/task-name.log'\""
```

## tmux Rules

`tmux` is the approved long-running process manager for this project.

Allowed operations:

- `tmux new-session`
- `tmux list-sessions`
- `tmux capture-pane`
- `tmux attach-session` if needed for inspection

Use one session per experiment.

Session naming format:

```text
lightningsearch-YYYYMMDD-task-name
```

Before launching, include the exact session name in the approval report. Do not kill a session without first reporting what it is and why it should be killed.

## Monitoring Rules

Approved monitoring methods:

- `nvidia-smi`
- `ps`
- `tmux list-sessions`
- `tmux capture-pane`
- `tail` on a specific log file under `/data/wzl/LightningSearch-RL/logs`
- `ls` / `du` on approved data, index, result, checkpoint, and run directories
- TensorBoard only with event logs under `/data/wzl/LightningSearch-RL/logs`, `/data/wzl/LightningSearch-RL/results`, `/data/wzl/LightningSearch-RL/runs`, or approved subdirectories

Example monitoring:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 '\
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader && \
  tmux list-sessions && \
  tail -n 100 /data/wzl/LightningSearch-RL/logs/task-name.log && \
  ls -lah /data/wzl/LightningSearch-RL/results/task-name && \
  ls -lah /data/wzl/LightningSearch-RL/checkpoints/task-name'
```

When reporting progress:

- show raw command output first
- then provide a short interpretation

Never claim a run is healthy without checking actual log output.

## Experiment Record Rules

After every experiment completes, record the effective information needed for future continuation, including:

- experiment goal and exact command or launcher
- repo, branch, commit, environment, GPU selection, and key runtime settings
- dataset, corpus, index, checkpoint, log, result, and run paths
- model, retrieval backend, search budget, reward configuration, seed, and rollout settings
- final status, metrics, errors, warnings, and relevant raw log excerpts
- analysis, conclusions, open risks, and next-step recommendations

Every completed experiment must be recorded locally under `D:\resume\Agent RL\docs\experiments` before treating the experiment as complete. A remote `EXPERIMENT_RECORD.md` is allowed and encouraged, but it does not replace the required local record.

The local record must include the raw result summary, exact commands used, links or paths to logs / results / checkpoints, observed metrics, failure notes if any, analysis of what the result means, and concrete next-step thoughts for the resume project.

Suggested record locations:

```bash
/data/wzl/LightningSearch-RL/results/<task-name>/EXPERIMENT_RECORD.md
D:\resume\Agent RL\docs\experiments\<task-name>.md
```

## Stop Rules

Allowed stop behavior:

- identify the exact `tmux` session
- identify the exact process or command being stopped
- report the reason
- wait for user approval before stopping unless the user already explicitly asked to stop it

Default stop command is intentionally non-destructive:

```bash
tmux list-sessions && echo 'Stop requires user approval for the exact lightningsearch-* session before running tmux kill-session.'
```

## Failure Rules

For sync, install, smoke test, dry run, launch, or monitoring failures:

- retry automatically at most once
- if the second attempt fails, stop and report
- include the exact failing command
- include the exact stderr or log excerpt
- do not silently change system config
- do not silently change SSH config
- do not silently switch package indexes, mirrors, models, datasets, GPUs, reward configs, or search budgets unless the user approves or the run config explicitly permits it

If model or dataset download fails once on the default Hugging Face endpoint, retry only when the user approves an explicit mirror such as `HF_ENDPOINT=https://hf-mirror.com`, and record that mirror in the log or approval payload.

## Forbidden Operations

Never do any of the following:

- `sudo`
- `su`
- `apt`
- `yum`
- `dnf`
- `reboot`
- `shutdown`
- `systemctl`
- `service`
- editing `.bashrc`
- editing `~/.ssh/config`
- changing the system CUDA driver
- installing into `base`
- global `pip install`
- broad deletion commands
- deleting `data`
- deleting `indexes`
- deleting `logs`
- deleting `checkpoints`
- deleting `results`
- deleting `runs`
- starting 4-GPU jobs without explicit approval
- starting real data preparation, SFT, rollout, GRPO, or evaluation jobs without a successful smoke test
- starting real data preparation, SFT, rollout, GRPO, or evaluation jobs without an approval report

## Default Command Patterns

Use explicit `cd` and explicit conda initialization in remote commands.

Example remote shell pattern:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 '\
  cd /data/wzl/LightningSearch-RL/repo && \
  source /home/user/anaconda3/etc/profile.d/conda.sh && \
  conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && \
  PYTHONNOUSERSITE=1 python --version'
```

Example GPU check:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 '\
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader'
```

Example log check:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 '\
  tail -n 100 /data/wzl/LightningSearch-RL/logs/task-name.log'
```

## Decision Rule

If an action is useful but not explicitly allowed here, do not assume permission. Stop and ask.
