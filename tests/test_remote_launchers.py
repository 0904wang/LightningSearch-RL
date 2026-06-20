from pathlib import Path


def test_phase6a_launcher_does_not_enable_nounset_before_conda_activation():
    script = Path("scripts/remote/phase6a_build_preference_pairs_from_phase5y.sh").read_text(
        encoding="utf-8"
    )

    conda_activate_index = script.index("conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl")
    before_activate = script[:conda_activate_index]

    assert "set -u" not in before_activate
    assert "set -euo pipefail" not in before_activate
    assert script.index("set -u") > conda_activate_index


def test_phase6c_launcher_uses_approved_env_and_paths():
    script = Path("scripts/remote/phase6c_synthetic_search_negatives_from_phase5w.sh").read_text(
        encoding="utf-8"
    )

    conda_activate_index = script.index("conda activate \"$ENV\"")
    before_activate = script[:conda_activate_index]

    assert "set -u" not in before_activate
    assert "set -euo pipefail" not in before_activate
    assert script.index("set -u") > conda_activate_index
    assert "PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli build-synthetic-search-preferences" in script
    assert "/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w" in script


def test_phase6d_gdpo_warmup_launcher_uses_approved_env_and_paths():
    script = Path("scripts/remote/phase6d_gdpo_warmup_from_phase5y.sh").read_text(encoding="utf-8")

    conda_activate_index = script.index("conda activate \"$ENV\"")
    before_activate = script[:conda_activate_index]

    assert "set -u" not in before_activate
    assert "set -euo pipefail" not in before_activate
    assert script.index("set -u") > conda_activate_index
    assert "CONFIG=configs/experiments/phase6d_gdpo_warmup_from_phase5y.yaml" in script
    assert "/data/wzl/LightningSearch-RL/results/phase6d-gdpo-warmup-from-phase5y" in script
    assert "/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y" in script
    assert "PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train" in script


def test_phase6e_grpo_warmstart_launcher_merges_gdpo_checkpoint_then_trains():
    script = Path("scripts/remote/phase6e_grpo_warmstart_from_phase6d_gdpo.sh").read_text(
        encoding="utf-8"
    )

    conda_activate_index = script.index("conda activate \"$ENV\"")
    before_activate = script[:conda_activate_index]

    assert "set -u" not in before_activate
    assert "set -euo pipefail" not in before_activate
    assert script.index("set -u") > conda_activate_index
    assert "CONFIG=configs/experiments/phase6e_grpo_warmstart_from_phase6d_gdpo.yaml" in script
    assert "/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/global_step_28/actor" in script
    assert "/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28" in script
    assert "PYTHONNOUSERSITE=1 python -m verl.model_merger merge" in script
    assert "PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train" in script
    assert "/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-from-phase6d-gdpo" in script
    assert "/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo" in script


def test_phase6e_eval_launcher_compares_sft_gdpo_and_grpo_warmstart():
    script = Path("scripts/remote/phase6e_grpo_warmstart_hard50_eval.sh").read_text(
        encoding="utf-8"
    )

    conda_activate_index = script.index("conda activate \"$ENV\"")
    before_activate = script[:conda_activate_index]

    assert "set -u" not in before_activate
    assert "set -euo pipefail" not in before_activate
    assert script.index("set -u") > conda_activate_index
    assert "/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval" in script
    assert "/data/wzl/LightningSearch-RL/logs/phase6e-grpo-warmstart-hard50-eval.log" in script
    assert "/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28" in script
    assert "/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/global_step_28/actor" in script
    assert "/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/hf_merged_global_step_28" in script
    assert "PYTHONNOUSERSITE=1 python -m verl.model_merger merge" in script
    assert script.count("PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli inspect-env-rollout") == 1
    assert 'run_eval "$SFT_MODEL" "$SFT_OUT" "sft baseline"' in script
    assert 'run_eval "$GDPO_MERGED" "$GDPO_OUT" "phase6d gdpo warmup"' in script
    assert 'run_eval "$GRPO_MERGED" "$GRPO_OUT" "phase6e grpo warmstart"' in script
    assert "phase6e_minus_sft" in script
    assert "phase6e_minus_phase6d" in script
    assert "comparison_summary.json" in script


def test_phase6e_stochastic_eval_launcher_samples_sft_gdpo_and_grpo_warmstart():
    script = Path("scripts/remote/phase6e_grpo_warmstart_hard50_stochastic_eval.sh").read_text(
        encoding="utf-8"
    )

    conda_activate_index = script.index("conda activate \"$ENV\"")
    before_activate = script[:conda_activate_index]

    assert "set -u" not in before_activate
    assert "set -euo pipefail" not in before_activate
    assert script.index("set -u") > conda_activate_index
    assert "/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-stochastic-eval" in script
    assert "/data/wzl/LightningSearch-RL/logs/phase6e-grpo-warmstart-hard50-stochastic-eval.log" in script
    assert "/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28" in script
    assert "/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/hf_merged_global_step_28" in script
    assert "PYTHONNOUSERSITE=1 python -m verl.model_merger merge" in script
    assert script.count("PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli inspect-env-rollout") == 1
    assert 'run_eval "$SFT_MODEL" "$SFT_OUT" "sft baseline stochastic"' in script
    assert 'run_eval "$GDPO_MERGED" "$GDPO_OUT" "phase6d gdpo warmup stochastic"' in script
    assert 'run_eval "$GRPO_MERGED" "$GRPO_OUT" "phase6e grpo warmstart stochastic"' in script
    assert "LIMIT=20" in script
    assert "TEMPERATURE=0.7" in script
    assert "TOP_P=0.9" in script
    assert "SAMPLE_TOP_K=40" in script
    assert "SEED=20260621" in script
    assert "--do-sample" in script
    assert "phase6e_minus_phase6d" in script
    assert "comparison_summary.json" in script
