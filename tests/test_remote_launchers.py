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
