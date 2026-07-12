import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments import campaign_seed_sweep as campaign  # noqa: E402


def _record(names, agree=0.346, cover=0.5, agree_fired=0.692):
    return {
        "admitted_names": list(names),
        "core_sw": {"agree": agree, "cover": cover, "agree_fired": agree_fired},
    }


def test_subprocess_env_fixes_hash_seed_and_varies_experimental_seed():
    first = campaign.build_subprocess_env(1)
    second = campaign.build_subprocess_env(16)
    assert first["PYTHONHASHSEED"] == second["PYTHONHASHSEED"] == "0"
    assert first["SEED_SWEEP_SEED"] == "1"
    assert second["SEED_SWEEP_SEED"] == "16"


def test_output_paths_are_isolated_and_never_the_frozen_b0():
    forbidden = (campaign.REPO / "data" / "wikitext_gated_rescue_b0.pt").resolve()
    for seed, rep in [(0, 0), (16, 0), (-1, 99)]:
        path = campaign.output_path(seed, rep)
        assert path != forbidden
        assert campaign.SWEEP_DIR.resolve() in path.parents
        assert path.suffix == ".json"


def test_composition_helpers_on_known_fixtures():
    records = [_record({"alpha [1]", "beta"}), _record({"alpha [2]", "beta"})]
    assert campaign.identical_compositions(records, normalize=True)
    assert not campaign.identical_compositions(records, normalize=False)
    assert campaign.jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    table = campaign.frequency_table({1: {"a", "b"}, 2: {"b", "c"}})
    assert table == [
        {"name": "b", "count": 2, "seeds": [1, 2]},
        {"name": "a", "count": 1, "seeds": [1]},
        {"name": "c", "count": 1, "seeds": [2]},
    ]


def test_consensus_threshold_uses_b3_eight_run_boundary():
    fixtures = {
        seed: ({"exactly-7"} if seed <= 7 else set()) | ({"only-6"} if seed <= 6 else set())
        for seed in range(1, 9)
    }
    assert campaign.consensus_membership(fixtures) == {"exactly-7"}


def test_part_a_binary_pass_and_fail_cases():
    passing = [_record({"alpha", "beta"}, agree=0.346 + offset) for offset in (0, 5e-7, -4e-7)]
    assert campaign.evaluate_part_a(passing)
    differing = passing[:2] + [_record({"alpha", "gamma"})]
    assert not campaign.evaluate_part_a(differing)
    drifting = passing[:2] + [_record({"alpha", "beta"}, agree=0.346002)]
    assert not campaign.evaluate_part_a(drifting)


def test_conditional_branch_scheduling_decision():
    assert campaign.conditional_conditions(True) == ()
    assert campaign.conditional_conditions(False) == ("cpu", "deterministic")


def test_worker_guard_rejects_frozen_artifact_and_outside_sweep():
    forbidden = campaign.REPO / "data" / "wikitext_gated_rescue_b0.pt"
    outside = campaign.REPO / "data" / "somewhere_else.json"
    for path in (forbidden, outside):
        try:
            campaign._guard_worker_path(Path(path))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected worker path guard to reject {path}")


def test_run_one_resumes_existing_json_without_subprocess(monkeypatch, tmp_path):
    sweep_dir = tmp_path / "seed_sweep"
    sweep_dir.mkdir()
    monkeypatch.setattr(campaign, "REPO", tmp_path)
    monkeypatch.setattr(campaign, "SWEEP_DIR", sweep_dir)
    existing = campaign.output_path(3, 0, "b3")
    payload = {"status": "ok", "device": "cuda:0", "marker": "existing"}
    existing.write_text(json.dumps(payload))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called for a resumed artifact")

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    record = campaign._run_one(
        0, 0, "b3", internal_seed=3, artifact_seed=3
    )
    assert record["marker"] == "existing"
    assert record["resumed"] is True
    assert record["external_seed"] == 0
    assert record["internal_seed"] == 3
    assert existing.read_text() == json.dumps(payload)


def test_internal_seed_patch_substitutes_zero_and_restores_identity():
    original_manual_seed = campaign.torch.manual_seed
    original_generator = campaign.torch.Generator
    campaign._apply_internal_seed_patch(37)
    try:
        generator = campaign.torch.Generator().manual_seed(0)
        assert generator.initial_seed() == 37
        campaign.torch.manual_seed(0)
        assert campaign.torch.initial_seed() == 37
    finally:
        campaign._restore_internal_seed_patch()
    assert campaign.torch.manual_seed is original_manual_seed
    assert campaign.torch.Generator is original_generator


def test_resumed_worker_metadata_distinguishes_device_and_both_seeds(
    monkeypatch, tmp_path
):
    sweep_dir = tmp_path / "seed_sweep"
    sweep_dir.mkdir()
    monkeypatch.setattr(campaign, "REPO", tmp_path)
    monkeypatch.setattr(campaign, "SWEEP_DIR", sweep_dir)
    existing = campaign.output_path(8, 0, "b3")
    existing.write_text(json.dumps({"status": "ok", "device": "cuda:0"}))
    record = campaign._run_one(
        0, 0, "b3", internal_seed=8, artifact_seed=8
    )
    assert record["device"] == "cuda:0"
    assert record["external_seed"] == 0
    assert record["internal_seed"] == 8
