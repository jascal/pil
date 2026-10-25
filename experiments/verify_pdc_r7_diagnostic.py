"""Independently reconstruct all frozen diagnostic bodies, margins and arms."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hashes-only", action="store_true", help="verify portable input mappings without numerical replay"
    )
    for name in ["data", "frames", "out"]:
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    out = Path(a.out)
    manifest = json.loads((out / "manifest.json").read_text())
    # Frozen manifest paths describe the original run machine. Resolve only
    # the enumerated file identities against this checkout and caller inputs.
    frames_root = Path(a.frames)
    repo_root = Path(__file__).resolve().parents[1]
    mapped = {
        "r7_fixed_data.npz": Path(a.data),
        "pdc_r7_diagnostic_prereg.md": repo_root / "docs/notes/pdc_r7_diagnostic_prereg.md",
        "pdc_r7_diagnostic.py": Path(__file__).with_name("pdc_r7_diagnostic.py"),
    }
    mapped.update(
        {
            name: frames_root / name
            for name in [
                "decode_initial.npz",
                "decode_final.npz",
                "admission_final.npz",
                "admission_initial.npz",
                "results.json",
            ]
        }
    )
    for name, wanted in manifest["sha256"].items():
        local = mapped[Path(name).name]
        assert hashlib.sha256(local.read_bytes()).hexdigest() == wanted, name
    if a.hashes_only:
        print(json.dumps({"input_hashes_checked": len(manifest["sha256"]), "failures": []}))
        return
    data = np.load(a.data)
    report = json.loads((out / "results.json").read_text())
    detail = json.loads((out / "body_diagnostics.json").read_text())
    random = np.random.default_rng(20261026).integers(0, 25, size=75).tolist()
    assert random == manifest["random_sources"]
    summaries = {}
    for name, filename in [
        ("initial", "decode_initial.npz"),
        ("decode", "decode_final.npz"),
        ("admission", "admission_final.npz"),
    ]:
        frame = np.load(Path(a.frames) / filename)
        U, b = frame["U"], frame["bias"]
        max_error = 0.0
        counts = {arm: 0 for arm in report["frames"][name]["arms"]}
        host_matches = 0
        strict_drift_matches = 0
        for i in range(75):
            target = int(data["teacher_indices"][i])
            full = U @ data["protected_residuals"][i] + b
            margins = []
            drifts = []
            target_scores = []
            for source in range(26):
                vector = data["native_sources"][i, source] if source < 25 else np.zeros(768)
                partial = U @ vector + b
                rivals = np.delete(partial, target)
                margin = float(partial[target] - rivals.max())
                # An independent full-minus-partial construction cancels bias.
                drift = float(np.ptp(full - partial))
                margins.append(margin)
                drifts.append(drift)
                target_scores.append(float(partial[target]))
                max_error = max(
                    max_error,
                    abs(margin - detail[name]["margin"][i][source]),
                    abs(drift - detail[name]["two_delta"][i][source]),
                )
            highest = int(np.argmax(target_scores[:25]))
            assert highest == detail[name]["highest_sources"][i]
            host_matches += any(m > 0 for m in margins[:25])
            strict_drift_matches += any(m > d for m, d in zip(margins[:25], drifts[:25], strict=True))
            for arm, summary in report["frames"][name]["arms"].items():
                indices = (
                    list(range(25))
                    if arm in ["original", "half_floor"]
                    else [highest if arm == "highest_plus_bias" else random[i], 25]
                )
                floor = 0.25 if arm == "half_floor" else 0.5
                valid = [j for j in indices if margins[j] > max(floor, drifts[j])]
                saved = summary["verdicts"][i]
                assert saved["bodies"] == indices
                assert saved["admitted"] == bool(valid)
                assert saved["witness"] == (valid[0] if valid else None)
                if valid:
                    j = valid[0]
                    assert margins[j] > saved["gamma"] > drifts[j]
                counts[arm] += bool(valid)
        logits = data["protected_residuals"] @ U.T + b
        full = np.c_[logits, data["outside_max"]]
        assert np.array_equal(full.argmax(1), data["teacher_indices"])
        for arm, count in counts.items():
            assert report["frames"][name]["arms"][arm]["admitted"] == count
        premises = report["frames"][name]["premise_context_counts"]
        assert host_matches == premises["any_singleton_host_match"]
        assert strict_drift_matches == premises["any_singleton_strict_drift"]
        assert max_error < 1e-8
        summaries[name] = dict(
            admitted=counts,
            protected_matches=183,
            body_values_checked=75 * 26,
            maximum_discrepancy=max_error,
            singleton_host_matches=host_matches,
            singleton_strict_drift_matches=strict_drift_matches,
        )
    verification = dict(
        schema="pdc.r7-diagnostic-verification.v1",
        frames=summaries,
        input_hashes_verified=True,
        new_model_calls=0,
        final_heldout_used=0,
        accounting_note="26 materialized bodies include one bias-only body requiring zero projection dots",
        accounting_per_frame=dict(
            unique_bodies=75 * 26,
            native_source_projection_row_dots=75 * 25 * 1703,
            protected_pool_projection_row_dots=183 * 1703,
            duplicate_train_tail_projection_row_dots=75 * 1703,
            body_score_annotations=75 * 26 * 1703,
        ),
    )
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
