"""One-shot offline four-arm diagnostic of the frozen R7 frames."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    data_path = Path(args.data)
    assert sha(data_path) == "ec7a855eaad8998c6eff6dac225e610b6dce641e283037399cfa93b0655ea195"
    frame_root = Path(args.frames)
    frame_paths = {
        name: frame_root / filename
        for name, filename in [
            ("initial", "decode_initial.npz"),
            ("decode", "decode_final.npz"),
            ("admission", "admission_final.npz"),
        ]
    }
    random_sources = np.random.default_rng(20261026).integers(0, 25, size=75).tolist()
    paths = [
        data_path,
        *frame_paths.values(),
        frame_root / "admission_initial.npz",
        frame_root / "results.json",
        Path(args.prereg),
        Path(__file__),
    ]
    manifest = dict(
        schema="pdc.r7-offline-diagnostic.v1",
        sha256={str(p): sha(p) for p in paths},
        random_sources=random_sources,
        random_seed=20261026,
        model_calls=0,
        final_heldout_used=0,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    data = np.load(data_path)
    D, r = data["native_sources"], data["protected_residuals"]
    targets = data["teacher_indices"]
    assert D.shape == (75, 25, 768) and r.shape == (183, 768)
    initial_other = np.load(frame_root / "admission_initial.npz")
    original = json.loads((frame_root / "results.json").read_text())
    results = {}
    details = {}
    for name, path in frame_paths.items():
        frame = np.load(path)
        U, b = frame["U"], frame["bias"]
        assert U.shape == (1703, 768) and b.shape == (1703,)
        assert np.isfinite(U).all() and np.isfinite(b).all()
        if name == "initial":
            assert np.array_equal(U, initial_other["U"]) and np.array_equal(b, initial_other["bias"])
        logits = r @ U.T + b
        full_logits = np.concatenate([logits, data["outside_max"][:, None]], axis=1)
        preserved = int((full_logits.argmax(1) == targets).sum())
        assert preserved == 183
        native_incidence = D @ U.T
        # Index 25 is the present empty/bias body, not an absent body.
        incidence = np.concatenate([native_incidence, np.zeros((75, 1, 1703))], axis=1)
        partial = incidence + b
        m = np.empty((75, 26))
        for i, t in enumerate(targets[:75]):
            rival_mask = np.arange(1703) != t
            m[i] = partial[i, :, t] - partial[i][:, rival_mask].max(1)
        tail = (r[:75] @ U.T)[:, None, :] - incidence
        delta2 = np.ptp(tail, axis=2)
        highest = native_incidence[np.arange(75), :, targets[:75]].argmax(1)
        assert highest.shape == (75,)
        arms = {}
        for arm, floor in [
            ("original", 0.5),
            ("half_floor", 0.25),
            ("highest_plus_bias", 0.5),
            ("random_plus_bias", 0.5),
        ]:
            verdicts = []
            for i, t in enumerate(targets[:75]):
                bodies = (
                    list(range(25))
                    if arm in {"original", "half_floor"}
                    else [int(highest[i]) if arm == "highest_plus_bias" else random_sources[i], 25]
                )
                admissible = [j for j in bodies if m[i, j] > max(floor, delta2[i, j])]
                witness = admissible[0] if admissible else None
                row = dict(admitted=witness is not None, bodies=bodies, witness=witness)
                if witness is not None:
                    j = witness
                    gamma = (m[i, j] + max(floor, delta2[i, j])) / 2
                    assert m[i, j] > gamma > delta2[i, j]
                    assert partial[i, j].argmax() == t
                    row.update(
                        margin=float(m[i, j]),
                        two_delta=float(delta2[i, j]),
                        gamma=float(gamma),
                        covered_host_agreement=True,
                    )
                verdicts.append(row)
            admitted = sum(v["admitted"] for v in verdicts)
            arms[arm] = dict(
                admitted=admitted,
                contexts=75,
                admission_rate=admitted / 75,
                protected_full_host_preserved=preserved,
                protected_full_host_fraction=preserved / 183,
                covered_agreement=1.0 if admitted else None,
                logical_body_evaluations=sum(len(v["bodies"]) for v in verdicts),
                absolute_margin_floor=floor,
                verdicts=verdicts,
            )
        expected = original["decode" if name == "initial" else name][
            "initial" if name == "initial" else "final"
        ]
        assert [v["admitted"] for v in arms["original"]["verdicts"]] == [v["admitted"] for v in expected]
        primary_positive = any(arms[a]["admission_rate"] >= 0.20 for a in ["half_floor", "highest_plus_bias"])
        negative = all(a["admission_rate"] < 0.05 for a in arms.values())
        host_matches = m[:, :25] > 0
        drift_ok = m[:, :25] > delta2[:, :25]
        floor_ok = m[:, :25] > 0.5
        results[name] = dict(
            arms=arms,
            classification="positive_local_sensitivity"
            if primary_positive
            else "negative_fixed_procedures"
            if negative
            else "inconclusive",
            premise_context_counts=dict(
                any_singleton_host_match=int(host_matches.any(1).sum()),
                any_singleton_original_floor=int(floor_ok.any(1).sum()),
                any_singleton_strict_drift=int(drift_ok.any(1).sum()),
                host_match_but_no_strict_drift=int((host_matches.any(1) & ~drift_ok.any(1)).sum()),
            ),
            actual_unique_body_evaluations=75 * 26,
            body_row_dot_calls=75 * 26 * 1703,
            protected_pool_row_dot_calls=183 * 1703,
        )
        details[name] = dict(margin=m.tolist(), two_delta=delta2.tolist(), highest_sources=highest.tolist())
    report = dict(
        schema="pdc.r7-offline-diagnostic-result.v1",
        frames=results,
        final_heldout_used=0,
        new_model_forwards=0,
        optimizer_steps=0,
        elapsed_seconds=time.perf_counter() - started,
        interpretation="fixed-frame diagnostic only; no generalization or theory claim",
    )
    (out / "body_diagnostics.json").write_text(json.dumps(details, indent=2) + "\n")
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                n: {
                    "classification": f["classification"],
                    "admitted": {a: v["admitted"] for a, v in f["arms"].items()},
                    "premises": f["premise_context_counts"],
                }
                for n, f in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "frames", "prereg", "out"]:
        p.add_argument("--" + name, required=True)
    run(p.parse_args())
