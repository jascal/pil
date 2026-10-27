"""Independent NumPy endpoint reconstruction; intermediate checks audit logs only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    d = np.load(a.data)
    results = json.loads((a.frames / 'results.json').read_text())
    reports = {}
    for arm in ['decode', 'admission']:
        steps = json.loads((a.frames / f'{arm}_steps.json').read_text())
        assert len(steps) == 200
        previous = np.zeros(100, dtype=bool)
        for i, s in enumerate(steps):
            assert s['step'] == i + 1
            assert s['protect_n'] == s['protected_native_matches'] == 208
            assert s['enforced_flips'] == 0 and s['premise_hold_rate'] == 1
            assert s['decided_rate'] == 1
            assert 2 * s['delta'] <= .9 * s['m_protect'] + 1e-10
            rho = results[arm]['summary']['rho']
            assert abs(s['delta'] - (rho * s['eps'] + s['beta'])) < 1e-10
            assert s['alpha'] == 0 or 2**-20 <= s['alpha'] <= 1
            mask = np.array(s['admission_mask'])
            assert np.all(~previous | mask) and mask.sum() == s['admitted']
            previous = mask
            g = s['geometry']
            assert g['fp_token'] + 1e-10 >= g['welch_token']
            assert g['updated_zero_rows'] == 0
        endpoints = {}
        for endpoint in ['initial', 'final']:
            f = np.load(a.frames / f'{arm}_{endpoint}.npz')
            U, b = f['U'], f['bias']
            if endpoint == 'initial':
                assert np.array_equal(U, d['initial_U'])
                assert np.array_equal(b, d['initial_bias'])
            H = d['protected_residuals'] @ U.T + b
            t = d['teacher_indices']
            scores = np.column_stack([H, d['outside_max']])
            assert np.array_equal(scores.argmax(1), t)
            target = scores[np.arange(208), t].copy()
            scores[np.arange(208), t] = -np.inf
            minimum_margin = float(np.min(target - scores.max(1)))
            assert minimum_margin > 0
            verdicts = results[arm][endpoint]
            max_error = 0.
            admitted = []
            saved = np.load(a.frames / f'{arm}_oracle.npz') if endpoint == 'final' else None
            for i in range(100):
                P = d['native_sources'][i] @ U.T + b
                tail = H[i] - P  # Bias cancels, independently of learner incidence algebra.
                margin = P[:, t[i]].copy()
                P[:, t[i]] = -np.inf
                margin -= P.max(1)
                delta = np.ptp(tail, axis=1)
                violation = np.maximum(.5, delta) - margin
                good = np.flatnonzero(violation < 0)
                j = int(good[0]) if len(good) else int(violation.argmin())
                v = verdicts[i]
                assert v['admitted'] == bool(len(good)) and v['source'] == j
                for key, value in [
                    ('margin', margin[j]),
                    ('two_delta', delta[j]),
                    ('violation', violation[j]),
                ]:
                    max_error = max(max_error, abs(v[key] - value))
                admitted.append(bool(len(good)))
                if saved is not None:
                    assert set(saved.files) == {'violation', 'margin', 'two_delta'}
                    for key, value in [('violation', violation), ('margin', margin), ('two_delta', delta)]:
                        max_error = max(max_error, float(np.max(np.abs(saved[key][i] - value))))
            assert max_error < 1e-9
            norm = np.linalg.norm(U, axis=1)
            assert np.all(norm > 0)
            G = d['outside_unit_gram'] + (U / norm[:, None]).T @ (U / norm[:, None])
            V = int(d['vocabulary_size'])
            fp = (np.sum(G * G) - V) / (V * (V - 1))
            welch = max(0, (V - U.shape[1]) / (U.shape[1] * (V - 1)))
            assert fp + 1e-10 >= welch
            assert sum(admitted) == results[arm][endpoint + '_admitted']
            if endpoint == 'final':
                assert abs(fp - results[arm]['geometry']['fp_token']) < 1e-10
                assert np.array_equal(admitted, previous)
                drift = np.linalg.norm(U - d['initial_U'], axis=1).max()
                assert drift <= sum(s['eps'] for s in steps) + 1e-9
                assert np.max(np.abs(b - d['initial_bias'])) <= sum(s['beta'] for s in steps) + 1e-9
            endpoints[endpoint] = dict(admitted=sum(admitted), protected_matches=208,
                minimum_global_margin=minimum_margin, fp=float(fp), welch=welch,
                maximum_oracle_error=max_error)
        reports[arm] = endpoints
    report = dict(gate='passed', arms=reports,
        intermediate_scope=(
            'Consistency checks of saved runtime logs; '
            'intermediate frames were not independently reconstructed.'
        ),
        endpoint_scope=(
            'Independent NumPy reconstruction of every source guard, '
            'full-host decisions and full-vocabulary frame potential.'
        ),
        data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(), new_model_forwards=0)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
