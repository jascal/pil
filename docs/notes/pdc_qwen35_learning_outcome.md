# Qwen3.5-4B guarded admission learning: fixed-budget negative

The preregistered run completed 200 optimizer proposals per arm. Both ordinary
cross-entropy training and the additional admission objective started and ended
with **0/100 admitted first-hop contexts**. The required five-context increase
and two-context advantage over the control were not met. This closes these
fixed-budget arms; it does not establish that admission is unlearnable.

The input embedding, backbone, source writes and gates stayed fixed. Only the
2261 selected output rows and their biases could move. Every runtime step
reported preservation of all 208 protected native decisions. Independent NumPy
endpoint reconstruction confirms 208/208 matches for both final frames, all
6500 source guards per frame, and full-vocabulary FP above its Welch bound.
Maximum oracle discrepancy is below 1.3e-13. The updated-pool Welch bound is
zero and supplies no useful geometric constraint. Intermediate checks inspect
saved runtime logs; intermediate frames were not independently reconstructed.

The minimum global protected margin fell from 0.394895 to 1.32e-5 in the control
and 1.89e-5 in the admission arm. Later proposals were restored to their prior
frames after the guard rejected their updates. This is consistent with a
binding preservation constraint, but does not isolate that constraint as the
sole cause of zero admission.

Fieldrun independently consumed the final frame artifacts using its unchanged
chart, certificate and arithmetic rules. Its chart-quality gate passed, while
the joint advance gate failed because admission did not rise. In the admission
arm, rederive and crystallize each changed from 11 affirmed / 14 hallucinated /
2 abstained to 12 / 14 / 1, for both readouts. Generated and tool arms retained
27 affirmed commitments each. None of this demonstrates actual admitted
residual replay: first-hop admissions and operational admitted commitments
remain zero. Host preservation includes the 30 wrong named second-hop outputs;
this experiment was not allowed to correct them.

The 100 first-hop cases, including their former held-out wordings, are now
learning development data. No fresh confirmation is claimed. The original
consumed 25-context final slice was not used. Authoritative run artifacts are
`results/pdc_qwen35_learning/`: manifest, both 200-step logs, initial/final
frames, complete endpoint oracle arrays, results and independent verification.
The artifact input SHA-256 is
`771a77f88c9b5074165570ec75ea7787660ab59b58fbcb96e46f24fda80d8e99`.
