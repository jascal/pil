# Provenance map: every number in the paper → its commit + artifact

All commits are on `jascal/pil` `main` (merged via PR #11) unless a repo is named; artifact logs
live in the workspace `WYLY_LM_ENDGAME_REVIEW_FABLE_artifacts/` directory.

| paper element | numbers | commit | artifact |
|---|---|---|---|
| Table 1 (battery) | induction 0.999/0.998, marker 1.000/1.000, khop2 0.920/0.224; transformer 30k chance / 1.000 marker | `55bc104` | `battery_v2.log` |
| Certificate → proved | 2048/2048; khop2 hardened 0.987/0.987; extraction layer-1/head-1 | `005f94f` | `harden.log` |
| Table 2 (curriculum), Fig 3 | 1.000×3 vs 0.010/0.002/0.007; sleep extends 0.806→0.960→0.995 | `73820ac` | `selfcompile.log` |
| v2 ladder | 0.199 vs floor 0.148; certified core 0.191, tax 0.008; program 0.614 vs head 0.022 | `1e7d39d`,`505fc04`,`f1c1e26` | `wyly_v2.log`, `certify.log`, `export.log` |
| v3 autonomous | 0.201; +0.058 vs hand +0.042; tier 0→2; spoke 12/12 + 3/3 | `9d1c74d` | `v3.log` |
| Teacher anchors / erratum | 12-tok true decode 0.244 (probe artifact 0.189); gold ladder 0.244..0.485 | `949b88f` | `v4.log`, teacher logs |
| Race table | learning 0.276 @99.2% vs analysis 0.025 @3.6% (0.688 fired); convergence 476/480 | `949b88f` | `race.log` |
| Fig 1 / Table 3 (scaling law) | gold/copy/big→T/student/core per rung; core/stu 74.4→84.5%; inter-scale 0.51→0.75 | `9a3ffeb` | `ladder2.log` |
| Library matrix (fixed covers) | base/ext/gates/online cells | `adeaacf`,`915633c`,`4fb2f4f`,`272ee95` | `ladder2_*.log` |
| Fig 2 / Table 4 (support-weighted) | wikitext 0.334/0.322/0.287/0.270; wt103 0.334/0.298; code 0.606/0.584; core>student on code | `ab33163` | `v5sw_*.log`, `ladder2_sw.log` |
| Mined frames | +0.005..+0.007 natural text; 191/187 frames on code; declined at 2.8b | `bbfb050` | `v5mined_*.log` |
| Relation kind | judge admits at 0.9444 fired-accuracy; runtime parity | pil `36121c2`; rosetta `fe3fb86`/`932ac4a`; sgiandubh `8debfc5` | `v5_relation_emit.log` |
| Federation | wiki→wiki, code→code, gibberish→refuse | pil `5676527`; sgiandubh `66254d7` | `claymore_demo.log` |
| §5 retention theorem (C9), §8 arbitration theorem (C10) | kernel-checked: `retention_by_compilation`, `certified_accuracy_invariant`, `cover_order_irrelevant`, `argmax_policy_optimal`, `miscalibration_bound` | i-orca `91ba50d` (merged `45480eb`, PR #24) | `examples/concept_grounding/{Retention,Arbitration}.thy`; `isabelle build … ConceptGrounding` green, Isabelle2025-2 |
| §9/§10 serving realization | served = learner core exactly: 0.605 @ 99.8% code (12000 win), 0.329 @ 99.1% wikitext (11477); C++ parity 200/200×2; mined-frame answer at conf 0.865 in the demo | pil `1550e90`; rosetta `3bb2a8d`; sgiandubh `dba9102` | `serve_eval_{wiki,code}.log`, `claymore_demo_sw.log`, `v5_swemit_*.log` |
| §10 campaign 2 (domain study → threshold dissolves) | study: threshold at gzip≈0.33 (Spearman gold −0.976); final matrix: all corpora 96–108%, five arc bests (wiki 0.342, wt103 0.350, math 0.381, legal 0.487, code 0.611), de 0.539 | pil `20e8bd6`→`cf26ce4` (PRs #15–#22); rosetta #32–35; sgiandubh #15–18; i-orca #24 | `v5_full_*.log`, `domain_structure.log`, `wyly_domain_structure.md`, `wyly_detection_concepts.md`; extractor certificates `wyly_{mate,derived}_certify.py` |
