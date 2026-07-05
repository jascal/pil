"""Figures for "The Self-Compiling Learner". Every number is a committed result; the provenance
map is NUMBERS.md (commit hash + artifact log per table/figure). Run:
    cd pil && .venv/bin/python paper/selfcompiler/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

# ---- Fig 1: the certifiability scaling law (commit 9a3ffeb, ladder2 logs) ----
params = [14, 70, 160, 410, 1000, 1400, 2800]
gold = [0.244, 0.311, 0.374, 0.427, 0.447, 0.463, 0.485]
student = [0.384, 0.352, 0.306, 0.282, 0.279, 0.270, 0.272]
core = [0.285, 0.276, 0.258, 0.245, 0.242, 0.231, 0.230]
crys = [74.4, 78.4, 84.3, 86.8, 86.7, 85.6, 84.5]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5))
a.semilogx(params, gold, "o-", color="#888", label="teacher gold top-1")
a.semilogx(params, student, "s-", color="#2b6cb0", label="student agreement")
a.semilogx(params, core, "^-", color="#c05621", label="certified core")
a.set_xlabel("teacher parameters (M)")
a.set_ylabel("top-1 / agreement")
a.legend(frameon=False, fontsize=7.5)
a.set_title("(a) no collapse across 200$\\times$ parameters", fontsize=9)
b.semilogx(params, crys, "d-", color="#276749")
b.set_xlabel("teacher parameters (M)")
b.set_ylabel("core / student (%)")
b.set_ylim(70, 90)
b.set_title("(b) crystallization rises to a $\\sim$85% plateau", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "scaling.pdf")
plt.close(fig)

# ---- Fig 2: cover variants across the matrix (commits ab33163, bbfb050, 272ee95, ladder2 logs) ----
cells = ["wiki\n14m", "wiki\n70m", "wiki\n410m", "wiki\n2.8b", "wt103\n70m", "wt103\n410m",
         "code\n70m", "code\n410m"]
base = [0.285, 0.276, 0.245, 0.230, 0.283, 0.257, 0.294, 0.294]
fixed_best = [0.287, 0.282, 0.251, 0.238, 0.292, 0.266, 0.376, 0.398]  # best fixed-order variant
sw = [0.334, 0.322, 0.287, 0.270, 0.334, 0.298, 0.604, 0.584]          # ext+ol, support-weighted
sw_mined = [0.339, 0.329, 0.288, 0.270, 0.341, 0.298, 0.605, 0.585]
studenta = [0.375, 0.347, 0.284, 0.271, 0.355, 0.302, 0.569, 0.541]    # soft student (sw runs)
import numpy as np
x = np.arange(len(cells))
w = 0.19
fig, ax = plt.subplots(figsize=(6.6, 2.6))
ax.bar(x - 1.5 * w, base, w, label="base library, fixed cover", color="#cbd5e0")
ax.bar(x - 0.5 * w, fixed_best, w, label="best fixed-order variant", color="#90cdf4")
ax.bar(x + 0.5 * w, sw_mined, w, label="support-weighted + mined", color="#c05621")
ax.plot(x, studenta, "k_", markersize=14, label="soft student (reference)")
ax.set_xticks(x, cells, fontsize=7.5)
ax.set_ylabel("certified-core agreement")
ax.legend(frameon=False, fontsize=7.5, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "library.pdf")
plt.close(fig)

# ---- Fig 3: sleep-as-compilation retention (commit 73820ac, selfcompile.log) ----
tasks = ["induction", "marker", "khop2"]
compile_arm = [1.000, 1.000, 1.000]
baseline = [0.010, 0.002, 0.007]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(3.2, 2.3))
ax.bar(x - 0.18, compile_arm, 0.36, label="sleep-compilation", color="#276749")
ax.bar(x + 0.18, baseline, 0.36, label="matched-budget baseline", color="#cbd5e0")
ax.set_xticks(x, tasks)
ax.set_ylabel("final accuracy (all 3 tasks)")
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "curriculum.pdf")
plt.close(fig)
print("figures ->", OUT)
