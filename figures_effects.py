"""How much each error costs each approach: change in catches from the nominal cell.

NEW FILE (friction sweep).  One row per condition, each a single change from the nominal
cell (sensor 2 mm, lateral toss 0.04 m/s, vertical toss 0.02 m/s, 30 ms close latency,
rubber ball), and one dot per approach: catches out of 100 at that condition minus
catches at nominal, on the same 100 tosses.  Puts the ball's material on the same scale
as the noise sources.  Reads results/robustness.csv and results/friction.csv; changes
nothing else.

    python figures_effects.py

writes results/figures/caught_change_by_condition.png and prints the numbers behind it.

A filled dot is a change McNemar's exact test finds significant on the paired tosses
(p < 0.05); a hollow dot is not, so it may be noise.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import figures
from figures import GRID, MUTED, SURFACE, TEXT
import replan_trials
import robustness
from robustness import A, APPROACHES, B, BASELINE
import robustness_friction as rf
import trials

ALPHA = 0.05  # significance level for filling a dot
DODGE = 0.22  # vertical offset between the three approaches within a row
ORDER = (BASELINE, A, B)  # top to bottom within each row


def conditions():
    """(group, label, rows_at(approach), nominal_rows(approach)).  Each condition is one
    change from nominal; its nominal comes from the same file, so the tosses pair up."""
    main = replan_trials.read_results(robustness.CSV_PATH)
    fric = rf.read_results(rf.CSV_PATH)
    noise_nominal = lambda approach: robustness.select(main, approach)
    rubber = lambda approach: rf.select(fric, approach, rf.RUBBER)

    def noise(**settings):
        return lambda approach: robustness.select(main, approach, **settings)

    def ball(mu):
        return lambda approach: rf.select(fric, approach, mu)

    return [
        ("Noise", "Sensor noise 2 → 5 mm", noise(sigma_p=0.005), noise_nominal),
        ("Noise", "Lateral toss noise 0.04 → 0.08 m/s", noise(sigma_lat=0.08), noise_nominal),
        ("Noise", "Vertical toss noise 0.02 → 0.05 m/s", noise(sigma_vz=0.05), noise_nominal),
        ("Timing", "Fingers 10 ms late (30 → 40 ms)", noise(latency=0.040, assumed_latency=robustness.ASSUMED_LATENCY),
         noise_nominal),
        ("Ball", "Smooth ball (μ 1.0 → 0.25)", ball(rf.MATERIALS["smooth"]), rubber),
        ("Ball", "Tennis ball (μ 1.0 → 0.6)", ball(rf.MATERIALS["tennis"]), rubber),
        ("Ball", "Sticky ball (μ 1.0 → 1.5)", ball(rf.MATERIALS["sticky"]), rubber),
    ]


def change(rows, nominal):
    """Catches at the condition minus at nominal, on the tosses both have, and McNemar's p."""
    at = {r.trial: r.caught for r in rows.results}
    base = {r.trial: r.caught for r in nominal.results}
    shared = sorted(set(at) & set(base))
    if not shared:
        return None
    _, _, p = trials.mcnemar([at[t] for t in shared], [base[t] for t in shared])
    return sum(at[t] for t in shared) - sum(base[t] for t in shared), p


def make(name: str = "caught_change_by_condition"):
    rows = conditions()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.set_facecolor(SURFACE)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)

    labels, y_of_row, y, previous_group = [], [], 0.0, None
    print(f"{'condition':44s} " + "  ".join(f"{robustness.APPROACHES[a]:>14s}" for a in ORDER))
    for group, label, at, nominal in rows:
        if previous_group is not None and group != previous_group:
            y += 0.6  # a gap between groups
            ax.axhline(y - 0.8, color=GRID, linewidth=0.8, zorder=0)
        previous_group = group
        printed = []
        for i, approach in enumerate(ORDER):
            yy = y + (i - 1) * DODGE
            if approach == BASELINE and "Sensor" in label:
                # The baseline never reads the sensor, so it has no sensor-noise row.
                ax.text(0.6, yy, "baseline ignores the sensor", color=MUTED, fontsize=7.5, va="center")
                printed.append(f"{'n/a':>14s}")
                continue
            result = change(at(approach), nominal(approach))
            if result is None:
                printed.append(f"{'-':>14s}")
                continue
            delta, p = result
            color = figures.COLORS[approach]
            ax.plot([0, delta], [yy, yy], color=color, linewidth=1.4, solid_capstyle="round", zorder=2)
            significant = p < ALPHA
            ax.plot([delta], [yy], marker="o", markersize=7.5, color=color if significant else SURFACE,
                    markeredgecolor=color, markeredgewidth=1.6, zorder=3)
            # The number sits beyond the dot's end, away from zero.
            ax.text(delta + (2.8 if delta >= 0 else -2.8), yy, f"{delta:+d}", color=TEXT, fontsize=7.5, va="center",
                    ha="left" if delta >= 0 else "right")
            printed.append(f"{delta:+5d} (p={p:.2g})".rjust(14))
        print(f"{label:44s} " + "  ".join(printed))
        labels.append(label)
        y_of_row.append(y)
        y += 1.0

    ax.set_yticks(y_of_row, labels)
    ax.set_ylim(y - 0.4, -0.6)  # first condition at the top
    ax.set_xlim(-95, 20)
    ax.set_xticks(range(-80, 21, 20))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for tick in ax.get_yticklabels():
        tick.set_color(TEXT)
    ax.set_xlabel("Change in catches out of 100, against the nominal setting (same tosses)", color=MUTED, fontsize=9.5)

    handles = [Line2D([], [], color=figures.COLORS[a], marker="o", markersize=7, linewidth=1.4, label=APPROACHES[a])
               for a in ORDER]  # same order as the dots, top to bottom
    handles.append(Line2D([], [], color=MUTED, marker="o", markersize=7, markerfacecolor=SURFACE, linewidth=0,
                          markeredgewidth=1.4, label=f"not significant (p ≥ {ALPHA:g})"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, 1.11), ncol=4, frameon=False, fontsize=8.5,
              labelcolor=TEXT, handlelength=1.6, columnspacing=1.2)
    fig.tight_layout()
    return figures._save(fig, name)


if __name__ == "__main__":
    print(make())
