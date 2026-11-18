"""The friction sweep's figures, from results/friction.csv, in the same style as figures.py.

NEW FILE (friction sweep).  figures.py is untouched; this reuses its colours and
helpers so the friction figures match the paper's other figures.

    python figures_friction.py          # from results/friction.csv
    python figures_friction.py quick    # from results/friction_quick.csv (the smoke test)

writes to results/figures/:
    caught_by_friction.png     caught of 100 against the ball's material, one line per approach
    friction_outcomes.png      how every toss ended, per material, one panel per approach
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import figures
from figures import GRID, MUTED, SURFACE, TEXT
import robustness_friction as rf
from robustness import A, APPROACHES, B, BASELINE

# Outcomes from best to worst, darkest to lightest: one neutral ramp, because the
# approach is already named by each panel's title, so colour only has to rank the outcome.
OUTCOME_COLORS = {
    "caught": "#2f2e2b",
    "gripped, then lost": "#7d7b75",
    "touched, never held": "#b3b1aa",
    "no contact": "#dcdad3",
}


def _levels(results) -> list[float]:
    """The materials that have results, in sweep order."""
    return [mu for mu in rf.FRICTIONS if rf.select(results, B, mu).results]


def _tick(mu: float) -> str:
    return f"{rf.NAME_OF[mu].capitalize()}\nμ = {mu:g}"


def _ticks(n: int) -> list[int]:
    """Whole-number y ticks: 0, 25, 50, 75, 100 for the full sweep."""
    return list(range(0, n + 1, max(1, n // 4)))


def _series(ax, labels, rows, approach: str, n: int) -> None:
    """figures._series, except the 95% interval is scaled to n tosses: figures._series
    assumes 100, which would draw the quick run's interval off the chart."""
    x = np.arange(len(labels))
    k = np.array([row.caught for row in rows])
    low, high = zip(*(rf.trials.wilson(int(ki), len(row.results)) for ki, row in zip(k, rows)))
    color = figures.COLORS[approach]
    ax.fill_between(x, n * np.array(low), n * np.array(high), color=color, alpha=0.10, linewidth=0)
    ax.plot(x, k, color=color, linewidth=figures.LINE_WIDTH, solid_capstyle="round", solid_joinstyle="round", marker="o",
            markersize=figures.MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=figures.RING, label=APPROACHES[approach])
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.25, len(labels) - 0.75)


def caught_by_friction(results, suffix: str = "") -> str:
    """Caught of 100 for each approach as the material changes: figures.by_factor's
    form, with rows picked by robustness_friction.select instead of robustness.select."""
    levels = _levels(results)
    n = len(rf.select(results, B, levels[0]).results)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for approach in (B, A, BASELINE):
        _series(ax, [_tick(mu) for mu in levels], [rf.select(results, approach, mu) for mu in levels], approach, n)
    figures._style(ax, "Ball–pad friction (material)", f"Caught, of {n}", ticks=_ticks(n), top=1.04 * n)
    figures._legend(ax)
    return figures._save(fig, "caught_by_friction" + suffix)


def friction_outcomes(results, suffix: str = "") -> str:
    """Stacked bars: for each material, how each approach's tosses ended.  Small
    multiples, one per approach, sharing the y axis, so the panels compare directly."""
    levels = _levels(results)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), sharey=True)
    x = np.arange(len(levels))
    for ax, approach in zip(axes, (BASELINE, A, B)):
        counts = [rf.outcome_counts(rf.select(results, approach, mu)) for mu in levels]
        n = max(sum(c.values()) for c in counts)
        bottom = np.zeros(len(levels))
        for name, color in OUTCOME_COLORS.items():
            height = np.array([c[name] for c in counts])
            # A thin surface-coloured edge separates stacked segments.
            ax.bar(x, height, bottom=bottom, width=0.62, color=color, edgecolor=SURFACE, linewidth=1.0)
            bottom += height
        figures._style(ax, "", f"Tosses, of {n}" if approach == BASELINE else None, ticks=_ticks(n), top=1.02 * n)
        ax.set_xticks(x, [rf.NAME_OF[mu].capitalize() for mu in levels])
        ax.set_title(APPROACHES[approach], color=TEXT, fontsize=10, pad=6)
    handles = [Patch(facecolor=color, edgecolor=GRID, label=name.capitalize()) for name, color in OUTCOME_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False, fontsize=9,
               labelcolor=TEXT, handlelength=1.2)
    fig.tight_layout()
    fig.supxlabel("Ball material", color=MUTED, fontsize=9.5, y=0.0)
    return figures._save(fig, "friction_outcomes" + suffix)


def make_all(results, suffix: str = "") -> list:
    return [caught_by_friction(results, suffix), friction_outcomes(results, suffix)]


if __name__ == "__main__":
    quick = sys.argv[1:] == ["quick"]
    path = rf.QUICK_CSV_PATH if quick else rf.CSV_PATH
    results = rf.read_results(path)
    if not results:
        sys.exit(f"no results in {path}: run robustness_friction.py{' quick' if quick else ''} first")
    for figure in make_all(results, "_quick" if quick else ""):
        print(figure)
