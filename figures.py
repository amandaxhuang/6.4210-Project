"""The paper's first figures, from results/robustness.csv.  The numbers behind each are
in results/robustness.md, so nothing here is only readable from the picture.

    python figures.py

writes PNGs to results/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import replan_trials
import robustness
from robustness import A, B, BASELINE
import trials

OUT = Path(__file__).parent / "results" / "figures"

# The first three slots of a palette checked for colorblind separation, in a fixed order, so
# each approach is the same color in every figure.  Aqua is below 3:1 against the surface,
# which is why every line also has a direct label.
COLORS = {A: "#2a78d6", B: "#eb6834", BASELINE: "#1baf7a"}
SURFACE = "#fcfcfb"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LINE_WIDTH, MARKER_SIZE, RING = 1.6, 6.5, 1.2  # points; about 2 px and 8 px at the figures' size


def _style(ax, xlabel: str, ylabel: str | None = "Caught, of 100") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 25))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)


def _series(ax, levels, rows: list[robustness.Row], approach: str, label_end: bool = True) -> None:
    """One approach's catches at each level, evenly spaced, with its 95% interval as a wash."""
    x = np.arange(len(levels))
    n = [len(r.results) for r in rows]
    k = np.array([r.caught for r in rows])
    low, high = zip(*(trials.wilson(int(ki), ni) for ki, ni in zip(k, n)))
    color = COLORS[approach]
    ax.fill_between(x, 100 * np.array(low), 100 * np.array(high), color=color, alpha=0.10, linewidth=0)
    ax.plot(x, k, color=color, linewidth=LINE_WIDTH, solid_capstyle="round", solid_joinstyle="round", marker="o",
            markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=RING, label=robustness.APPROACHES[approach])
    if label_end:
        ax.annotate(f"{robustness.APPROACHES[approach]} {k[-1]}", (x[-1], k[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=TEXT, fontsize=9)
    ax.set_xticks(x, levels)
    ax.set_xlim(-0.25, len(levels) - 0.25 + (0.75 if label_end else 0))


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _legend(ax) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=9, labelcolor=TEXT, handlelength=1.6)


def by_factor(results, name: str, title_x: str, setting: str, levels, labels) -> Path:
    """Catches as one factor moves, for each approach."""
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for approach in (B, A, BASELINE):
        _series(ax, labels, [robustness.select(results, approach, **{setting: level}) for level in levels], approach)
    _style(ax, title_x)
    _legend(ax)
    return _save(fig, name)


def by_both_noises(results) -> Path:
    """Toss noise and sensor noise together: one panel for each toss noise."""
    fig, axes = plt.subplots(1, len(robustness.CROSS_SIGMAS_LAT), figsize=(8.4, 3.0), sharey=True)
    labels = [f"{1e3 * s:.0f}" for s in robustness.CROSS_SIGMAS_P]
    for ax, sigma_lat in zip(axes, robustness.CROSS_SIGMAS_LAT):
        for approach in (B, A, BASELINE):
            rows = [robustness.select(results, approach, sigma_lat=sigma_lat, sigma_p=s) for s in robustness.CROSS_SIGMAS_P]
            _series(ax, labels, rows, approach, label_end=False)
        _style(ax, "Sensor noise (mm)", "Caught, of 100" if ax is axes[0] else None)
        ax.set_title(f"Lateral toss noise {sigma_lat} m/s", color=TEXT, fontsize=9.5)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False, fontsize=9, labelcolor=TEXT, handlelength=1.6)
    return _save(fig, "caught_by_both_noises")


def by_target_moved(results) -> Path:
    """Spec section 11.3's check on the mechanism: A's catches fall off where B had to move its target the most."""
    a = {r.trial: r for r in robustness.select(results, A).results}
    b = sorted((r for r in robustness.select(results, B).results if r.plans_adopted > 1), key=lambda r: r.plan_drift)
    groups = [[b[i] for i in part] for part in np.array_split(np.arange(len(b)), 4)]
    labels = [f"{100 * g[0].plan_drift:.1f} to {100 * g[-1].plan_drift:.1f}" for g in groups]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(len(groups))
    for approach, share in ((B, [sum(r.caught for r in g) / len(g) for g in groups]),
                            (A, [sum(a[r.trial].caught for r in g) / len(g) for g in groups])):
        y = 100 * np.array(share)
        ax.plot(x, y, color=COLORS[approach], linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE,
                markeredgewidth=RING, label=robustness.APPROACHES[approach])
        ax.annotate(f"{robustness.APPROACHES[approach]} {y[-1]:.0f}%", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    color=TEXT, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.25, len(groups) - 0.25 + 0.75)
    _style(ax, "How far B moved its target (cm): the same tosses, in four equal groups", "Caught (%)")
    _legend(ax)
    return _save(fig, "caught_by_target_moved")


def make_all(results) -> list[Path]:
    mm = [f"{1e3 * s:.0f}" for s in robustness.SIGMAS_P]
    return [
        by_factor(results, "caught_by_sensor_noise", "Sensor noise (mm)", "sigma_p", robustness.SIGMAS_P, mm),
        by_factor(results, "caught_by_lateral_toss_noise", "Lateral toss noise (m/s)", "sigma_lat", robustness.SIGMAS_LAT,
                  [str(s) for s in robustness.SIGMAS_LAT]),
        by_factor(results, "caught_by_vertical_toss_noise", "Vertical toss noise (m/s)", "sigma_vz", robustness.SIGMAS_VZ,
                  [str(s) for s in robustness.SIGMAS_VZ]),
        by_factor(results, "caught_by_close_latency", "Real close latency, with the trigger assuming 30 ms (ms)", "latency", robustness.LATENCIES,
                  [f"{1e3 * s:.0f}" for s in robustness.LATENCIES]),
        by_both_noises(results),
        by_target_moved(results),
    ]


if __name__ == "__main__":
    for path in make_all(replan_trials.read_results(robustness.CSV_PATH)):
        print(path)
