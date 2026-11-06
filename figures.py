"""The paper's first figures, from results/robustness.csv.  The numbers behind each are
in results/robustness.md, so nothing here is only readable from the picture.

    python figures.py

writes PNGs to results/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def _style(ax, xlabel: str, ylabel: str | None = "Caught, of 100", ticks=range(0, 101, 25), top: float = 104) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, top)
    ax.set_yticks(ticks)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)


def _series(ax, levels, rows: list[robustness.Row], approach: str) -> None:
    """One approach's catches at each level, evenly spaced, with its 95% interval as a wash.

    The counts themselves are not written on the lines: every figure here has a table behind
    it in results/robustness.md, which is also what relieves the baseline's colour of having
    to carry identity on its own -- it sits below 3:1 against the surface.
    """
    x = np.arange(len(levels))
    n = [len(r.results) for r in rows]
    k = np.array([r.caught for r in rows])
    low, high = zip(*(trials.wilson(int(ki), ni) for ki, ni in zip(k, n)))
    color = COLORS[approach]
    ax.fill_between(x, 100 * np.array(low), 100 * np.array(high), color=color, alpha=0.10, linewidth=0)
    ax.plot(x, k, color=color, linewidth=LINE_WIDTH, solid_capstyle="round", solid_joinstyle="round", marker="o",
            markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=RING, label=robustness.APPROACHES[approach])
    ax.set_xticks(x, levels)
    ax.set_xlim(-0.25, len(levels) - 0.25)


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _legend(ax) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=9, labelcolor=TEXT, handlelength=1.6)


def by_factor(results, name: str, title_x: str, setting: str, levels, labels, width: float = 5.4) -> Path:
    """Catches as one factor moves, for each approach."""
    fig, ax = plt.subplots(figsize=(width, 3.3))
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
            _series(ax, labels, rows, approach)
        _style(ax, "Sensor noise (mm)", "Caught, of 100" if ax is axes[0] else None)
        ax.set_title(f"Lateral toss noise {sigma_lat} m/s", color=TEXT, fontsize=9.5)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False, fontsize=9, labelcolor=TEXT, handlelength=1.6)
    return _save(fig, "caught_by_both_noises")


def by_target_moved(results) -> Path:
    """Spec section 11.3's check on the mechanism: catches against how far B moved its target, on the same tosses."""
    a = {r.trial: r for r in robustness.select(results, A).results}
    b = sorted((r for r in robustness.select(results, B).results if r.plans_adopted > 1), key=lambda r: r.plan_drift)
    groups = [[b[i] for i in part] for part in np.array_split(np.arange(len(b)), 4) if len(part)]
    labels = [f"{100 * g[0].plan_drift:.2f}–{100 * g[-1].plan_drift:.2f}" for g in groups]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(len(groups))
    for approach, share in ((B, [sum(r.caught for r in g) / len(g) for g in groups]),
                            (A, [sum(a[r.trial].caught for r in g) / len(g) for g in groups])):
        y = 100 * np.array(share)
        ax.plot(x, y, color=COLORS[approach], linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE,
                markeredgewidth=RING, label=robustness.APPROACHES[approach])
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.25, len(groups) - 0.25)
    _style(ax, f"How far B moved its target (cm), on the {len(b)} tosses where it did, in four equal groups", "Caught (%)")
    _legend(ax)
    return _save(fig, "caught_by_target_moved")


def _caught_share(rows) -> float:
    """Caught, as a percentage of the rows given."""
    return 100 * sum(r.caught for r in rows) / len(rows) if rows else float("nan")


def by_latency_cell(results) -> Path:
    """A delay drawn per toss against fixed delays either side of its mean, so the cost of
    the spread alone can be read off.  The two kinds of cell are separated on the axis
    because one is a constant and the other is a distribution: they are not a sequence."""
    fixed = [(f"{1e3 * level:.0f} ms\nfixed", robustness._latency_settings(level)) for level in (0.025, 0.030)]
    drawn = [(f"mean {1e3 * robustness.LATENCY_MEAN:.1f} ms\nsigma {1e3 * sigma:.1f} ms", robustness._noise_settings(sigma))
             for sigma in robustness.LATENCY_SIGMAS]
    cells = fixed + drawn
    x = np.array([0, 1, 2.6, 3.6])  # the gap marks where constants end and distributions begin
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    # Dodged, because approaches tie exactly where a delay is hopeless or harmless -- at 0 ms
    # none of them catches anything -- and markers on one spot would hide each other.
    for approach, dodge in zip((B, A, BASELINE), (-0.13, 0.0, 0.13)):
        y = [_caught_share(robustness.select(results, approach, **settings).results) for _, settings in cells]
        ax.plot(x + dodge, y, color=COLORS[approach], linewidth=0, marker="o", markersize=MARKER_SIZE + 1.5,
                markeredgecolor=SURFACE, markeredgewidth=RING, label=robustness.APPROACHES[approach])
    ax.axvline(1.8, color=GRID, linewidth=0.9)
    ax.set_xticks(x, [label for label, _ in cells])
    ax.set_xlim(-0.4, 4.0)
    _style(ax, "A fixed delay, and a delay drawn per toss", "Caught (%)")
    _legend(ax)
    return _save(fig, "caught_by_latency_cell")


def by_realized_latency(results, sigma: float | None = None, bins: int = 5) -> Path:
    """The mechanism: does a drawn delay behave like the fixed delay it happened to draw?

    One panel per approach, each with the fixed-delay curve against the drawn cell's trials
    binned by the delay they actually drew.  Both are on a real latency axis, in ms, so the
    two are directly comparable.  If a drawn delay is just the fixed curve sampled, the dots
    land on the line; if the spread costs something of its own, they fall below it.
    """
    sigma = max(robustness.LATENCY_SIGMAS) if sigma is None else sigma  # the narrow cell has no range to bin
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), sharey=True)
    for ax, approach in zip(axes, (BASELINE, A, B)):
        color = COLORS[approach]
        levels = [level for level in robustness.LATENCIES
                  if robustness.select(results, approach, **robustness._latency_settings(level)).results]
        ax.plot([1e3 * level for level in levels],
                [_caught_share(robustness.select(results, approach, **robustness._latency_settings(level)).results)
                 for level in levels],
                color=color, linewidth=LINE_WIDTH, solid_capstyle="round", label="Fixed delay")
        rows = sorted(robustness.select(results, approach, **robustness._noise_settings(sigma)).results,
                      key=lambda r: r.latency_drawn)
        if rows:
            groups = [[rows[i] for i in part] for part in np.array_split(np.arange(len(rows)), bins) if len(part)]
            ax.plot([1e3 * np.median([r.latency_drawn for r in g]) for g in groups],
                    [_caught_share(g) for g in groups],
                    color=color, linewidth=0, marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE,
                    markeredgewidth=RING)
        _style(ax, "Close latency (ms)" if ax is axes[1] else "", "Caught (%)" if ax is axes[0] else None, top=118)
        ax.set_title(robustness.APPROACHES[approach], color=TEXT, fontsize=9.5, pad=14)
    # One legend for all three panels, in text ink: it tells fixed from drawn, while the color
    # and the title already say which approach a panel is.  A hue here would claim a meaning.
    handles = [Line2D([], [], color=MUTED, linewidth=LINE_WIDTH, label="A fixed delay, at each level"),
               Line2D([], [], color=MUTED, linewidth=0, marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE,
                      markeredgewidth=RING, label=f"Drawn per toss, sigma {1e3 * sigma:.1f} ms, binned by the delay drawn")]
    fig.tight_layout(rect=(0, 0.1, 1, 1))  # keep the bottom tenth clear for the shared legend
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=9,
               labelcolor=TEXT, handlelength=1.8)
    return _save(fig, "caught_by_realized_latency")


# The secondary metrics worth watching across the latency sweep: a label, how to read it off a
# cell's rows, and the y scale it needs.  Each gets its own panel, never a second y axis.
LATENCY_PANELS = (
    ("Timing error, 1 sigma (ms)", lambda rows: 1e3 * float(np.nanstd([r.timing_error for r in rows])), None),
    ("Caught ball ends along\nthe fingers, median (cm)",
     lambda rows: 100 * float(np.median([r.end_fingers for r in rows if r.caught])) if any(r.caught for r in rows) else float("nan"), None),
    ("Caught on the pad, of 100", lambda rows: sum(r.caught_on_pad for r in rows), (range(0, 101, 25), 104)),
    ("Contact, of 100", lambda rows: sum(r.contact for r in rows), (range(0, 101, 25), 104)),
)


def latency_metrics_panels(results) -> Path:
    """What moves behind the catch rate as the delay grows, one panel per metric."""
    levels = robustness.LATENCIES
    x = np.arange(len(levels))
    fig, axes = plt.subplots(1, len(LATENCY_PANELS), figsize=(11.0, 2.9))
    for ax, (label, read, scale) in zip(axes, LATENCY_PANELS):
        for approach in (B, A, BASELINE):
            rows = [robustness.select(results, approach, **robustness._latency_settings(level)).results for level in levels]
            # A cell with no rows is a gap, but a cell whose count is really 0 is a zero.
            y = [read(group) if group else float("nan") for group in rows]
            ax.plot(x, y, color=COLORS[approach], linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE - 1.5,
                    markeredgecolor=SURFACE, markeredgewidth=RING, label=robustness.APPROACHES[approach])
        if scale:
            _style(ax, "Close latency (ms)", label, *scale)
        else:
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.7)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=MUTED, length=0, labelsize=9)
            ax.set_xlabel("Close latency (ms)", color=MUTED, fontsize=9.5)
            ax.set_ylabel(label, color=MUTED, fontsize=9.5)
        ax.set_xticks(x, [f"{1e3 * level:.0f}" for level in levels], fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))  # the top tenth carries one legend for all four panels
    fig.legend(handles=axes[0].get_lines()[:3], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, fontsize=9, labelcolor=TEXT, handlelength=1.6)
    return _save(fig, "latency_metrics_panels")


def latency_draws(results) -> Path:
    """What the sampler drew, as an ECDF per cell with the 5th and 95th percentiles marked.
    The entity here is a distribution rather than an approach, so these wear text ink and are
    told apart by their dash and their label: the approach colors keep their own meaning."""
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    styles = ["-", "--", ":"]
    drew = False
    for (sigma, dash) in zip(robustness.LATENCY_SIGMAS, styles):
        rows = robustness.select(results, BASELINE, **robustness._noise_settings(sigma)).results
        if not rows:
            continue
        drew = True
        drawn = np.sort(1e3 * np.array([r.latency_drawn for r in rows]))
        share = 100 * np.arange(1, len(drawn) + 1) / len(drawn)
        ax.step(drawn, share, where="post", color=TEXT, linestyle=dash, linewidth=LINE_WIDTH,
                label=f"sigma {1e3 * sigma:.1f} ms: 5th {np.percentile(drawn, 5):.1f}, 95th {np.percentile(drawn, 95):.1f} ms")
    for level in (5.0, 50.0):  # what the task asked the 5th and 95th percentiles to be
        ax.axvline(level, color=GRID, linewidth=0.9)
        ax.annotate(f"{level:.0f} ms", xy=(level, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", color=MUTED, fontsize=8)
    _style(ax, "Delay drawn (ms)", "Trials at or below (%)")
    if drew:  # below the axes: both curves are steep near the mean, so any corner is in the way
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=8.5, labelcolor=TEXT,
                  handlelength=2.2)
    return _save(fig, "latency_draws")


def make_all(results) -> list[Path]:
    mm = [f"{1e3 * s:.0f}" for s in robustness.SIGMAS_P]
    return [
        by_factor(results, "caught_by_sensor_noise", "Sensor noise (mm)", "sigma_p", robustness.SIGMAS_P, mm),
        by_factor(results, "caught_by_lateral_toss_noise", "Lateral toss noise (m/s)", "sigma_lat", robustness.SIGMAS_LAT,
                  [f"{s:g}" for s in robustness.SIGMAS_LAT]),
        by_factor(results, "caught_by_vertical_toss_noise", "Vertical toss noise (m/s)", "sigma_vz", robustness.SIGMAS_VZ,
                  [f"{s:g}" for s in robustness.SIGMAS_VZ]),
        by_factor(results, "caught_by_close_latency", "Real close latency, with the trigger assuming 30 ms (ms)", "latency", robustness.LATENCIES,
                  [f"{1e3 * s:.0f}" for s in robustness.LATENCIES], width=6.6),
        by_both_noises(results),
        by_target_moved(results),
        by_latency_cell(results),
        by_realized_latency(results),
        latency_metrics_panels(results),
        latency_draws(results),
    ]


if __name__ == "__main__":
    for path in make_all(replan_trials.read_results(robustness.CSV_PATH)):
        print(path)
