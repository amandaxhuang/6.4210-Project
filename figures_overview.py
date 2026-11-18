"""One figure with every factor side by side: how each approach holds up as one thing
gets worse at a time, on the same "caught, of 100" axis.

NEW FILE (friction sweep).  Reads results/robustness.csv (sensor noise, toss noise,
close latency) and results/friction.csv (the ball's material), and reuses figures.py's
colours and helpers, so it matches the single-factor figures.  Nothing else changes.

    python figures_overview.py

writes results/figures/caught_overview.png.

Every panel varies one factor with the others at the nominal cell (sensor 2 mm,
lateral toss 0.04 m/s, vertical toss 0.02 m/s, 30 ms close latency, rubber ball), and
every panel's nominal level is the same 100 tosses, so the shaded column is the same
three numbers (44 / 62 / 87) in every panel.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figures
from figures import GRID, MUTED, SURFACE, TEXT
import replan_trials
import robustness
from robustness import A, APPROACHES, B, BASELINE
import robustness_friction as rf

NOMINAL_SHADE = "#efeee9"  # a recessive column behind each panel's nominal level


def _panel(ax, title: str, labels: list[str], rows_of, nominal_index: int, approaches=(B, A, BASELINE),
           every: int = 1, flat_baseline: bool = False) -> None:
    """One factor: each approach's catches at each level, with the nominal level shaded.
    rows_of(approach) gives that approach's rows, one per level.  every > 1 labels only
    every that-many-th tick, for factors with many levels.  flat_baseline draws the
    baseline dashed: its value is the nominal one repeated, not re-run at each level."""
    ax.axvspan(nominal_index - 0.4, nominal_index + 0.4, color=NOMINAL_SHADE, zorder=0, linewidth=0)
    for approach in approaches:
        figures._series(ax, labels, rows_of(approach), approach)
        if flat_baseline and approach == BASELINE:
            ax.lines[-1].set_linestyle((0, (4, 3)))  # dashed: the same 100 results at every level
    ax.set_xticks(range(len(labels)), [label if i % every == 0 else "" for i, label in enumerate(labels)])
    ax.set_xlim(-0.5, len(labels) - 0.5)  # even margins on both sides
    ax.set_title(title, color=TEXT, fontsize=9.5, loc="left", pad=6)


def make(path_name: str = "caught_overview"):
    main = replan_trials.read_results(robustness.CSV_PATH)
    friction = rf.read_results(rf.CSV_PATH)
    nominal = robustness.NOMINAL

    def factor(setting, levels):
        return lambda approach: [robustness.select(main, approach, **{setting: level}) for level in levels]

    def sensor_rows(approach):
        # The baseline never reads the sensor, and the sensor's noise comes from its own
        # random stream, so its simulation is identical at every sensor level: the sweep
        # ran it once (recorded as sigma_p = 0).  Repeat that nominal result at each level.
        if approach == BASELINE:
            return [robustness.select(main, BASELINE)] * len(robustness.SIGMAS_P)
        return factor("sigma_p", robustness.SIGMAS_P)(approach)

    panels = [
        ("Sensor noise (mm)", [f"{1e3 * s:.0f}" for s in robustness.SIGMAS_P],
         sensor_rows, robustness.SIGMAS_P.index(nominal["sigma_p"]), (B, A, BASELINE), 1, True),
        ("Lateral toss noise (m/s)", [f"{s:g}" for s in robustness.SIGMAS_LAT],
         factor("sigma_lat", robustness.SIGMAS_LAT), robustness.SIGMAS_LAT.index(nominal["sigma_lat"]), (B, A, BASELINE), 1,
         False),
        ("Vertical toss noise (m/s)", [f"{s:g}" for s in robustness.SIGMAS_VZ],
         factor("sigma_vz", robustness.SIGMAS_VZ), robustness.SIGMAS_VZ.index(nominal["sigma_vz"]), (B, A, BASELINE), 1,
         False),
        ("Gripper close latency (ms)", [f"{1e3 * s:.0f}" for s in robustness.LATENCIES],
         factor("latency", robustness.LATENCIES), robustness.LATENCIES.index(0.030), (B, A, BASELINE), 2, False),
        ("Ball material (ball–pad friction)", [f"{rf.NAME_OF[mu].capitalize()}\n{mu:g}" for mu in rf.FRICTIONS],
         lambda approach: [rf.select(friction, approach, mu) for mu in rf.FRICTIONS], rf.FRICTIONS.index(rf.RUBBER),
         (B, A, BASELINE), 1, False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharey=True)
    for ax, (title, labels, rows_of, nominal_index, approaches, every, flat) in zip(axes.flat, panels):
        _panel(ax, title, labels, rows_of, nominal_index, approaches, every, flat)
        figures._style(ax, "", "Caught, of 100" if ax in axes[:, 0] else None)
    axes[0, 0].text(2.1, 49, "baseline: 44 at every level\n(never reads the sensor)", color=MUTED, fontsize=7.5,
                    va="bottom")

    # The sixth slot holds the key: which line is which, and what the shaded column means.
    key = axes.flat[-1]
    key.axis("off")
    handles = [Line2D([], [], color=figures.COLORS[a], linewidth=figures.LINE_WIDTH, marker="o",
                      markersize=figures.MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=figures.RING,
                      label=APPROACHES[a]) for a in (B, A, BASELINE)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=NOMINAL_SHADE, label="Nominal setting"))
    key.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=10, labelcolor=TEXT,
               handlelength=2.0)
    # NEW: the nominal setting's values, and what every approach catches there.
    caught = {a: robustness.select(main, a).caught for a in (BASELINE, A, B)}
    key.text(0.02, 0.6, "Nominal setting (shaded in every panel)", color=TEXT, fontsize=9, va="top",
             transform=key.transAxes)
    key.text(0.02, 0.52,
             f"Sensor noise {1e3 * nominal['sigma_p']:.0f} mm · lateral toss {nominal['sigma_lat']:g} m/s\n"
             f"Vertical toss {nominal['sigma_vz']:g} m/s · close latency 30 ms\n"
             f"Rubber ball (μ = {rf.RUBBER:g})\n"
             f"Caught there: baseline {caught[BASELINE]}, A {caught[A]}, B {caught[B]}\n"
             "One factor varies per panel; the rest stay nominal.\n"
             "Same 100 tosses at every point; bands are 95% intervals.\n"
             "Latency: the close trigger always assumes 30 ms.",
             transform=key.transAxes, color=MUTED, fontsize=8.2, va="top", linespacing=1.45)

    fig.tight_layout(h_pad=2.2, w_pad=1.6)
    return figures._save(fig, path_name)


if __name__ == "__main__":
    print(make())
