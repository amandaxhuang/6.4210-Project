"""How much error the catch can take, and how well the arm tracks its plan.

1. Gripper response.  How long the pads take to reach the ball's width, against
   where the command lands in the WSG controller's 50 ms update.
2. Trigger lead (spec deliverable A3).  Whether a perfect toss is caught, against
   when the pads get there.
3. Sideways tolerance.  Whether it is caught with the hand deliberately aimed
   off the ball, along the closing axis and across the pads.
4. Tracking (spec section 14).  How far the pinch point is from where the plan
   commands it, through a catch motion.
5. Timing the close on noisy tosses, to choose how to trigger it.

The sweeps in 2 and 3 run the open-loop baseline on a noiseless toss, so that the
only error is the one being swept.

    HEADLESS=1 python catch_timing.py [response lead offsets tracking trigger]
"""

import multiprocessing
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")  # the workers are the parallelism

import numpy as np

import catch
import environment as env
import feasibility
import toss

N_WORKERS = 7
LEADS = [l / 1000 for l in range(50, 91)]  # s before t_catch, including the latency
OFFSETS = [o / 1000 for o in range(-40, 41, 5)]  # m of deliberate aim error
OFFSET_LEADS = [0.055, 0.056, 0.058, 0.062, 0.066]  # s; 55 is the last on the late side of the gap
# Approach A on noisy tosses, by how the close is timed: a fixed lead, or a pad lead
# looked up through the gripper's response.
NOISE_LEVELS = [0.002, 0.005]  # m
FIXED_LEADS = [0.055, 0.060]  # s
PAD_LEADS = [0.0, 0.003, 0.005, 0.008, 0.012]  # s
N_TRIALS = 60


class _AimedOff(catch.OpenLoopPolicy):
    """The baseline, aiming the hand off the ball by a fixed offset."""

    def __init__(self, planner, offset):
        super().__init__(planner)
        planner.nominal = catch.Nominal(planner.nominal.t_catch, planner.nominal.p_catch + offset)


def init_worker():
    # make_scenario rewrites the SDFs on every run, so each worker needs its own.
    env.ASSET_DIR = Path(tempfile.mkdtemp(prefix="catch_timing_"))


def verdict(outcome) -> str:
    return "caught" if outcome.caught else ("touched" if outcome.contact else "missed")


def pads_at_ball(run):
    """When (s) the pads reach ball width after the close command, or nan."""
    t_command = run.controller.t_close + run.controller.latency
    reached = (run.log.t >= t_command) & (run.log.finger_width <= env.FINGER_CONTACT)
    return run.log.t[np.argmax(reached)] if reached.any() else np.nan


def run_response(k):
    r = feasibility.measure_close(0.300 + k / 1000)
    return k, r["contact"]


def run_lead(lead):
    run = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0, lead=lead)
    plan = run.plans[-1]
    t_command = run.controller.t_close + run.controller.latency
    return dict(lead=lead, command=plan.t_catch - t_command,
                phase=t_command % 0.05, pads=pads_at_ball(run) - plan.t_catch,
                verdict=verdict(run.outcome), hold=run.outcome.hold_time, end=run.outcome.end_in_hand,
                z_end=run.log.ball_p[-1, 2])


def run_offset(args):
    lead, axis, offset, direction = args
    run = catch.simulate_catch(lambda planner: _AimedOff(planner, offset * direction), sigma_p=0.0, lead=lead)
    plan = run.plans[-1]
    if not plan.solve_ok:
        return dict(lead=lead, axis=axis, offset=offset, verdict="no plan")
    return dict(lead=lead, axis=axis, offset=offset, verdict=verdict(run.outcome), end=run.outcome.end_in_hand,
                in_hand=run.outcome.miss_in_hand)


def run_noisy(args):
    sigma_p, lead, pad_lead, trial = args
    run = catch.simulate_catch(catch.PredictOncePolicy, sigma_p=sigma_p, lead=lead,
                               pad_lead=catch.PAD_LEAD if pad_lead is None else pad_lead, trial=trial)
    o, plan = run.outcome, run.plans[-1]
    return dict(sigma_p=sigma_p, lead=lead, pad_lead=pad_lead, trial=trial, caught=o.caught, contact=o.contact,
                solve_ok=plan.solve_ok, rejected=plan.intercept_rejected, timing=o.timing_error,
                predicted=o.predicted_miss, arrival=o.arm_arrival_error, on_pad=abs(o.end_in_hand[1]) <= 0.01)


def report_response(pool):
    print("## 1. Gripper response\n")
    print("From the close command reaching the fingers to the pads at ball width, with the arm "
          "still and no ball, by where the command lands after a WSG controller update.\n")
    rows = pool.map(run_response, range(50))
    response = {k: t for k, t in rows}
    print("| command, ms after an update | reaches ball width after (ms) | arrives, ms after that update |")
    print("|---|---|---|")
    for k in [0, 1, 10, 20, 27, 28, 32, 35, 38, 41, 44, 47, 49]:
        print(f"| {k} | {1e3 * response[k]:.0f} | {k + 1e3 * response[k]:.0f} |")
    print()
    print("GRIPPER_RESPONSE_MS =", [round(1e3 * response[k]) for k in range(50)])
    print()


def report_lead(pool):
    print("## 2. Trigger lead\n")
    print("A perfect toss, by how long before t_catch the decision to close is taken.  The latency "
          f"of {1e3 * catch.T_LATENCY:.0f} ms is included, so the fingers get the command that much "
          "later.  'Ball ends' is relative to the pinch point, in cm along the fingers and across "
          "the pads; the pad is 2 cm square, so anything past 1 cm has its center off the pad.\n")
    rows = pool.map(run_lead, LEADS)
    print("| lead (ms) | command before t_catch (ms) | ms after update | pads vs t_catch (ms) | result | held (s) | ball ends, fingers | across | height (m) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {1e3 * r['lead']:.0f} | {1e3 * r['command']:.0f} | {1e3 * r['phase']:.1f} | {1e3 * r['pads']:+.1f} "
              f"| {r['verdict']} | {r['hold']:.2f} | {100 * r['end'][1]:+.1f} | {100 * r['end'][2]:+.1f} | {r['z_end']:.3f} |")
    print()
    return rows


def report_offsets(pool):
    print("## 3. Sideways tolerance\n")
    print("A perfect toss with the hand aimed off the ball, horizontally, by the offset (cm).  The "
          "axes are the closing axis and the across-pad axis at the nominal catch, flattened.  "
          "C caught, t touched, . missed, x no plan found.\n")
    nominal = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0)
    i = np.searchsorted(nominal.log.t, nominal.plans[-1].t_catch)
    R = nominal.log.R_WG[i]
    tilt = np.degrees(np.arccos(np.clip(R[2, 1], -1, 1)))
    v = nominal.log.ball_v[np.searchsorted(nominal.log.t, nominal.plans[-1].t_catch - 0.02)]  # before any contact
    off_axis = np.degrees(np.arccos(np.clip(-v @ R[:, 1] / np.linalg.norm(v), -1, 1)))
    print(f"At the nominal catch the gripper is {tilt:.1f} deg from up, and the ball comes in "
          f"{off_axis:.1f} deg off the finger axis.\n")
    axes = {}
    for name, column in (("closing", 0), ("across pad", 2)):
        flat = R[:, column] * np.array([1, 1, 0])
        axes[name] = flat / np.linalg.norm(flat)
    jobs = [(lead, name, o, direction) for lead in OFFSET_LEADS
            for name, direction in axes.items() for o in OFFSETS]
    rows = pool.map(run_offset, jobs)
    marks = {"caught": "C", "touched": "t", "missed": ".", "no plan": "x"}
    print("| lead (ms) | axis | " + " | ".join(f"{100 * o:+.1f}" for o in OFFSETS) + " |")
    print("|---|---|" + "---|" * len(OFFSETS))
    for lead in OFFSET_LEADS:
        for name in axes:
            cells = [marks[r["verdict"]] for r in rows if r["lead"] == lead and r["axis"] == name]
            print(f"| {1e3 * lead:.0f} | {name} | " + " | ".join(cells) + " |")
    print()
    return rows


def report_trigger(pool):
    print("## 5. Timing the close on noisy tosses\n")
    print(f"Approach A, {N_TRIALS} trials per row, the same noise in every row.  'Lookup' aims the pads "
          "that long before t_catch through the measured gripper response; 'fixed' is the spec's "
          "constant lead.  'On pad' counts catches with the ball's center within 1 cm of the pad's.\n")
    settings = [(lead, None) for lead in FIXED_LEADS] + [(None, pad) for pad in PAD_LEADS]
    jobs = [(sigma, lead, pad, trial) for sigma in NOISE_LEVELS for lead, pad in settings
            for trial in range(N_TRIALS)]
    rows = pool.map(run_noisy, jobs)
    print("| sigma_p | trigger | caught | on pad | contact | no plan | timing error, 1 sigma (ms) | predicted miss, median (cm) | arm arrival error, median / worst (cm) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sigma in NOISE_LEVELS:
        for lead, pad in settings:
            r = [x for x in rows if x["sigma_p"] == sigma and x["lead"] == lead and x["pad_lead"] == pad]
            name = f"fixed {1e3 * lead:.0f} ms" if lead is not None else f"lookup {1e3 * pad:.0f} ms"
            arrival = [x["arrival"] for x in r if x["solve_ok"]]
            print(f"| {1e3 * sigma:.0f} mm | {name} | {sum(x['caught'] for x in r)} | "
                  f"{sum(x['caught'] and x['on_pad'] for x in r)} | {sum(x['contact'] for x in r)} | "
                  f"{sum(not x['solve_ok'] for x in r)} | {1e3 * np.std([x['timing'] for x in r]):.1f} | "
                  f"{100 * np.median([x['predicted'] for x in r]):.1f} | "
                  f"{100 * np.median(arrival):.2f} / {100 * np.max(arrival):.2f} |")
    print()


def report_tracking(pool=None):
    print("## 4. Tracking through a catch motion\n")
    run = catch.simulate_catch(catch.OpenLoopPolicy, sigma_p=0.0)
    log, plan, t_release = run.log, run.plans[-1], run.toss_plan.t_release
    kin = catch.shared_kinematics()
    error = np.linalg.norm(log.pinch - np.array([kin.pinch(q) for q in log.q_cmd]), axis=1)
    v_max, effort_max = toss.arm_limits(run.plant)
    during = (log.t >= plan.q_traj.start_time()) & (log.t <= plan.t_catch)

    print("The nominal catch, from the arm's measured state and its commanded one.\n")
    print("| s after release | pinch point vs command (cm) | joint 6 error (mrad) | joint 6 torque / limit | fastest joint / limit |")
    print("|---|---|---|---|---|")
    moments = [("0.101 (hand-off)", 0.101), ("0.150", 0.150), ("0.200", 0.200), ("0.300", 0.300), ("0.400", 0.400),
               ("0.500", 0.500), (f"{plan.t_catch - t_release:.3f} (t_catch)", plan.t_catch - t_release),
               ("0.625", 0.625), ("0.650", 0.650), ("0.700", 0.700), ("0.800", 0.800)]
    for name, t in moments:
        i = np.searchsorted(log.t, t_release + t)
        print(f"| {name} | {100 * error[i]:.2f} | {1e3 * (log.q_cmd[i, 5] - log.arm_q[i, 5]):+.1f} "
              f"| {log.arm_effort[i, 5] / effort_max[5]:+.2f} | {np.abs(log.arm_v[i] / v_max).max():.2f} |")
    print()

    effort = np.abs(log.arm_effort[during]) / effort_max
    saturated = log.t[during][effort[:, 5] >= 0.99] - t_release
    ts = np.linspace(plan.q_traj.start_time(), plan.t_catch, 400)
    planned = np.array([plan.q_traj.EvalDerivative(t, 2).ravel() for t in ts])
    a_max = catch.accel_limits(kin)["brake"]
    print(f"- Arm arrival error: {100 * run.outcome.arm_arrival_error:.2f} cm from the planned catch point")
    print(f"- Joint 6 at 99% of its torque limit for {len(saturated)} simulator steps, "
          f"{saturated.min():.3f} to {saturated.max():.3f} s after release; no other joint gets there "
          f"(next highest {np.delete(effort.max(axis=0), 5).max():.2f})")
    print(f"- Peak joint speed: {np.round(np.abs(log.arm_v[during]).max(axis=0) / v_max, 3)} of limit")
    print(f"- Planned peak acceleration: {np.round(np.abs(planned).max(axis=0) / a_max, 2)} of the limit; "
          f"joint 6 still decelerating at {abs(planned[-1, 5]):.1f} rad/s^2 on arrival")
    print()


def run(sections):
    with multiprocessing.get_context("spawn").Pool(N_WORKERS, initializer=init_worker) as pool:
        for name, section in (("response", report_response), ("lead", report_lead), ("offsets", report_offsets),
                              ("tracking", report_tracking), ("trigger", report_trigger)):
            if name in sections:
                section(pool)


if __name__ == "__main__":
    run(sys.argv[1:] or ["response", "lead", "offsets", "tracking", "trigger"])
