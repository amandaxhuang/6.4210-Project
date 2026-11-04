"""Can the arm get to the catch in time?  Three checks for planning the catch.

1. Gripper close time.  How long the WSG takes, in the full simulation, from
   the close command to the pads reaching the ball's width (FINGER_CONTACT).

2. Minimum time to the catch pose.  The fastest the arm can get from the toss
   to any pose that puts the pinch point at the planned catch point with the
   gripper within CATCH_TILT of up, and stop there, for tosses at TOSS_SPEED
   and each of TOSS_ANGLES that it can reach.  Two hand-offs:
     - post-brake: from rest at the end of the toss's brake, as toss.py does now
     - release:    straight from the release state (RELEASE_POSE moving at u),
                   skipping the brake
   The model has no acceleration limits, so each is solved under three:
     - "brake": VELOCITY_MARGIN * v_max / BRAKE, what the toss's brake already
                commands (about 10 v_max per second)
     - "swing": VELOCITY_MARGIN * v_max / SWING, what the toss's swing commands
                (about 2 v_max per second)
     - "none":  velocity limits only, a lower bound
   Velocities stay within VELOCITY_MARGIN of the limits, as in toss.py.

3. Correction envelope, for the tosses in ENVELOPE_ANGLES (ones the arm can
   reach in time), under the "brake" acceleration limit.  While the arm follows
   a nominal catch motion (the smoothest one arriving exactly when the ball
   does), how far the catch point can still move, along +-x and +-y, if the new
   target is learned t_c seconds after release.  Next to it, how far off a
   least-squares prediction from the observations up to t_c is expected to be,
   so the two can be compared.

Each motion is found by trajectory optimization over N_KNOTS knots, with the
pinch point kept at or below the ball (so the ball leaves the hand and later
comes down into it).  A solve that fails is not proof that no motion exists,
so each minimum time is an upper bound on the true minimum, and each envelope
a lower bound on the true envelope.

    HEADLESS=1 python feasibility.py
"""

import numpy as np
from manipulation.station import MakeHardwareStation
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    EventStatus,
    PiecewisePolynomial,
    RigidTransform,
    Simulator,
    TrajectorySource,
)

import environment as env
from planning_core import Kinematics, accel_limits, plan_reach
from sensing import OBS_RATE, SIGMA_P, SIGMA_V  # the prediction-error column's observation model
import toss

BISECT_STEPS = 11  # minimum-time search resolution: 1 s / 2^11, about 0.5 ms

# When the close command is sent, in seconds; the offsets spread it across the
# WSG controller's 50 ms update period.
CLOSE_COMMAND_TIMES = [0.300, 0.310, 0.320, 0.330, 0.340]
FORCE_LIMITS = [20.0, 40.0, 80.0]  # N; 40 is the controller's default

# Degrees above horizontal.  At TOSS_SPEED the arm is at its limit, so other
# angles may not be reachable: those get a row saying so.
TOSS_ANGLES = [70, 75, toss.TOSS_ANGLE, 80, 85]
ENVELOPE_ANGLES = [toss.TOSS_ANGLE]
CUTOFFS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]  # s after release
DIRECTIONS = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0]}



# ---------------------------------------------------------------------------
# 1. Gripper close time
# ---------------------------------------------------------------------------


def measure_close(t_command: float, force_limit: float | None = None) -> dict:
    """Simulate the fingers closing from fully open with no ball in them.

    The arm is held still and the ball is parked on the floor out of reach.
    The fingers are commanded to FINGER_CLOSED at t_command, as when catching.
    Returns the delay until the fingers start to move, the time to reach
    FINGER_CONTACT (where the pads would touch a centered ball), and the peak
    closing speed of the opening.
    """
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(env.make_scenario(), meshcat=None))
    plant = station.GetSubsystemByName("plant")
    iiwa = plant.GetModelInstanceByName("iiwa")
    hold = builder.AddSystem(ConstantVectorSource(plant.GetDefaultPositions(iiwa)))
    builder.Connect(hold.get_output_port(), station.GetInputPort("iiwa.position"))

    wsg = plant.GetModelInstanceByName("wsg")
    plant.SetDefaultPositions(wsg, [-toss.FINGER_OPEN / 2, toss.FINGER_OPEN / 2])
    t_end = t_command + 0.3
    command = builder.AddSystem(TrajectorySource(PiecewisePolynomial.ZeroOrderHold(
        [0.0, t_command, t_end],
        np.array([[toss.FINGER_OPEN, env.FINGER_CLOSED, env.FINGER_CLOSED]]),
    )))
    builder.Connect(command.get_output_port(), station.GetInputPort("wsg.position"))
    if force_limit is not None:
        limit = builder.AddSystem(ConstantVectorSource([force_limit]))
        builder.Connect(limit.get_output_port(), station.GetInputPort("wsg.force_limit"))
    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    ball = plant.GetBodyByName(env.BODY, plant.GetModelInstanceByName("ball"))
    plant.SetFreeBodyPose(
        plant.GetMyMutableContextFromRoot(context), ball,
        RigidTransform([1.5, 1.5, env.BALL_RADIUS]),
    )

    state_port = station.GetOutputPort("wsg.state_measured")
    log = []  # (t, opening, opening rate)

    def record(root_context):
        width, rate = state_port.Eval(station.GetMyContextFromRoot(root_context))
        log.append((root_context.get_time(), width, rate))
        return EventStatus.Succeeded()

    simulator.set_monitor(record)
    simulator.AdvanceTo(t_end)

    t, width, rate = (np.array(column) for column in zip(*log))
    after = t >= t_command
    moving = after & (width < toss.FINGER_OPEN - 0.001)
    touching = after & (width <= env.FINGER_CONTACT)
    return {
        "start": t[np.argmax(moving)] - t_command if moving.any() else np.nan,
        "contact": t[np.argmax(touching)] - t_command if touching.any() else np.nan,
        "peak_speed": -rate[after].min(),
        "open_width": width[~after][-1],
    }


def report_close():
    print("## 1. Gripper close time (no ball, FINGER_OPEN -> FINGER_CLOSED command)\n")
    results = [measure_close(t_command) for t_command in CLOSE_COMMAND_TIMES]
    print(f"Open width before the command: {results[0]['open_width']:.4f} m; "
          f"pads touch a centered ball at {env.FINGER_CONTACT:.3f} m.\n")
    print("| command at (s) | force limit (N) | starts moving after (ms) "
          "| reaches ball width after (ms) | peak closing speed (m/s) |")
    print("|---|---|---|---|---|")
    for t_command, r in zip(CLOSE_COMMAND_TIMES, results):
        print(f"| {t_command:.3f} | 40 (default) | {1e3 * r['start']:.0f} "
              f"| {1e3 * r['contact']:.0f} | {r['peak_speed']:.2f} |")
    for force in FORCE_LIMITS:
        if force == 40.0:
            continue
        r = measure_close(CLOSE_COMMAND_TIMES[0], force)
        print(f"| {CLOSE_COMMAND_TIMES[0]:.3f} | {force:.0f} | {1e3 * r['start']:.0f} "
              f"| {1e3 * r['contact']:.0f} | {r['peak_speed']:.2f} |")
    print()


# ---------------------------------------------------------------------------
# 2 and 3. Reaching the catch
# ---------------------------------------------------------------------------


def min_time(kin, plan, q0, v0, t0, p_target, a_max, t_max=1.0):
    """Shortest T for which plan_reach finds a motion, by bisection.  Returns
    (T, motion), or (inf, None) if none is found even at t_max."""
    best = plan_reach(kin, plan, q0, v0, t0, t_max, p_target, a_max)
    if best is None:
        return np.inf, None
    lo, hi = 0.0, t_max
    for _ in range(BISECT_STEPS):
        mid = (lo + hi) / 2
        motion = plan_reach(kin, plan, q0, v0, t0, mid, p_target, a_max,
                            guess=best.stretched(mid))
        if motion is None:
            motion = plan_reach(kin, plan, q0, v0, t0, mid, p_target, a_max)
        if motion is None:
            lo = mid
        else:
            hi, best = mid, motion
    return hi, best


def saturated_joints(kin, motion) -> str:
    """Joints that reach their velocity limit somewhere in the motion."""
    v_lim = toss.VELOCITY_MARGIN * kin.v_max
    hit = np.flatnonzero((np.abs(motion.v) >= 0.98 * v_lim).any(axis=0))
    return ", ".join(str(j + 1) for j in hit) or "none"


def prediction_error(t_cutoff: float, t_catch: float, with_velocity: bool, t_first: float = 0.0) -> float:
    """Expected (1 sigma) error, along one horizontal axis, of the catch point
    predicted by least squares from observations at OBS_RATE from t_first to
    t_cutoff.

    Horizontal motion is p0 + v0 t, so the fit is linear in (p0, v0).  With
    with_velocity, each observation also has a velocity reading.  The report's
    default t_first = 0 counts the sample at release, when the ball is still in
    the fingers; the catch controller's fit starts at t_first = 0.01.
    """
    times = np.arange(t_first, t_cutoff + 1e-9, 1 / OBS_RATE)
    rows = [np.array([1.0, ti]) / SIGMA_P for ti in times]
    if with_velocity:
        rows += [np.array([0.0, 1.0]) / SIGMA_V for _ in times]
    A = np.array(rows)
    if np.linalg.matrix_rank(A) < 2:
        return np.inf
    x = np.array([1.0, t_catch])
    return float(np.sqrt(x @ np.linalg.inv(A.T @ A) @ x))


def hand_offs(plan):
    """(name, t0 after release, q0, v0) for the two ways of starting the catch."""
    q_stop = plan.q_traj.value(plan.t_release + toss.BRAKE).ravel()
    return [
        ("post-brake", toss.BRAKE, q_stop, np.zeros(7)),
        ("release", 0.0, toss.RELEASE_POSE, plan.qdot_release),
    ]


def report_min_time(kin):
    limits = accel_limits(kin)
    print("## 2. Minimum time to the catch pose\n")
    print(f"Tosses at {toss.TOSS_SPEED} m/s; catch height CATCH_DROP = {toss.CATCH_DROP} m below "
          f"release.  Acceleration limits (rad/s^2): brake {np.round(limits['brake'], 1)}, "
          f"swing {np.round(limits['swing'], 1)}.  Times are seconds after release; a dash "
          "means no motion was found.\n")
    print("| toss angle | catch point (m) | ball arrives | post-brake: brake | swing | none "
          "| release: brake | swing | none | joints at velocity limit (post-brake, brake) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for angle in TOSS_ANGLES:
        try:
            plan = toss.plan_toss(kin.plant, toss.release_velocity(toss.TOSS_SPEED, angle))
        except ValueError:
            print(f"| {angle} | not reachable at {toss.TOSS_SPEED} m/s | | | | | | | | |")
            continue
        t_catch, p_catch = toss.catch_point(plan)
        arrivals, saturated = [], ""
        for name, t0, q0, v0 in hand_offs(plan):
            for level, a_max in limits.items():
                T, motion = min_time(kin, plan, q0, v0, t0, p_catch, a_max)
                if motion is None:
                    arrivals.append("-")
                    continue
                mark = "" if t0 + T <= t_catch else " (late)"
                arrivals.append(f"{t0 + T:.3f}{mark}")
                if name == "post-brake" and level == "brake":
                    saturated = saturated_joints(kin, motion)
        print(f"| {angle} | {np.round(p_catch, 3)} | {t_catch:.3f} | " + " | ".join(arrivals)
              + f" | {saturated} |")
    print()


def report_envelope(kin):
    a_max = accel_limits(kin)["brake"]
    print("## 3. Correction envelope (brake acceleration limit)\n")
    print("How far (cm) the catch point can still move, from a nominal motion that arrives "
          "exactly with the ball, if the new target is known at t_c (s after release).  "
          "Prediction error: 1 sigma per horizontal axis from least squares on observations "
          f"up to t_c (sigma_p = {1e3 * SIGMA_P:.0f} mm, sigma_v = {SIGMA_V} m/s, "
          f"{OBS_RATE:.0f} Hz).\n")
    print("| toss angle | hand-off | t_c | time left | +x | -x | +y | -y "
          "| prediction error, position only | with velocity |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for angle in ENVELOPE_ANGLES:
        plan = toss.plan_toss(kin.plant, toss.release_velocity(toss.TOSS_SPEED, angle))
        t_catch, p_catch = toss.catch_point(plan)
        for name, t0, q0, v0 in hand_offs(plan):
            base = plan_reach(kin, plan, q0, v0, t0, t_catch - t0, p_catch, a_max)
            if base is None:
                print(f"| {angle} | {name} | | | no nominal motion found | | | | | |")
                continue
            for t_c in CUTOFFS:
                if t_c < t0:
                    continue
                q_c, v_c = base.state(t_c)
                reach = []
                for d in DIRECTIONS.values():
                    motion = plan_reach(kin, plan, q_c, v_c, t_c, t_catch - t_c, p_catch, a_max,
                                        direction=d, guess=base.window(t_c, t_catch))
                    reach.append("fail" if motion is None else f"{100 * motion.delta:.1f}")
                err_p = prediction_error(t_c, t_catch, with_velocity=False)
                err_v = prediction_error(t_c, t_catch, with_velocity=True)
                print(f"| {angle} | {name} | {t_c:.2f} | {t_catch - t_c:.3f} | " + " | ".join(reach)
                      + f" | {100 * err_p:.1f} | {100 * err_v:.1f} |")
    print()


def run():
    report_close()
    kin = Kinematics()
    report_min_time(kin)
    report_envelope(kin)


if __name__ == "__main__":
    run()
