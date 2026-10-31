"""Can the arm get to the catch in time?  Three checks for planning the catch.

1. Gripper close time.  How long the WSG takes, in the full simulation, from
   the close command to the pads reaching the ball's width (FINGER_CONTACT).

2. Minimum time to the catch pose.  The fastest the arm can get from the toss
   to any pose that puts the pinch point at the planned catch point with the
   gripper within CATCH_TILT of up, and stop there, for tosses at TOSS_SPEED
   and each of TOSS_ANGLES.  Two hand-offs:
     - post-brake: from rest at the end of the toss's brake, as toss.py does now
     - release:    straight from the release state (RELEASE_POSE moving at u),
                   skipping the brake
   The model has no acceleration limits, so each is solved under three:
     - "brake": VELOCITY_MARGIN * v_max / BRAKE, what the toss's brake already
                commands (about 18 v_max per second)
     - "swing": VELOCITY_MARGIN * v_max / SWING, what the toss's swing commands
                (about 3.6 v_max per second)
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

from dataclasses import dataclass

import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation, MakeMultibodyPlant
from pydrake.all import (
    AngleBetweenVectorsConstraint,
    ConstantVectorSource,
    DiagramBuilder,
    EventStatus,
    MathematicalProgram,
    PiecewisePolynomial,
    PositionConstraint,
    RigidTransform,
    Simulator,
    Solve,
    TrajectorySource,
)

import environment as env
import toss

N_KNOTS = 24  # trajectory intervals per motion
BISECT_STEPS = 11  # minimum-time search resolution: 1 s / 2^11, about 0.5 ms
POSITION_TOL = 0.003  # how close the pinch point must get to the target (m), as in catch_pose

# When the close command is sent, in seconds; the offsets spread it across the
# WSG controller's 50 ms update period.
CLOSE_COMMAND_TIMES = [0.300, 0.310, 0.320, 0.330, 0.340]
FORCE_LIMITS = [20.0, 40.0, 80.0]  # N; 40 is the controller's default

TOSS_ANGLES = [65, 70, 75, 80, 85, 90]  # degrees above horizontal; 65 was toss.TOSS_ANGLE
ENVELOPE_ANGLES = [75, 80, 85]
CUTOFFS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.32]  # s after release
DIRECTIONS = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0]}

# Observation model for the prediction-error column (week 1 spec, section 8).
OBS_RATE = 100.0  # Hz
SIGMA_P = 0.005  # m, per axis
SIGMA_V = 0.05  # m/s, per axis


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


class Kinematics:
    """The iiwa and gripper alone (no floor, no ball), for planning.  Positions
    are the 7 arm joints followed by the 2 finger joints."""

    def __init__(self):
        self.plant = MakeMultibodyPlant(LoadScenario(data=env.ROBOT))
        assert toss.arm_slice(self.plant) == slice(0, 7)
        self.context = self.plant.CreateDefaultContext()
        self.plant_ad = self.plant.ToAutoDiffXd()
        self.context_ad = self.plant_ad.CreateDefaultContext()
        arm = toss.arm_slice(self.plant)
        self.v_max = self.plant.GetVelocityUpperLimits()[arm]
        self.q_lo = self.plant.GetPositionLowerLimits()[arm]
        self.q_hi = self.plant.GetPositionUpperLimits()[arm]

    def pinch(self, q_arm):
        """Pinch point in world, for float or AutoDiffXd joint angles."""
        if q_arm.dtype == float:
            plant, context = self.plant, self.context
        else:
            plant, context = self.plant_ad, self.context_ad
        plant.SetPositions(context, plant.GetModelInstanceByName("iiwa"), q_arm)
        X_WG = plant.CalcRelativeTransform(context, plant.world_frame(), toss.gripper_frame(plant))
        return X_WG.translation() + X_WG.rotation().matrix() @ env.PINCH


@dataclass
class Motion:
    """Knots of a planned arm motion: positions and velocities, with velocity
    linear and position quadratic between knots."""

    t: np.ndarray  # (N + 1,) seconds after release
    q: np.ndarray  # (N + 1, 7)
    v: np.ndarray  # (N + 1, 7)
    delta: float = 0.0  # how far the target was moved, for envelope solves

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity at time t, consistent with the knots."""
        k = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2))
        h, s = self.t[k + 1] - self.t[k], t - self.t[k]
        a = (self.v[k + 1] - self.v[k]) / h
        return self.q[k] + self.v[k] * s + a * s**2 / 2, self.v[k] + a * s

    def stretched(self, T: float) -> "Motion":
        """The same path, retimed to take T, as an initial guess."""
        rate = (self.t[-1] - self.t[0]) / T
        return Motion(self.t[0] + (self.t - self.t[0]) / rate, self.q, self.v * rate)

    def window(self, t0: float, t1: float, n: int = N_KNOTS) -> "Motion":
        """The part of the motion from t0 to t1, on n intervals of its own, as
        an initial guess."""
        t = np.linspace(t0, t1, n + 1)
        q, v = zip(*(self.state(ti) for ti in t))
        return Motion(t, np.array(q), np.array(v))


def ball_height(plan: toss.TossPlan, t):
    """Planned ball height t seconds after release."""
    return toss.ballistic_position(plan.p_release, plan.v_release, t)[..., 2]


def initial_guess(kin, q0, v0, t0, T, p_target) -> Motion:
    """Cubic from (q0, v0) to a catch pose at rest, as a starting point."""
    q_end = toss.catch_pose(kin.plant, p_target)
    if q_end is None:
        q_end = toss.RELEASE_POSE
    curve = PiecewisePolynomial.CubicHermite(
        [t0, t0 + T], np.column_stack([q0, q_end]), np.column_stack([v0, np.zeros(7)])
    )
    t = np.linspace(t0, t0 + T, N_KNOTS + 1)
    return Motion(t, np.array([curve.value(ti).ravel() for ti in t]),
                  np.array([curve.EvalDerivative(ti, 1).ravel() for ti in t]))


def plan_reach(kin, plan, q0, v0, t0, T, p_target, a_max, direction=None, guess=None):
    """A motion from (q0, v0) at t0 (s after release) that stops at t0 + T with
    the pinch point at p_target + delta * direction and the gripper within
    CATCH_TILT of up, keeping the pinch point at or below the ball throughout.

    With no direction, delta is 0 and the smoothest such motion is returned.
    With a direction, delta is made as large as possible.  a_max is a per-joint
    acceleration limit (rad/s^2), or None for velocity limits only.  Returns a
    Motion, or None if the solver finds none.
    """
    n, h = N_KNOTS, T / N_KNOTS
    t = t0 + h * np.arange(n + 1)
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(n + 1, 7, "q")
    v = prog.NewContinuousVariables(n + 1, 7, "v")
    fingers = prog.NewContinuousVariables(2, "fingers")  # only to fill the plant's position vector
    delta = prog.NewContinuousVariables(1, "delta")
    prog.AddBoundingBoxConstraint(0, 0, fingers)

    prog.AddBoundingBoxConstraint(q0, q0, q[0])
    prog.AddBoundingBoxConstraint(v0, v0, v[0])
    prog.AddBoundingBoxConstraint(0, 0, v[n])
    v_lim = toss.VELOCITY_MARGIN * kin.v_max
    for k in range(n):
        prog.AddLinearEqualityConstraint(q[k + 1] - q[k] - h / 2 * (v[k] + v[k + 1]), np.zeros(7))
        prog.AddBoundingBoxConstraint(-v_lim, v_lim, v[k + 1])
        prog.AddBoundingBoxConstraint(kin.q_lo, kin.q_hi, q[k + 1])
        if a_max is not None:
            prog.AddLinearConstraint(v[k + 1] - v[k], -h * a_max, h * a_max)

    # Pinch point at or below the ball between the hand-off and the catch.
    # Once the ball has come down past the catch height it is gone, so motions
    # longer than that are only held below it until then.
    W, G = kin.plant.world_frame(), toss.gripper_frame(kin.plant)
    t_catch, _ = toss.catch_point(plan)
    for k in range(1, n):
        if t[k] >= t_catch:
            break
        below = PositionConstraint(
            kin.plant, W, np.full(3, -np.inf), np.array([np.inf, np.inf, ball_height(plan, t[k])]),
            G, env.PINCH, kin.context,
        )
        prog.AddConstraint(below, np.concatenate([q[k], fingers]))

    # End: pinch point at the (moved) target, gripper pointing up.
    d = np.zeros(3) if direction is None else np.asarray(direction, dtype=float)
    prog.AddConstraint(
        lambda z: kin.pinch(z[:7]) - z[7] * d,
        p_target - POSITION_TOL, p_target + POSITION_TOL, np.concatenate([q[n], delta]),
    )
    tilt = AngleBetweenVectorsConstraint(
        kin.plant, W, np.array([0.0, 0.0, 1.0]), G, np.array([0.0, 1.0, 0.0]),
        0.0, np.radians(toss.CATCH_TILT), kin.context,
    )
    prog.AddConstraint(tilt, np.concatenate([q[n], fingers]))

    # Cost: smoothness (normalized squared accelerations), and, when moving the
    # target, mostly how far it moves.
    scale = 1 / (h * (a_max if a_max is not None else 30 * kin.v_max))
    for k in range(n):
        dv = (v[k + 1] - v[k]) * scale
        prog.AddQuadraticCost(1e-3 * dv.dot(dv) / n)
    if direction is None:
        prog.AddBoundingBoxConstraint(0, 0, delta)
    else:
        prog.AddBoundingBoxConstraint(0, 0.5, delta)
        prog.AddLinearCost(-100 * delta[0])

    if guess is None:
        guess = initial_guess(kin, q0, v0, t0, T, p_target)
    prog.SetInitialGuess(q, guess.q)
    prog.SetInitialGuess(v, guess.v)
    prog.SetInitialGuess(delta, [0.0])
    result = Solve(prog)
    if not result.is_success():
        return None
    return Motion(t, result.GetSolution(q), result.GetSolution(v), float(result.GetSolution(delta)[0]))


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


def prediction_error(t_cutoff: float, t_catch: float, with_velocity: bool) -> float:
    """Expected (1 sigma) error, along one horizontal axis, of the catch point
    predicted by least squares from observations at OBS_RATE up to t_cutoff.

    Horizontal motion is p0 + v0 t, so the fit is linear in (p0, v0).  With
    with_velocity, each observation also has a velocity reading.
    """
    times = np.arange(0.0, t_cutoff + 1e-9, 1 / OBS_RATE)
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


def accel_limits(kin):
    margin = toss.VELOCITY_MARGIN * kin.v_max
    return {"brake": margin / toss.BRAKE, "swing": margin / toss.SWING, "none": None}


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
        plan = toss.plan_toss(kin.plant, toss.release_velocity(toss.TOSS_SPEED, angle))
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
