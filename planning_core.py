"""Planning the arm's motion to a catch: shared by the feasibility report
(feasibility.py) and the catch controller (catch.py).

Each motion is found by trajectory optimization over N_KNOTS knots, with the
pinch point kept at or below the ball (so the ball leaves the hand and later
comes down into it).  A solve that fails is not proof that no motion exists.
"""

from dataclasses import dataclass

import numpy as np
from manipulation.station import LoadScenario, MakeMultibodyPlant
from pydrake.all import (
    AngleBetweenVectorsConstraint,
    MathematicalProgram,
    PiecewisePolynomial,
    PositionConstraint,
    Solve,
)

import environment as env
import toss

N_KNOTS = 24  # trajectory intervals per motion
POSITION_TOL = 0.003  # how close the pinch point must get to the target (m), as in catch_pose


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


def initial_guess(kin, q0, v0, t0, T, p_target, n=N_KNOTS) -> Motion:
    """Cubic from (q0, v0) to a catch pose at rest, on n intervals, as a starting point."""
    q_end = toss.catch_pose(kin.plant, p_target)
    if q_end is None:
        q_end = toss.RELEASE_POSE
    curve = PiecewisePolynomial.CubicHermite(
        [t0, t0 + T], np.column_stack([q0, q_end]), np.column_stack([v0, np.zeros(7)])
    )
    t = np.linspace(t0, t0 + T, n + 1)
    return Motion(t, np.array([curve.value(ti).ravel() for ti in t]),
                  np.array([curve.EvalDerivative(ti, 1).ravel() for ti in t]))


def plan_reach(kin, plan, q0, v0, t0, T, p_target, a_max, direction=None, guess=None, speed_margin=toss.VELOCITY_MARGIN,
               n_knots=N_KNOTS):
    """A motion from (q0, v0) at t0 (s after release) that stops at t0 + T with
    the pinch point at p_target + delta * direction and the gripper within
    CATCH_TILT of up, keeping the pinch point at or below the ball throughout.

    With no direction, delta is 0 and the smoothest such motion is returned.
    With a direction, delta is made as large as possible.  a_max is a per-joint
    acceleration limit (rad/s^2), or None for velocity limits only.  speed_margin
    is the fraction of each joint's rated speed it may use.  A guess must be on
    n_knots intervals.  Returns a Motion, or None if the solver finds none.
    """
    n, h = n_knots, T / n_knots
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
    v_lim = speed_margin * kin.v_max
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
        guess = initial_guess(kin, q0, v0, t0, T, p_target, n)
    prog.SetInitialGuess(q, guess.q)
    prog.SetInitialGuess(v, guess.v)
    prog.SetInitialGuess(delta, [0.0])
    result = Solve(prog)
    if not result.is_success():
        return None
    return Motion(t, result.GetSolution(q), result.GetSolution(v), float(result.GetSolution(delta)[0]))


def accel_limits(kin):
    margin = toss.VELOCITY_MARGIN * kin.v_max
    return {"brake": margin / toss.BRAKE, "swing": margin / toss.SWING, "none": None}
