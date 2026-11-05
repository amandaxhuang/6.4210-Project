"""Catching the ball: one controller, with one decision left to a policy.

CatchController does everything the approaches share.  It follows the toss,
feeds the sensor's observations to the fit, follows the catch plan, and drives
the fingers.  A policy decides only which CatchPlan to follow and when, so a
difference between approaches can only come from the policy:

    OpenLoopPolicy     the reference baseline: no observations, one plan to where
                       the nominal toss comes down
    PredictOncePolicy  Approach A: one plan from a fit over an early window

replan.py adds Approach B to these.

The timeline, in seconds after release: the brake ends at 0.100, the decision
comes on the first control tick after it, the close command goes out at about
0.52, and the ball crosses the catch plane at about 0.575.

    HEADLESS=1 python catch.py [open_loop | predict_once]
"""

from dataclasses import dataclass, field
from functools import lru_cache
import math
import sys
import time
from typing import Protocol

import numpy as np
from pydrake.all import (
    AbstractValue,
    DiagramBuilder,
    EventStatus,
    LeafSystem,
    PiecewisePolynomial,
    Simulator,
    SpatialVelocity,
)

import environment as env
from planning_core import Kinematics, Motion, N_KNOTS, accel_limits, plan_reach
import sensing
from sensing import BallisticFit, BallSensor, Estimate, Observation
import toss
from utils.viz import HEADLESS, keep_meshcat_open, publish_recording, start_recording

T_WINDOW = 0.10  # s after release; how long Approach A observes before deciding
T_MIN_MOTION = 0.41  # s; the fastest the arm reaches the catch from rest at the toss's speed (results/catch_feasibility.md)
# Of each joint's rated speed.  Drake doesn't enforce the rating, and the arm overshoots its plan by up to about 20%,
# so a plan at the toss's margin relies on speeds the arm isn't rated for: results/approach_a.md.
PLAN_SPEED_LIMIT = 0.85
T_MAX_CATCH = 1.0  # s after release; an intercept later than this is not believed
DECISION_DEADLINE = 0.165  # s after release; past this no motion from rest can arrive in time

T_LATENCY = 0.030  # s from deciding to close to the command reaching the fingers
PAD_LEAD = 0.003  # s before t_catch that the pads should reach the ball's width (results/catch_timing.md)

# From the close command reaching the fingers to the pads at ball width (ms), by
# how many ms after a WSG controller update (every 50 ms) the command lands.
# Measured by catch_timing.py.
GRIPPER_RESPONSE_MS = [
    20, 28, 28, 28, 28, 28, 28, 28, 28, 28,
    28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
    28, 28, 28, 28, 28, 28, 28, 28, 27, 27,
    27, 27, 26, 26, 26, 25, 25, 25, 24, 24,
    24, 23, 23, 23, 22, 22, 22, 21, 21, 21,
]

# The controller runs at 200 Hz, one plant step after each multiple of 5 ms.
# Everything that happens at one instant sees the state from before that instant,
# so a controller on the sensor's own ticks would see each sample a tick late.
CONTROL_PERIOD = 0.005
CONTROL_OFFSET = 0.001

SEED_ROOT = 4210

# Toss noise goes straight onto the ball's velocity this long after release (s).  The
# ball has left the fingers by about 6 ms and the first observation is at 10 ms, so
# the fit only ever sees the perturbed flight.  It can't go on the commanded release,
# as the toss is already at the arm's joint speed limits.  Spec section 12.
T_TOSS_NOISE = 0.008

# The simulated toss comes down a repeatable distance from the planned one (the
# ball leaves slower toward -x than planned).  The baseline is allowed to know
# this: it stands in for a catcher that has seen the nominal toss before.  If the
# toss changes, re-measure with measure_nominal_offset().
NOMINAL_TIME_OFFSET = 0.00067  # s
NOMINAL_OFFSET = np.array([0.01375, -0.00180, 0.0])  # m

# Grip force below this counts as not gripping (N).  The spec says only "non-zero",
# and fingers closed on nothing read under 0.05 N.  This sets the early edge of the
# catch window, though it barely changes noisy catch rates: results/catch_timing.md.
GRIP_FORCE_MIN = 1.0
HOLD_TIME = 0.5  # s the ball must stay gripped to count as caught
PAD_HALF_SIZE = 0.01  # m; the fingertip pads are 2 cm square
FINGER_STOPPED_SPEED = 0.02  # m/s of finger opening rate
FINGER_SETTLED_ROWS = 10  # simulator steps it must stay below that, to count as stopped


@dataclass(frozen=True)
class Nominal:
    """Where the nominal toss comes down."""

    t_catch: float  # s after release
    p_catch: np.ndarray


def nominal_catch(plan: toss.TossPlan) -> Nominal:
    """Where this toss comes down in simulation, from the plan alone."""
    t, p = toss.catch_point(plan)
    return Nominal(t + NOMINAL_TIME_OFFSET, p + NOMINAL_OFFSET)


def measure_nominal_offset() -> tuple[float, np.ndarray]:
    """Simulate a toss with no catch and measure how far its crossing of the
    catch plane is from the planned one.  Returns (time offset, position offset)."""
    plan, log, _ = toss.simulate()
    fly = toss.free_flight(plan, log)
    t, p = toss.catch_point(plan)
    return fly.t_down - plan.t_release - t, fly.p_down - p


def ball_on_pad(in_hand: np.ndarray) -> bool:
    """Whether the ball's center is on the pads rather than off their edges, from
    its position relative to the pinch point in the gripper frame."""
    return bool(abs(in_hand[1]) <= PAD_HALF_SIZE and abs(in_hand[2]) <= PAD_HALF_SIZE)


def toss_velocity_noise(trial: int, sigma_lat: float, sigma_vz: float) -> np.ndarray:
    """What this trial's toss adds to the ball's velocity (m/s, world frame).  It
    depends only on the trial, so a trial is the same toss under every policy, and
    a larger sigma scales the same draw."""
    draw = np.random.default_rng([SEED_ROOT, trial, 0]).normal(size=3)
    return draw * np.array([sigma_lat, sigma_lat, sigma_vz])


def intercept(est: Estimate, z_catch: float, t_rel: float, t_min_motion: float = T_MIN_MOTION) -> tuple[float, np.ndarray] | None:
    """When (s after release) and where the fitted flight comes down through
    z_catch, or None if it never does, or does so too soon to reach or
    implausibly late.  t_min_motion is the least time to reach it from t_rel, from
    rest unless given otherwise.  Unreachable targets inside those times are left
    for the planner to fail on: that is a result, not something to hide here."""
    if est.v0[2] ** 2 - 2 * toss.G_ACCEL * (z_catch - est.p0[2]) < 0:
        return None
    t = toss.time_to_height(est.p0, est.v0, z_catch)
    if not t_rel + t_min_motion <= t <= T_MAX_CATCH:
        return None
    return t, toss.ballistic_position(est.p0, est.v0, t)


def finger_command_time(t_catch: float, pad_lead: float = PAD_LEAD) -> float:
    """When the close command should reach the fingers for the pads to reach the
    ball's width pad_lead before t_catch, looked up from the measured response.
    Both are simulation times.

    The pads can't arrive 20 to 29 ms after a controller update: a command just
    before an update arrives at 20 ms, and one just after it at 29 ms.  A target in
    that gap goes to the nearer side.  A command landing exactly on an update could
    fall on either side by rounding, so none does.
    """
    target = (t_catch - pad_lead) * 1000
    last = int(np.ceil(target))
    commands = [k for k in range(last - 60, last + 1) if k % 50 != 0]
    best = min(commands, key=lambda k: abs(k + GRIPPER_RESPONSE_MS[k % 50] - target))
    return (best - 0.5) / 1000  # half a step early, so the step at ms `best` sees it


def close_command_time(t_catch: float, lead: float | None = None, latency: float = T_LATENCY,
                       pad_lead: float = PAD_LEAD) -> float:
    """When to decide to close, for a ball crossing the catch plane at t_catch: a
    latency before the fingers should get the command, or lead before t_catch if
    that is given.  Both are simulation times."""
    if lead is not None:
        return t_catch - lead
    return finger_command_time(t_catch, pad_lead) - latency


def motion_to_trajectory(motion: Motion, t_release: float) -> PiecewisePolynomial:
    """A Motion as a trajectory in simulation time.  Velocity is linear between
    knots, so the cubic Hermite through them is that same quadratic."""
    return PiecewisePolynomial.CubicHermite(motion.t + t_release, motion.q.T, motion.v.T)


@dataclass
class CatchPlan:
    """One planned motion to a catch."""

    q_traj: PiecewisePolynomial | None  # in simulation time; None if the solve failed
    t_catch: float  # simulation time the ball reaches the catch plane
    p_catch: np.ndarray
    est: Estimate | None  # what it was planned from; None for the baseline
    solve_time: float  # wall-clock seconds
    solve_ok: bool
    fit_rank_deficient: bool = False  # fell back to nominal: too few observations
    intercept_rejected: bool = False  # fell back to nominal: unbelievable intercept
    motion: Motion | None = None  # the knots behind q_traj, to warm-start another solve from


class CatchPlanner:
    """Turns a target into a CatchPlan.  Every policy plans through this, so every
    policy gets the same optimizer, limits and catch plane."""

    def __init__(self, kin: Kinematics, toss_plan: toss.TossPlan):
        self.kin = kin
        self.toss_plan = toss_plan
        # The catch relies on accelerations the model doesn't limit, so the plan must.
        self.a_max = accel_limits(kin)["brake"]
        self.z_catch = toss_plan.p_release[2] - toss.CATCH_DROP
        self.nominal = nominal_catch(toss_plan)

    def plan(self, t_rel, q0, v0, t_catch, p_catch, est=None, guess=None, n_knots=N_KNOTS, **flags) -> CatchPlan:
        """Plan from (q0, v0) at t_rel to rest at p_catch at t_catch (both in s
        after release).  A failed solve is reported, not retried: a retry one
        approach gets and another doesn't would break the comparison."""
        start = time.perf_counter()
        motion = plan_reach(self.kin, self.toss_plan, q0, v0, t_rel, t_catch - t_rel,
                            p_catch, self.a_max, guess=guess, speed_margin=PLAN_SPEED_LIMIT, n_knots=n_knots)
        solve_time = time.perf_counter() - start
        t_release = self.toss_plan.t_release
        q_traj = None if motion is None else motion_to_trajectory(motion, t_release)
        return CatchPlan(q_traj, t_release + t_catch, p_catch, est, solve_time,
                         motion is not None, motion=motion, **flags)


class CatchPolicy(Protocol):
    def update(self, t_rel: float, est: Estimate | None, q_cmd: np.ndarray,
               qdot_cmd: np.ndarray, plan: CatchPlan | None) -> CatchPlan | None:
        """A new plan to adopt, or None to keep the current one.

        t_rel     seconds since release
        est       the current fit, or None if there isn't one yet
        q_cmd     the position currently being commanded (not measured)
        qdot_cmd  its derivative
        plan      the plan being followed, or None before the first decision

        A plan takes over when its trajectory starts, which may be later than now.
        A policy may name the estimator it wants in fit_class; the default is
        BallisticFit.
        """


class OpenLoopPolicy:
    """The reference baseline: uses no observations, and plans once, as soon as
    the arm is free, to where the nominal toss comes down."""

    def __init__(self, planner: CatchPlanner):
        self.planner = planner
        self.decided = False

    def update(self, t_rel, est, q_cmd, qdot_cmd, plan):
        if self.decided or t_rel < toss.BRAKE:
            return None
        self.decided = True
        nominal = self.planner.nominal
        return self.planner.plan(t_rel, q_cmd, qdot_cmd, nominal.t_catch, nominal.p_catch)


class PredictOncePolicy:
    """Approach A: at t_window, fit the flight so far, and plan once from that."""

    def __init__(self, planner: CatchPlanner, t_window: float = T_WINDOW):
        self.planner = planner
        self.t_window = t_window
        self.decided = False

    def update(self, t_rel, est, q_cmd, qdot_cmd, plan):
        if self.decided or t_rel < self.t_window:
            return None
        self.decided = True
        nominal = self.planner.nominal
        if est is None:
            return self.planner.plan(t_rel, q_cmd, qdot_cmd, nominal.t_catch, nominal.p_catch,
                                     fit_rank_deficient=True)
        target = intercept(est, self.planner.z_catch, t_rel)
        if target is None:
            return self.planner.plan(t_rel, q_cmd, qdot_cmd, nominal.t_catch, nominal.p_catch,
                                     est=est, intercept_rejected=True)
        return self.planner.plan(t_rel, q_cmd, qdot_cmd, *target, est=est)


@dataclass(frozen=True)
class _Decision:
    """The controller's abstract state: what it is following right now."""

    plan: CatchPlan | None = None
    t_close: float = np.inf  # simulation time of the decision to close
    pending: CatchPlan | None = None  # planned, but its compute time hasn't passed
    t_adopt: float = np.inf


class CatchController(LeafSystem):
    """Follows the toss, then whichever CatchPlan the policy last returned.

    Runs at 200 Hz, but its commands are functions of time, so the arm sees the
    same smooth trajectories it would from a TrajectorySource.  The policy and
    the fit are per-run objects: a simulator only ever runs this forward.

    latency delays the finger command by a fixed time, for every policy.  The
    trigger allows for it, unless assumed_latency says it allows for another value.
    lead overrides how long before t_catch the decision to close is taken.
    pad_lead is how long before t_catch the pads should reach the ball's width.
    compute_latency delays when a plan may first be acted on, and a plan is never
    acted on before its own trajectory starts.
    """

    def __init__(self, toss_plan, policy, fit, latency=T_LATENCY, lead=None, compute_latency=0.0,
                 pad_lead=PAD_LEAD, assumed_latency=None):
        LeafSystem.__init__(self)
        self.set_name("catch_controller")
        self.toss_plan = toss_plan
        self.policy = policy
        self.fit = fit
        self.latency = latency
        self.lead = lead
        self.compute_latency = compute_latency
        self.pad_lead = pad_lead
        self.assumed_latency = latency if assumed_latency is None else assumed_latency
        self.t_close = np.inf  # the latest decision to close, for the trial record
        self.t_close_first = np.inf  # and the first
        self.adoption_times: list[float] = []  # when each plan took over, in simulation time
        self.t_last_observation = -np.inf
        self.plans: list[CatchPlan] = []  # every plan the policy returned
        self.policy_calls = 0
        self.plans_adopted = 0

        self.observation_port = self.DeclareAbstractInputPort(
            "observation", AbstractValue.Make(Observation(-1.0, np.zeros(3), None)))
        self.decision_index = self.DeclareAbstractState(AbstractValue.Make(_Decision()))
        self.DeclarePeriodicUnrestrictedUpdateEvent(CONTROL_PERIOD, CONTROL_OFFSET, self._tick)
        # The commands depend on time and state, and not on the observation.
        depends = {self.time_ticket(), self.abstract_state_ticket(self.decision_index)}
        self.DeclareVectorOutputPort("iiwa_position", 7, self._calc_arm, prerequisites_of_calc=depends)
        self.DeclareVectorOutputPort("wsg_position", 1, self._calc_fingers, prerequisites_of_calc=depends)

    def _trajectory(self, plan):
        """What the arm is following: the toss, which ends holding still at the
        end of the brake, until there is a plan to follow instead."""
        if plan is None or plan.q_traj is None:
            return self.toss_plan.q_traj
        return plan.q_traj

    def _decision(self, context) -> _Decision:
        return context.get_abstract_state(self.decision_index).get_value()

    def _tick(self, context, state):
        t = context.get_time()
        t_release = self.toss_plan.t_release
        t_rel = t - t_release

        # Only samples strictly after release: at release the ball is in the fingers.
        observation = self.observation_port.Eval(context)
        if observation.t > max(self.t_last_observation, t_release + 1e-9):
            self.fit.add(observation)
            self.t_last_observation = observation.t
        if t_rel < 0:
            return

        d = self._decision(context)
        plan, t_close, pending, t_adopt = d.plan, d.t_close, d.pending, d.t_adopt
        if pending is not None and t >= t_adopt - 1e-9:
            plan, t_close, pending = pending, self._t_close(t, pending, t_close), None
            self._adopted(t)

        trajectory = self._trajectory(plan)
        new = self.policy.update(t_rel, self.fit.estimate(), trajectory.value(t).ravel(),
                                 trajectory.EvalDerivative(t, 1).ravel(), plan)
        self.policy_calls += 1
        if new is not None:
            self.plans.append(new)
            t_due = t + self.compute_latency
            if new.q_traj is not None:
                t_due = max(t_due, new.q_traj.start_time())
            if t_due <= t + 1e-9:
                plan, t_close = new, self._t_close(t, new, t_close)
                self._adopted(t)
            else:
                pending, t_adopt = new, t_due
        if t_rel >= DECISION_DEADLINE and not self.plans:
            raise RuntimeError(f"the policy returned no plan by {DECISION_DEADLINE} s after release")
        state.get_mutable_abstract_state(self.decision_index).set_value(
            _Decision(plan, t_close, pending, t_adopt))

    def _adopted(self, t):
        self.plans_adopted += 1
        self.adoption_times.append(t)

    def _t_close(self, t, plan, t_close):
        """The decision to close, from this plan.  Once it has been taken it stands:
        the fingers are already moving.  A failed solve never closes."""
        if t >= t_close or not plan.solve_ok:
            return t_close
        self.t_close = close_command_time(plan.t_catch, self.lead, self.assumed_latency, self.pad_lead)
        if self.t_close_first == np.inf:
            self.t_close_first = self.t_close
        return self.t_close

    def _calc_arm(self, context, output):
        t = context.get_time()
        output.SetFromVector(self._trajectory(self._decision(context).plan).value(t).ravel())

    def _calc_fingers(self, context, output):
        t = context.get_time()
        if t >= self._decision(context).t_close + self.latency:
            output.SetFromVector([env.FINGER_CLOSED])
        else:
            output.SetFromVector(self.toss_plan.wsg_traj.value(t).ravel())


@dataclass
class CatchLog(toss.TossLog):
    """A simulated toss and catch, one row per simulator step."""

    q_cmd: np.ndarray = None  # arm joint positions commanded
    finger_width: np.ndarray = None  # measured finger opening
    finger_rate: np.ndarray = None  # its rate
    grip_force: np.ndarray = None  # measured grip force
    R_WG: np.ndarray = None  # gripper orientation, (n, 3, 3)


@dataclass
class CatchOutcome:
    """How one catch went."""

    caught: bool  # gripped by both fingers, with force, for HOLD_TIME
    contact: bool  # the ball touched the gripper after leaving it
    hold_time: float = 0.0  # s gripped continuously from when the fingers stopped
    t_fingers_stopped: float | None = None  # simulation time
    t_first_contact: float | None = None  # simulation time
    timing_error: float = np.nan  # s: real crossing of the catch plane - planned
    miss_horizontal: float = np.nan  # m: ball to pinch point at that crossing, horizontally
    miss_vertical: float = np.nan  # m: pinch point height error at that crossing
    # ball - pinch point in the gripper frame, m: closing axis, along the fingers, across the pad
    miss_in_hand: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))  # at that crossing
    end_in_hand: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))  # at the end of the run
    predicted_miss: float = np.nan  # m: |planned catch point - real crossing|, the estimator alone
    arm_arrival_error: float = np.nan  # m: |pinch at t_catch - planned catch point|, which these two combine into:
    plan_stop_error: float = np.nan  # m: |where the plan stops - planned catch point|
    tracking_error: float = np.nan  # m: |pinch at t_catch - where the plan stops|


@dataclass
class CatchRun:
    """Everything one simulate_catch produced."""

    toss_plan: toss.TossPlan
    log: CatchLog
    controller: CatchController
    plant: object
    outcome: CatchOutcome
    toss_noise: np.ndarray  # what was added to the ball's velocity

    @property
    def plans(self) -> list[CatchPlan]:
        return self.controller.plans

    @property
    def peak_speed(self) -> float:
        """The fastest any joint went from the start of the motion to t_catch, as a
        fraction of its rated speed, or nan if there was no motion."""
        plan = self.plans[-1]
        if not plan.solve_ok:
            return math.nan
        v_max, _ = toss.arm_limits(self.plant)
        first = next(p for p in self.plans if p.solve_ok)
        during = (self.log.t >= first.q_traj.start_time()) & (self.log.t <= plan.t_catch)
        return float((np.abs(self.log.arm_v[during]) / v_max).max())


@lru_cache(maxsize=1)
def shared_kinematics() -> Kinematics:
    return Kinematics()


def build_catch_system(policy_factory, rng_obs, sigma_p, sigma_v, obs_mode, latency, lead, compute_latency,
                       pad_lead, assumed_latency):
    """The scene with a sensor and a CatchController wired in, extending
    toss.add_toss_station.  Returns (diagram, toss plan, controller)."""
    builder = DiagramBuilder()
    station, plan = toss.add_toss_station(builder)
    plant = station.GetSubsystemByName("plant")
    ball = plant.GetBodyByName(env.BODY, plant.GetModelInstanceByName("ball"))

    sensor = builder.AddSystem(BallSensor(ball.index(), rng_obs, sigma_p, sigma_v, obs_mode))
    builder.Connect(station.GetOutputPort("body_poses"), sensor.GetInputPort("body_poses"))
    builder.Connect(station.GetOutputPort("body_spatial_velocities"),
                    sensor.GetInputPort("body_spatial_velocities"))

    policy = policy_factory(CatchPlanner(shared_kinematics(), plan))
    fit = getattr(policy, "fit_class", BallisticFit)(plan.t_release, sigma_p, sigma_v)
    controller = builder.AddSystem(CatchController(plan, policy, fit, latency, lead, compute_latency, pad_lead,
                                                   assumed_latency))
    builder.Connect(sensor.GetOutputPort("observation"), controller.GetInputPort("observation"))
    builder.Connect(controller.GetOutputPort("iiwa_position"), station.GetInputPort("iiwa.position"))
    builder.Connect(controller.GetOutputPort("wsg_position"), station.GetInputPort("wsg.position"))
    return builder.Build(), plan, controller


def simulate_catch(policy_factory, *, sigma_p=sensing.SIGMA_P, sigma_v=sensing.SIGMA_V,
                   obs_mode=sensing.POSITION_ONLY, latency=T_LATENCY, lead=None,
                   compute_latency=0.0, pad_lead=PAD_LEAD, assumed_latency=None, sigma_lat=0.0, sigma_vz=0.0,
                   trial=0) -> CatchRun:
    """Simulate one toss and catch.  policy_factory takes a CatchPlanner and
    returns a policy, so a policy class will do.  The toss and observation noise
    depend only on the trial number, so a trial is the same under every policy."""
    rng_obs = np.random.default_rng([SEED_ROOT, trial, 1])
    noise = toss_velocity_noise(trial, sigma_lat, sigma_vz)
    system, plan, controller = build_catch_system(policy_factory, rng_obs, sigma_p, sigma_v, obs_mode,
                                                  latency, lead, compute_latency, pad_lead, assumed_latency)
    toss.draw_planned_flight(plan)
    simulator = Simulator(system)
    context = simulator.get_mutable_context()
    env.place_ball_in_gripper(system, context)

    station = system.GetSubsystemByName("station")
    plant = station.GetSubsystemByName("plant")
    iiwa = plant.GetModelInstanceByName("iiwa")
    ball = plant.GetBodyByName(env.BODY, plant.GetModelInstanceByName("ball"))
    G = toss.gripper_frame(plant).body()
    effort = station.GetOutputPort("iiwa_net_actuation")
    fingers = station.GetOutputPort("wsg.state_measured")
    force = station.GetOutputPort("wsg.force_measured")
    command = controller.GetOutputPort("iiwa_position")
    rows = []

    def record(root_context):
        plant_context = plant.GetMyContextFromRoot(root_context)
        station_context = station.GetMyContextFromRoot(root_context)
        X_WB = plant.EvalBodyPoseInWorld(plant_context, ball)
        V_WB = plant.EvalBodySpatialVelocityInWorld(plant_context, ball)
        X_WG = plant.EvalBodyPoseInWorld(plant_context, G)
        width, rate = fingers.Eval(station_context)
        rows.append((
            root_context.get_time(), X_WB.translation(), V_WB.translational(),
            toss.ball_contacts(plant, plant_context, ball), X_WG @ env.PINCH,
            plant.GetPositions(plant_context, iiwa), plant.GetVelocities(plant_context, iiwa),
            effort.Eval(station_context),
            command.Eval(controller.GetMyContextFromRoot(root_context)),
            width, rate, force.Eval(station_context)[0], X_WG.rotation().matrix(),
        ))
        return EventStatus.Succeeded()

    simulator.set_monitor(record)
    if not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    start_recording()
    simulator.AdvanceTo(plan.t_release + T_TOSS_NOISE)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    V_WB = plant.EvalBodySpatialVelocityInWorld(plant_context, ball)
    plant.SetFreeBodySpatialVelocity(plant_context, ball,
                                     SpatialVelocity(V_WB.rotational(), V_WB.translational() + noise))
    simulator.AdvanceTo(plan.q_traj.end_time())
    publish_recording()

    t, ball_p, ball_v, touching, pinch, arm_q, arm_v, arm_effort, q_cmd, width, rate, grip, R_WG = zip(*rows)
    log = CatchLog(np.array(t), np.array(ball_p), np.array(ball_v), list(touching), np.array(pinch),
                   np.array(arm_q), np.array(arm_v), np.array(arm_effort), np.array(q_cmd), np.array(width),
                   np.array(rate), np.array(grip), np.array(R_WG))
    return CatchRun(plan, log, controller, plant, score(plan, log, controller), noise)


def _at(t, times, values):
    """values, one row per entry of times, interpolated at t."""
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    a = (t - times[i - 1]) / (times[i] - times[i - 1])
    return values[i - 1] + a * (values[i] - values[i - 1])


def first_settled(rate: np.ndarray, rows: int = FINGER_SETTLED_ROWS) -> int | None:
    """The first index from which the finger rate stays slow for this many rows,
    or None.  A finger pushing the ball can pause for a moment on its way in."""
    slow = np.abs(rate) < FINGER_STOPPED_SPEED
    for i in range(len(rate) - rows + 1):
        if slow[i:i + rows].all():
            return i
    return None


def hold_after_closing(log, controller, grip_force_min: float = GRIP_FORCE_MIN) -> tuple[float, float | None]:
    """How long (s) both fingers keep gripping the ball, from when they stop
    closing, and the simulation time they stop, or None if they never do."""
    closing = np.flatnonzero(log.t >= controller.t_close + controller.latency)
    moved = closing[log.finger_rate[closing] < -FINGER_STOPPED_SPEED]
    if len(moved) == 0:
        return 0.0, None
    settled = first_settled(log.finger_rate[moved[0]:])
    if settled is None:
        return 0.0, None
    i_stop = moved[0] + settled
    i_end = i_stop
    while (i_end < len(log.t)
           and {"wsg::left_finger", "wsg::right_finger"} <= log.touching[i_end]
           and log.grip_force[i_end] >= grip_force_min):
        i_end += 1
    held = log.t[i_end - 1] - log.t[i_stop] if i_end > i_stop else 0.0
    return held, log.t[i_stop]


def score(plan: toss.TossPlan, log: CatchLog, controller: CatchController) -> CatchOutcome:
    """Decide whether the ball was caught, and measure how the catch went."""
    fly = toss.free_flight(plan, log)
    if fly.i_free is None:
        return CatchOutcome(False, False)
    gripper = [i for i in range(fly.i_free, len(log.t))
               if any(name.startswith("wsg::") for name in log.touching[i])]
    outcome = CatchOutcome(False, bool(gripper))
    outcome.end_in_hand = log.R_WG[-1].T @ (log.ball_p[-1] - log.pinch[-1])
    if gripper:
        outcome.t_first_contact = log.t[gripper[0]]

    chosen = controller.plans[-1] if controller.plans else None
    if chosen is None:
        return outcome
    if fly.t_down is not None:
        outcome.timing_error = fly.t_down - chosen.t_catch
        pinch = _at(fly.t_down, log.t, log.pinch)
        outcome.miss_horizontal = np.linalg.norm((fly.p_down - pinch)[:2])
        outcome.miss_vertical = pinch[2] - fly.p_down[2]
        R_WG = log.R_WG[np.searchsorted(log.t, fly.t_down)]
        outcome.miss_in_hand = R_WG.T @ (fly.p_down - pinch)
        outcome.predicted_miss = np.linalg.norm(chosen.p_catch - fly.p_down)
    arrived = _at(chosen.t_catch, log.t, log.pinch)
    outcome.arm_arrival_error = np.linalg.norm(arrived - chosen.p_catch)

    if chosen.solve_ok:
        stopped = shared_kinematics().pinch(chosen.q_traj.value(chosen.t_catch).ravel())
        outcome.plan_stop_error = np.linalg.norm(stopped - chosen.p_catch)
        outcome.tracking_error = np.linalg.norm(arrived - stopped)
        outcome.hold_time, outcome.t_fingers_stopped = hold_after_closing(log, controller)
        outcome.caught = outcome.hold_time >= HOLD_TIME
    return outcome


def report(run: CatchRun) -> None:
    """Print one trial's record."""
    o, plans = run.outcome, run.plans
    print(f"caught: {o.caught}   contact: {o.contact}   held {o.hold_time:.3f} s")
    print(f"policy calls {run.controller.policy_calls}, plans adopted {run.controller.plans_adopted}; "
          f"toss noise {np.round(run.toss_noise, 4)} m/s")
    if plans:
        plan = plans[-1]
        t_release = run.toss_plan.t_release
        print(f"plan:   solve_ok {plan.solve_ok} in {plan.solve_time * 1e3:.0f} ms; catch at "
              f"{plan.t_catch - t_release:.4f} s after release at {np.round(plan.p_catch, 4)} m")
    print(f"timing error {o.timing_error * 1e3:.2f} ms; predicted miss {o.predicted_miss * 1e2:.2f} cm; "
          f"arm arrival error {o.arm_arrival_error * 1e2:.2f} cm")
    print(f"miss horizontal {o.miss_horizontal * 1e2:.2f} cm, vertical {o.miss_vertical * 1e2:.2f} cm; "
          f"in hand (closing, fingers, pad) {np.round(o.miss_in_hand * 1e2, 2)} cm")
    if o.t_first_contact is not None:
        print(f"first contact {o.t_first_contact - run.toss_plan.t_release:.4f} s after release")
    if o.t_fingers_stopped is not None:
        print(f"fingers stopped {o.t_fingers_stopped - run.toss_plan.t_release:.4f} s after release")
    print(f"ball ends at {np.round(run.log.ball_p[-1], 3)} m; in hand {np.round(o.end_in_hand * 1e2, 2)} cm")


POLICIES = {"open_loop": OpenLoopPolicy, "predict_once": PredictOncePolicy}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "open_loop"
    report(simulate_catch(POLICIES[name], sigma_p=0.0))
    keep_meshcat_open()
