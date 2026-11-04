"""Find the release pose that tosses the ball highest within the joint speed limits.

How fast the hand can go at release depends only on the pose and the joint
velocity limits at that instant, so the search is over poses.  At each pose a
small linear program picks the joint velocities that add up to the most upward
speed.  Joint 1 turns about a vertical axis and the ball sits on joint 7's axis,
so joints 2 to 6 are the ones that can add height.

1. Pose search, no physics.  For each band of release heights and each cap on
   how far the gripper may point off the toss, a seeded multi-start search for
   the highest peak.  The heading is free, so joint 1 just puts the release on
   the +x axis.  Run twice: for height alone, which shows the ceiling, and with a
   rough estimate of the catch in the cost, also choosing how far the toss may
   lean, which is what finds tosses that can be caught.  Also for height alone
   with a fixed 80 degree toss, to compare.
2. Simulation gates, for the best poses of each cell.  The commanded joint speed
   cap and the brake time are adjusted per pose: the cap until the measured joint
   speeds stay within the limits, the brake until the torque stays below them.
3. Catch check on the poses that pass, highest first, by trajectory optimization.
4. Swing sweep on the winner: does starting lower change anything?

Results go to results/release_search.md.

    python release_search.py
"""

import multiprocessing
import os
import tempfile
from pathlib import Path

os.environ["HEADLESS"] = "1"  # No Meshcat: this runs many simulations at once.

import numpy as np
from manipulation.station import LoadScenario
from manipulation.utils import ConfigureParser
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseKinematics,
    JacobianWrtVariable,
    ModelDirectives,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    Solve,
)
from scipy.optimize import linprog, minimize

import environment as env
import feasibility as fz
import toss

SEED = 4210
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

SEARCH_SWING = 0.25  # s; the swing used until the sweep in stage 4

# Stage 1
RELEASE_BANDS = [0.80, 0.85, 0.90, 0.95, 1.00]  # lower edge of each band (m); the last is open
BAND_WIDTH = 0.05
GRIPPER_CAPS = [45, 60, 75, 90, None]  # degrees between the palm axis and the toss
CATCH_CAPS = [45, 60, 75, None]  # the same, for the search that includes the catch
OFF_VERTICAL = 15.0  # how far from vertical the toss may be (degrees)
FINGER_TILT = 5.0  # how far from square to the toss the fingers' closing axis may be (degrees)
MIN_RISE = 0.25  # below this the search just lifts the hand and barely throws (m)
MIN_REACH, MAX_REACH = 0.3, 0.8  # pinch point's distance from the base axis at release (m)
MIN_CLEARANCE = 0.01  # between arm and gripper collision shapes (m)
MIN_LINK_HEIGHT = 0.1  # lowest arm link origin at the start pose (m)
LIMIT_MARGIN = 0.02  # stay this far inside the joint position limits (rad)
N_SAMPLES = 20000
N_RANKED = 400  # samples ranked with the linear program
N_STARTS = 16  # local searches per cell
N_CATCH_STARTS = 24  # the same, with the catch estimate in the cost, which needs more to find anything
CATCH_ESTIMATE_CUSHION = 0.02  # s on top of MIN_CATCH_SPARE, as the estimate is rough
TOP_PER_CELL = 5
FIXED_TOSS_ANGLE = 80.0

# Stage 2
BRAKES = [0.10, 0.15, 0.20]  # s; 0.05 already saturates torque at 2 m/s
MAX_SPEED_STEPS = 5
SPEED_FRACTION = 0.999  # of the pose's top speed, so the release velocity QP stays feasible
EFFORT_CEILING = 0.995  # the simulator clips torque at the limit, so reaching it is a failure
MAX_PEAK_ERROR = 0.02  # simulated vs planned peak (m)
MAX_GRIP_OFFSET = 0.015  # ball center vs pinch point before release (m)
MAX_LEAVE_TIME = 0.03  # the ball must stop touching the fingers this soon after release (s)
BALL_MASSES = [0.01, 0.1]  # kg
AFTER = 1.0  # s simulated after the brake

# Stage 3
MIN_CATCH_SPARE = 0.05  # s, covers the gripper's close time and some latency
N_HEIGHT_ONLY_CHECKS = 5  # of the height-only poses, to show how far off they are

# Stage 4
SWINGS = [0.25, 0.30, 0.40, 0.50, 0.60, 0.75]  # s; multiples of the WSG controller's 50 ms update
MIN_PEAK_FOR_ADOPTION = 1.1  # m; below this, stop and look at what is limiting the toss

NOISE_LEVELS = [0.02, 0.05]  # m/s per axis
N_NOISE_DRAWS = 300

OCTAGON = np.radians(np.arange(0, 360, 45))
RESULTS = Path(__file__).parent / "results" / "release_search.md"


class Arm:
    """Kinematics and a collision check for the iiwa and gripper, for the search."""

    def __init__(self):
        self.kin = fz.Kinematics()
        self.plant = self.kin.plant
        self.context = self.kin.context
        self.iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.G = toss.gripper_frame(self.plant)
        self.W = self.plant.world_frame()
        self.v_max = self.kin.v_max
        self.lo, self.hi = self.kin.q_lo, self.kin.q_hi
        self.links = [self.plant.GetBodyByName(f"iiwa_link_{i}", self.iiwa) for i in range(2, 8)]
        self._build_collision_model()

    def _build_collision_model(self):
        """The same arm with collision shapes, which the simulated scene's arm
        does not have, plus a free ball to measure the ball's path against."""
        builder = DiagramBuilder()
        self.cplant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        parser = Parser(self.cplant)
        ConfigureParser(parser)
        data = env.ROBOT.replace("iiwa14_no_collision", "iiwa14_polytope_collision")
        directives = LoadScenario(data=data.split("model_drivers")[0]).directives
        ProcessModelDirectives(ModelDirectives(directives=directives), self.cplant, parser)
        parser.AddModelsFromString(f"""<?xml version="1.0"?>
<sdf version="1.7"><model name="path_ball"><link name="{env.BODY}">
  <collision name="c"><geometry><sphere><radius>{env.BALL_RADIUS}</radius></sphere></geometry></collision>
</link></model></sdf>""", "sdf")
        self.cplant.Finalize()
        diagram = builder.Build()
        self.ccontext = diagram.CreateDefaultContext()
        self.cplant_context = self.cplant.GetMyMutableContextFromRoot(self.ccontext)
        self.sg_context = self.scene_graph.GetMyContextFromRoot(self.ccontext)
        self.c_iiwa = self.cplant.GetModelInstanceByName("iiwa")
        self.path_ball = self.cplant.GetBodyByName(env.BODY, self.cplant.GetModelInstanceByName("path_ball"))
        # Fingers open, so they are not each other's nearest shape.
        self.cplant.SetPositions(self.cplant_context, self.cplant.GetModelInstanceByName("wsg"), [-0.055, 0.055])
        self._put_ball(np.array([0.0, 0.0, 100.0]))

    def _put_ball(self, p):
        self.cplant.SetFreeBodyPose(self.cplant_context, self.path_ball, RigidTransform(p))

    def clearance(self, q) -> float:
        """Smallest distance between arm and gripper shapes (negative: inside each other)."""
        self.cplant.SetPositions(self.cplant_context, self.c_iiwa, q)
        query = self.scene_graph.get_query_output_port().Eval(self.sg_context)
        return min((r.distance for r in query.ComputeSignedDistancePairwiseClosestPoints(0.1)), default=0.1)

    def ball_clearance(self, plan, t_end: float, n: int = 40) -> float:
        """Smallest distance from the planned ball to the arm links, up to t_end
        seconds after release.  In the simulated scene the arm has no collision
        shapes, so a ball going through the arm would not be noticed there."""
        self.cplant.SetPositions(self.cplant_context, self.cplant.GetModelInstanceByName("wsg"), [-0.055, 0.055])
        inspector = None
        smallest = np.inf
        for t in np.linspace(0.02, t_end, n):
            self.cplant.SetPositions(self.cplant_context, self.c_iiwa, plan.q_traj.value(plan.t_release + t).ravel())
            self._put_ball(toss.ballistic_position(plan.p_release, plan.v_release, t))
            query = self.scene_graph.get_query_output_port().Eval(self.sg_context)
            inspector = query.inspector()
            for r in query.ComputeSignedDistancePairwiseClosestPoints(0.2):
                bodies = [self.cplant.GetBodyFromFrameId(inspector.GetFrameId(g)) for g in (r.id_A, r.id_B)]
                names = {self.cplant.GetModelInstanceName(b.model_instance()) for b in bodies}
                if names == {"path_ball", "iiwa"}:
                    smallest = min(smallest, r.distance)
        self._put_ball(np.array([0.0, 0.0, 100.0]))
        return smallest

    def kinematics(self, q):
        """Pinch point, gripper rotation and pinch point Jacobian (3 x 7) at q."""
        self.plant.SetPositions(self.context, self.iiwa, q)
        X = self.plant.CalcRelativeTransform(self.context, self.W, self.G)
        J = self.plant.CalcJacobianSpatialVelocity(
            self.context, JacobianWrtVariable.kV, self.G, env.PINCH, self.W, self.W
        )[3:, :7]
        return X @ env.PINCH, X.rotation().matrix(), J

    def catch_arrival(self, q_stop, p_catch, v_lim, a_lim):
        """Rough time from rest at q_stop to rest at a catch pose, or None if
        inverse kinematics finds no catch pose.  The slowest joint sets it, moving
        on a trapezoid at its velocity and acceleration limits."""
        ik = InverseKinematics(self.plant)
        q = ik.q()
        ik.AddPositionConstraint(self.G, env.PINCH, self.W, p_catch - 0.003, p_catch + 0.003)
        ik.AddAngleBetweenVectorsConstraint(
            self.W, np.array([0.0, 0.0, 1.0]), self.G, np.array([0.0, 1.0, 0.0]),
            0.0, np.radians(toss.CATCH_TILT),
        )
        seed = np.r_[q_stop, 0.0, 0.0]
        weights = np.r_[1 / v_lim**2, 0.0, 0.0]
        ik.prog().AddQuadraticErrorCost(np.diag(weights), seed, q)
        ik.prog().SetInitialGuess(q, seed)
        result = Solve(ik.prog())
        if not result.is_success():
            return None
        moves = np.abs(result.GetSolution(q)[:7] - q_stop)
        return np.max(np.where(moves > v_lim**2 / a_lim, moves / v_lim + v_lim / a_lim,
                               2 * np.sqrt(moves / a_lim)))

    def lowest_link(self, q) -> float:
        self.plant.SetPositions(self.context, self.iiwa, q)
        return min(self.plant.EvalBodyPoseInWorld(self.context, b).translation()[2] for b in self.links)


_arm = None


def get_arm() -> Arm:
    """One Arm per process."""
    global _arm
    if _arm is None:
        _arm = Arm()
    return _arm


def init_worker():
    # Each worker writes its own ball SDF, since the ball's mass varies by run.
    env.ASSET_DIR = Path(tempfile.mkdtemp(prefix="release_search_"))


# ---------------------------------------------------------------------------
# 1. Pose search
# ---------------------------------------------------------------------------


def fastest_upward(J, R, v_lim, cap, tan_off=np.tan(np.radians(OFF_VERTICAL))):
    """Joint velocities giving the most upward pinch speed, with the toss near
    vertical, the fingers' closing axis near square to it, and (if cap is set)
    the palm axis within cap degrees of it.  Heading is free.  The cone
    constraints use the upward speed in place of the total speed, a close match
    for a steep toss; angles are measured properly afterward."""
    up = J[2]
    finger = R[:, 0] @ J
    tan_finger = np.tan(np.radians(FINGER_TILT))
    rows = [np.cos(h) * J[0] + np.sin(h) * J[1] - tan_off * up for h in OCTAGON]
    rows += [finger - tan_finger * up, -finger - tan_finger * up]
    if cap is not None:
        rows.append(np.cos(np.radians(cap)) * up - R[:, 1] @ J)
    result = linprog(-up, A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                     bounds=np.c_[-v_lim, v_lim], method="highs")
    return result.x if result.status == 0 else np.zeros(7)


def fastest_along(J, d, v_lim):
    """Joint velocities giving the most pinch speed exactly along d."""
    result = linprog(np.r_[np.zeros(7), -1.0], A_eq=np.c_[J, -d], b_eq=np.zeros(3),
                     bounds=[*np.c_[-v_lim, v_lim], (0, None)], method="highs")
    return result.x[:7] if result.status == 0 else np.zeros(7)


def angles(R, v):
    """Degrees between the palm axis and v, and between the fingers' closing
    axis and square to v."""
    unit = v / max(np.linalg.norm(v), 1e-9)
    palm = np.degrees(np.arccos(np.clip(R[:, 1] @ unit, -1, 1)))
    finger = abs(np.degrees(np.arcsin(np.clip(R[:, 0] @ unit, -1, 1))))
    return palm, finger


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def on_x_axis(arm, x):
    """The pose with arm joints 2 to 7 from x and joint 1 turned so the pinch
    point is on the +x axis.  Only for a free heading: turning joint 1 turns the
    toss with it and changes nothing else."""
    q = np.r_[0.0, x]
    p, _, _ = arm.kinematics(q)
    q[0] = wrap(-np.arctan2(p[1], p[0]))
    return q


def evaluate(arm, x, band, cap, fixed, catch=False):
    """Peak height for the pose x, less penalties for what it breaks.  With catch,
    the last entry of x is how far the toss may lean (tangent of the angle), and
    the penalties include a rough estimate of the time to the catch."""
    tan_off = np.tan(np.radians(OFF_VERTICAL))
    if catch:
        tan_off = np.clip(x[-1], 0.0, tan_off)
        x = x[:-1]
    q = np.array(x) if fixed else on_x_axis(arm, x)
    p, R, J = arm.kinematics(q)
    if fixed:
        u = fastest_along(J, toss.release_velocity(1.0, FIXED_TOSS_ANGLE, 0.0), arm.v_max)
    else:
        u = fastest_upward(J, R, arm.v_max, cap, tan_off)
    u[6] = 0.0  # Joint 7 moves nothing here, so the linear program can leave it anywhere.
    v = J @ u
    rise = max(v[2], 0.0) ** 2 / (2 * toss.G_ACCEL)
    palm, finger = angles(R, v)

    q_start, q_stop = q - u * SEARCH_SWING / 2, q + u * BRAKES[0] / 2
    corners = np.array([q, q_start, q_stop])
    pen = np.sum(np.maximum(0, arm.lo + LIMIT_MARGIN - corners.min(axis=0)))
    pen += np.sum(np.maximum(0, corners.max(axis=0) - arm.hi + LIMIT_MARGIN))
    pen += 5 * max(0, band - p[2])
    if band < RELEASE_BANDS[-1]:
        pen += 5 * max(0, p[2] - band - BAND_WIDTH)
    reach = np.linalg.norm(p[:2])
    pen += 5 * (max(0, MIN_REACH - reach) + max(0, reach - MAX_REACH))
    pen += 5 * max(0, MIN_RISE - rise)
    pen += 5 * max(0, MIN_LINK_HEIGHT - arm.lowest_link(q_start))
    clear = min(arm.clearance(q), arm.clearance(q_start), arm.clearance(q_stop))
    pen += 5 * max(0, MIN_CLEARANCE - clear)
    if fixed:
        pen += 0.05 * max(0, finger - FINGER_TILT)
        if cap is not None:
            pen += 0.05 * max(0, palm - cap)
    spare = np.nan
    if catch and v[2] > 0:
        t_catch = toss.time_to_height(p, v, p[2] - toss.CATCH_DROP)
        arrival = arm.catch_arrival(q_stop, toss.ballistic_position(p, v, t_catch), arm.v_max,
                                    arm.v_max / BRAKES[0])
        if arrival is None:
            pen += 2.0
        else:
            spare = t_catch - BRAKES[0] - arrival
            pen += 5 * max(0, MIN_CATCH_SPARE + CATCH_ESTIMATE_CUSHION - spare)
    elif catch:
        pen += 2.0
    info = dict(q=q, u=u, v=v, z=p[2], reach=reach, rise=rise, peak=p[2] + rise, palm=palm, finger=finger,
                clear=clear, band=band, cap=cap, pen=pen, catch=catch, spare_estimate=spare)
    return p[2] + rise - pen, info


def search_cell(task):
    """The best few poses for one band of release heights and one gripper cap."""
    band, cap, fixed, catch = task
    arm = get_arm()
    rng = np.random.default_rng([SEED, int(100 * band), 0 if cap is None else cap, int(fixed), int(catch)])
    # Without a fixed heading joint 1 is not searched over: see on_x_axis.
    lo, hi = (arm.lo, arm.hi) if fixed else (arm.lo[1:], arm.hi[1:])
    samples = rng.uniform(lo, hi, (N_SAMPLES, len(lo)))
    if catch:  # Start with the toss leaning about 8 degrees.
        samples = np.c_[samples, np.full(N_SAMPLES, np.tan(np.radians(8.0)))]

    # Rank cheaply by what every joint could add upward if nothing else mattered,
    # then properly for the best of those, and search from the best of those.
    rough = []
    for i, x in enumerate(samples):
        q = np.array(x) if fixed else np.r_[0.0, x[:6]]
        p, _, J = arm.kinematics(q)
        if band - 0.1 < p[2] and (band >= RELEASE_BANDS[-1] or p[2] < band + BAND_WIDTH + 0.1):
            rough.append((np.sum(np.abs(J[2] * arm.v_max)), i))
    rough.sort(reverse=True)
    # (What every joint could add says little about the catch, so with it, rank them all.)
    to_rank = rough if catch else rough[:N_RANKED]
    ranked = sorted(((evaluate(arm, samples[i], band, cap, fixed, catch)[0], i) for _, i in to_rank), reverse=True)

    found = []
    for _, i in ranked[:N_CATCH_STARTS if catch else N_STARTS]:
        cost = lambda x: -evaluate(arm, x, band, cap, fixed, catch)[0]
        result = minimize(cost, samples[i], method="Nelder-Mead",
                          options=dict(maxiter=1500, xatol=1e-3, fatol=1e-4, adaptive=True))
        result = minimize(cost, result.x, method="Powell", options=dict(maxiter=6, xtol=1e-3, ftol=1e-4))
        info = evaluate(arm, result.x, band, cap, fixed, catch)[1]
        if info["pen"] < 1e-3:
            found.append(info)
    found.sort(key=lambda f: -f["peak"])
    kept = []
    for f in found:
        if all(np.linalg.norm(f["q"] - k["q"]) > 0.05 for k in kept):
            kept.append(f)
    print(f"  stage 1: band {band:.2f}, cap {cap}, fixed {fixed}, catch {catch}: {len(found)} found; "
          f"best peak {kept[0]['peak']:.3f} m" if kept else
          f"  stage 1: band {band:.2f}, cap {cap}, fixed {fixed}, catch {catch}: none found", flush=True)
    return task, kept[:TOP_PER_CELL]


# ---------------------------------------------------------------------------
# 2. Simulation gates
# ---------------------------------------------------------------------------


def configure(pose, margin, brake, swing=None, mass=None):
    """Set toss.py and environment.py up for this pose.  The velocity scales with
    the cap, because every bound in the linear program does."""
    toss.RELEASE_POSE = pose["q"]
    toss.VELOCITY_MARGIN = margin
    toss.BRAKE = brake
    toss.SWING = toss.SWING if swing is None else swing
    toss.AFTER = AFTER
    if mass is not None:
        env.BALL_MASS = mass
    return pose["v"] * margin * SPEED_FRACTION


def measure(pose, margin, brake, swing=SEARCH_SWING, mass=BALL_MASSES[0]) -> dict:
    """Simulate the toss and measure everything the gates look at."""
    v = configure(pose, margin, brake, swing, mass)
    plan, log, plant = toss.simulate(lambda p: toss.plan_toss(p, v))
    v_max, effort_max = toss.arm_limits(plant)
    fly = toss.free_flight(plan, log)
    t_plan, p_plan = toss.catch_point(plan)
    before = log.t < plan.t_release
    i_rel = np.argmin(np.abs(log.t - plan.t_release))
    m = dict(
        speed=np.linalg.norm(v), vz=v[2], margin=margin, brake=brake, swing=swing, mass=mass,
        speed_peak=np.abs(log.arm_v).max(axis=0) / v_max,
        effort_peak=np.abs(log.arm_effort).max(axis=0) / effort_max,
        effort_swing=np.abs(log.arm_effort[before]).max(axis=0) / effort_max,
        grip_offset=np.linalg.norm(log.ball_p[before] - log.pinch[before], axis=1).max(),
        start_z=log.ball_p[0, 2], tracking=np.abs(log.arm_q[i_rel] - plan.q_traj.value(log.t[i_rel]).ravel()).max(),
        release_v=log.ball_v[fly.i_release], planned_peak=plan.p_release[2] + plan.v_release[2] ** 2 / (2 * toss.G_ACCEL),
        planned_t_catch=t_plan, p_catch=p_plan, p_release=plan.p_release, v_release=plan.v_release,
        qdot_release=plan.qdot_release, q_start=plan.q_start,
        lowest=get_arm().lowest_link(plan.q_start),
        ball_clear=get_arm().ball_clearance(plan, plan.v_release[2] / toss.G_ACCEL),
    )
    m["left"] = fly.i_free is not None
    if m["left"]:
        hit = fly.i_hit < len(log.t)
        m.update(
            leave_time=log.t[fly.i_free] - log.t[fly.i_release], apex=fly.apex, off=fly.off,
            extrapolated=fly.extrapolated, reached_catch=fly.t_down is not None,
            first_contact=", ".join(sorted(log.touching[fly.i_hit])) if hit else "nothing",
            contact_time=log.t[fly.i_hit] - log.t[fly.i_release] if hit else np.nan,
            flight=(fly.t_down - log.t[fly.i_release]) if fly.t_down is not None else np.nan,
            miss=np.linalg.norm((fly.p_down - p_plan)[:2]) if fly.t_down is not None else np.nan,
        )
    return m


def failure(m) -> str | None:
    """The first thing wrong with a measured toss, or None."""
    if m["speed_peak"].max() > 1.0:
        return f"joint speed {m['speed_peak'].max():.3f} of limit"
    if m["effort_peak"].max() >= EFFORT_CEILING:
        return f"torque at limit (joint {m['effort_peak'].argmax() + 1})"
    if m["grip_offset"] > MAX_GRIP_OFFSET:
        return f"ball slipped {100 * m['grip_offset']:.1f} cm before release"
    if not m["left"] or m["leave_time"] > MAX_LEAVE_TIME:
        return "ball did not leave the fingers"
    # After the peak, the ball can come down near the arm parked by the brake.
    # That's not a failure, because the catch motion moves it out of the way.
    if m["contact_time"] < m["vz"] / toss.G_ACCEL:
        return f"ball hit {m['first_contact']} {m['contact_time']:.3f} s after release, on its way up"
    if abs(m["apex"] - m["planned_peak"]) > MAX_PEAK_ERROR:
        return f"peak {100 * abs(m['apex'] - m['planned_peak']):.1f} cm from plan"
    if m["ball_clear"] < 0:
        return "ball's path up goes through the arm"
    return None


def gate_pose(pose) -> dict:
    """Find the highest toss from this pose that stays within the limits as
    measured, then check it at each ball mass."""
    margin, m, why = 1.0, None, None
    try:
        for brake in BRAKES:
            for _ in range(MAX_SPEED_STEPS):
                m = measure(pose, margin, brake)
                top = m["speed_peak"].max()
                if top <= 1.0:
                    break
                margin *= 0.9995 / top
            if m["effort_peak"].max() < EFFORT_CEILING:
                break
        why = failure(m)
        others = []
        if why is None:
            others = [measure(pose, margin, m["brake"], mass=mass) for mass in BALL_MASSES[1:]]
            why = next((f"at {o['mass']} kg: {failure(o)}" for o in others if failure(o)), None)
    except ValueError as e:
        why = str(e)[:80]
    result = dict(pose=pose, measured=m, passed=why is None, why=why)
    peak = m["apex"] if m is not None and m["left"] else np.nan
    print(f"  stage 2: palm {pose['palm']:3.0f}, release {pose['z']:.2f} m: peak {peak:.3f} m, "
          f"{'pass' if why is None else why}", flush=True)
    return result


# ---------------------------------------------------------------------------
# 3. Catch check
# ---------------------------------------------------------------------------


def catch_check(candidate) -> dict:
    """How soon the arm can be at the catch and stopped, from the end of the brake."""
    pose, m = candidate["pose"], candidate["measured"]
    v = configure(pose, m["margin"], m["brake"], m["swing"])
    arm = get_arm()
    kin = arm.kin
    plan = toss.plan_toss(kin.plant, v)
    t_catch, p_catch = toss.catch_point(plan)
    limits = fz.accel_limits(kin)
    _, t0, q0, v0 = fz.hand_offs(plan)[0]
    out = dict(candidate, t_catch=t_catch, p_catch=p_catch, reachable=toss.catch_pose(kin.plant, p_catch) is not None)
    for level in ["brake", "swing"]:
        T, motion = fz.min_time(kin, plan, q0, v0, t0, p_catch, limits[level])
        out[f"arrive_{level}"] = t0 + T if motion is not None else np.nan
    out["spare"] = t_catch - out["arrive_brake"]
    out["catchable"] = out["reachable"] and out["spare"] >= MIN_CATCH_SPARE
    print(f"  stage 3: peak {m['apex']:.3f} m: arrives {out['arrive_brake']:.3f} s, ball {t_catch:.3f} s, "
          f"spare {out['spare']:+.3f} s, reachable {out['reachable']}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 4. Swing sweep, and how much noise the winner can take
# ---------------------------------------------------------------------------


def swing_row(task) -> dict:
    pose, margin, brake, swing = task
    try:
        m = measure(pose, margin, brake, swing)
        m["why"] = failure(m)
        return m
    except ValueError as e:
        return dict(swing=swing, why=str(e)[:80])


def noise_row(task):
    """How many noisy tosses the arm cannot reach, as Approach A's spec adds noise."""
    pose, margin, brake, sigma = task
    v = configure(pose, margin, brake, swing=SEARCH_SWING)
    kin = get_arm().kin
    rng = np.random.default_rng([SEED, int(1000 * sigma)])
    bad = 0
    for _ in range(N_NOISE_DRAWS):
        try:
            toss.plan_toss(kin.plant, v + rng.normal(0, sigma, 3))
        except ValueError:
            bad += 1
    return sigma, bad


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def cap_name(cap):
    return "no cap" if cap is None else f"{cap}°"


def cell_table(cells, bands, caps):
    """Upward speed / rise / peak of the best pose in each cell, as table rows."""
    lines = ["| release height | " + " | ".join(cap_name(c) for c in caps) + " |", "|---|" + "---|" * len(caps)]
    for band in bands:
        row = []
        for cap in caps:
            found = cells[(band, cap)]
            row.append(f"{found[0]['v'][2]:.2f} m/s / {found[0]['rise']:.2f} m / {found[0]['peak']:.2f} m"
                       if found else "-")
        label = f"{band:.2f} m and up" if band == bands[-1] else f"{band:.2f}–{band + BAND_WIDTH:.2f} m"
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return lines


def gate_table(gated, limit=None):
    """Simulated tosses, highest peak first, as table rows."""
    lines = ["| peak | release height | palm off toss | toss speed | commanded cap | brake | "
             "time to catch height | catch point offset | result |", "|---|---|---|---|---|---|---|---|---|"]
    def key(g):
        m = g["measured"]
        return -(m["apex"] if m is not None and m["left"] else -1)
    for g in sorted(gated, key=key)[:limit]:
        p, m = g["pose"], g["measured"]
        if m is None:
            lines.append(f"| - | {p['z']:.2f} m | {p['palm']:.0f}° | - | - | - | - | - | {g['why']} |")
            continue
        peak = f"{m['apex']:.3f} m" if m["left"] else "-"
        lines.append(f"| {peak} | {p['z']:.2f} m | {p['palm']:.0f}° | {m['speed']:.2f} m/s | {m['margin']:.3f} × limit | "
                     f"{m['brake']:.2f} s | {m.get('flight', np.nan):.3f} s | {100 * m.get('miss', np.nan):.1f} cm | "
                     f"{g['why'] or 'pass'} |")
    return lines


def write_results(cells, fixed_cells, catch_cells, gated, checked, winner, sweep, noise, joints):
    lines = ["# Release pose search", "",
             "Generated by `python release_search.py`. Times are seconds after release.", "",
             "The adopted toss came from a follow-up search just below these bands, with a longer swing: see `results/catch_feasibility.md`. The winner here is the best with release at 0.80 m and up.", ""]

    lines += ["## 1. Best poses (no physics)", "",
              f"Upward speed at 1.0 × limit / rise / peak. Toss within {OFF_VERTICAL:.0f}° of vertical, "
              f"fingers within {FINGER_TILT:.0f}° of square, rise at least {MIN_RISE} m. "
              "A dash means nothing was found.", "", "For height alone, by the cap on how far the gripper may point off the toss:", ""]
    lines += cell_table(cells, RELEASE_BANDS, GRIPPER_CAPS)
    lines += ["", f"With the catch included, needing about {MIN_CATCH_SPARE + CATCH_ESTIMATE_CUSHION:.2f} s to spare "
              "on a rough estimate:", ""]
    lines += cell_table(catch_cells, RELEASE_BANDS, CATCH_CAPS)
    lines += ["", f"For height alone, free direction against a fixed {FIXED_TOSS_ANGLE:.0f}° toss, release {RELEASE_BANDS[0]:.2f}–"
              f"{RELEASE_BANDS[0] + BAND_WIDTH:.2f} m:", "",
              "| gripper cap | fixed: speed | fixed: peak | free: speed | free: peak |", "|---|---|---|---|---|"]
    for cap, found in fixed_cells.items():
        free = cells[(RELEASE_BANDS[0], cap)]
        a = f"{np.linalg.norm(found[0]['v']):.2f} m/s | {found[0]['peak']:.2f} m" if found else "- | -"
        b = f"{np.linalg.norm(free[0]['v']):.2f} m/s | {free[0]['peak']:.2f} m" if free else "- | -"
        lines.append(f"| {cap_name(cap)} | {a} | {b} |")

    lines += ["", "## 2. Simulated tosses", "",
              "The cap and brake are what each pose needed to stay within the joint speed and torque limits as measured.", "",
              "Poses from the search for height alone, the top 10:", ""]
    lines += gate_table([g for g in gated if not g["pose"]["catch"]], 10)
    lines += ["", "Poses from the search that includes the catch:", ""]
    lines += gate_table([g for g in gated if g["pose"]["catch"]])

    lines += ["", "## 3. Catch check", "",
              "Arrival is the earliest the arm can be at the catch point and stopped, from the end of the brake, "
              "by trajectory optimization, at each toss's brake-level and swing-level acceleration. A dash means no motion "
              f"was found. Needs {MIN_CATCH_SPARE} s to spare at brake level. The first rows are the best poses for height alone.", "",
              "| peak | ball at catch height | arrives, brake level | spare | arrives, swing level | reachable |",
              "|---|---|---|---|---|---|"]
    def time_or_dash(t, form="{:.3f} s"):
        return form.format(t) if np.isfinite(t) else "-"
    for c in checked:
        lines.append(f"| {c['measured']['apex']:.3f} m | {c['t_catch']:.3f} s | {time_or_dash(c['arrive_brake'])} | "
                     f"{time_or_dash(c['spare'], '{:+.3f} s')} | {time_or_dash(c['arrive_swing'])} | {c['reachable']} |")

    if winner is not None:
        p, m = winner["pose"], winner["measured"]
        v = m["v_release"]
        heading = np.degrees(np.arctan2(v[1], v[0]))
        angle = np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2])))
        lines += ["", "## 4. Winner", "",
                  f"- `RELEASE_POSE` = {np.round(p['q'], 3).tolist()}",
                  f"- `TOSS_SPEED` = {m['speed']:.3f} m/s, `TOSS_ANGLE` = {angle:.2f}°, `TOSS_HEADING` = {heading:.2f}°",
                  f"- `VELOCITY_MARGIN` = {m['margin']:.3f}, `BRAKE` = {m['brake']:.2f} s",
                  f"- Release at {np.round(m['p_release'], 3).tolist()} m, gripper {p['palm']:.0f}° off the toss",
                  f"- Simulated peak {m['apex']:.3f} m, catch height after {m['flight']:.3f} s",
                  f"- Catch at {np.round(winner['p_catch'], 3).tolist()} m, {winner['spare']:.3f} s to spare at brake level", "",
                  "What each joint adds:", "",
                  "| joint | limit (rad/s) | used at release | measured peak | adds upward (m/s) | adds sideways (m/s) | peak torque |",
                  "|---|---|---|---|---|---|---|"]
        for row in joints:
            lines.append("| {joint} | {limit:.2f} | {used:.3f} × limit | {peak:.3f} × limit | {up:+.2f} | {side:.2f} | "
                         "{effort:.2f} × limit |".format(**row))
        lines.append(f"| total | | | | {sum(r['up'] for r in joints):+.2f} | | |")

        lines += ["", "## 5. Starting lower", "",
                  "The same release pose and speed with a longer wind-up.", "",
                  "| swing | ball start height | peak | release speed | tracking error at release | "
                  "peak joint speed | peak torque, swing | peak torque, all | result |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s in sweep:
            if "speed_peak" not in s:
                lines.append(f"| {s['swing']:.2f} s | - | - | - | - | - | - | - | {s['why']} |")
                continue
            lines.append(f"| {s['swing']:.2f} s | {s['start_z']:.3f} m | {s['apex']:.3f} m | {np.linalg.norm(s['release_v']):.3f} m/s | "
                         f"{1000 * s['tracking']:.1f} mrad | {s['speed_peak'].max():.3f} | {s['effort_swing'].max():.3f} | "
                         f"{s['effort_peak'].max():.3f} | {s['why'] or 'pass'} |")

        lines += ["", "## 6. Toss noise", "",
                  f"Out of {N_NOISE_DRAWS} draws of velocity noise added to the commanded release, as Approach A's spec does, "
                  "how many the arm cannot reach:", "", "| noise per axis | unreachable |", "|---|---|"]
        for sigma, bad in noise:
            lines.append(f"| {sigma} m/s | {bad} of {N_NOISE_DRAWS} |")
    RESULTS.write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS}")


def joint_table(winner) -> list[dict]:
    """What each joint contributes to the winning toss."""
    pose, m = winner["pose"], winner["measured"]
    arm = get_arm()
    _, _, J = arm.kinematics(pose["q"])
    rows = []
    for i in range(7):
        add = J[:, i] * m["qdot_release"][i]
        rows.append(dict(joint=i + 1, limit=arm.v_max[i], used=abs(m["qdot_release"][i]) / arm.v_max[i],
                         peak=m["speed_peak"][i], up=add[2], side=np.linalg.norm(add[:2]), effort=m["effort_peak"][i]))
    return rows


def stage_search(pool):
    print("stage 1: pose search", flush=True)
    tasks = [(band, cap, False, False) for band in RELEASE_BANDS for cap in GRIPPER_CAPS]
    tasks += [(RELEASE_BANDS[0], cap, True, False) for cap in [45, None]]
    tasks += [(band, cap, False, True) for band in RELEASE_BANDS for cap in CATCH_CAPS]
    done = pool.map(search_cell, tasks, chunksize=1)
    cells = {(band, cap): found for (band, cap, fixed, catch), found in done if not fixed and not catch}
    fixed_cells = {cap: found for (band, cap, fixed, catch), found in done if fixed}
    catch_cells = {(band, cap): found for (band, cap, fixed, catch), found in done if catch}
    return cells, fixed_cells, catch_cells


def stage_gates(pool, cells, catch_cells):
    poses = []
    for found in [*cells.values(), *catch_cells.values()]:
        for f in found:
            if all(np.linalg.norm(f["q"] - k["q"]) > 0.05 or f["catch"] != k["catch"] for k in poses):
                poses.append(f)
    print(f"stage 2: simulating {len(poses)} poses", flush=True)
    return pool.map(gate_pose, poses, chunksize=1)


def stage_catch(pool, gated):
    passed = sorted((g for g in gated if g["passed"]), key=lambda g: -g["measured"]["apex"])
    to_check = [g for g in passed if not g["pose"]["catch"]][:N_HEIGHT_ONLY_CHECKS]
    to_check += [g for g in passed if g["pose"]["catch"]]
    print(f"stage 3: {len(passed)} poses passed; checking the catch for {len(to_check)}", flush=True)
    return pool.map(catch_check, to_check, chunksize=1)


def stage_winner(pool, checked):
    """The highest peak that can be caught, then the swing sweep and noise for it."""
    catchable = [c for c in checked if c["catchable"]]
    if not catchable:
        return None, [], [], []
    winner = max(catchable, key=lambda c: (round(c["measured"]["apex"], 2), c["spare"]))
    pose, m = winner["pose"], winner["measured"]
    print(f"stage 4: winner peaks at {m['apex']:.3f} m; sweeping the swing", flush=True)
    sweep = pool.map(swing_row, [(pose, m["margin"], m["brake"], s) for s in SWINGS], chunksize=1)
    noise = pool.map(noise_row, [(pose, m["margin"], m["brake"], s) for s in NOISE_LEVELS], chunksize=1)
    return winner, sweep, noise, joint_table(winner)


def run():
    with multiprocessing.get_context("spawn").Pool(N_WORKERS, initializer=init_worker) as pool:
        cells, fixed_cells, catch_cells = stage_search(pool)
        gated = stage_gates(pool, cells, catch_cells)
        checked = stage_catch(pool, gated)
        winner, sweep, noise, joints = stage_winner(pool, checked)
    write_results(cells, fixed_cells, catch_cells, gated, checked, winner, sweep, noise, joints)
    if winner is None:
        print("no pose passed every gate")
    elif winner["measured"]["apex"] < MIN_PEAK_FOR_ADOPTION:
        print(f"best peak {winner['measured']['apex']:.3f} m is below {MIN_PEAK_FOR_ADOPTION} m: "
              "look at what is limiting it before changing the toss")


if __name__ == "__main__":
    run()
