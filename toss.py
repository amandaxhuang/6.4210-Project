"""An underhand toss: swing the ball up and let go.

Arm starts low, swings up through the release pose with its joints at their
velocity limits, opens the fingers, and brakes hard so ball flies on in a
parabola.

"""

from dataclasses import dataclass

import numpy as np
from manipulation.station import MakeHardwareStation
from pydrake.all import (
    DiagramBuilder,
    EventStatus,
    InverseKinematics,
    JacobianWrtVariable,
    MathematicalProgram,
    PiecewisePolynomial,
    RigidTransform,
    Rgba,
    Simulator,
    Solve,
    Sphere,
    TrajectorySource,
)

import environment as env
from utils.viz import (
    HEADLESS,
    get_meshcat,
    keep_meshcat_open,
    publish_recording,
    start_recording,
)

G_ACCEL = 9.81
TOSS_SPEED = 2.720 # m/s; about the most RELEASE_POSE can do within VELOCITY_MARGIN
TOSS_ANGLE = 76.50 # degrees above horizontal
TOSS_HEADING = -123.74 # Direction of the toss around z, degrees from +x
CATCH_DROP = 0.1 # Ball rises about 36 cm and comes down CATCH_DROP below release height

# Arm pose at the instant of release, from release_search.py: the highest peak
# it found with the pinch point 0.75 to 0.80 m up and time left to catch the
# ball.  Joints 2 to 6 all run at their limits and add up to about 2.65 m/s
# upward.  The gripper is about 67 degrees off the toss, with the fingers'
# closing axis within 5 degrees of square to it, so the ball leaves between the
# open fingers.  The pinch point is at about [0.80, 0, 0.77] m.
RELEASE_POSE = np.array([-2.483, -0.92, -1.028, 0.716, 2.091, 1.023, -1.452])

# Timing of the motion, in seconds.
SETTLE = 0.5  # hold still at the start so the grip settles
SWING = 0.5  # constant acceleration from rest at the start pose to release speed.  SETTLE + SWING
              # is kept on a multiple of 50 ms, so the open command lands on a WSG controller update.
BRAKE = 0.1  # constant deceleration from release speed to rest; at 0.05 s the joints hit their torque limits
AFTER = 1.5   # keep simulating after arm stops, to watch the ball

# When to send the finger-open command, this long before the release instant.
OPEN_LEAD = 0.0 # Can tune this for gripper delay layer?

# The release joint velocities may use at most this fraction of each joint's
# velocity limit.  The joints overshoot by a little more than this leaves just
# after release, so measured speeds peak right at the limit: see report().
VELOCITY_MARGIN = 0.986
# How much to discourage spinning the gripper at release, relative to joint
# effort (1 / (rad/s)^2).  Spin is not forbidden: the fingers are already
# opening, so a little rotation barely affects the ball.
SPIN_WEIGHT = 0.1

# Finger opening once the ball is let go: fully open.
FINGER_OPEN = 0.107

# How far from straight up the gripper may point when catching (degrees).
CATCH_TILT = 25.0


def release_velocity(
    speed: float = TOSS_SPEED, angle: float = TOSS_ANGLE, heading: float = TOSS_HEADING
) -> np.ndarray:
    """Ball velocity at release, in world coordinates: `speed` m/s at `angle`
    degrees above horizontal, aimed `heading` degrees around z from +x."""
    a, h = np.radians(angle), np.radians(heading)
    return speed * np.array([np.cos(a) * np.cos(h), np.cos(a) * np.sin(h), np.sin(a)])


@dataclass
class TossPlan:
    """A planned toss: the arm's joint trajectory and the finger commands."""

    q_traj: PiecewisePolynomial  # iiwa joint positions over time
    wsg_traj: PiecewisePolynomial  # finger opening command over time
    q_start: np.ndarray  # underhand start pose
    t_release: float  # when the hand reaches release speed at RELEASE_POSE
    p_release: np.ndarray  # pinch point (ball center) at release, world frame
    v_release: np.ndarray  # planned ball velocity at release, world frame
    qdot_release: np.ndarray  # joint velocities at release


def arm_slice(plant) -> slice:
    """The iiwa's entries in the plant's position and velocity vectors."""
    iiwa = plant.GetModelInstanceByName("iiwa")
    return slice(
        plant.GetJointByName("iiwa_joint_1", iiwa).velocity_start(),
        plant.GetJointByName("iiwa_joint_7", iiwa).velocity_start() + 1,
    )


def gripper_frame(plant):
    """Frame G, the WSG gripper's body frame."""
    return plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")).body_frame()


def release_joint_velocity(plant, plant_context, v_WP: np.ndarray) -> np.ndarray:
    """Joint velocities that move the pinch point at exactly v_WP.

    Solves a small QP: hit v_WP exactly, keep every joint within
    VELOCITY_MARGIN of its limit, and otherwise use as little of each joint's
    limit as possible, with a small penalty on spinning the gripper.  Plain
    pseudo-inverse treats every joint alike and asks too much of the slow
    ones.  Raises if no joint velocities can reach v_WP within the limits.
    """
    W = plant.world_frame()
    J = plant.CalcJacobianSpatialVelocity(
        plant_context, JacobianWrtVariable.kV, gripper_frame(plant), env.PINCH, W, W
    )
    arm = arm_slice(plant)
    J_spin, J_move = J[:3, arm], J[3:, arm]  # angular rows, translational rows
    v_max = plant.GetVelocityUpperLimits()[arm]

    prog = MathematicalProgram()
    qdot = prog.NewContinuousVariables(len(v_max), "qdot")
    prog.AddLinearEqualityConstraint(J_move, v_WP, qdot)
    prog.AddBoundingBoxConstraint(-VELOCITY_MARGIN * v_max, VELOCITY_MARGIN * v_max, qdot)
    # Cost: sum of (qdot_i / v_max_i)^2, plus SPIN_WEIGHT * |gripper angular velocity|^2.
    Q = np.diag(1 / v_max**2) + SPIN_WEIGHT * J_spin.T @ J_spin
    prog.AddQuadraticCost(2 * Q, np.zeros(len(v_max)), qdot)
    result = Solve(prog)
    if not result.is_success():
        raise ValueError(
            f"no joint velocities reach {np.round(v_WP, 2)} m/s within "
            f"{VELOCITY_MARGIN:.0%} of the velocity limits at this pose; "
            "toss slower or change RELEASE_POSE"
        )
    return result.GetSolution(qdot)


def plan_toss(plant, v_release: np.ndarray | None = None) -> TossPlan:
    """Plan an underhand toss that releases at RELEASE_POSE.

    Arm moves in a straight line in joint space along the release joint
    velocity u.  Accelerating at a constant rate from rest to u over SWING
    seconds covers u * SWING / 2, so the start pose is that far back:
    q_start = RELEASE_POSE - u * SWING / 2.  Swinging from there reaches u
    exactly at the release pose, where the gripper's velocity is
    J(RELEASE_POSE) u = v_release.  Because the release velocity points mostly
    up, the start pose is lower: the underhand wind-up.  Braking to rest over BRAKE seconds then decelerates the hand far
    faster than gravity, so the ball leaves it.

    Also sets the plant's default arm positions to the start pose, so a
    simulation built on this plant begins there.
    """
    if v_release is None:
        v_release = release_velocity()
    iiwa = plant.GetModelInstanceByName("iiwa")
    context = plant.CreateDefaultContext()
    q_rel = RELEASE_POSE
    plant.SetPositions(context, iiwa, q_rel)
    p_release = plant.CalcRelativeTransform(
        context, plant.world_frame(), gripper_frame(plant)
    ) @ env.PINCH
    u = release_joint_velocity(plant, context, v_release)

    q_start = q_rel - u * SWING / 2
    q_stop = q_rel + u * BRAKE / 2
    arm = arm_slice(plant)
    lo = plant.GetPositionLowerLimits()[arm]
    hi = plant.GetPositionUpperLimits()[arm]
    for name, q in [("start", q_start), ("stop", q_stop)]:
        if np.any(q < lo) or np.any(q > hi):
            raise ValueError(f"the {name} pose is outside the joint limits")
    plant.SetDefaultPositions(iiwa, q_start)

    # Knots: (time, position, velocity).  Between knots, cubic Hermite
    # interpolation reproduces the constant-acceleration segments exactly.
    zero = np.zeros_like(q_rel)
    t_release = SETTLE + SWING
    t_stop = t_release + BRAKE
    t_end = t_stop + AFTER
    knots = [
        (0.0, q_start, zero),
        (SETTLE, q_start, zero),
        (t_release, q_rel, u),
        (t_stop, q_stop, zero),
        (t_end, q_stop, zero),
    ]
    q_traj = PiecewisePolynomial.CubicHermite(
        [k[0] for k in knots],
        np.column_stack([k[1] for k in knots]),
        np.column_stack([k[2] for k in knots]),
    )

    # Fingers: squeeze until OPEN_LEAD before release, then open.
    t_open = t_release - OPEN_LEAD
    wsg_traj = PiecewisePolynomial.ZeroOrderHold(
        [0.0, t_open, t_end],
        np.array([[env.FINGER_CLOSED, FINGER_OPEN, FINGER_OPEN]]),
    )
    return TossPlan(q_traj, wsg_traj, q_start, t_release, p_release, np.array(v_release), u)


def add_toss_station(builder, plan_fn=plan_toss):
    """Add the scene to builder, ready for a toss, with its command inputs left
    unconnected.  Returns (station, plan).  The station is built first so the
    plan can use its plant.
    """
    station = builder.AddSystem(
        MakeHardwareStation(env.make_scenario(), meshcat=get_meshcat())
    )
    station.set_name("station")
    plant = station.GetSubsystemByName("plant")

    # Fingers start touching the ball, as in environment.build_system.
    wsg = plant.GetModelInstanceByName("wsg")
    plant.SetDefaultPositions(wsg, [-env.FINGER_CONTACT / 2, env.FINGER_CONTACT / 2])
    return station, plan_fn(plant)


def build_toss_system(plan_fn=plan_toss):
    """The scene with the arm and fingers following a planned toss.

    Returns (diagram, plan).
    """
    builder = DiagramBuilder()
    station, plan = add_toss_station(builder, plan_fn)
    arm = builder.AddSystem(TrajectorySource(plan.q_traj))
    builder.Connect(arm.get_output_port(), station.GetInputPort("iiwa.position"))
    fingers = builder.AddSystem(TrajectorySource(plan.wsg_traj))
    builder.Connect(fingers.get_output_port(), station.GetInputPort("wsg.position"))
    return builder.Build(), plan


def ballistic_position(p0: np.ndarray, v0: np.ndarray, t) -> np.ndarray:
    """Where a ball leaving p0 at velocity v0 is t seconds later, under gravity
    alone.  t may be an array, giving one row per time."""
    t = np.asarray(t, dtype=float)[..., None]
    return p0 + v0 * t + 0.5 * np.array([0.0, 0.0, -G_ACCEL]) * t**2


def time_to_height(p0: np.ndarray, v0: np.ndarray, z: float) -> float:
    """When a ball leaving p0 at velocity v0 comes down through height z: the
    later root of z0 + vz t - g t^2 / 2 = z."""
    vz, dz = v0[2], z - p0[2]
    return (vz + np.sqrt(vz**2 - 2 * G_ACCEL * dz)) / G_ACCEL


def catch_point(plan: TossPlan) -> tuple[float, np.ndarray]:
    """When (seconds after release) and where the planned flight comes down
    through the catch height, CATCH_DROP below the release."""
    t = time_to_height(plan.p_release, plan.v_release, plan.p_release[2] - CATCH_DROP)
    return t, ballistic_position(plan.p_release, plan.v_release, t)


def catch_pose(plant, p_catch: np.ndarray) -> np.ndarray | None:
    """Arm joint angles that put the pinch point at p_catch with the gripper
    within CATCH_TILT of pointing up, or None if inverse kinematics finds none.
    Seeded from the release pose, so the answer continues the toss motion."""
    ik = InverseKinematics(plant)
    q = ik.q()
    G, W = gripper_frame(plant), plant.world_frame()
    ik.AddPositionConstraint(G, env.PINCH, W, p_catch - 0.003, p_catch + 0.003)
    ik.AddAngleBetweenVectorsConstraint(
        W, np.array([0.0, 0.0, 1.0]), G, np.array([0.0, 1.0, 0.0]),
        0.0, np.radians(CATCH_TILT),
    )
    arm = arm_slice(plant)
    q_seed = plant.GetPositions(plant.CreateDefaultContext())
    q_seed[arm] = RELEASE_POSE
    ik.prog().AddQuadraticErrorCost(np.eye(len(q_seed)), q_seed, q)
    ik.prog().SetInitialGuess(q, q_seed)
    result = Solve(ik.prog())
    if not result.is_success():
        return None
    return result.GetSolution(q)[arm]


def draw_planned_flight(plan: TossPlan, n: int = 40) -> None:
    """Draw the planned parabola in Meshcat as a line, with a small sphere at
    the catch point."""
    meshcat = get_meshcat()
    if meshcat is None:
        return
    t_catch, p_catch = catch_point(plan)
    points = ballistic_position(plan.p_release, plan.v_release, np.linspace(0, t_catch, n))
    meshcat.SetLine("toss/planned_flight", points.T, 2.0, Rgba(0.2, 0.4, 1.0, 1))
    meshcat.SetObject("toss/catch_point", Sphere(0.01), Rgba(0.2, 0.4, 1.0, 0.6))
    meshcat.SetTransform("toss/catch_point", RigidTransform(p_catch))


def ball_contacts(plant, plant_context, ball) -> frozenset[str]:
    """Names ("model::body") of the bodies touching the ball."""
    results = plant.get_contact_results_output_port().Eval(plant_context)
    inspector = plant.get_geometry_query_input_port().Eval(plant_context).inspector()
    pairs = []
    for i in range(results.num_hydroelastic_contacts()):
        surface = results.hydroelastic_contact_info(i).contact_surface()
        pairs.append([plant.GetBodyFromFrameId(inspector.GetFrameId(g))
                      for g in (surface.id_M(), surface.id_N())])
    for i in range(results.num_point_pair_contacts()):
        info = results.point_pair_contact_info(i)
        pairs.append([plant.get_body(info.bodyA_index()), plant.get_body(info.bodyB_index())])
    names = set()
    for a, b in pairs:
        if ball.index() in (a.index(), b.index()):
            other = b if a.index() == ball.index() else a
            names.add(f"{plant.GetModelInstanceName(other.model_instance())}::{other.name()}")
    return frozenset(names)


@dataclass
class TossLog:
    """A simulated toss, one row per simulator step."""

    t: np.ndarray
    ball_p: np.ndarray  # ball center, world frame
    ball_v: np.ndarray  # ball velocity, world frame
    touching: list[frozenset[str]]  # bodies touching the ball
    pinch: np.ndarray  # pinch point, world frame
    arm_q: np.ndarray  # arm joint positions
    arm_v: np.ndarray  # arm joint velocities
    arm_effort: np.ndarray  # torque applied at each arm joint


def simulate(plan_fn=plan_toss):
    """Simulate one toss.  Returns (plan, log, plant)."""
    system, plan = build_toss_system(plan_fn)
    draw_planned_flight(plan)
    simulator = Simulator(system)
    context = simulator.get_mutable_context()
    env.place_ball_in_gripper(system, context)

    station = system.GetSubsystemByName("station")
    plant = station.GetSubsystemByName("plant")
    iiwa = plant.GetModelInstanceByName("iiwa")
    ball = plant.GetBodyByName(env.BODY, plant.GetModelInstanceByName("ball"))
    G = gripper_frame(plant).body()
    effort = station.GetOutputPort("iiwa_net_actuation")
    rows = []

    def record(root_context):
        plant_context = plant.GetMyContextFromRoot(root_context)
        X = plant.EvalBodyPoseInWorld(plant_context, ball)
        V = plant.EvalBodySpatialVelocityInWorld(plant_context, ball)
        rows.append((
            root_context.get_time(), X.translation(), V.translational(),
            ball_contacts(plant, plant_context, ball),
            plant.EvalBodyPoseInWorld(plant_context, G) @ env.PINCH,
            plant.GetPositions(plant_context, iiwa), plant.GetVelocities(plant_context, iiwa),
            effort.Eval(station.GetMyContextFromRoot(root_context)),
        ))
        return EventStatus.Succeeded()

    simulator.set_monitor(record)
    if not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    start_recording()
    simulator.AdvanceTo(plan.q_traj.end_time())
    publish_recording()

    t, ball_p, ball_v, touching, pinch, arm_q, arm_v, arm_effort = zip(*rows)
    log = TossLog(np.array(t), np.array(ball_p), np.array(ball_v), list(touching),
                  np.array(pinch), np.array(arm_q), np.array(arm_v), np.array(arm_effort))
    return plan, log, plant


def run():
    """Simulate one toss, then compare the ball's flight with the plan."""
    plan, log, plant = simulate()
    report(plan, log, plant)


def arm_limits(plant) -> tuple[np.ndarray, np.ndarray]:
    """Each arm joint's velocity limit (rad/s) and effort limit (N m)."""
    iiwa = plant.GetModelInstanceByName("iiwa")
    effort = [plant.get_joint_actuator(i).effort_limit() for i in plant.GetJointActuatorIndices(iiwa)]
    return plant.GetVelocityUpperLimits()[arm_slice(plant)], np.array(effort)


@dataclass
class Flight:
    """What the ball did after release, from a TossLog.  Only the free flight is
    measured: from when the ball stops touching the fingers to when it next
    touches anything."""

    i_release: int  # row where the ball leaves the hand
    i_free: int | None  # first row after release touching nothing; None if it never left
    i_hit: int  # next row touching something; len(log.t) if nothing
    apex: float = np.nan  # highest point of the free flight (m)
    off: float = np.nan  # farthest from the planned parabola (m)
    t_down: float | None = None  # simulation time at catch height, None if never
    p_down: np.ndarray | None = None  # where it came down through catch height
    extrapolated: bool = False  # the crossing came after the ball hit something


def free_flight(plan: TossPlan, log: TossLog) -> Flight:
    """Measure the ball's free flight against the plan.  If the ball touches
    something before the catch height (with a steep toss, it can come down onto
    the open fingers, parked where the brake left them), the crossing is
    extrapolated from its state just before."""
    t, p, v, touching = log.t, log.ball_p, log.ball_v, log.touching

    # The ball's fastest moment near the planned release is when it leaves the
    # hand.  (Only near it: the ball is faster still when it hits the floor.)
    near = np.abs(t - plan.t_release) <= 0.1
    i_rel = np.flatnonzero(near)[np.argmax(np.linalg.norm(v[near], axis=1))]

    # Free flight: from the first sample after release touching nothing, to the
    # next sample touching something.
    free = [i for i in range(i_rel, len(t)) if not touching[i]]
    if not free:
        return Flight(i_rel, None, len(t))
    i_free = free[0]
    i_hit = next((i for i in range(i_free, len(t)) if touching[i]), len(t))
    flight = slice(i_free, i_hit)
    i_apex = i_free + np.argmax(p[flight, 2])
    planned = ballistic_position(plan.p_release, plan.v_release, t[flight] - plan.t_release)
    result = Flight(i_rel, i_free, i_hit, p[i_apex, 2],
                    np.linalg.norm(p[flight] - planned, axis=1).max())

    # First free-flight sample at or below the catch height on the way down,
    # interpolated; or, if the ball touched something before getting there,
    # the crossing extrapolated from its last free-flight state.
    z_catch = catch_point(plan)[1][2]
    below = np.flatnonzero(p[i_apex:i_hit, 2] <= z_catch)
    if len(below) and i_apex + below[0] > i_free:
        k = i_apex + below[0]
        a = (p[k - 1, 2] - z_catch) / (p[k - 1, 2] - p[k, 2])
        result.t_down = t[k - 1] + a * (t[k] - t[k - 1])
        result.p_down = p[k - 1] + a * (p[k] - p[k - 1])
    elif p[i_hit - 1, 2] > z_catch:
        k = i_hit - 1
        dt = time_to_height(p[k], v[k], z_catch)
        result.t_down = t[k] + dt
        result.p_down = ballistic_position(p[k], v[k], dt)
        result.extrapolated = True
    return result


def report(plan: TossPlan, log: TossLog, plant):
    """Print the planned and simulated release and flight, how close the arm
    came to its limits, and whether it can reach the catch point."""
    t, p, v = log.t, log.ball_p, log.ball_v
    fly = free_flight(plan, log)
    i_rel = fly.i_release

    t_plan, p_plan = catch_point(plan)
    apex_plan = plan.p_release[2] + plan.v_release[2] ** 2 / (2 * G_ACCEL)
    print(f"start:     ball at {np.round(p[0], 3)} m")
    print(f"planned:   release at t = {plan.t_release:.3f} s from {np.round(plan.p_release, 3)} m "
          f"at {np.round(plan.v_release, 3)} m/s")
    print(f"           apex {apex_plan:.3f} m; at catch height after {t_plan:.3f} s "
          f"at {np.round(p_plan, 3)} m")
    print(f"simulated: release at t = {t[i_rel]:.3f} s from {np.round(p[i_rel], 3)} m "
          f"at {np.round(v[i_rel], 3)} m/s")

    if fly.i_free is None:
        print("           the ball never left the fingers")
    else:
        print(f"           free flight {t[fly.i_free] - t[i_rel]:.3f} to {t[fly.i_hit - 1] - t[i_rel]:.3f} s "
              f"after release, at most {100 * fly.off:.1f} cm from the planned parabola")
        if fly.t_down is None:
            print(f"           apex {fly.apex:.3f} m; never came down to the catch height")
        else:
            miss = np.linalg.norm((fly.p_down - p_plan)[:2])
            how = ", extrapolated from free flight" if fly.extrapolated else ""
            print(f"           apex {fly.apex:.3f} m; at catch height after "
                  f"{fly.t_down - t[i_rel]:.3f} s at {np.round(fly.p_down, 3)} m "
                  f"({100 * miss:.1f} cm from the planned point{how})")
        if fly.i_hit < len(t):
            print(f"first contact after release: {', '.join(sorted(log.touching[fly.i_hit]))} at "
                  f"{t[fly.i_hit] - t[i_rel]:.3f} s after release, ball at {np.round(p[fly.i_hit], 3)} m")

    v_max, effort_max = arm_limits(plant)
    speed, effort = np.abs(log.arm_v).max(axis=0) / v_max, np.abs(log.arm_effort).max(axis=0) / effort_max
    print(f"arm:       peak joint speed {speed.max():.3f} of limit (joint {speed.argmax() + 1}); "
          f"peak torque {effort.max():.3f} of limit (joint {effort.argmax() + 1})")

    q_catch = catch_pose(plant, p_plan)
    if q_catch is None:
        print("catch point: NOT reachable with the gripper pointing up")
    else:
        print(f"catch point: reachable with the gripper within {CATCH_TILT:.0f} deg of up; "
              f"hand {np.linalg.norm(p_plan[:2]):.2f} m out from the base vs "
              f"{np.linalg.norm(p[0, :2]):.2f} m at the start")
    print(f"ball ends at {np.round(p[-1], 3)}")


if __name__ == "__main__":
    run()
    keep_meshcat_open()