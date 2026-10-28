"""An underhand toss: swing the ball up and forward, and let go.

Arm starts low with the palm up, swings up and forward through the
release pose, opens the fingers, and brakes hard so  ball flies on in a
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
TOSS_SPEED = 1.5 # 1.5 m/s
TOSS_ANGLE = 65.0 # 65 degrees
CATCH_DROP = 0.1 # Ball rises about 9 cm and comes down CATCH_DROP below release height

# Arm pose at the instant of release.  Found by searching for a pose whose
# hand can move fastest along the toss direction within the joint velocity
# limits (about 1.67 m/s at 65 degrees here) while the gripper points within
# 25 degrees of the toss, so the ball leaves along the palm.  The pinch point
# is at about [0.65, 0, 0.54] m. 
RELEASE_POSE = np.array([-2.34, -1.39, 1.99, -1.45, -1.62, 1.61, -1.0])

# Timing of the motion, in seconds.
SETTLE = 0.5  # hold still at the start so the grip settles
SWING = 0.25  # constant acceleration from rest at the start pose to release speed
BRAKE = 0.05  # constant deceleration from release speed to rest
AFTER = 1.5   # keep simulating after arm stops, to watch the ball

# When to send the finger-open command, this long before the release instant.
OPEN_LEAD = 0.0 # Can tune this for gripper delay layer?

# The release joint velocities may use at most this fraction of each joint's
# velocity limit, leaving headroom for tracking error.
VELOCITY_MARGIN = 0.9
# How much to discourage spinning the gripper at release, relative to joint
# effort (1 / (rad/s)^2).  Spin is not forbidden: the fingers are already
# opening, so a little rotation barely affects the ball.
SPIN_WEIGHT = 0.1

# Finger opening once the ball is let go: fully open.
FINGER_OPEN = 0.107

# How far from straight up the gripper may point when catching (degrees).
CATCH_TILT = 25.0


def release_velocity(speed: float = TOSS_SPEED, angle: float = TOSS_ANGLE) -> np.ndarray:
    """Ball velocity at release, in world coordinates: `speed` m/s at `angle`
    degrees above horizontal, aimed in +x."""
    a = np.radians(angle)
    return speed * np.array([np.cos(a), 0.0, np.sin(a)])


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
    J(RELEASE_POSE) u = v_release.  Because the release velocity points up and
    forward, the start pose is lower and closer to the robot: the underhand
    wind-up.  Braking to rest over BRAKE seconds then decelerates the hand far
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


def build_toss_system(plan_fn=plan_toss):
    """The scene with the arm and fingers following a planned toss.

    Returns (diagram, plan).  The station is built first so the plan can use
    its plant; the trajectory sources are then wired to its command inputs.
    """
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeHardwareStation(env.make_scenario(), meshcat=get_meshcat())
    )
    station.set_name("station")
    plant = station.GetSubsystemByName("plant")

    # Fingers start touching the ball, as in environment.build_system.
    wsg = plant.GetModelInstanceByName("wsg")
    plant.SetDefaultPositions(wsg, [-env.FINGER_CONTACT / 2, env.FINGER_CONTACT / 2])

    plan = plan_fn(plant)
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


def run():
    """Simulate one toss, then compare the ball's flight with the plan."""
    system, plan = build_toss_system()
    draw_planned_flight(plan)
    simulator = Simulator(system)
    context = simulator.get_mutable_context()
    env.place_ball_in_gripper(system, context)

    plant = system.GetSubsystemByName("station").GetSubsystemByName("plant")
    ball = plant.GetBodyByName(env.BODY, plant.GetModelInstanceByName("ball"))
    log = []  # (t, ball position, ball velocity)

    def record(root_context):
        plant_context = plant.GetMyContextFromRoot(root_context)
        X = plant.EvalBodyPoseInWorld(plant_context, ball)
        V = plant.EvalBodySpatialVelocityInWorld(plant_context, ball)
        log.append((root_context.get_time(), X.translation(), V.translational()))
        return EventStatus.Succeeded()

    simulator.set_monitor(record)
    if not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    start_recording()
    simulator.AdvanceTo(plan.q_traj.end_time())
    publish_recording()

    report(plan, log, plant)


def report(plan: TossPlan, log, plant):
    """Print the planned and simulated release and flight, and whether the arm
    can reach the catch point."""
    t = np.array([entry[0] for entry in log])
    p = np.array([entry[1] for entry in log])
    v = np.array([entry[2] for entry in log])

    # The ball's fastest moment near the planned release is when it leaves the
    # hand.  (Only near it: the ball is faster still when it hits the floor.)
    near = np.abs(t - plan.t_release) <= 0.1
    i_rel = np.flatnonzero(near)[np.argmax(np.linalg.norm(v[near], axis=1))]
    i_apex = i_rel + np.argmax(p[i_rel:, 2])

    t_plan, p_plan = catch_point(plan)
    apex_plan = plan.p_release[2] + plan.v_release[2] ** 2 / (2 * G_ACCEL)
    print(f"start:     ball at {np.round(p[0], 3)} m")
    print(f"planned:   release at t = {plan.t_release:.3f} s from {np.round(plan.p_release, 3)} m "
          f"at {np.round(plan.v_release, 3)} m/s")
    print(f"           apex {apex_plan:.3f} m; at catch height after {t_plan:.3f} s "
          f"at {np.round(p_plan, 3)} m")
    print(f"simulated: release at t = {t[i_rel]:.3f} s from {np.round(p[i_rel], 3)} m "
          f"at {np.round(v[i_rel], 3)} m/s")

    # First sample at or below the catch height on the way down, interpolated.
    z_catch = p_plan[2]
    below = np.flatnonzero(p[i_apex:, 2] <= z_catch)
    if len(below):
        k = i_apex + below[0]
        a = (p[k - 1, 2] - z_catch) / (p[k - 1, 2] - p[k, 2])
        t_down = t[k - 1] + a * (t[k] - t[k - 1])
        p_down = p[k - 1] + a * (p[k] - p[k - 1])
        miss = np.linalg.norm((p_down - p_plan)[:2])
        print(f"           apex {p[i_apex, 2]:.3f} m; at catch height after "
              f"{t_down - t[i_rel]:.3f} s at {np.round(p_down, 3)} m "
              f"({100 * miss:.1f} cm from the planned point)")
    else:
        print(f"           apex {p[i_apex, 2]:.3f} m; never came down to the catch height")

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