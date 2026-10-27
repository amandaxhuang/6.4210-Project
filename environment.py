"""
An iiwa14 with a gripper, a floor, and a ball held in the gripper.
The arm is held at a fixed pose and the fingers squeeze the ball: this is starting point for tossing and catching.
"""

from pathlib import Path

import numpy as np
from manipulation.station import (
    AppendDirectives,
    LoadScenario,
    MakeHardwareStation,
    MakeMultibodyPlant,
    Scenario,
)
from pydrake.all import (
    AddFrameTriadIllustration,
    ConstantVectorSource,
    DiagramBuilder,
    Integrator,
    InverseKinematics,
    JacobianWrtVariable,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Solve,
    TrajectorySource,
)

from utils.viz import HEADLESS, get_meshcat, keep_meshcat_open

# Generated SDFs for the floor and ball in assets dir; rewritten every run.
ASSET_DIR = Path(__file__).parent.resolve() / "assets"
BODY = "base_link"

# Floor box (x, y, z) in meters - top surface at z = 0, flush with  iiwa's base.  
FLOOR_SIZE = (4.0, 4.0, 0.1)

# Ball size and mass.  BALL_RADIUS must satisfy two gripper limits: 0.0125 m <= BALL_RADIUS <= 0.0575 m
#   - the pads must open wide enough to fit it: FINGER_CONTACT <= 0.11 m
#     (each WSG finger travels 0.055 m), so BALL_RADIUS <= 0.0575 m;
#   - the closed command must not be negative: FINGER_CLOSED >= 0,
#     so BALL_RADIUS >= 0.0125 m.
BALL_RADIUS = 0.04
BALL_MASS = 0.1
PINCH = np.array([0.0, 0.108, 0.0]) # Point between WSG fingertap pads, in gripper body frame G
FINGER_CONTACT = 2 * BALL_RADIUS - 0.005 # Finger opening (dist betwen 2 finger joints) - each pad's inner face is 2.5 mm farther out than its joint
FINGER_SQUEEZE = 0.02 # 2 cm past contact to squeeze the ball
FINGER_CLOSED = FINGER_CONTACT - FINGER_SQUEEZE

# Robot: iiwa14 joined to world origin, with a WSG gripper welded joined to its last link.  
# default_joint_positions is the pose the arm starts in and is held at
ROBOT = """directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.4]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.6]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.2]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90] }

model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {}
"""

def write_sdf(name: str, geometry: str, center: str, mass: float, inertia, rgba) -> Path:
    """Write a one-link SDF with compliant hydroelastic contact.
    name:     model name, and the file is ASSET_DIR/<name>.sdf (overwritten).
    geometry: an SDF <geometry> body, e.g. "<sphere><radius>0.04</radius></sphere>".
    center:   SDF pose "x y z roll pitch yaw" of the shape in the link frame;
              the inertia, collision, and visual all use this same pose.
    inertia:  (ixx, iyy, izz) about the center; off-diagonal terms are zero,
              which is exact for a sphere or an axis-aligned box.
    rgba:     visual color, each component 0 to 1.

    The collision is compliant hydroelastic with modulus 1e6 Pa, so contacts
    are soft and give smooth forces, which helps the grip stay stable.
    """
    ixx, iyy, izz = inertia
    path = ASSET_DIR / f"{name}.sdf"
    path.write_text(f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <link name="{BODY}">
      <inertial>
        <pose>{center}</pose>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy}</iyy><iyz>0</iyz>
          <izz>{izz}</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <pose>{center}</pose>
        <geometry>{geometry}</geometry>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>1.0e6</drake:hydroelastic_modulus>
          <drake:mesh_resolution_hint>0.01</drake:mesh_resolution_hint>
        </drake:proximity_properties>
      </collision>
      <visual name="visual">
        <pose>{center}</pose>
        <geometry>{geometry}</geometry>
        <material><diffuse>{" ".join(map(str, rgba))}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
""")
    return path


def floor_directive() -> str:
    """A box welded down with its top surface at z = 0.
    The box's link frame is at its top surface, and the shape is shifted down
    by half its thickness. 
    """
    sx, sy, sz = FLOOR_SIZE
    m = 20.0
    sdf = write_sdf(
        "floor",
        f"<box><size>{sx} {sy} {sz}</size></box>",
        f"0 0 {-sz / 2} 0 0 0",
        m,
        # Solid box inertia about its center.
        (m * (sy**2 + sz**2) / 12, m * (sx**2 + sz**2) / 12, m * (sx**2 + sy**2) / 12),
        (0.55, 0.50, 0.45, 1),
    )
    return f"""directives:
- add_model:
    name: floor
    file: file://{sdf}
- add_weld:
    parent: world
    child: floor::{BODY}
"""


def ball_directive() -> str:
    """A free sphere, body frame at its center.
    No starting pose is set: where ball belongs depends on where gripper is,
    so place_ball_in_gripper moves it into the fingers in the simulation's context.  
    Until then it sits at the world origin.
    """
    # Solid sphere: I = 2/5 m r^2 about every axis through its center.
    i = 2 / 5 * BALL_MASS * BALL_RADIUS**2 # Inertia of solid uniform sphere: 2/5 * mr^2
    sdf = write_sdf(
        "ball",
        f"<sphere><radius>{BALL_RADIUS}</radius></sphere>",
        "0 0 0 0 0 0",
        BALL_MASS,
        (i, i, i),
        (0.9, 0.2, 0.2, 1),
    )
    return f"""directives:
- add_model:
    name: ball
    file: file://{sdf}
"""


def make_scenario():
    """The full scenario: robot, then floor, then ball.  Writes floor and
    ball SDFs into ASSET_DIR."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    scenario = LoadScenario(data=ROBOT)
    scenario = AppendDirectives(scenario, data=floor_directive())
    scenario = AppendDirectives(scenario, data=ball_directive())
    return scenario


def build_system():
    """The scene, with the arm held at its starting pose and the fingers
    closed on the ball.
    """
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeHardwareStation(make_scenario(), meshcat=get_meshcat())
    )
    station.set_name("station")
    plant = station.GetSubsystemByName("plant")

    # Command arm to stay at default joint positions (from ROBOT).
    iiwa = plant.GetModelInstanceByName("iiwa")
    hold_arm = builder.AddSystem(ConstantVectorSource(plant.GetDefaultPositions(iiwa)))
    builder.Connect(hold_arm.get_output_port(), station.GetInputPort("iiwa.position"))

    # Start the pads touching the ball and command them past it to squeeze
    wsg = plant.GetModelInstanceByName("wsg")
    plant.SetDefaultPositions(wsg, [-FINGER_CONTACT / 2, FINGER_CONTACT / 2]) # Set gripper's starting finger positions
    fingers = builder.AddSystem(ConstantVectorSource([FINGER_CLOSED])) # Constant target opening of 0.055 (2 cm inside ball)
    builder.Connect(fingers.get_output_port(), station.GetInputPort("wsg.position"))

    return builder.Build()


def place_ball_in_gripper(system, context):
    """Put the ball's center at the gripper's pinch point.
    
    Called after creating the Simulator and before advancing it.  
    Ball's orientation is left as identity: sphere looks + behaves same either way.
    """
    plant = system.GetSubsystemByName("station").GetSubsystemByName("plant")
    plant_context = plant.GetMyMutableContextFromRoot(context)
    G = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))
    ball = plant.GetBodyByName(BODY, plant.GetModelInstanceByName("ball"))
    X_WG = plant.EvalBodyPoseInWorld(plant_context, G)
    plant.SetFreeBodyPose(plant_context, ball, RigidTransform(X_WG @ PINCH))


def run(duration: float = 3.0):
    """Simulate the held ball for `duration` seconds and print where it ends.

    If the grip works, the ball ends where it started, about 0.30 m up.  A
    final height near BALL_RADIUS (0.04 m) means it was dropped onto the floor.
    """
    system = build_system()
    simulator = Simulator(system)
    place_ball_in_gripper(system, simulator.get_mutable_context())
    if not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(duration)

    plant = system.GetSubsystemByName("station").GetSubsystemByName("plant")
    ball = plant.GetBodyByName(BODY, plant.GetModelInstanceByName("ball"))
    X_WB = plant.EvalBodyPoseInWorld(
        plant.GetMyContextFromRoot(simulator.get_context()), ball
    )
    print(f"ball ends at {np.round(X_WB.translation(), 3)}")


if __name__ == "__main__":
    run()
    keep_meshcat_open()
