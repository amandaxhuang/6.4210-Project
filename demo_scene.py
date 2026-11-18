"""Replay one toss and catch in Meshcat, for figures and the video.

    python demo_scene.py b 3              # Approach B on trial 3, nominal noise
    python demo_scene.py a 3              # Approach A on the same toss
    python demo_scene.py baseline 3       # the baseline on the same toss
    python demo_scene.py baseline --clean # no noise at all: the clean catch for Fig. 1
    python demo_scene.py b 3 --latency 0.04   # fingers 10 ms later than the trigger assumes
    python demo_scene.py b 3 --no-path    # hide the blue planned flight line and catch point
    python demo_scene.py b 3 --friction smooth   # a different ball: smooth, tennis, rubber, sticky, or a number

Every run draws the catch plane (green) and the planned flight (blue) and uses the same camera, so screenshots and
recordings of different runs line up.  Run each approach in its own terminal:
the second and third get the next Meshcat ports (7001, 7002).
"""

import argparse

import numpy as np
from pydrake.all import Box, Rgba, RigidTransform

import catch
import environment_friction  # NEW: for --friction
import replan
import toss
from utils.viz import get_meshcat, keep_meshcat_open

APPROACHES = {"baseline": catch.OpenLoopPolicy, "a": replan.OneCorrectionPolicy, "b": replan.MovingStartPolicy}

# One fixed view for every run.  Change these to reframe; all runs follow.
CAMERA = np.array([2.2, -2.4, 1.4])  # m, where the camera sits
TARGET = np.array([0.6, -0.15, 0.75])  # m, what it looks at

PLANE_SIZE = 0.6  # m, side of the drawn catch plane
DASH = 0.03  # m, length of each dash in its outline
# Green, so it stands apart from the blue planned flight line.
PLANE_FILL = Rgba(0.1, 0.75, 0.3, 0.3)
PLANE_EDGE = Rgba(0.0, 0.6, 0.2, 1.0)


def draw_catch_plane(meshcat, plan) -> float:
    """A faint square at the catch height, centered under the nominal catch point,
    with a dashed outline.  Returns the height."""
    _, p_catch = toss.catch_point(plan)
    z = plan.p_release[2] - toss.CATCH_DROP
    center = np.array([p_catch[0], p_catch[1], z])
    meshcat.SetObject("catch_plane/fill", Box(PLANE_SIZE, PLANE_SIZE, 0.0005), PLANE_FILL)
    meshcat.SetTransform("catch_plane/fill", RigidTransform(center))

    h = PLANE_SIZE / 2
    corners = center + np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    starts, ends = [], []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        n = int(np.linalg.norm(b - a) / DASH)
        for k in range(0, n, 2):  # every other segment: a dashed edge
            starts.append(a + (b - a) * k / n)
            ends.append(a + (b - a) * (k + 1) / n)
    meshcat.SetLineSegments("catch_plane/outline", np.array(starts).T, np.array(ends).T, 2.0, PLANE_EDGE)
    return z


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("approach", choices=APPROACHES)
    parser.add_argument("trial", nargs="?", type=int, default=3)
    parser.add_argument("--clean", action="store_true", help="no sensor or toss noise")
    parser.add_argument("--latency", type=float, help="real gripper delay (s); the trigger still assumes 0.03")
    parser.add_argument("--no-path", action="store_true", help="hide the blue planned flight line and catch point")
    # NEW: the ball's material.  Left out, the ball is environment.py's own (rubber, 1.0).
    parser.add_argument("--friction", help="ball-pad friction: smooth, tennis, rubber, sticky, or a number such as 0.4")
    args = parser.parse_args()

    # NEW: set the ball's friction before the scene is built.
    material = ""
    if args.friction is not None:
        if args.friction in environment_friction.MATERIALS:
            environment_friction.set_material(args.friction)
        else:
            environment_friction.set_friction(float(args.friction))
        material = f", {args.friction} ball (friction {environment_friction.current():g})"

    noise = dict(sigma_p=0.0) if args.clean else dict(sigma_lat=0.04, sigma_vz=0.02)
    if args.latency is not None:
        noise.update(latency=args.latency, assumed_latency=catch.T_LATENCY)
    run = catch.simulate_catch(APPROACHES[args.approach], trial=args.trial, **noise)

    meshcat = get_meshcat()
    if meshcat is not None:
        if args.no_path:
            meshcat.Delete("toss")
        z = draw_catch_plane(meshcat, run.toss_plan)
        meshcat.SetCameraPose(CAMERA, TARGET)
        print(f"catch plane drawn at z = {z:.3f} m")

    t_release = run.toss_plan.t_release
    print(f"{args.approach}, trial {args.trial}{' (no noise)' if args.clean else ''}{material}: "
          f"{'CAUGHT' if run.outcome.caught else 'missed'}")
    print(f"animation times: release {t_release:.2f} s, catch plane about {t_release + 0.575:.2f} s")
    keep_meshcat_open()


if __name__ == "__main__":
    main()
