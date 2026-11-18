"""environment.py with the ball's friction as an option.  environment.py itself is untouched.

NEW FILE (friction sweep).  Why this isn't a full copy of environment.py: toss.py and
catch.py import `environment` by name and call `environment.make_scenario()`, so a
renamed copy would never be used.  Instead this module reuses everything in
environment.py and, when `set_friction` is called, swaps in a `make_scenario` whose
ball carries a friction coefficient.  catch_timing.py swaps the scenario the same way
for its time-step check.

    import environment_friction
    environment_friction.set_friction(0.25)        # or set_material("smooth")
    run = catch.simulate_catch(...)                # now uses the slippery ball

Without a call to set_friction, nothing changes: the simulation is exactly what
environment.py builds.

What "friction" means here: the coefficient between the ball and the gripper's
fingertip pads, which is what decides whether a catch holds.  Drake combines two
surfaces' coefficients as  mu = 2 mu_ball mu_pad / (mu_ball + mu_pad),  and the pads
are fixed at PAD_FRICTION, so this sets the ball's own mu to whatever gives the
requested ball-pad value.  Only friction changes: size, mass and softness stay as
environment.py has them, so any change in the results is friction's alone.
"""

import environment as env

# The WSG fingertip pads' coefficient: <drake:mu_dynamic>1.0</drake:mu_dynamic> on
# left_tip and right_tip in manipulation/models/hydro/schunk_wsg_50_with_tip.sdf.
PAD_FRICTION = 1.0

# Ball-pad coefficients for the sweep.  Representative values, not measurements:
# say so in the paper.  "rubber" is what environment.py already simulates (the ball
# sets no friction, so Drake gives it 1.0, against pads at 1.0), so every result
# produced before this file existed is the rubber ball.
MATERIALS = {
    "smooth": 0.25,  # polished plastic on rubber pads
    "tennis": 0.6,  # felt on rubber pads
    "rubber": 1.0,  # rubber on rubber: the existing results
    "sticky": 1.5,  # tacky surface: tests the ball dragging on the fingers at release
}
DEFAULT = MATERIALS["rubber"]

_original_make_scenario = env.make_scenario  # environment.py's own, kept to restore
_friction = DEFAULT  # the ball-pad coefficient the next scenario gets


def ball_mu(target: float, pad: float = PAD_FRICTION) -> float:
    """The ball's own coefficient that makes the ball-pad coefficient `target`.
    Inverts Drake's rule target = 2 b p / (b + p):  b = target p / (2 p - target).
    Only targets below 2 p are possible, since the combination never reaches 2 p."""
    if not 0 < target < 2 * pad:
        raise ValueError(f"ball-pad friction {target} is out of reach with pads at {pad}: it must be in (0, {2 * pad})")
    return target * pad / (2 * pad - target)


def combined(mu_ball: float, pad: float = PAD_FRICTION) -> float:
    """Drake's ball-pad coefficient for a ball with coefficient mu_ball."""
    return 2 * mu_ball * pad / (mu_ball + pad)


def ball_directive() -> str:
    """CHANGED from environment.ball_directive: the ball's SDF also sets friction.

    Everything else (radius, mass, inertia, colour, hydroelastic contact) is
    environment.py's.  Static and dynamic friction get the same value: the discrete
    contact solver this simulation uses reads only the dynamic one.
    """
    i = 2 / 5 * env.BALL_MASS * env.BALL_RADIUS**2  # solid sphere, as in environment.py
    path = env.write_sdf(
        "ball",
        f"<sphere><radius>{env.BALL_RADIUS}</radius></sphere>",
        "0 0 0 0 0 0",
        env.BALL_MASS,
        (i, i, i),
        (0.9, 0.2, 0.2, 1),
    )
    mu = ball_mu(_friction)
    # NEW: add the friction tags to the proximity properties write_sdf produced.
    tags = (f"<drake:mu_dynamic>{mu:.12g}</drake:mu_dynamic>\n"
            f"          <drake:mu_static>{mu:.12g}</drake:mu_static>\n"
            "        </drake:proximity_properties>")
    text = path.read_text()
    assert text.count("</drake:proximity_properties>") == 1, "environment.write_sdf's template changed"
    path.write_text(text.replace("</drake:proximity_properties>", tags))
    return f"""directives:
- add_model:
    name: ball
    file: file://{path}
"""


def make_scenario():
    """environment.make_scenario, with the friction-carrying ball."""
    env.ASSET_DIR.mkdir(parents=True, exist_ok=True)
    scenario = env.LoadScenario(data=env.ROBOT)
    scenario = env.AppendDirectives(scenario, data=env.floor_directive())
    scenario = env.AppendDirectives(scenario, data=ball_directive())  # CHANGED: this module's ball
    return scenario


def set_friction(target: float) -> None:
    """Use a ball whose ball-pad coefficient is `target` in every simulation from
    now on, in this process.  Each sweep worker is its own process, so a sweep
    calls this at the start of every trial."""
    global _friction
    ball_mu(target)  # check it is reachable before changing anything
    _friction = target
    env.make_scenario = make_scenario


def set_material(name: str) -> None:
    """set_friction by name: one of MATERIALS."""
    set_friction(MATERIALS[name])


def reset() -> None:
    """Back to environment.py's own scenario."""
    global _friction
    _friction = DEFAULT
    env.make_scenario = _original_make_scenario


def current() -> float:
    return _friction
