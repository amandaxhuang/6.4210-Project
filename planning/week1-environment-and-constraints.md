# Week 1 Spec — Drake Environment & Initial Constraint Exploration

**Project:** Effect of Continual Planning on Robotic Arm Catching Robustness (6.4210, Team 4.4)
**Week:** 1 (of 2026-10-26), toward the final deadline of 2026-12-02
**Source:** `6.4210 project planning.pdf`, §6 Timeline, Week 1
**Status:** Draft for team review

---

## 1. Purpose and Scope

The proposal commits Week 1 to four things:

1. Project setup: GitHub, Drake environment with 1 arm and 1 sphere
2. Initial exploration into constraints, limitations, or assumptions we will make
3. All team members familiar with setup and pushing to GitHub
4. Outline paper structure

This spec covers items 1–3. Item 4 (paper outline) is tracked separately and is not
specified here.

The proposal also flags one risk by name, and it is the single most important question
Week 1 has to answer:

> "Joint velocity limits may cap the release speed and therefore the toss height, so we
> will check the limits for our model in week 1, and may relax them in simulation."

### Explicitly out of scope for Week 1

Nothing in Week 1 plans, executes, or evaluates a toss or a catch. Those are
approach-specific and belong to Weeks 2–3:

- Approach A (predict once, then intercept)
- Approach B (closed-loop replanning)
- The open-loop reference baseline
- The trajectory estimator (least squares / RLS / Kalman filter)
- Release-state optimization and toss trajectory generation
- The evaluation harness and the 100-toss sweeps

What Week 1 *does* build is the approach-agnostic foundation all three approaches share:
a working environment, a physics-only scene, measured limits, and a written set of
assumptions.

### Current state of the repo

Worth knowing before starting, because none of it is project-ready:

- **No Python environment exists.** `pydrake` is not importable from any interpreter on
  the author's PATH. A working course venv exists at `~/Documents/64210/.venv`
  (Python 3.13.15, drake 1.45.0, manipulation 2026.9.17.post1), but it belongs to the
  pset clone and is not shared with teammates.
- **`iiwa_code.py` is a pset skeleton, not project code.** It references `BLOCK_LAYOUT`,
  `TABLE_SIZE`, `BLOCK_SIZE`, `TABLE_COLOR`, `BLOCK_MASS`, and `TABLE_MASS`, none of
  which are defined in the file, and five of its functions still
  `raise NotImplementedError`. It models a table-and-blocks pick-and-place scene. It is
  still useful — its IK helper, its Jacobian column slicing, and its SDF-generation and
  weld directives are all patterns we reuse below — but it is reference material.
- **Seven `.pyc` files under `utils/__pycache__/` are tracked in git**, and there is no
  `.gitignore`.
- **The GitHub remote already exists** (`github.com/amandaxhuang/6.4210-Project`), so
  item 3 above is about documented setup steps and a verified clone, not repo creation.

---

## 2. Deliverables

Four deliverables, each with an acceptance criterion that someone other than the author
can check.

| ID | Deliverable | Acceptance criterion |
|----|-------------|----------------------|
| **D1** | Repo scaffolding: `.gitignore`, README setup section, directory layout, smoke test | All three teammates clone fresh, follow the README, and `scripts/smoke_test.py` passes on their machine |
| **D2** | Physics-only scene module (`tosscatch/scene.py`, `tosscatch/constants.py`) | Scene loads in Meshcat: arm standing, ball resting on the ground, no parser warnings, ball settles rather than jitters |
| **D3** | Limits report script (`scripts/check_limits.py`) | Writes `results/week1_limits.md` containing a yes/no answer to "can this arm produce a 0.5–0.65 s flight?", with the numbers behind it |
| **D4** | Assumptions & constraints doc (`planning/assumptions.md`) | Every row of the parameter table in §8 has a value, a basis, and a status; no row reads "TBD" without an owner and a date |

---

## 3. Environment Setup (D1)

We use a **fresh `uv` virtual environment inside this repo**, mirroring the course
README's Workflow A so that it is already familiar, plus a pinned `requirements.txt` so
all three of us resolve to identical versions.

```sh
cd ~/path/to/6.4210-Project
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
uv pip install drake "manipulation[letter]"
uv pip install scipy rtree
uv pip freeze > requirements.txt     # first person to set up only; commit this
```

Everyone after the first person installs from the pin instead:

```sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

Also needed, outside Python: **Graphviz** (the program, for rendering system block
diagrams) — `brew install graphviz` on macOS, `sudo apt install graphviz` on
Ubuntu/WSL.

**Windows:** there are no Windows builds of Drake. Use WSL with Ubuntu and follow the
Linux path inside it.

### Known-good versions

These are the versions verified working on the course venv as of 2026-10-27, and are
what `requirements.txt` should pin unless there is a reason to move:

- Python 3.13.15
- drake 1.45.0
- manipulation 2026.9.17.post1

### Warning: no spaces in the venv path

Drake downloads its robot models (~170 MB, once, into `~/.cache/drake`) the first time
anything loads a `package://drake_models/...` URI. It does this by handing a command
line to `/bin/sh` with the path of a helper script pasted in **unquoted**. A space —
or an apostrophe, parenthesis, `&`, `;`, `$`, or backtick — anywhere in the venv path
makes the fetch fail with a message that says nothing about paths:

```
PackageMap: when downloading 'drake_models', the downloader experienced an error:
returncode == 512
```

`utils/drake_models.py` documents this in full and provides
`explain_model_download_error` for the `except` around our first model load — **use
it**, so that whoever hits this gets a message they can act on.

The author's path (`/Users/amandahuang/Documents/6.4210-Project`) is clean. A
teammate's may not be — a clone under `~/Robotic Manipulation/` on any OS, or under
`/mnt/c/Users/First Last/` on WSL, will break. The fix is to put the venv in the home
directory instead:

```sh
uv venv --python 3.13 ~/venv4210
uv pip install --python ~/venv4210 -r requirements.txt
source ~/venv4210/bin/activate
```

### Smoke test

`scripts/smoke_test.py`, the thing each teammate runs to prove their setup works:

1. Import `pydrake` and `manipulation`; print both versions and the Python version.
2. Build the D2 scene.
3. Advance the simulator 0.1 s with `HEADLESS=1`.
4. Print the number of positions and velocities in the plant, and the ball's pose.
5. Exit 0 on success.

Wrap the first model load in the `try`/`except` that calls
`explain_model_download_error`, so a path problem reports itself.

---

## 4. Repo Layout and Scaffolding (D1)

```
6.4210-Project/
  planning/                 # this spec, assumptions.md, later specs
  tosscatch/                # the project package
    __init__.py
    constants.py            # every tunable number, one place
    scene.py                # the physics-only world (D2)
  scripts/
    smoke_test.py           # env verification
    check_limits.py         # D3
  reference/
    iiwa_code.py            # the pset skeleton, kept for reference
  utils/                    # unchanged, inherited from the course repo
  results/                  # gitignored: generated reports and data
  assets/                   # gitignored: SDFs written at run time
  requirements.txt
  README.md
  .gitignore
```

### `.gitignore`

```
__pycache__/
*.py[cod]
.venv
venv/
assets/
results/
.DS_Store
.ruff_cache/
.pytest_cache/
```

### Untrack the committed bytecode

Seven `.pyc` files are currently tracked. They must come out, and they will keep
reappearing in diffs until they do:

```sh
git rm -r --cached utils/__pycache__
git commit -m "Untrack bytecode; add .gitignore"
```

### What to do with `iiwa_code.py`

Move it, do not delete it:

```sh
mkdir reference
git mv iiwa_code.py reference/iiwa_code.py
```

Add a header comment saying it is the ps2 pick-and-place skeleton, kept for the patterns
we reuse (`inv_kin`, `arm_jacobian_columns`, `write_box_sdf`, the weld directives), and
that it does not run as-is because several module-level constants are missing.

### README

The README should get: a one-paragraph project description, the §3 setup commands, how
to run the smoke test, how to run with `HEADLESS=1`, and the repo layout above.

### GitHub workflow

The proposal's "all team members familiar with setup and pushing to GitHub" is satisfied
when **each of the three of us has pushed at least one commit**. Suggested convention,
to keep three people out of each other's way once Week 2 starts:

- Work on a branch named for the piece of work (`toss-opt`, `approach-a`, `estimator`).
- Open a PR into `main`; anyone other than the author merges it.
- `main` always runs: if `scripts/smoke_test.py` fails on `main`, that is the priority.

---

## 5. The Scene (D2)

`tosscatch/scene.py` defines the world and **nothing else**. No controller, no
trajectory, no gripper logic, no estimator. All three approaches in Weeks 2–3 load this
same scene, which is what makes their comparison a controlled one.

### Arm: iiwa14

The proposal specifies the **iiwa 14**, and that is what the scene uses:

```
package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf
```

welded `world` → `iiwa::iiwa_link_0`.

**Note the discrepancy:** `iiwa_code.py` loads `iiwa7_no_collision.sdf`, because the
pset it came from used the iiwa7. This matters for us, because the iiwa7 has higher
joint velocity limits than the iiwa14 and would therefore support a faster release.
Rather than assume, **D3 reports the limits for both arms**, so that if the iiwa14
cannot produce the flight time we need, we have the iiwa7 number in hand and can make
the switch as a documented decision with evidence behind it (and amend the proposal
text accordingly).

**Verify the URI resolves.** The exact filename under drake 1.45.0 is an acceptance
criterion of D2, not an assumption of this spec — the `drake_models` package downloads
on first load, and the path should be confirmed against what actually ships rather than
taken from here.

### Gripper: Schunk WSG 50

```
package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
```

welded to `iiwa::iiwa_link_7` at translation `[0, 0, 0.09]`, rotation
`!Rpy { deg: [90, 0, 90] }`. This is copied verbatim from `reference/iiwa_code.py`
(lines 57–68), where it is known to work, and verified against the installed
`manipulation` package.

The hydroelastic variant is the right one: the ball makes a curved-on-flat contact with
the finger pads, which point contact handles badly.

Useful constants already established in the reference code:

- `FINGER_OPEN = 0.107` m — just under the WSG 50's 110 mm maximum opening
- `GRIPPER_REACH = 0.108` m — how far the middle of the finger pads sits from the
  gripper body frame `G`, along the axis the fingers reach out on

### Ball: a generated sphere SDF

Follow the pattern of `write_box_sdf` in `reference/iiwa_code.py` (line 78): generate
the SDF at run time into `assets/`, rather than checking a model in. The sphere needs:

- Explicit mass and inertia (uniform sphere: `I = 2/5 · m · r²` on each diagonal)
- A collision geometry with `<drake:compliant_hydroelastic/>` and
  `<drake:hydroelastic_modulus>`, matching the gripper's hydroelastic contact
- Explicit `<drake:mu_static>` / `<drake:mu_dynamic>` — friction is what makes the catch
  hold rather than squirt out, so it should be a stated number, not a default
- A visual geometry in an obvious color
- Added via `default_free_body_pose`, so it is a free body the arm can throw

Unlike the boxes in the reference code, put the body frame at the **center** of the
sphere, not at the bottom — the center is what the estimator tracks and what ballistic
equations are written about.

### Ground

A welded box with its top surface at z = 0, following `table_directive` in
`reference/iiwa_code.py` (line 137). Its only job is to stop a dropped ball from
falling forever and to make failed catches visible and terminal.

### Plant configuration

```yaml
plant_config:
    time_step: 0.001
    contact_model: hydroelastic_with_fallback
    discrete_contact_approximation: sap
```

See §8 for why 1 ms, and §7 for what needs validating.

### Drivers

```yaml
model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {}
```

This matches the course code and gives us the same ports the psets use. **It may not
survive contact with the toss** — see §7, which documents why and what the fallbacks
are. Keep the driver choice in `constants.py` or behind a scene-builder argument so
that switching it later is a one-line change, not a refactor.

Ports this exposes, worth knowing before Week 2:

- In: `iiwa.position`, `wsg.position`, `wsg.force_limit`
- Out: `iiwa.position_measured`, `iiwa.velocity_estimated`, `iiwa.state_estimated`,
  `iiwa.torque_measured`, `iiwa.torque_external`, `wsg.state_measured`,
  `wsg.force_measured`

---

## 6. Limits Investigation (D3)

`scripts/check_limits.py` answers the proposal's Week 1 risk question with numbers.
It runs headless, prints a table, and writes `results/week1_limits.md`.

### 6.1 Joint limits, for iiwa14 *and* iiwa7

Straight out of the plant, for each of the 7 joints:

- Position limits — `plant.GetPositionLowerLimits()` / `GetPositionUpperLimits()`
- Velocity limits — `plant.GetVelocityUpperLimits()`
- Effort limits — per joint actuator

Build the plant for this with `MakeMultibodyPlant(LoadScenario(data=ROBOT))`, as
`inv_kin` does in `reference/iiwa_code.py` (line 294): arm and gripper only, no ball
and no ground, so the indices line up with the arm without slicing.

### 6.2 Maximum gripper speed at candidate release poses

The real question is not what the joints can do but what the **gripper** can do. For a
configuration `q` and a desired release direction `d̂`, the fastest the gripper can
travel along `d̂` is the solution of a small linear program:

```
maximize    s
subject to  J_G(q)[3:, arm_columns] · q̇ = s · d̂      (translational rows)
            |q̇ᵢ| ≤ v_max,ᵢ        for each joint i
```

Sample this over a set of candidate release configurations — IK solutions at a range of
release heights and arm extensions — and report the best, along with the `q` that
achieves it.

Two implementation notes:

- `J_G` comes from `plant.CalcJacobianSpatialVelocity(..., JacobianWrtVariable.kV, G,
  [0,0,0], W, W)` and is **6 × N**, angular rows first, translational rows second.
- `N` is larger than 7, because the ball and any other free bodies contribute velocity
  coordinates. Slice out the arm's seven columns with the
  `arm_jacobian_columns` pattern from `reference/iiwa_code.py` (line 354), which reads
  the velocity start indices of `iiwa_joint_1` through `iiwa_joint_7`.

### 6.3 Translate the speed into a toss

For a purely vertical toss with release and catch at the same height:

```
t_flight = 2·v_z / g           apex above release = v_z² / (2g)
```

The proposal's target window of **0.5–0.65 s** therefore requires:

| Flight time | Required v_z | Apex above release |
|---|---|---|
| 0.50 s | 2.45 m/s | 0.31 m |
| 0.65 s | 3.19 m/s | 0.52 m |

So the headline number D3 must produce is: **can the arm release at 2.45–3.19 m/s
vertically, at a pose it can also catch from?** The report should say yes or no in one
sentence, with the achieved number next to the required one.

If the answer is no, the options, in the order we should prefer them, are: pick a
release pose with a better Jacobian; shorten the target flight time; relax the joint
velocity limits in simulation (the proposal already permits this, and it must then be
recorded as a limitation in the paper); or switch to the iiwa7.

### 6.4 Reach map, for choosing the catch plane

The catch intercepts the ball where it crosses a horizontal **catch plane**. We have to
pick that plane's height, and it has to be a height the arm can actually reach across a
useful lateral area, because toss variability means the ball will not come straight back
down.

For a grid of candidate heights `z` and lateral offsets `(x, y)` within some radius of
the arm, run IK (reuse `inv_kin` from `reference/iiwa_code.py`, line 287) with the
gripper oriented for a catch, and record which succeed. Report the reachable area at
each candidate height, and recommend a catch plane height.

### 6.5 Gripper closing time

The gripper's own closing speed is the hardware half of the catch timing budget, and it
is not something we get to choose — it is a property of the WSG 50 model and the
`SchunkWsgPositionController`. Measure it:

- Command the fingers from `FINGER_OPEN` to a closed width around the ball diameter
- Record the time from the command to the fingers actually reaching that width
- Sweep `wsg.force_limit`, since grip force and closing speed trade off

Report the closing time. The total latency budget for the catch is this number **plus**
the artificial decision-to-action delay from §8, and Weeks 2–3 need both.

### 6.6 Tracking error at toss speed

Command the arm along a fast trajectory approaching the release speed from §6.3, with
and without the ball held, and record the gap between `iiwa.position` (commanded) and
`iiwa.position_measured`, and between the commanded and actual gripper velocity.

This is the measurement that decides the driver question in §7. Report the peak error
and whether it is small enough that a toss planned open-loop will release where we asked.

### 6.7 Output

Printed table plus `results/week1_limits.md`, containing at minimum:

1. The joint limit tables for both arms
2. Max gripper speed achieved and the configuration that achieves it
3. The yes/no flight-time verdict from §6.3
4. The recommended catch plane height and its reachable area
5. Measured gripper closing time vs. force limit
6. Peak tracking error at toss speed, with and without the ball

---

## 7. Drake-Specific Constraints Found While Writing This Spec

This section is the substance of the proposal's "initial exploration into constraints
and limitations." Each item below is grounded in the installed `manipulation/station.py`
or the pydrake docs, and each one has the potential to cost us a week if we discover it
in Week 3 instead.

### 7.1 The controller does not know the ball is there

`IiwaDriver` builds its controller plant from the iiwa **plus frozen welded children
only** (`manipulation/station.py`, `_ApplyDriverConfigSim`). The gripper is welded, so
it is included. The ball is a **free body**, so it is not.

That means during the toss — the one part of the cycle where the arm is accelerating
hard while holding a mass — the controller is doing inverse dynamics for a system that
is 60 g lighter than reality, at the end of a long lever arm. The resulting tracking
error grows with acceleration, which is precisely where the toss needs to be accurate.

This is not a bug to fix so much as a property to measure (§6.6) and then either
calibrate around or design away. Note that the real system has the same problem, so
modeling it is arguably more honest than not.

### 7.2 Position-only commanding adds velocity lag

In `control_mode: position_only`, commands pass through `SimIiwaDriver`, which
deliberately mimics the real iiwa's LCM command/status interface — it takes a commanded
position and derives the velocity command from it. For slow pick-and-place this is
invisible. For a 0.5 s toss where the release *velocity* is the thing we are trying to
control, it may not be.

If §6.6 shows the error is too large, the fallbacks, in increasing order of departure
from the real robot's interface:

1. `control_mode: position_and_torque` — adds a feedforward torque input alongside the
   position command.
2. `InverseDynamicsDriver` — exports an `iiwa.desired_state` input port carrying **both
   q and q̇**, i.e. true velocity feedforward. Simulation-only, and further from the real
   iiwa's interface, but it is the right tool if commanded velocity has to be exact.

Whichever we end up on must be recorded in `assumptions.md` and mentioned in the paper,
since it affects how transferable the result is.

### 7.3 The arm has no collision geometry

`iiwa14_no_collision.sdf` has visual geometry only. The ball passes straight through
the links, and the arm cannot self-collide.

This is fine, and probably desirable, for now: it keeps the simulation fast and avoids
spurious contacts during fast motion. It becomes wrong the moment we want a near-miss to
bounce off the forearm rather than through it, or want to trust that a planned catch
configuration is self-collision-free. If we need that, switch to
`iiwa14_primitive_collision.urdf` and expect the simulation to slow down.

Decide this consciously; do not let it be discovered by a video reviewer.

### 7.4 Headless running is mandatory for the evaluation sweeps

`utils/viz.py` reads `HEADLESS` from the environment:

```sh
HEADLESS=1 python scripts/check_limits.py
```

With it set there is no Meshcat server, no waiting for a browser, and no pacing to
wall-clock time. Week 4 runs ~100 tosses per approach per noise setting; at real-time
pacing that is hours of watching a browser tab. Every script we write from Week 1
onward should work headless, and `check_limits.py` is the first one to prove it.

### 7.5 The free-body ball changes every index in the plant

Adding the ball adds 7 positions and 6 velocities to the plant. Anything that indexes
into `q`, `v`, or a Jacobian by position must account for it. The
`arm_jacobian_columns` helper (`reference/iiwa_code.py`, line 354) exists for exactly
this reason, and the same care applies to `plant.GetPositions(context, model_instance)`
calls — always pass the model instance.

### 7.6 Model download happens once, on first load

The first `package://drake_models/...` load of a fresh machine downloads ~170 MB into
`~/.cache/drake`. It is slow, it needs network, and it fails opaquely on paths with
spaces (§3). Expect the smoke test to be slow the first time each teammate runs it, and
make sure the README says so, or someone will think it has hung.

---

## 8. Provisional Parameters

These are **starting values with reasons**, not decisions. Each is marked provisional
and is validated or revised during Weeks 1–2. They seed `planning/assumptions.md` (D4),
which is the document the final paper's methods section gets written from.

Put every one of these in `tosscatch/constants.py`, named, with the justification as a
comment. Numbers scattered through five files is how a controlled comparison stops
being controlled.

| Quantity | Proposed value | Basis | Status |
|---|---|---|---|
| Ball radius | 0.035 m | Tennis ball. Diameter 70 mm against the WSG 50's 107 mm open width leaves ~18 mm clearance per side — enough tolerance for an imperfect catch, small enough to grip | Provisional |
| Ball mass | 0.06 kg | Tennis ball. Light enough not to dominate the arm's dynamics, heavy enough that contact is well-conditioned | Provisional |
| Aerodynamic drag | Neglected | At 3 m/s with C_d ≈ 0.5: F ≈ 0.010 N against a weight of 0.589 N, under 2%, worth a few mm over a 0.6 s flight — below our measurement noise | Per proposal; justified above |
| Spin / Magnus | Neglected | Per proposal. Also means the release grasp should be as symmetric as we can make it, so we are not quietly imparting spin | Per proposal |
| Gravity | 9.81 m/s² | Drake default | Fixed |
| Plant time step | 1e-3 s | At 3.2 m/s the ball moves 3.2 mm per step, far under its 35 mm radius, so no tunneling through the fingers | Provisional — validate against 5e-4 |
| Contact model | `hydroelastic_with_fallback` | Curved-on-flat ball/pad contact is what hydroelastic is for; fallback covers anything without a hydroelastic representation | Provisional |
| Hydroelastic modulus | 1e6 Pa | Matches the value used for the boxes in the course code | Provisional |
| Friction (ball ↔ pads) | µ_s = µ_d = 1.0 | Rubber on rubber. This is what makes the catch hold rather than squirt out; tune it in Week 2 if catches fail by slipping | Provisional |
| Control / replan rate | 200 Hz (5 ms) | Matches the real iiwa's command rate, so Approach B's replanning budget is realistic | Provisional |
| Observation rate | 100 Hz | Depth-camera / mocap scale rate | Provisional |
| Observed position | Ball position, Gaussian, σ_p = 5 mm per axis | Depth-camera scale noise | Provisional |
| Observed velocity | Ball velocity, Gaussian, σ_v = 0.05 m/s per axis | Available to all three approaches; see "Observation model" below for why σ_v is a sensor spec and not derived from σ_p | Provisional |
| Position noise sweep | σ_p ∈ {0, 2, 5, 10, 20} mm | 0 is the no-noise control from the proposal's baseline goal; 20 mm is deliberately past where one-shot prediction should break | Provisional |
| Velocity noise sweep | σ_v ∈ {0, 0.02, 0.05, 0.1, 0.2} m/s | Same shape: 0 is the control, 0.2 m/s is ~7% of the release speed and should break the one-shot fit | Provisional |
| Observation mode | `POSITION_AND_VELOCITY` (default), `POSITION_ONLY` available | A flag in `constants.py`, not a code change; see below | Provisional |
| Gripper decision→action latency | 30 ms fixed delta | The proposal's "small, fixed delta." Applied **on top of** the WSG's own closing time measured in §6.5 | Provisional |
| Latency sweep | {0, 20, 50} ms | Brackets the plausible range for a real pipeline | Provisional |
| Target flight time | 0.5–0.65 s | Per proposal | Per proposal |
| Required release speed | 2.45–3.19 m/s vertical | Derived from the above; see §6.3 | Derived |
| Catch plane height | Set from the §6.4 reach map | — | **Open — D3 decides** |
| Release pose | Set from the §6.2 speed map | — | **Open — D3 decides** |
| Random seed policy | One seed per trial, derived from a trial index, logged | Identical tosses across approaches is what makes the comparison controlled — the proposal says "identical tosses" explicitly | Provisional |

### Observation model

The ball sensor reports **position and velocity**, both with Gaussian noise, at 100 Hz.
Velocity is included because Approach A needs it: a one-shot fit over an early window of
measurements has very little data to work with, and giving it a velocity channel is what
makes it a fair reference point for Approach B rather than a straw man.

Two things about this are worth getting right in the code.

**σ_v is a sensor specification, not something derived from σ_p.** It is tempting to
model the velocity reading as a finite difference of two position readings, but that
would be a bad sensor: at σ_p = 5 mm and 100 Hz, differencing gives

```
σ_v = σ_p · √2 / Δt = 0.005 · 1.414 / 0.01 ≈ 0.71 m/s
```

which is roughly a quarter of the entire release speed — unusable, and not what a real
vision or motion-capture system delivers, since those track across many frames rather
than differencing two. So σ_v is its own number (0.05 m/s provisionally), swept on its
own axis, and the two noise channels are sampled independently.

**Make the mode a flag, not a fork in the code.** `constants.py` carries an
`OBSERVATION_MODE` setting with `POSITION_AND_VELOCITY` as the default and
`POSITION_ONLY` as an alternative; the sensor system reads it and zeroes or omits the
velocity channel accordingly. All three approaches read whatever the sensor offers.

The reason to keep `POSITION_ONLY` working, even though it is not the default: the
project's claim is about *continual replanning* versus *one-shot prediction*. If
Approach A is handed a clean velocity reading, part of what it gains comes from the
sensor rather than from the planning strategy, and the gap we measure between A and B
narrows for a reason that has nothing to do with our hypothesis. Running the headline
comparison in both modes costs one extra sweep and lets the paper say which part of the
difference is strategy and which part is sensing. If the schedule only allows one, run
`POSITION_AND_VELOCITY` and state the caveat in the limitations section.

### Success criteria, to define now rather than in Week 3

The proposal reports two metrics. Both need a definition precise enough to code, and
agreeing on them now avoids re-running sweeps later:

- **Catch (primary).** Proposed: the ball is still between the fingers, with non-zero
  grip force, 0.5 s after the gripper finished closing. The hold duration is what
  distinguishes a catch from a deflection.
- **Contact (secondary).** Proposed: any contact at all between the ball geometry and
  either finger geometry at any point during the trial.
- **Consecutive cycles.** Proposed: the count of successive catches before the first
  failure, per the proposal's "consecutive successful cycles."

---

## 9. Open Questions Carried Into Week 2

These are answered by the end of Week 1, each by a deliverable:

1. **iiwa14 or iiwa7?** — answered by D3 §6.1–6.3. iiwa14 per the proposal unless its
   velocity limits cannot produce the required release speed.
2. **Do we need to relax joint velocity limits in simulation?** — answered by D3 §6.3.
   If yes, it is a stated limitation in the paper, not a quiet change.
3. **Which driver: `IiwaDriver` position-only, position-and-torque, or
   `InverseDynamicsDriver`?** — answered by D3 §6.6 and §7.2.
4. **What is the catch plane height?** — answered by D3 §6.4.
5. **How does the ball get into the gripper for cycle 1?** — not yet addressed anywhere.
   Options: start it already between the fingers with the gripper closed on it; or place
   it on the ground and pick it up once. The first is simpler and avoids spending Week 2
   on a pick-and-place that is not the project. **Needs a decision before the toss work
   starts.**
6. **Is 1 ms a small enough time step for the catch contact?** — validate against 5e-4
   when the first real catch exists.
7. **Do we run the headline A-vs-B comparison in both observation modes?** — the default
   is `POSITION_AND_VELOCITY`; `POSITION_ONLY` isolates how much of A's performance comes
   from the sensor rather than the strategy (§8, "Observation model"). Decide in Week 3
   when the sweep budget is known. **Needs a decision before the Week 4 runs.**
8. **Is σ_v = 0.05 m/s the right sensor spec?** — it is a stand-in for a vision or mocap
   system we are not modeling. Sanity-check it in Week 3 against how much the estimator
   actually relies on the velocity channel.

---

## 10. Week 1 Acceptance Checklist

- [ ] `uv` venv builds from scratch on a clean clone, per §3
- [ ] `requirements.txt` committed and pinned
- [ ] `.gitignore` committed; `utils/__pycache__/*.pyc` untracked
- [ ] `iiwa_code.py` moved to `reference/` with an explanatory header
- [ ] README has setup steps, smoke-test instructions, and the layout
- [ ] `scripts/smoke_test.py` passes **on all three teammates' machines**
- [ ] All three teammates have pushed at least one commit to `main`
- [ ] `tosscatch/scene.py` loads in Meshcat: arm standing, ball resting, no warnings
- [ ] iiwa14 model URI confirmed to resolve under the installed Drake
- [ ] `tosscatch/constants.py` holds every §8 value with its justification, including
      `OBSERVATION_MODE` and both noise sigmas
- [ ] `scripts/check_limits.py` runs headless and writes `results/week1_limits.md`
- [ ] The report answers §6.3 in one sentence: can we get a 0.5–0.65 s flight?
- [ ] `planning/assumptions.md` written, every row filled in
- [ ] §9 questions 1–4 answered; questions 5–8 have owners
