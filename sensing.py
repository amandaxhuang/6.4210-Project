"""Seeing the ball: a noisy, rate-limited sensor and the one-shot ballistic fit.

BallSensor reports the ball at OBS_RATE with Gaussian noise.  BallisticFit fits
a free flight to its reports by least squares.  Gravity is known, so after
subtracting it the fit is linear in the release position and velocity, and each
axis is the same 2 x 2 problem:

    p_obs(t_i) - g t_i^2 / 2 = [1  t_i] [p0; v0]

This is the estimator feasibility.prediction_error() models, so its error
numbers apply here, as long as both use the same sample times.
"""

from dataclasses import dataclass

import numpy as np
from pydrake.all import RigidTransform, SpatialVelocity

from toss import G_ACCEL
from utils.dtsystems import Abstract, DTSystem

OBS_RATE = 100.0  # Hz
SIGMA_P = 0.005  # m, per axis
SIGMA_V = 0.05  # m/s, per axis; a sensor specification, not a difference of positions

POSITION_ONLY = "position_only"
POSITION_AND_VELOCITY = "position_and_velocity"

# Noiseless readings still need a weight in the fit, so sigmas are floored here.
SIGMA_FLOOR = 1e-6

GRAVITY = np.array([0.0, 0.0, -G_ACCEL])


@dataclass(frozen=True)
class Observation:
    """One sensor reading."""

    t: float  # simulation time of the sample
    p: np.ndarray  # noisy ball position, world frame
    v: np.ndarray | None  # noisy ball velocity, or None in POSITION_ONLY


@dataclass(frozen=True)
class Estimate:
    """A fitted free flight."""

    p0: np.ndarray  # fitted ball position at t_anchor
    v0: np.ndarray  # fitted ball velocity at t_anchor
    cov: np.ndarray  # (6, 6) covariance of (p0, v0)
    n_obs: int
    t_anchor: float  # simulation time that p0 and v0 refer to


class BallSensor(DTSystem):
    """The ball's position (and, by mode, velocity) at OBS_RATE, with noise.

    Noise is drawn once per sample, so reading the output twice between samples
    gives the same observation.  Both channels are drawn in every mode, so
    switching the velocity channel on or off leaves the position noise alone.

    Inputs are the station's body_poses and body_spatial_velocities.  The first
    sample is at t = 0.  ball_index is the ball body's index in the plant.
    """

    def __init__(self, ball_index, rng, sigma_p=SIGMA_P, sigma_v=SIGMA_V, mode=POSITION_ONLY):
        dt = 1 / OBS_RATE
        i = int(ball_index)

        def sample(previous, inputs):
            poses, velocities = inputs
            p_noise, v_noise = rng.normal(size=3), rng.normal(size=3)
            v = None
            if mode == POSITION_AND_VELOCITY:
                v = velocities[i].translational() + sigma_v * v_noise
            k = round(previous.t * OBS_RATE) + 1
            return Observation(k * dt, poses[i].translation() + sigma_p * p_noise, v)

        before_first = Observation(-dt, np.zeros(3), None)
        super().__init__(
            input_spec=[Abstract([RigidTransform()]),
                        Abstract([SpatialVelocity(np.zeros(3), np.zeros(3))])],
            state_spec=Abstract(before_first),
            output_spec=Abstract(before_first),
            next_state_fun=sample,
            output_fun=lambda observation, _: observation,
            dt=dt,
            input_port_name=["body_poses", "body_spatial_velocities"],
            output_port_name="observation",
            name="ball_sensor",
        )


class BallisticFit:
    """Batch least squares over every observation added so far.

    Observation times are measured from t_anchor, and the estimate is the state
    at t_anchor.  The ball leaves the fingers a few ms after the commanded
    release; the fit is unaffected as long as the anchor is the same throughout.
    """

    def __init__(self, t_anchor: float, sigma_p: float = SIGMA_P, sigma_v: float = SIGMA_V):
        self.t_anchor = t_anchor
        self.position_weight = 1 / max(sigma_p, SIGMA_FLOOR) ** 2
        self.velocity_weight = 1 / max(sigma_v, SIGMA_FLOOR) ** 2
        self.observations: list[Observation] = []

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)

    def estimate(self) -> Estimate | None:
        """The fit so far, or None if the observations can't determine it."""
        n00 = n01 = n11 = 0.0  # the normal matrix, the same for every axis
        r0, r1 = np.zeros(3), np.zeros(3)  # its right-hand side, per axis
        for o in self.observations:
            t = o.t - self.t_anchor
            w = self.position_weight
            y = o.p - 0.5 * GRAVITY * t**2
            n00, n01, n11 = n00 + w, n01 + w * t, n11 + w * t**2
            r0, r1 = r0 + w * y, r1 + w * t * y
            if o.v is not None:
                w = self.velocity_weight
                n11 += w
                r1 = r1 + w * (o.v - GRAVITY * t)

        det = n00 * n11 - n01**2
        if not self.observations or det <= 1e-12 * n00 * n11:
            return None
        i00, i01, i11 = n11 / det, -n01 / det, n00 / det
        identity = np.eye(3)
        cov = np.block([[i00 * identity, i01 * identity], [i01 * identity, i11 * identity]])
        return Estimate(i00 * r0 + i01 * r1, i01 * r0 + i11 * r1, cov,
                        len(self.observations), self.t_anchor)


class RecursiveFit:
    """Approach B's estimator: planning/approach-b-closed-loop-replanning.md, section 6."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RecursiveFit belongs to Approach B and isn't written yet")
