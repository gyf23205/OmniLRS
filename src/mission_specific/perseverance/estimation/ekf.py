__license__ = "BSD-3-Clause"
__status__ = "development"

"""
Navigation EKF for the perseverance rover.

Prediction runs on the IMU (models.propagate). Updates are scalar and sequential, one per sensor
channel, which is what lets every channel carry its own innovation and its own chi-square gate:

    6 x wheel rolling speed      encoder rate * wheel radius against the state's predicted wheel speed
    6 x wheel side slip          pseudo-measurement 0: a wheel does not slide along its axle
    1 x gyro z                   omega + gyro bias
    1 x heading                  IMU yaw with the mounting offset removed

Every innovation is reported whether or not it was used, because the innovations ARE the product:
a gated-out wheel is exactly the wheel a fault detector wants to hear about.

Gating rule for wheels. Wheels whose innovation fails the gate are left out of the update, so a
stuck wheel cannot drag the state - but at most max_gated_wheels of them, the worst first. When more
fail together the likelier explanation is that the state is wrong (the IMU-only prediction drifted,
or the filter just started), so the remaining wheels pull it back, and on the next step only the
genuinely bad wheel is still out. Accepting ALL wheels in that case instead would let a bad wheel
into a compromise state that keeps several wheels failing - a lock-in seen in testing. The decision
is only about the state update; the reported residuals are unaffected.

Wheels do not update heading, yaw rate or gyro bias (a "consider" update: their gain on those states
is zero). The gyro and the precise imu heading already fix them - the gyro bias is observable through
the heading - while wheel yaw is only good to a few percent (scrub). Letting the wheels in made every
wheel fault leak into the imu channels in the fault runs: a sunk or slipping rover pulled omega off
the gyro, the gyro bias estimate drifted 0.05-0.12 rad/s on a healthy gyro, and the heading residual
reached 10-20 sigma. Wheel residuals are still judged against the gyro's omega.

Reacquisition for the single-sensor channels (gyro, heading). A channel rejected continuously for
reacquire_s is either facing a wrong state (after a transient) or has genuinely shifted (a bias
fault). Rejecting it forever would freeze a wrong state in the first case - a lock-out seen in
testing. So the channel's own state (gyro bias, heading) is opened up by the size of the innovation
and the measurement is taken: a transient heals, and a real bias shows as an onset spike in the
innovation followed by a jump in the bias estimate. Each one is counted in `reacquisitions`.

Pure numpy, no omni imports.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.mission_specific.perseverance.estimation import models as M

# Chi-square, 1 degree of freedom, p = 0.999.
CHI2_1DOF_999 = 10.828


@dataclass
class MeasurementNoise:
    """One-sigma measurement noise, fitted from nominal data."""

    sigma_wheel: float = 0.01        # m/s, rolling speed from the encoder rate (terrain bumps, vibration)
    sigma_sideslip: float = 0.02     # m/s, normal scrub a healthy wheel shows along its axle
    sigma_gyro: float = 0.005        # rad/s
    sigma_heading: float = math.radians(0.5)
    # Relative uncertainty of the wheels' yaw geometry (wheel_yaw_scale). Scrub differs between point
    # turns and arcs, so in a turn a wheel's reading is less exact than on a straight: its variance
    # gets (sigma_yaw_scale * omega * lever)^2 added. Without it the wheels pull omega off the gyro in
    # every turn and the heading drifts (heading NIS ~80 in the calibration run).
    sigma_yaw_scale: float = 0.2


@dataclass
class Innovation:
    """One scalar innovation nu = z - h(x_prior), with its predicted variance S."""

    nu: float
    S: float
    used: bool

    @property
    def normalized(self) -> float:
        return self.nu / math.sqrt(self.S)

    @property
    def nis(self) -> float:
        return self.nu * self.nu / self.S


class NavEKF:
    def __init__(
        self,
        wheel_positions: Mapping[str, Tuple[float, float]],
        process_noise: Optional[M.ProcessNoise] = None,
        measurement_noise: Optional[MeasurementNoise] = None,
        gravity: float = 1.62,
        gate: float = CHI2_1DOF_999,
        max_gated_wheels: int = 2,
        reacquire_s: float = 3.0,
        wheel_yaw_scale: float = 1.0,
    ):
        self.wheel_names: Tuple[str, ...] = tuple(wheel_positions)
        self._positions = dict(wheel_positions)
        self.process_noise = process_noise or M.ProcessNoise()
        self.noise = measurement_noise or MeasurementNoise()
        self.gravity = float(gravity)
        self.gate = float(gate)
        self.max_gated_wheels = int(max_gated_wheels)
        self.reacquire_s = float(reacquire_s)
        self.wheel_yaw_scale = float(wheel_yaw_scale)

        self.x = np.zeros(M.STATE_DIM)
        self.P = np.eye(M.STATE_DIM)
        self.reset()

    def reset(self, heading: float = 0.0, heading_sigma: float = math.radians(2.0)) -> None:
        """Origin at the current position, heading from the IMU, at rest, biases unknown but small."""
        self._clock = 0.0
        self._rejected_since: Dict[str, float] = {}
        self.reacquisitions = 0
        self.x = np.zeros(M.STATE_DIM)
        self.x[M.PSI] = M.wrap_angle(heading)
        self.P = np.diag([
            1e-6, 1e-6,               # position: the origin is defined here
            heading_sigma ** 2,
            0.05 ** 2, 0.02 ** 2,     # v, v_lat: settled after the episode reset
            0.05 ** 2,                # omega
            0.02 ** 2,                # gyro bias
            0.05 ** 2,                # accel bias
        ])

    # ── prediction ────────────────────────────────────────────────────────────────
    def predict(self, imu: M.ImuInput, dt: float) -> None:
        self._clock += dt
        self.x, F = M.propagate(self.x, imu, dt, self.gravity)
        self.P = F @ self.P @ F.T + M.process_noise(self.process_noise, dt)
        self.P = 0.5 * (self.P + self.P.T)

    # ── generic scalar update ─────────────────────────────────────────────────────
    def innovation(self, z: float, h: float, H: np.ndarray, R: float, angle: bool = False) -> Innovation:
        """The innovation against the current (prior) state, without applying it."""
        nu = M.wrap_angle(z - h) if angle else z - h
        S = float(H @ self.P @ H) + R
        return Innovation(nu=float(nu), S=S, used=False)

    # States the wheels may not correct; see the module docstring.
    WHEEL_CONSIDER_STATES = (M.PSI, M.OMEGA, M.BG)

    def apply(self, innovation: Innovation, H: np.ndarray, R: float, frozen=()) -> None:
        """
        Kalman update with the Joseph form, which stays symmetric positive definite for any gain -
        including one with the `frozen` states' rows zeroed (a consider update).
        """
        K = (self.P @ H) / innovation.S
        for index in frozen:
            K[index] = 0.0
        self.x = self.x + K * innovation.nu
        self.x[M.PSI] = M.wrap_angle(self.x[M.PSI])
        I_KH = np.eye(M.STATE_DIM) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + R * np.outer(K, K)
        innovation.used = True

    def _update_channel(self, channel: str, z: float, model, R: float, own_state: int, angle: bool = False) -> Innovation:
        """Gated scalar update with reacquisition; see the module docstring."""
        h, H = model(self.x)
        inn = self.innovation(z, h, H, R, angle)
        if inn.nis <= self.gate:
            self._rejected_since.pop(channel, None)
            self.apply(inn, H, R)
            return inn

        since = self._rejected_since.setdefault(channel, self._clock)
        if self._clock - since >= self.reacquire_s:
            self.P[own_state, own_state] += inn.nu ** 2
            h, H = model(self.x)
            self.apply(self.innovation(z, h, H, R, angle), H, R)
            inn.used = True
            self._rejected_since.pop(channel)
            self.reacquisitions += 1
        return inn

    # ── sensor updates ────────────────────────────────────────────────────────────
    def update_wheels(
        self, rolling_speeds: Mapping[str, float], steer_angles: Mapping[str, float]
    ) -> Tuple[Dict[str, Innovation], Dict[str, Innovation]]:
        """
        rolling_speeds: wheel radius * measured joint rate, m/s, keyed by wheel name.
        steer_angles: measured steer angle, rad, for the steered corners; missing wheels are unsteered.
        Returns (rolling innovations, side-slip innovations), each keyed by wheel name.
        """
        rolling_models = {
            name: (lambda x, p=self._positions[name], d=steer_angles.get(name, 0.0):
                   M.wheel_rolling(x, p, d, self.wheel_yaw_scale))
            for name in self.wheel_names
        }
        sideslip_models = {
            name: (lambda x, p=self._positions[name], d=steer_angles.get(name, 0.0):
                   M.wheel_sideslip(x, p, d, self.wheel_yaw_scale))
            for name in self.wheel_names
        }
        rolling = self._update_group(rolling_speeds, rolling_models,
                                     self._wheel_variance(rolling_models, self.noise.sigma_wheel),
                                     max_excluded=self.max_gated_wheels)
        sideslip = self._update_group({name: 0.0 for name in self.wheel_names}, sideslip_models,
                                      self._wheel_variance(sideslip_models, self.noise.sigma_sideslip),
                                      max_excluded=len(self.wheel_names))
        return rolling, sideslip

    def _wheel_variance(self, models, sigma: float) -> Dict[str, float]:
        """Per-wheel R: sensor noise plus the yaw-geometry uncertainty at the current yaw rate."""
        variance = {}
        for name in self.wheel_names:
            _, H = models[name](self.x)
            yaw_part = self.noise.sigma_yaw_scale * H[M.OMEGA] * self.x[M.OMEGA]
            variance[name] = sigma ** 2 + yaw_part ** 2
        return variance

    def _update_group(self, measurements, models, R: Mapping[str, float], max_excluded: int) -> Dict[str, Innovation]:
        """
        Update with one scalar measurement per wheel.

        The reported innovations are all taken against the same state, before any of them is applied,
        so every wheel is judged against the same prediction and the order of the sequential update
        cannot make one wheel look better than another. Gating is decided on those same innovations.
        """
        reported = {}
        for name in self.wheel_names:
            h, H = models[name](self.x)
            reported[name] = self.innovation(measurements[name], h, H, R[name])
        failed = sorted((name for name, inn in reported.items() if inn.nis > self.gate),
                        key=lambda name: reported[name].nis, reverse=True)
        excluded = set(failed[:max_excluded])

        for name in self.wheel_names:
            if name in excluded:
                continue
            h, H = models[name](self.x)
            self.apply(self.innovation(measurements[name], h, H, R[name]), H, R[name],
                       frozen=self.WHEEL_CONSIDER_STATES)
            reported[name].used = True
        return reported

    def update_gyro(self, rate: float) -> Innovation:
        return self._update_channel("gyro", rate, M.gyro_z, self.noise.sigma_gyro ** 2, own_state=M.BG)

    def update_heading(self, heading: float) -> Innovation:
        return self._update_channel("heading", heading, M.heading, self.noise.sigma_heading ** 2,
                                    own_state=M.PSI, angle=True)

    # ── monitor-only ──────────────────────────────────────────────────────────────
    def lateral_force_residual(self, imu: M.ImuInput) -> float:
        """
        Sideways specific force the estimated motion does not explain, m/s^2. Monitor-only.

        With grip, the wheels supply exactly the force that cancels the cross-slope gravity and turns
        the rover, so a_l = g*sin(roll)*cos(pitch) + omega*v and this is zero-mean noise. When grip is
        lost the wheels can no longer supply it: on a cross-slope the rover slides downhill, in a turn
        it understeers, and the shortfall persists for as long as the slide does. Sliding at constant
        speed on flat ground needs no force, so it is invisible here - and physically does not happen.
        """
        c_th = math.cos(imu.pitch)
        expected = self.gravity * math.sin(imu.roll) * c_th + self.x[M.OMEGA] * self.x[M.V]
        return float(imu.a_l - expected)

    # ── readout ───────────────────────────────────────────────────────────────────
    def state(self) -> Dict[str, float]:
        return {name: float(value) for name, value in zip(M.STATE_NAMES, self.x)}

    def sigma(self) -> Dict[str, float]:
        return {name: float(math.sqrt(max(value, 0.0))) for name, value in zip(M.STATE_NAMES, np.diag(self.P))}
