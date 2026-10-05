__license__ = "BSD-3-Clause"
__status__ = "development"

"""
Monitor-only residuals and their statistics for the perseverance navigation EKF.

Monitor-only means evaluated against the EKF estimate but never fed back into it. Feeding motor
effort back would let the filter explain a slipping wheel (too little torque) as accelerometer bias,
and the residual would vanish while the fault persisted.

    DriveTorqueModel    expected drive torque per wheel; residual = excess torque in the direction of motion
    tracking_residual   commanded wheel rate the wheel failed to reach
    WindowStats         1 s mean / extreme of every normalized residual, for the 1 Hz downlink
    Cusum               two-sided persistence test on the window means: noise averages out, a fault does not

Pure numpy, no omni imports.
"""

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# ── drive torque model ─────────────────────────────────────────────────────────
TORQUE_FEATURES = ("specific_force", "rolling", "viscous", "inertia", "scrub")

# Below this joint rate (rad/s) the rolling-resistance term fades to zero instead of flipping sign.
ROLLING_RATE_SCALE = 0.05


def torque_features(specific_force: float, wheel_rate: float, wheel_accel: float, yaw_rate: float) -> np.ndarray:
    """
    Regressors for one wheel's drive torque, linear in the unknown coefficients:

        specific_force  a_f - b_a: acceleration plus the slope's gravity pull, which the wheels must supply
        rolling         smoothed sign of the wheel rate: rolling resistance opposes motion
        viscous         wheel rate: joint and soil damping
        inertia         wheel angular acceleration
        scrub           |yaw rate|: steered wheels scrub in a turn
    """
    return np.array([
        specific_force,
        math.tanh(wheel_rate / ROLLING_RATE_SCALE),
        wheel_rate,
        wheel_accel,
        abs(yaw_rate),
    ])


def fit_torque_model(features: np.ndarray, torques: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares coefficients for one wheel from nominal data, and the residual sigma."""
    coefficients, *_ = np.linalg.lstsq(features, torques, rcond=None)
    residuals = torques - features @ coefficients
    dof = max(len(torques) - features.shape[1], 1)
    return coefficients, float(math.sqrt(residuals @ residuals / dof))


class DriveTorqueModel:
    """Expected drive torque per wheel, from coefficients fitted on nominal episodes."""

    def __init__(self, coefficients: Mapping[str, Sequence[float]], sigma: Mapping[str, float]):
        self._coefficients = {name: np.asarray(c, dtype=float) for name, c in coefficients.items()}
        self._sigma = {name: float(s) for name, s in sigma.items()}

    @classmethod
    def from_config(cls, config: Mapping) -> Optional["DriveTorqueModel"]:
        """None until the calibration has written coefficients for every wheel."""
        coefficients = (config or {}).get("coefficients") or {}
        sigma = (config or {}).get("sigma") or {}
        if not coefficients or set(coefficients) != set(sigma):
            return None
        return cls(coefficients, sigma)

    @property
    def wheels(self) -> Tuple[str, ...]:
        return tuple(self._coefficients)

    def expected(self, name: str, features: np.ndarray) -> float:
        return float(self._coefficients[name] @ features)

    def residual(self, name: str, measured: float, features: np.ndarray, direction: float) -> float:
        """
        Normalized excess torque in the direction of motion. direction is the sign of the commanded
        wheel rate, so the meaning holds when driving backwards: slip (spins too easily) is negative,
        sink or a stuck joint (too much resistance) is positive.
        """
        excess = (measured - self.expected(name, features)) / self._sigma[name]
        return math.copysign(1.0, direction) * excess if direction != 0.0 else excess


def tracking_residual(commanded: float, measured: float, sigma: float) -> float:
    """
    Normalized shortfall of the wheel rate against the command, positive when the wheel turns slower
    than commanded in either direction. A velocity-controlled wheel that slips still tracks (about 0);
    one that is sunk, stuck or torque-limited cannot (positive).
    """
    direction = math.copysign(1.0, commanded) if commanded != 0.0 else 0.0
    return direction * (commanded - measured) / sigma if direction else -abs(measured) / sigma


# ── statistics ─────────────────────────────────────────────────────────────────
class WindowStats:
    """
    Per-key mean and extreme over a downlink window. The downlink is 1 Hz and the filter 30 Hz; a
    single sample per second would miss transients and carry the full per-sample noise.
    """

    def __init__(self):
        self._sum: Dict[str, float] = {}
        self._count: Dict[str, int] = {}
        self._extreme: Dict[str, float] = {}

    def add(self, key: str, value: float) -> None:
        if not math.isfinite(value):
            return
        self._sum[key] = self._sum.get(key, 0.0) + value
        self._count[key] = self._count.get(key, 0) + 1
        if abs(value) > abs(self._extreme.get(key, 0.0)):
            self._extreme[key] = value

    def add_many(self, prefix: str, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self.add(f"{prefix}.{name}", value)

    def flush(self) -> Dict[str, Dict[str, float]]:
        """{key: {"mean", "max", "n"}} for the window, then start a new one. "max" is the signed extreme."""
        out = {
            key: {"mean": self._sum[key] / self._count[key], "max": self._extreme.get(key, 0.0), "n": self._count[key]}
            for key in self._sum
        }
        self._sum.clear()
        self._count.clear()
        self._extreme.clear()
        return out


class Cusum:
    """
    Two-sided CUSUM on a standardized statistic (nominally N(0, 1)): flags a sustained mean shift of
    more than `drift` sigma. Fed once per window with the window mean divided by its own nominal
    sigma, so it tests "has this residual been off for a while", not "was one sample large".
    """

    def __init__(self, drift: float = 0.5, threshold: float = 10.0):
        self.drift = float(drift)
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self.high = 0.0
        self.low = 0.0

    def update(self, z: float) -> bool:
        if math.isfinite(z):
            self.high = max(0.0, self.high + z - self.drift)
            self.low = max(0.0, self.low - z - self.drift)
        return self.alarm

    @property
    def alarm(self) -> bool:
        return self.high > self.threshold or self.low > self.threshold


def cusum_mean_run_length(drift: float, threshold: float, shift: float = 0.0) -> float:
    """
    Siegmund's approximation of the average number of updates before a one-sided CUSUM alarms, for
    a true mean shift of `shift` sigma. shift = 0 gives the false-alarm interval. Used to pick the
    threshold for a target false-alarm rate.
    """
    delta = shift - drift
    b = threshold + 1.166
    if abs(delta) < 1e-9:
        return b * b
    return (math.exp(-2.0 * delta * b) + 2.0 * delta * b - 1.0) / (2.0 * delta * delta)


class CusumBank:
    """One Cusum per residual key, with the nominal sigma of each key's window mean."""

    def __init__(self, drift: float = 0.5, threshold: float = 10.0, window_sigma: Optional[Mapping[str, float]] = None,
                 samples_per_window: int = 30, thresholds: Optional[Mapping[str, float]] = None):
        self.drift = drift
        self.threshold = threshold
        # Per-residual thresholds, fitted so nominal driving never alarms. Residuals such as motor
        # effort carry slow terrain-dependent offsets that a white-noise threshold reads as faults.
        self._thresholds = dict(thresholds or {})
        # Nominal sigma of a window mean. Default assumes white, unit-variance samples; calibration
        # replaces it with the measured value, because real residuals are correlated in time.
        self._window_sigma = dict(window_sigma or {})
        self._default_sigma = 1.0 / math.sqrt(samples_per_window)
        self._cusums: Dict[str, Cusum] = {}

    def reset(self) -> None:
        for cusum in self._cusums.values():
            cusum.reset()

    def update(self, window_means: Mapping[str, float]) -> Dict[str, bool]:
        alarms = {}
        for key, mean in window_means.items():
            cusum = self._cusums.setdefault(key, Cusum(self.drift, self._thresholds.get(key, self.threshold)))
            alarms[key] = cusum.update(mean / self._window_sigma.get(key, self._default_sigma))
        return alarms

    def statistics(self) -> Dict[str, float]:
        """Signed CUSUM level per key: +high or -low, whichever is larger. Useful as a continuous feature."""
        return {key: (c.high if c.high >= c.low else -c.low) for key, c in self._cusums.items()}


def alarm_bitmask(alarms: Mapping[str, bool], order: Iterable[str]) -> int:
    """Pack alarms into an int, bit i for the i-th key of `order`, for a fixed-width downlink field."""
    mask = 0
    for bit, key in enumerate(order):
        if alarms.get(key, False):
            mask |= 1 << bit
    return mask
