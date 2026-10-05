__license__ = "BSD-3-Clause"
__status__ = "development"

"""
Onboard navigation filter: reads the rover's own sensors every physics step, runs the EKF and the
monitor-only residuals, and publishes one telemetry snapshot per window (1 s of sim time).

Only what the rover could measure goes in: IMU, drive joint rates and efforts, steer angles, and the
wheel rates the drive controller commanded. Ground truth is touched in exactly one place, the
optional calibration dump, which records it BESIDE the sensor data for offline fitting and never
feeds it to the filter.

Update order within a step, and why:
    1. predict on the IMU
    2. gyro, heading           pin omega and psi first, so the wheels are judged against a good state
    3. 6 wheel rolling speeds  (chi-square gated per wheel)
    4. 6 wheel side slips      (pseudo-measurement 0)
    5. monitor-only residuals  lateral force, drive torque, command tracking: evaluated, not applied

Windows close on sim time inside update(), so the snapshot does not depend on when, or whether, the
downlink asks for it. telemetry() only returns the last closed window.

No omni imports: the robot is duck-typed, like the dataset modules.
"""

import math
import os
from typing import Dict, List, Mapping, Optional

import numpy as np

from src.mission_specific.perseverance.control.ackermann_model import (
    ALL_WHEELS, STEERED_WHEELS, RoverGeometry,
)
from src.mission_specific.perseverance.estimation import models as M
from src.mission_specific.perseverance.estimation import monitor as MON
from src.mission_specific.perseverance.estimation.ekf import MeasurementNoise, NavEKF

# Fixed residual order for the cusum_alarm bitmask. Appending is safe; reordering changes the meaning
# of archived values.
RESIDUAL_KEYS = (
    [f"wheel_rolling.{w}" for w in ALL_WHEELS]
    + [f"wheel_sideslip.{w}" for w in ALL_WHEELS]
    + [f"wheel_effort.{w}" for w in ALL_WHEELS]
    + [f"wheel_tracking.{w}" for w in ALL_WHEELS]
    + ["gyro", "heading", "lateral_force"]
)

# How the imu is mounted: the rover's forward and left axes as unit vectors in the imu frame. The imu
# prim is aligned with the body link (x = left, y = backward, z = up; the rover drives along body -y, see
# forward_axis_sign in the robot config). Everything else is derived from these two vectors, so there
# is no per-channel mapping to get wrong: specific forces are projections of the accelerometer
# vector, yaw rate is the gyro about forward x left, and pitch, roll and heading come from the full
# orientation. Confirmed by the calibration fit (scripts/fit_nav_estimator.py).
DEFAULT_IMU_MOUNT = {"forward": (0.0, -1.0, 0.0), "left": (1.0, 0.0, 0.0)}


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def imu_orientation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    The imu's orientation in the world, rebuilt from Robot.get_imu_readings()'s angles (radians).

    Those angles are NOT body roll and pitch. robot.py takes scipy's as_euler('zyx') - lowercase is
    an extrinsic sequence about the fixed world axes - and reports roll = -a_x, pitch = -a_y,
    yaw = a_z. So its "roll" and "pitch" are tilts about the world x and y axes and mix together as
    the rover turns (found in the calibration run: they correlated 0.07 / 0.10 with the true body
    pitch / roll). Extrinsic z-y-x composes as Rx(a_x) Ry(a_y) Rz(a_z).
    """
    return _rot_x(-roll) @ _rot_y(-pitch) @ _rot_z(yaw)


class NavFilter:
    def __init__(
        self,
        robot,
        drive_controller,
        geometry: RoverGeometry,
        config: Optional[Mapping] = None,
        steer_sign: float = 1.0,
        gravity: float = 1.62,
        truth_fn=None,
    ):
        """
        robot: Robot (get_imu_readings, get_wheel_joint_velocities, get_wheel_joint_efforts, get_steer_angles).
        drive_controller: exposes last_wheel_command, or None to skip the tracking residual.
        truth_fn: () -> (position, orientation_wxyz), used ONLY by the calibration dump.
        """
        config = dict(config or {})
        self._robot = robot
        self._drive = drive_controller
        self._geometry = geometry
        self._steer_sign = float(steer_sign)
        self._truth_fn = truth_fn

        imu = dict(config.get("imu") or {})
        self._forward = np.asarray(imu.get("forward", DEFAULT_IMU_MOUNT["forward"]), dtype=float)
        self._left = np.asarray(imu.get("left", DEFAULT_IMU_MOUNT["left"]), dtype=float)
        self._up = np.cross(self._forward, self._left)

        self.ekf = NavEKF(
            geometry.wheel_positions(),
            M.ProcessNoise(**(config.get("process_noise") or {})),
            MeasurementNoise(**(config.get("measurement_noise") or {})),
            gravity=gravity,
            gate=float(config.get("gate", 10.828)),
            max_gated_wheels=int(config.get("max_gated_wheels", 2)),
            reacquire_s=float(config.get("reacquire_s", 3.0)),
            wheel_yaw_scale=float(config.get("wheel_yaw_scale", 1.0)),
        )
        self._window_s = float(config.get("window_s", 1.0))
        self._tracking_sigma = float(config.get("tracking_sigma", 0.2))       # rad/s at the drive joint
        self._accel_filter = float(config.get("wheel_accel_filter", 0.2))     # EMA weight on d(rate)/dt
        self._torque = MON.DriveTorqueModel.from_config(config.get("torque_model"))
        cusum = dict(config.get("cusum") or {})
        self._cusum = MON.CusumBank(
            drift=float(cusum.get("drift", 0.5)),
            threshold=float(cusum.get("threshold", 10.0)),
            window_sigma=cusum.get("window_sigma"),
            thresholds=cusum.get("thresholds"),
            samples_per_window=max(int(round(self._window_s * 30)), 1),
        )
        self._dump_path = config.get("calibration_dump")
        self._window = MON.WindowStats()
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """New episode or teleport: local origin here, heading from the IMU, statistics cleared."""
        self._time = 0.0
        self._window_start = 0.0
        self._window.flush()
        self._cusum.reset()
        self._snapshot: Dict = {}
        self._prev_rates: Optional[Dict[str, float]] = None
        self._wheel_accel = {name: 0.0 for name in ALL_WHEELS}
        self._initialized = False
        self._dump_rows: List[Dict] = []

    def _read_imu(self):
        accel, gyro, orientation = self._robot.get_imu_readings()
        channels = {**accel, **gyro, **{k: math.radians(v) for k, v in orientation.items()}}
        a = np.array([channels["ax"], channels["ay"], channels["az"]])
        w = np.array([channels["gx"], channels["gy"], channels["gz"]])

        rotation = imu_orientation_matrix(channels["roll"], channels["pitch"], channels["yaw"])
        forward_world = rotation @ self._forward
        left_world = rotation @ self._left
        imu = M.ImuInput(
            a_f=float(a @ self._forward),
            a_l=float(a @ self._left),
            roll=math.asin(max(-1.0, min(1.0, left_world[2]))),       # left side up positive
            pitch=math.asin(max(-1.0, min(1.0, forward_world[2]))),   # nose up positive
            pitch_rate=float(w @ self._left),
        )
        heading = math.atan2(forward_world[1], forward_world[0])
        return imu, float(w @ self._up), heading, channels

    def _read_steer(self) -> Dict[str, float]:
        angles = self._robot.get_steer_angles()
        names = [n.replace("steer_joint_", "") for n in getattr(self._robot, "_steer_joint_names", [])] or list(STEERED_WHEELS)
        return {name: float(angle) / self._steer_sign for name, angle in zip(names, angles)}

    # ── per step ─────────────────────────────────────────────────────────────────
    def update(self, dt: float) -> None:
        imu, yaw_rate, heading, raw_imu = self._read_imu()
        rates = dict(zip(ALL_WHEELS, (float(r) for r in self._robot.get_wheel_joint_velocities(list(ALL_WHEELS)))))
        efforts = dict(zip(ALL_WHEELS, (float(e) for e in self._robot.get_wheel_joint_efforts(list(ALL_WHEELS)))))
        steer = self._read_steer()
        commanded = getattr(self._drive, "last_wheel_command", None) if self._drive is not None else None

        if not self._initialized:
            self.ekf.reset(heading=heading)
            self._initialized = True
        else:
            self.ekf.predict(imu, dt)
        self._time += dt

        # Wheel angular acceleration for the torque model's inertia term, lightly smoothed.
        if self._prev_rates is not None and dt > 0.0:
            for name in ALL_WHEELS:
                raw = (rates[name] - self._prev_rates[name]) / dt
                self._wheel_accel[name] += self._accel_filter * (raw - self._wheel_accel[name])
        self._prev_rates = rates

        gyro = self.ekf.update_gyro(yaw_rate)
        head = self.ekf.update_heading(heading)
        rolling_speeds = {name: rates[name] * self._geometry.wheel_radius for name in ALL_WHEELS}
        rolling, sideslip = self.ekf.update_wheels(rolling_speeds, steer)

        w = self._window
        w.add_many("wheel_rolling", {n: inn.normalized for n, inn in rolling.items()})
        w.add_many("wheel_sideslip", {n: inn.normalized for n, inn in sideslip.items()})
        w.add("gyro", gyro.normalized)
        w.add("heading", head.normalized)
        w.add("lateral_force", self.ekf.lateral_force_residual(imu) / self.ekf.process_noise.sigma_accel)
        w.add("nis_wheels", float(np.mean([inn.nis for inn in rolling.values()])))
        w.add("nis_gyro", gyro.nis)
        w.add("nis_heading", head.nis)

        state = self.ekf.x
        if self._torque is not None:
            specific_force = imu.a_f - state[M.BA]
            for name in self._torque.wheels:
                features = MON.torque_features(specific_force, rates[name], self._wheel_accel[name], state[M.OMEGA])
                direction = (commanded or {}).get(name, rates[name])
                w.add(f"wheel_effort.{name}", self._torque.residual(name, efforts[name], features, direction))
        if commanded:
            for name in ALL_WHEELS:
                if name in commanded:
                    w.add(f"wheel_tracking.{name}", MON.tracking_residual(commanded[name], rates[name], self._tracking_sigma))

        if self._dump_path:
            self._record_dump(dt, raw_imu, rates, efforts, steer, commanded)

        if self._time - self._window_start >= self._window_s - 1e-9:
            self._close_window()

    def _close_window(self) -> None:
        stats = self._window.flush()
        means = {key: s["mean"] for key, s in stats.items() if key in RESIDUAL_KEYS}
        alarms = self._cusum.update(means)
        x = self.ekf.x

        def per_wheel(prefix, field):
            values = [stats.get(f"{prefix}.{name}", {}).get(field) for name in ALL_WHEELS]
            return None if any(v is None for v in values) else [float(v) for v in values]

        def scalar(key, field="mean"):
            return float(stats[key][field]) if key in stats else float("nan")

        snapshot = {
            "pose": {"x": float(x[M.PX]), "y": float(x[M.PY]), "yaw": math.degrees(x[M.PSI])},
            "motion": {"speed": float(x[M.V]), "lateral_speed": float(x[M.VLAT]), "yaw_rate": float(x[M.OMEGA])},
            "bias": {"gyro": float(x[M.BG]), "accel": float(x[M.BA])},
            "imu_residual": {
                "gyro_mean": scalar("gyro"), "gyro_max": scalar("gyro", "max"),
                "heading_mean": scalar("heading"), "heading_max": scalar("heading", "max"),
                "lateral_force_mean": scalar("lateral_force"), "lateral_force_max": scalar("lateral_force", "max"),
            },
            "nis": {"wheels": scalar("nis_wheels"), "gyro": scalar("nis_gyro"), "heading": scalar("nis_heading")},
            "cusum_alarm": MON.alarm_bitmask(alarms, RESIDUAL_KEYS),
            "reacquisitions": int(self.ekf.reacquisitions),
        }
        for prefix in ("wheel_rolling", "wheel_sideslip", "wheel_effort", "wheel_tracking"):
            for field in ("mean", "max"):
                values = per_wheel(prefix, field)
                if values is not None:
                    snapshot[f"{prefix}_{field}"] = values
        self._snapshot = snapshot
        self._window_start = self._time

    def telemetry(self) -> Dict:
        """The last closed window. Empty until the first window of the episode closes."""
        return self._snapshot

    # ── calibration dump ─────────────────────────────────────────────────────────
    def _record_dump(self, dt, raw_imu, rates, efforts, steer, commanded) -> None:
        row = {"t": self._time, "dt": dt}
        row.update({f"imu.{k}": v for k, v in raw_imu.items()})
        row.update({f"rate.{k}": v for k, v in rates.items()})
        row.update({f"effort.{k}": v for k, v in efforts.items()})
        row.update({f"steer.{k}": v for k, v in steer.items()})
        row.update({f"cmd.{k}": v for k, v in (commanded or {}).items()})
        row.update({f"ekf.{k}": v for k, v in self.ekf.state().items()})
        if self._truth_fn is not None:
            position, orientation = self._truth_fn()
            row.update({f"truth.p{i}": float(v) for i, v in enumerate(position)})
            row.update({f"truth.q{i}": float(v) for i, v in enumerate(orientation)})
        self._dump_rows.append(row)

    def save_dump(self, tag: str) -> Optional[str]:
        """Write this episode's calibration rows to <calibration_dump>/<tag>.npz. Returns the path."""
        if not self._dump_path or not self._dump_rows:
            return None
        os.makedirs(self._dump_path, exist_ok=True)
        keys = sorted({k for row in self._dump_rows for k in row})
        arrays = {k: np.array([row.get(k, np.nan) for row in self._dump_rows], dtype=float) for k in keys}
        path = os.path.join(self._dump_path, f"{tag}.npz")
        np.savez_compressed(path, **arrays)
        self._dump_rows = []
        return path
