#!/usr/bin/env python3
"""
Host-runnable checks for the NavFilter adapter. No Isaac Sim, no omni imports.

    python3 test/test_nav_filter.py

A fake robot reports its sensors the way Robot does - imu in body axes (x left, y backward, z up) with
orientation in degrees, joint rates, efforts, steer angles in steer_joints order - while driving a
known arc. This checks the plumbing: axis mapping, window timing, snapshot shape, and that a stuck
wheel reaches the telemetry. The real axis conventions are confirmed by the calibration run.
"""

import math
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.control.ackermann_model import (
    ALL_WHEELS, AckermannModel, RoverGeometry,
)
from src.mission_specific.perseverance.estimation.nav_filter import RESIDUAL_KEYS, NavFilter
# The transmitter imports Robot only for a type hint; stub it so omni is not needed (as test_fault_tc_path does).
_stub = types.ModuleType("src.robots.robot")
_stub.Robot = object
sys.modules["src.robots.robot"] = _stub
from src.mission_specific.perseverance.tmtc.perseverance_transmitter import PerseveranceTransmitter

failures = []


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


GEOMETRY = RoverGeometry.from_config({
    "wheel_radius": 0.2625, "corner_half_track": 1.062, "mid_half_track": 1.184,
    "front_offset": 1.185, "rear_offset": -1.075,
}, scale=0.35)
ACKERMANN = AckermannModel(GEOMETRY)
DT = 1.0 / 30.0


class FakeDrive:
    def __init__(self):
        self.last_wheel_command = None


class FakeRobot:
    """Drives a steady left arc at 0.3 m/s; heading is the world angle of body -y."""

    _steer_joint_names = ["steer_joint_front_left", "steer_joint_front_right",
                          "steer_joint_rear_left", "steer_joint_rear_right"]

    def __init__(self, drive, heading=0.7, stuck=None):
        self.drive = drive
        self.heading = heading
        self.stuck = stuck
        self.rng = np.random.default_rng(0)
        self.v, self.omega = 0.3, 0.1
        self.steer, self.speeds = ACKERMANN.solve(self.v, self.omega / self.v)
        drive.last_wheel_command = dict(self.speeds)

    def step(self):
        self.heading += self.omega * DT

    def get_imu_readings(self):
        n = lambda s: float(self.rng.normal(0, s))
        # Body axes: x left, y backward. Steady arc: centripetal acceleration omega*v toward the left (+x).
        accel = {"ax": self.omega * self.v + n(0.05), "ay": n(0.05), "az": 1.62 + n(0.05)}
        gyro = {"gx": n(0.005), "gy": n(0.005), "gz": self.omega + n(0.005)}
        # Euler yaw of body +x (left) = heading of forward + 90 deg.
        orientation = {"roll": 0.0, "pitch": 0.0, "yaw": math.degrees(self.heading) + 90.0 + n(0.5)}
        return accel, gyro, orientation

    def get_wheel_joint_velocities(self, names):
        rates = [self.speeds[w] + float(self.rng.normal(0, 0.01 / GEOMETRY.wheel_radius)) for w in names]
        if self.stuck in names:
            rates[names.index(self.stuck)] *= 0.1
        return rates

    def get_wheel_joint_efforts(self, names):
        return [2.0] * len(names)

    def get_steer_angles(self):
        return [self.steer[n.replace("steer_joint_", "")] for n in self._steer_joint_names]


def run(seconds, stuck=None):
    drive = FakeDrive()
    robot = FakeRobot(drive, stuck=stuck)
    nav = NavFilter(robot, drive, GEOMETRY, {})
    snapshots = []
    for k in range(int(seconds / DT)):
        robot.step()
        nav.update(DT)
        if nav.telemetry() and (not snapshots or nav.telemetry() is not snapshots[-1]):
            snapshots.append(nav.telemetry())
    return robot, nav, snapshots


print("Healthy arc")
robot, nav, snapshots = run(20.0)
check(f"one snapshot per second of sim time ({len(snapshots)} in 20 s)", 19 <= len(snapshots) <= 20)
last = snapshots[-1]
expected_fields = {"pose", "motion", "bias", "imu_residual", "nis", "cusum_alarm",
                   "wheel_rolling_mean", "wheel_rolling_max", "wheel_sideslip_mean", "wheel_sideslip_max",
                   "wheel_tracking_mean", "wheel_tracking_max"}
check("snapshot has every field except the uncalibrated torque residual",
      expected_fields <= set(last) and "wheel_effort_mean" not in last)
yaw_err = (last["pose"]["yaw"] - math.degrees(robot.heading) + 180) % 360 - 180
check(f"heading from imu yaw with the -90 deg mounting offset: error {yaw_err:+.2f} deg", abs(yaw_err) < 1.0)
check(f"speed {last['motion']['speed']:.3f} m/s (true 0.300), yaw rate {last['motion']['yaw_rate']:.3f} (true 0.100)",
      abs(last["motion"]["speed"] - 0.3) < 0.01 and abs(last["motion"]["yaw_rate"] - 0.1) < 0.01)
worst = max(abs(v) for s in snapshots[3:] for v in s["wheel_rolling_mean"])
check(f"wheel rolling window means stay small (worst {worst:.2f} sigma)", worst < 1.5)
lateral = max(abs(s["imu_residual"]["lateral_force_mean"]) for s in snapshots[3:])
check(f"centripetal force of the arc is explained (worst lateral window mean {lateral:.2f} sigma)", lateral < 1.5)
check("no persistent-shift alarms", all(s["cusum_alarm"] == 0 for s in snapshots))

print("Front-right wheel at 10 % of its command")
robot, nav, snapshots = run(40.0, stuck="front_right")
last = snapshots[-1]
idx = ALL_WHEELS.index("front_right")
check(f"rolling innovation of front_right {last['wheel_rolling_mean'][idx]:+.1f} sigma",
      last["wheel_rolling_mean"][idx] < -10)
check(f"tracking shortfall of front_right {last['wheel_tracking_mean'][idx]:+.1f} sigma",
      last["wheel_tracking_mean"][idx] > 10)
bit = RESIDUAL_KEYS.index("wheel_rolling.front_right")
check("its CUSUM alarm bit is set", bool(last["cusum_alarm"] >> bit & 1))
others = [i for i in range(6) if i != idx]
check("other wheels' rolling alarms are clear",
      not any(last["cusum_alarm"] >> RESIDUAL_KEYS.index(f"wheel_rolling.{ALL_WHEELS[i]}") & 1 for i in others))

print("Downlink")
sent = {}
conf = {f"estimator_{f}": f"/Rover/estimator/{f}" for f in PerseveranceTransmitter.ESTIMATOR_FIELDS}
transmitter = PerseveranceTransmitter(lambda path, value: sent.__setitem__(path, value), None, None, None,
                                      "perseverance", conf, nav_filter=nav)
transmitter.transmit_estimator()
check("transmit_estimator sends the snapshot and skips the absent effort residual",
      "/Rover/estimator/pose" in sent and "/Rover/estimator/wheel_effort_mean" not in sent)
check("wheel arrays are six floats", len(sent["/Rover/estimator/wheel_rolling_mean"]) == 6)

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("all passed")
