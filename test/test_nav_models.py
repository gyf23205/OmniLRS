#!/usr/bin/env python3
"""
Host-runnable checks for the navigation EKF models. No Isaac Sim, no omni imports.

    python3 test/test_nav_models.py

The wheel models are the Ackermann solver inverted, so the strongest check is a round trip: every
wheel command AckermannModel produces must be predicted exactly by the measurement models, with zero
side slip. The Jacobians are checked against finite differences.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.control.ackermann_model import (
    ALL_WHEELS, AckermannModel, RoverGeometry,
)
from src.mission_specific.perseverance.estimation import models as M

failures = []


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


# The yaml geometry at the dataset scale: forward-positive offsets, as the controller uses them.
GEOMETRY = RoverGeometry.from_config({
    "wheel_radius": 0.2625, "corner_half_track": 1.062, "mid_half_track": 1.184,
    "front_offset": 1.185, "rear_offset": -1.075,
    "max_steer_angle_deg": 90.0, "max_wheel_speed": 1e9,   # unclamped, so the round trip is exact
}, scale=0.35)
MODEL = AckermannModel(GEOMETRY)
POSITIONS = GEOMETRY.wheel_positions()


def state(v=0.0, v_lat=0.0, omega=0.0, psi=0.3, b_g=0.0, b_a=0.0):
    x = np.zeros(M.STATE_DIM)
    x[M.PSI], x[M.V], x[M.VLAT], x[M.OMEGA], x[M.BG], x[M.BA] = psi, v, v_lat, omega, b_g, b_a
    return x


def round_trip_error(v, omega, steer, speeds):
    """Largest |predicted - commanded| over rolling speeds, and largest |side slip|."""
    x = state(v=v, omega=omega)
    worst_roll = worst_slip = 0.0
    for name in ALL_WHEELS:
        delta = steer.get(name, 0.0)
        rolling, _ = M.wheel_rolling(x, POSITIONS[name], delta)
        slip, _ = M.wheel_sideslip(x, POSITIONS[name], delta)
        worst_roll = max(worst_roll, abs(rolling - speeds[name] * GEOMETRY.wheel_radius))
        worst_slip = max(worst_slip, abs(slip))
    return worst_roll, worst_slip


print("Ackermann round trip")
for v, curvature in [(0.3, 0.0), (-0.2, 0.0), (0.3, 0.5), (0.3, -0.5), (0.15, 1.0), (-0.25, 0.8), (0.3, 2.5)]:
    steer, speeds = MODEL.solve(v, curvature)
    roll, slip = round_trip_error(v, v * curvature, steer, speeds)
    check(f"solve(v={v}, k={curvature}): rolling err {roll:.1e}, side slip {slip:.1e}", roll < 1e-9 and slip < 1e-9)

for yaw_rate in [math.radians(15.0), math.radians(-30.0)]:
    steer, speeds = MODEL.solve_point_turn(yaw_rate)
    roll, slip = round_trip_error(0.0, yaw_rate, steer, speeds)
    check(f"point turn {math.degrees(yaw_rate):+.0f} deg/s (mid wheels fold to counter-rotation): "
          f"rolling err {roll:.1e}, side slip {slip:.1e}", roll < 1e-9 and slip < 1e-9)

print("Side slip")
x = state(v=0.3, v_lat=0.05)
slip, _ = M.wheel_sideslip(x, POSITIONS["mid_left"], 0.0)
check("mid wheel side slip equals v_lat", abs(slip - 0.05) < 1e-12)
steer, speeds = MODEL.solve(0.3, 0.5)
_, slip = round_trip_error(0.3, 0.15, {**steer, "front_left": steer["front_left"] + 0.2}, speeds)
check("a steer angle that disagrees with the motion shows as side slip", slip > 0.01)

print("Jacobians vs finite differences")
rng = np.random.default_rng(0)
imu = M.ImuInput(a_f=0.12, a_l=-0.07, roll=0.1, pitch=-0.15, pitch_rate=0.05)
eps = 1e-6
for trial in range(5):
    x0 = rng.normal(size=M.STATE_DIM) * np.array([3, 3, 1, 0.3, 0.05, 0.3, 0.01, 0.05])
    _, F = M.propagate(x0, imu, 0.0333, 1.62)
    F_num = np.zeros_like(F)
    for j in range(M.STATE_DIM):
        dx = np.zeros(M.STATE_DIM)
        dx[j] = eps
        plus, _ = M.propagate(x0 + dx, imu, 0.0333, 1.62)
        minus, _ = M.propagate(x0 - dx, imu, 0.0333, 1.62)
        diff = plus - minus
        diff[M.PSI] = M.wrap_angle(diff[M.PSI])
        F_num[:, j] = diff / (2 * eps)
    check(f"process F, trial {trial}: max err {np.abs(F - F_num).max():.1e}", np.abs(F - F_num).max() < 1e-7)

x0 = state(v=0.25, v_lat=0.02, omega=0.1, b_g=0.01)
for label, fn in [
    ("wheel_rolling", lambda x: M.wheel_rolling(x, POSITIONS["front_left"], 0.4)),
    ("wheel_sideslip", lambda x: M.wheel_sideslip(x, POSITIONS["rear_right"], -0.3)),
    ("gyro_z", M.gyro_z),
    ("heading", M.heading),
]:
    _, H = fn(x0)
    H_num = np.array([(fn(x0 + e * eps)[0] - fn(x0 - e * eps)[0]) / (2 * eps) for e in np.eye(M.STATE_DIM)])
    check(f"{label} H: max err {np.abs(H - H_num).max():.1e}", np.abs(H - H_num).max() < 1e-7)

print("Process model physics")
x_parked = state(psi=0.0)
tilted = M.ImuInput(a_f=1.62 * math.sin(0.2), a_l=0.0, roll=0.0, pitch=0.2)
x_next, _ = M.propagate(x_parked, tilted, 1.0, 1.62)
check("parked nose-up on a slope: accelerometer gravity is removed, speed stays 0", abs(x_next[M.V]) < 1e-12)
x_turn = state(v=0.3, omega=0.2, psi=0.0)
steady = M.ImuInput(a_f=0.0, a_l=0.3 * 0.2, roll=0.0, pitch=0.0)
x_next, _ = M.propagate(x_turn, steady, 0.1, 1.62)
check("steady turn: centripetal acceleration does not create side slip", abs(x_next[M.VLAT]) < 1e-12)
check("steady turn: heading advances by omega*dt", abs(x_next[M.PSI] - 0.02) < 1e-12)

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("all passed")
