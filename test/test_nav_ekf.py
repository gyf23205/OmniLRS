#!/usr/bin/env python3
"""
Host-runnable checks for the navigation EKF. No Isaac Sim, no omni imports.

    python3 test/test_nav_ekf.py

Runs the filter on synthetic drives whose sensor readings are generated from a known truth, then
checks the four things the filter is for: it is statistically consistent on a healthy rover (NIS
near its degrees of freedom), it learns a gyro bias, one bad wheel is flagged without corrupting the
state, and sideways drift shows up in the side-slip innovations.
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
from src.mission_specific.perseverance.estimation.ekf import MeasurementNoise, NavEKF

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
POSITIONS = GEOMETRY.wheel_positions()
DT = 1.0 / 30.0
GRAVITY = 1.62
NOISE = MeasurementNoise(sigma_yaw_scale=0.0)   # the generator has no scrub: wheel yaw is exact
SIGMA_ACCEL = 0.05


def smooth_step(t, t0, t1):
    """0 before t0, 1 after t1, a cosine ramp between."""
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * (t - t0) / (t1 - t0))


def profile(t):
    """Truth (v, omega): accelerate, drive straight, arc left, straight, arc right."""
    v = 0.3 * smooth_step(t, 1.0, 4.0)
    omega = 0.12 * (smooth_step(t, 15.0, 17.0) - smooth_step(t, 28.0, 30.0)) \
        - 0.1 * (smooth_step(t, 40.0, 42.0) - smooth_step(t, 50.0, 52.0))
    return v, omega


def simulate(duration=60.0, seed=1, gyro_bias=lambda t: 0.0, wheel_scale=lambda t, name: 1.0,
             v_lat_profile=lambda t: 0.0, accel_bias=0.0, roll=0.0):
    """Run the filter over a synthetic drive; return per-step logs."""
    rng = np.random.default_rng(seed)
    # Q matched to this generator: smooth steering on flat ground (the defaults are fitted to the
    # rougher simulator terrain and would overstate the uncertainty here).
    ekf = NavEKF(POSITIONS, M.ProcessNoise(sigma_accel=SIGMA_ACCEL, q_omega=3e-4, q_heading=1e-6), NOISE,
                 gravity=GRAVITY)
    heading0 = 0.4
    ekf.reset(heading=heading0)

    psi, px, py = heading0, 0.0, 0.0
    logs = {"t": [], "rolling": [], "sideslip": [], "gyro": [], "heading": [], "lateral": [],
            "state": [], "truth": []}
    steps = int(duration / DT)
    for k in range(steps):
        t = k * DT
        v, omega = profile(t)
        v_lat = v_lat_profile(t)
        v_next, _ = profile(t + DT)
        v_dot = (v_next - v) / DT
        v_lat_dot = (v_lat_profile(t + DT) - v_lat) / DT

        # Accelerometer: body acceleration in the rotating frame plus the cross-slope gravity it feels.
        imu = M.ImuInput(
            a_f=v_dot - omega * v_lat + accel_bias + rng.normal(0, SIGMA_ACCEL),
            a_l=v_lat_dot + omega * v + GRAVITY * math.sin(roll) + rng.normal(0, SIGMA_ACCEL),
            roll=roll, pitch=0.0,
        )
        ekf.predict(imu, DT)

        # Truth after this step
        px += (v * math.cos(psi) - v_lat * math.sin(psi)) * DT
        py += (v * math.sin(psi) + v_lat * math.cos(psi)) * DT
        psi = M.wrap_angle(psi + omega * DT)
        v, omega = v_next, profile(t + DT)[1]
        v_lat = v_lat_profile(t + DT)
        truth = np.zeros(M.STATE_DIM)
        truth[M.PX], truth[M.PY], truth[M.PSI] = px, py, psi
        truth[M.V], truth[M.VLAT], truth[M.OMEGA] = v, v_lat, omega

        # Wheels: the controller steers for the commanded arc (no slip assumed), the ground decides the rest.
        if abs(v) > 1e-3:
            steer, _ = ACKERMANN.solve(v, omega / v)
        else:
            steer, _ = ACKERMANN.stop()
        rolling = {
            name: M.wheel_rolling(truth, POSITIONS[name], steer.get(name, 0.0))[0] * wheel_scale(t, name)
            + rng.normal(0, NOISE.sigma_wheel)
            for name in ALL_WHEELS
        }
        # Same order as the adapter: gyro and heading pin omega and psi before the wheels are judged.
        gyro_inn = ekf.update_gyro(omega + gyro_bias(t) + rng.normal(0, NOISE.sigma_gyro))
        head_inn = ekf.update_heading(M.wrap_angle(psi + rng.normal(0, NOISE.sigma_heading)))
        roll_inn, slip_inn = ekf.update_wheels(rolling, steer)
        logs["lateral"].append(ekf.lateral_force_residual(imu))

        logs["t"].append(t)
        logs["rolling"].append(roll_inn)
        logs["sideslip"].append(slip_inn)
        logs["gyro"].append(gyro_inn)
        logs["heading"].append(head_inn)
        logs["state"].append(ekf.x.copy())
        logs["truth"].append(truth)
    logs["t"] = np.array(logs["t"])
    logs["lateral"] = np.array(logs["lateral"])
    logs["state"] = np.array(logs["state"])
    logs["truth"] = np.array(logs["truth"])
    return logs


def window(logs, t0, t1):
    return (logs["t"] >= t0) & (logs["t"] < t1)


def mean_nis(innovations, mask):
    return float(np.mean([inn.nis for inn, m in zip(innovations, mask) if m]))


def wheel_normalized(logs, name, mask, kind="rolling"):
    return np.array([step[name].normalized for step, m in zip(logs[kind], mask) if m])


print("Healthy rover: consistency and accuracy")
logs = simulate()
settled = window(logs, 5.0, 60.0)
for label, innovations in [("gyro", logs["gyro"]), ("heading", logs["heading"])]:
    nis = mean_nis(innovations, settled)
    check(f"{label} mean NIS {nis:.2f} is near 1", 0.5 < nis < 1.6)
wheel_nis = np.mean([step[n].nis for step, m in zip(logs["rolling"], settled) if m for n in ALL_WHEELS])
check(f"wheel rolling mean NIS {wheel_nis:.2f} is near 1", 0.5 < wheel_nis < 1.6)
error = logs["state"] - logs["truth"]
check(f"speed error rms {np.sqrt(np.mean(error[settled, M.V] ** 2)):.4f} m/s < 0.01",
      np.sqrt(np.mean(error[settled, M.V] ** 2)) < 0.01)
drift = math.hypot(error[-1, M.PX], error[-1, M.PY])
distance = np.sum(np.abs(logs["truth"][:, M.V])) * DT
check(f"position drift {drift:.3f} m over {distance:.1f} m (< 1%)", drift < 0.01 * distance)

print("Small gyro bias step at 30 s (inside the gate): learned")
logs = simulate(gyro_bias=lambda t: 0.02 if t >= 30.0 else 0.0)
estimated = logs["state"][-1, M.BG]
check(f"gyro bias estimated {estimated:.4f} rad/s (true 0.0200)", abs(estimated - 0.02) < 0.003)
onset = window(logs, 30.0, 31.0)
peak = max(abs(inn.normalized) for inn, m in zip(logs["gyro"], onset) if m)
check(f"gyro innovation at onset peaks at {peak:.1f} sigma", peak > 2.0)

print("Injector-sized gyro bias step at 30 s (0.15 rad/s, severity 0.3): onset spike, then reacquired as bias")
logs = simulate(gyro_bias=lambda t: 0.15 if t >= 30.0 else 0.0)
onset = window(logs, 30.0, 33.0)
spike = np.mean([abs(inn.normalized) for inn, m in zip(logs["gyro"], onset) if m])
check(f"gyro innovation during the first 3 s: {spike:.0f} sigma", spike > 10)
settled = window(logs, 40.0, 60.0)
after = np.mean([abs(inn.normalized) for inn, m in zip(logs["gyro"], settled) if m])
check(f"after reacquisition the gyro innovation is back to {after:.1f} sigma", after < 2)
check(f"gyro bias estimate jumped to {logs['state'][-1, M.BG]:.3f} rad/s (true 0.150)", abs(logs["state"][-1, M.BG] - 0.15) < 0.01)
w_err = np.sqrt(np.mean((logs["state"][settled, M.OMEGA] - logs["truth"][settled, M.OMEGA]) ** 2))
check(f"yaw rate unharmed: error rms {w_err:.4f} rad/s", w_err < 0.01)

print("Front-left wheel turns at 20 % of ground speed from 20 s (stuck / sunk)")
logs = simulate(wheel_scale=lambda t, name: 0.2 if (t >= 20.0 and name == "front_left") else 1.0)
faulted = window(logs, 22.0, 60.0)
bad = wheel_normalized(logs, "front_left", faulted)
check(f"front_left rolling innovation mean {bad.mean():.1f} sigma (strongly negative)", bad.mean() < -10.0)
others = max(abs(wheel_normalized(logs, n, faulted).mean()) for n in ALL_WHEELS if n != "front_left")
check(f"other wheels stay quiet (worst |mean| {others:.2f} sigma)", others < 1.0)
v_err = np.sqrt(np.mean((logs["state"][faulted, M.V] - logs["truth"][faulted, M.V]) ** 2))
check(f"state not dragged: speed error rms {v_err:.4f} m/s", v_err < 0.01)

print("Cross-slope slide: grip lost from 20 to 30 s on a 9 deg side slope, sliding downhill")
# Sliding with too little friction: the sideways speed keeps growing, 2 cm/s every second.
logs = simulate(roll=0.15, v_lat_profile=lambda t: -0.02 * min(max(t - 20.0, 0.0), 10.0) * (1.0 - smooth_step(t, 30.0, 31.0)))
sliding = window(logs, 21.0, 29.0)
baseline = window(logs, 5.0, 19.0)
slide_mean = logs["lateral"][sliding].mean()
base_mean = logs["lateral"][baseline].mean()
check(f"lateral force residual mean {slide_mean:+.4f} m/s^2 while sliding (true -0.020), "
      f"{base_mean:+.4f} before", abs(slide_mean + 0.02) < 0.004 and abs(base_mean) < 0.004)
# A gentle slide is ~2 sigma per 1 s window; the evidence accumulates, which is what the CUSUM is for.
slide_sigma = SIGMA_ACCEL / math.sqrt(sliding.sum())
check(f"over the 8 s slide the mean is {abs(slide_mean) / slide_sigma:.1f} sigma", abs(slide_mean) / slide_sigma > 5.0)

print("Steer stuck 0.3 rad off at front_left from 20 s")
def stuck_steer_logs():
    """Re-run with the front-left steer encoder reading a wrong angle, as a stuck corner would."""
    original = ACKERMANN.solve
    def solve(v, k):
        steer, speeds = original(v, k)
        steer = dict(steer)
        steer["front_left"] += 0.3
        return steer, speeds
    ACKERMANN.solve = solve
    try:
        return simulate()
    finally:
        ACKERMANN.solve = original
logs = stuck_steer_logs()
active = window(logs, 5.0, 60.0)
bad = wheel_normalized(logs, "front_left", active, "sideslip")
others = max(abs(wheel_normalized(logs, n, active, "sideslip").mean()) for n in ALL_WHEELS if n != "front_left")
check(f"front_left side-slip innovation mean {bad.mean():+.1f} sigma, others worst {others:.2f}", abs(bad.mean()) > 3.0 and others < 1.0)

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("all passed")
