#!/usr/bin/env python3
"""
Host-runnable checks for the navigation monitor-only residuals. No Isaac Sim, no omni imports.

    python3 test/test_nav_monitor.py

The torque model is fitted on synthetic nominal data with known coefficients, then must read slip
as negative and sink as positive excess torque, in both driving directions. The CUSUM must stay quiet
on pure noise and fire on a sustained shift.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.estimation import monitor as MON

failures = []


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


rng = np.random.default_rng(3)
TRUE = np.array([15.0, 0.8, 0.3, 0.05, 1.5])   # N*m per regressor
SIGMA = 0.2


def nominal_samples(n):
    """Varied driving: speeds both ways, slopes, accelerations, turns."""
    rows, torques = [], []
    for _ in range(n):
        f = MON.torque_features(
            specific_force=rng.normal(0, 0.3),
            wheel_rate=rng.uniform(-4, 4),
            wheel_accel=rng.normal(0, 2),
            yaw_rate=rng.normal(0, 0.15),
        )
        rows.append(f)
        torques.append(TRUE @ f + rng.normal(0, SIGMA))
    return np.array(rows), np.array(torques)


print("Torque model fit")
X, tau = nominal_samples(5000)
coefficients, sigma = MON.fit_torque_model(X, tau)
check(f"coefficients recovered {np.round(coefficients, 2).tolist()}", np.allclose(coefficients, TRUE, atol=0.1))
check(f"residual sigma {sigma:.3f} (true {SIGMA})", abs(sigma - SIGMA) < 0.02)

model = MON.DriveTorqueModel({"front_left": coefficients}, {"front_left": sigma})
X_test, tau_test = nominal_samples(2000)
nominal = np.array([model.residual("front_left", t, f, direction=math.copysign(1, f[2])) for f, t in zip(X_test, tau_test)])
check(f"held-out nominal residual mean {nominal.mean():+.3f}, std {nominal.std():.3f}", abs(nominal.mean()) < 0.1 and abs(nominal.std() - 1) < 0.1)

print("Excess torque in the direction of motion")
climbing = MON.torque_features(specific_force=0.25, wheel_rate=3.0, wheel_accel=0.0, yaw_rate=0.0)
expected = model.expected("front_left", climbing)
check("slip forward (wheel delivers 40 % of the torque) is negative",
      model.residual("front_left", 0.4 * expected, climbing, direction=+1.0) < -5)
check("sink forward (extra 3 N*m of resistance) is positive",
      model.residual("front_left", expected + 3.0, climbing, direction=+1.0) > 5)
reversing = MON.torque_features(specific_force=-0.25, wheel_rate=-3.0, wheel_accel=0.0, yaw_rate=0.0)
expected_rev = model.expected("front_left", reversing)
check("slip in reverse is still negative",
      model.residual("front_left", 0.4 * expected_rev, reversing, direction=-1.0) < -5)
check("sink in reverse is still positive",
      model.residual("front_left", expected_rev - 3.0, reversing, direction=-1.0) > 5)
check("uncalibrated config gives no model (residual omitted, not faked)", MON.DriveTorqueModel.from_config({}) is None)

print("Command tracking")
check("tracking wheel: ~0", abs(MON.tracking_residual(3.0, 2.98, 0.05)) < 1)
check("sunk wheel forward: positive", MON.tracking_residual(3.0, 1.0, 0.05) > 10)
check("sunk wheel reverse: positive", MON.tracking_residual(-3.0, -1.0, 0.05) > 10)
check("wheel overspinning is negative", MON.tracking_residual(3.0, 3.5, 0.05) < -5)

print("Window statistics")
stats = MON.WindowStats()
for v in [0.1, -2.0, 0.3, float("nan")]:
    stats.add("x", v)
out = stats.flush()
check("mean ignores NaN, max is the signed extreme", abs(out["x"]["mean"] - (-1.6 / 3)) < 1e-12 and out["x"]["max"] == -2.0 and out["x"]["n"] == 3)
check("flush starts a new window", stats.flush() == {})

print("CUSUM on 1 s window means")
false_alarm_s = MON.cusum_mean_run_length(0.5, 10.0) / 2          # two-sided: either side alarms
check(f"h = 10: one false alarm per {false_alarm_s / 3600:.1f} h per residual (nominal)", false_alarm_s > 3600 * 5)
bank = MON.CusumBank(drift=0.5, threshold=10.0, samples_per_window=30)
hours = 3
alarms = 0
for _ in range(hours * 3600):
    mean = rng.normal(0, 1, 30).mean()                           # 30 white N(0,1) samples per window
    alarms += bank.update({"r": mean})["r"]
check(f"{hours} h of pure noise: {alarms} alarm windows", alarms == 0)

bank = MON.CusumBank(drift=0.5, threshold=10.0, samples_per_window=30)
delay = None
for second in range(120):
    mean = (rng.normal(0, 1, 30) + 0.3).mean()                   # 0.3 sigma per sample = 1.6 sigma per window
    if bank.update({"r": mean})["r"]:
        delay = second + 1
        break
predicted = MON.cusum_mean_run_length(0.5, 10.0, shift=0.3 * math.sqrt(30))
check(f"0.3 sigma per-sample shift detected after {delay} s (predicted {predicted:.0f} s)", delay is not None and delay < 20)

check("bitmask packs alarms in order", MON.alarm_bitmask({"a": True, "c": True}, ["a", "b", "c"]) == 0b101)

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("all passed")
