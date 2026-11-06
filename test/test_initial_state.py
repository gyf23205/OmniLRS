#!/usr/bin/env python3
"""
Host-runnable checks of the initial-state hooks a dataset episode uses to start mid-mission rather
than from a power-on. No Isaac Sim, no omni imports.

    python3 test/test_initial_state.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.estimation import models as M
from src.mission_specific.perseverance.estimation.ekf import NavEKF
from src.mission_specific.perseverance.subsystems.perseverance_subsystems_handler import (
    PerseveranceSubsystemsHandler,
)
from src.subsystems.robot_enums import ObcState

failures = []


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


print("\n=== battery state of charge ===")
handler = PerseveranceSubsystemsHandler()
handler.set_battery_state_of_charge(0.4)
model = handler._power_model
check("charge is the fraction of capacity", abs(model._battery_charge_wh - 0.4 * handler.BATTERY_CAPACITY_WH) < 1e-9)
check("voltage follows the curve (13.5 V at 20%, 15.0 V at 40%)", abs(model._battery_voltage() - 15.0) < 1e-9)
handler.set_battery_state_of_charge(1.7)
check("a fraction above 1 is clamped to full", model._battery_charge_wh == handler.BATTERY_CAPACITY_WH)

print("\n=== obc uptime on a settable clock ===")
now = [100.0]
handler = PerseveranceSubsystemsHandler()
handler.set_obc_clock(lambda: now[0])
handler.set_obc_state(ObcState.OFF)
handler.get_obc_status()
handler.set_obc_state(ObcState.IDLE)
handler.get_obc_status()
handler.set_obc_uptime(3600.0)
check("uptime starts at the set value", handler.get_obc_status()["uptime"] == 3600)
now[0] += 42.0
check("and counts the clock, not wall time", handler.get_obc_status()["uptime"] == 3642)
handler.set_obc_state(ObcState.MOTOR)
check("a non-OFF transition keeps it", handler.get_obc_status()["uptime"] == 3642)
handler.set_obc_state(ObcState.OFF)
check("an OFF transition restarts it", handler.get_obc_status()["uptime"] == 0)

print("\n=== thermal history ===")
handler = PerseveranceSubsystemsHandler()
sun = (1000.0, 0.0, 300.0)
handler.set_sun_position(sun)
thermal = handler._thermal_model
position = (0.0, 0.0, 0.0)
now_eq = thermal.equilibrium_temperatures(position, sun, 0.0)
before_eq = thermal.equilibrium_temperatures(position, sun, 180.0)
check("the equilibrium depends on the heading", abs(now_eq["+X"] - before_eq["+X"]) > 50.0)

handler.set_thermal_history(position, [(180.0, 0.0), (0.0, 0.0)])
check("just turned: still at the old equilibrium",
      all(abs(thermal._node_temps[f] - before_eq[f]) < 1e-9 for f in now_eq))
handler.set_thermal_history(position, [(180.0, 0.0), (0.0, 1e6)])
check("long ago: at the new equilibrium",
      all(abs(thermal._node_temps[f] - now_eq[f]) < 1e-6 for f in now_eq))

# The closed form against the model's own integration: settled at 180 deg, then 120 s at 90 deg and
# 300 s at 0 deg, stepped one second at a time.
segments = [(180.0, 0.0), (90.0, 120.0), (0.0, 300.0)]
handler.set_thermal_history(position, segments[:1])
for yaw, seconds in segments[1:]:
    for _ in range(int(seconds)):
        handler.get_thermal_status(position, yaw, 1.0)
stepped = dict(thermal._node_temps)
handler.set_thermal_history(position, segments)
worst = max(abs(thermal._node_temps[f] - stepped[f]) for f in stepped)
check(f"the closed form matches 420 one-second steps of the model (worst {worst:.3f} C)", worst < 0.2)

print("\n=== navigation filter start ===")
wheels = {"front_left": (0.4, 0.4), "front_right": (-0.4, 0.4), "rear_left": (0.4, -0.4), "rear_right": (-0.4, -0.4)}
ekf = NavEKF(wheels)
ekf.reset(heading=0.3, position=(12.5, -7.0), reacquisitions=17)
check("the position starts where it is told", ekf.x[M.PX] == 12.5 and ekf.x[M.PY] == -7.0)
check("the heading is unaffected", abs(ekf.x[M.PSI] - 0.3) < 1e-12)
check("the counter starts where it is told", ekf.reacquisitions == 17)
ekf.reset(heading=0.0)
check("the defaults are the old origin and zero",
      ekf.x[M.PX] == 0.0 and ekf.x[M.PY] == 0.0 and ekf.reacquisitions == 0)

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for label in failures:
        print(f"  - {label}")
    sys.exit(1)

print("all initial-state checks passed")
