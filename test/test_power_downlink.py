#!/usr/bin/env python3
"""
Check that the power telemetry downlink does not advance the battery a second time.

    python3 test/test_power_downlink.py

The launch scripts and the dataset runner advance the power model every physics step. The downlink
used to advance it again by the downlink interval, which drained the battery exactly twice as fast
as the logged net power implies. Drives the real PerseveranceSubsystemsHandler and power model.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# src.robots.robot pulls in omni for a type annotation the transmitter never calls.
_stub = types.ModuleType("src.robots.robot")
_stub.Robot = object
sys.modules["src.robots.robot"] = _stub

import yaml

from src.mission_specific.perseverance.subsystems.perseverance_subsystems_handler import PerseveranceSubsystemsHandler
from src.mission_specific.perseverance.tmtc.perseverance_transmitter import PerseveranceTransmitter
from src.subsystems.robot_enums import ObcState

TMTC_CFG = yaml.safe_load((ROOT / "cfg/controller/perseverance-controller.yaml").read_text())
PARAMS = TMTC_CFG["robots_settings"]["yamcs_tmtc"]["parameters"]

DT = 1.0 / 30.0
STEPS = 300                 # 10 s of simulation
POSITION = (10.0, 10.0, 0.3)

failures = []


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


class FakeRobot:
    def __init__(self, subsystems):
        self.subsystems = subsystems

    def get_imu_readings(self):
        return {}, {}, {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}


class FakeRobotGroup:
    def get_pose_of_base_link(self):
        return list(POSITION), [1.0, 0.0, 0.0, 0.0]


def charge_wh(subsystems):
    return subsystems._power_model._battery_charge_wh


def run(advance_power_model):
    """Step the model at the physics rate, as the launch loops do, and downlink once per second."""
    subsystems = PerseveranceSubsystemsHandler()
    subsystems.set_sun_position((1000.0, 0.0, 500.0))
    subsystems.set_obc_state(ObcState.MOTOR)
    sent = {}
    transmitter = PerseveranceTransmitter(
        lambda name, value: sent.__setitem__(name, value), None, FakeRobot(subsystems),
        FakeRobotGroup(), "perseverance", PARAMS, advance_power_model=advance_power_model,
    )
    start = charge_wh(subsystems)
    expected = 0.0
    for step in range(1, STEPS + 1):
        status = subsystems.get_power_status(POSITION, 0.0, DT, ObcState.MOTOR)
        expected += status["net_power"] * DT / 3600.0
        if step % 30 == 0:
            transmitter.transmit_power_info(1.0)
    return charge_wh(subsystems) - start, expected, sent


print("\n=== the downlink only reads the power model ===")
change, expected, sent = run(advance_power_model=False)
check(f"battery change {change:+.4f} Wh matches the net power integral {expected:+.4f} Wh",
      abs(change - expected) < 1e-6 and expected < 0)
check("power telemetry is still downlinked",
      PARAMS["battery_charge"] in sent and PARAMS["net_power"] in sent and len(sent[PARAMS["motor_current"]]) == 6)

print("\n=== advance_power_model=True, for a loop that never steps the model ===")
change, expected, _ = run(advance_power_model=True)
check(f"the downlink adds its interval on top: {change / expected:.2f}x the physics-rate integral",
      abs(change / expected - 2.0) < 0.05)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all power downlink checks passed")
