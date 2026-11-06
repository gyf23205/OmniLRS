#!/usr/bin/env python3
"""
Check of the joint setpoint downlink without Isaac Sim.

    python3 test/test_joint_setpoints.py

Drives the real PerseveranceDriveController against a fake rover and checks what
PerseveranceTransmitter.transmit_joint_setpoints sends: the mean setpoint over the downlink
interval, in the encoders' order and units, with every physics step counted exactly once.
"""

import math
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

from src.mission_specific.perseverance.control.ackermann_model import AckermannModel, RoverGeometry
from src.mission_specific.perseverance.control.drive_controller import PerseveranceDriveController
from src.mission_specific.perseverance.tmtc.perseverance_transmitter import PerseveranceTransmitter
from src.subsystems.device import HealthState, PowerState
from src.subsystems.robot_enums import GoNogoState

TMTC_CFG = yaml.safe_load((ROOT / "cfg/controller/perseverance-controller.yaml").read_text())
PARAMS = TMTC_CFG["robots_settings"]["yamcs_tmtc"]["parameters"]
MDB = (ROOT / "cfg/mdb/perseverance.xml").read_text()

failures = []


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b)) and len(a) == len(b)


class FakeSubsystems:
    def get_go_nogo_state(self):
        return GoNogoState.GO

    def get_device_power_state(self, device):
        return PowerState.ON

    def get_device_health_state(self, device):
        return HealthState.NOMINAL

    def set_obc_state(self, state):
        pass


class FakeRobot:
    def __init__(self):
        self.subsystems = FakeSubsystems()

    def set_steer_angles(self, angles):
        pass

    def set_wheel_velocities(self, velocities):
        pass


class FakeRobotGroup:
    """A rover that never moves, so a command never completes."""

    def get_pose_of_base_link(self):
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]


def build(steer_sign=1.0):
    geometry = RoverGeometry()
    drive = PerseveranceDriveController(
        FakeRobot(), FakeRobotGroup(), AckermannModel(geometry), {"steer_sign": steer_sign},
    )
    sent = {}
    transmitter = PerseveranceTransmitter(
        lambda name, value: sent.__setitem__(name, value),
        None, None, None, "perseverance", PARAMS, drive_controller=drive,
    )
    return geometry, drive, transmitter, sent


WHEEL = PARAMS["wheel_speed_setpoint"]
STEER = PARAMS["steer_angle_setpoint"]

print("\n=== wiring ===")
check("both parameters are declared in the mdb",
      'Parameter name="wheel_speed_setpoint"' in MDB and 'Parameter name="steer_angle_setpoint"' in MDB)
check("both parameters are observable paths", WHEEL.startswith("/Rover/") and "/faults/" not in WHEEL
      and STEER.startswith("/Rover/") and "/faults/" not in STEER)

print("\n=== before any command ===")
geometry, drive, transmitter, sent = build()
transmitter.transmit_joint_setpoints()
check("zero wheel speeds, one per wheel", sent[WHEEL] == [0.0] * 6)
check("zero steer angles, one per corner", sent[STEER] == [0.0] * 4)

print("\n=== a straight drive ===")
speed = 0.2
drive.command_straight(speed, 100.0)
for _ in range(30):
    drive.update()
transmitter.transmit_joint_setpoints()
expected = speed / geometry.wheel_radius
# The first update() samples the stop setpoint the command was issued under; the other 29 drive.
check("mean counts the step before the first drive setpoint",
      close(sent[WHEEL], [expected * 29 / 30] * 6))
check("wheels straight", close(sent[STEER], [0.0] * 4))

for _ in range(30):
    drive.update()
transmitter.transmit_joint_setpoints()
check("a new window does not carry the previous one", close(sent[WHEEL], [expected] * 6))

print("\n=== a command that ends mid-window ===")
for _ in range(10):
    drive.update()
drive.command_stop()
for _ in range(20):
    drive.update()
transmitter.transmit_joint_setpoints()
check("mean is a third of the drive speed", close(sent[WHEEL], [expected / 3] * 6))

print("\n=== an empty window ===")
transmitter.transmit_joint_setpoints()
check("falls back to the setpoint held now", sent[WHEEL] == [0.0] * 6)

print("\n=== manual control, mirrored steering ===")
geometry, drive, transmitter, sent = build(steer_sign=-1.0)
steer, speeds = AckermannModel(geometry).solve(0.2, 0.5)
for _ in range(10):
    active = drive.manual(0.2, 0.5)
    if not active:
        drive.update()
transmitter.transmit_joint_setpoints()
order = ("front_left", "front_right", "mid_left", "mid_right", "rear_left", "rear_right")
corners = ("front_left", "front_right", "rear_left", "rear_right")
# Step 1 samples the stop setpoint held before the first manual write.
check("wheel speeds in motor_effort order (FL, FR, ML, MR, RL, RR)",
      close(sent[WHEEL], [speeds[name] * 9 / 10 for name in order]))
check("steer angles in degrees, in the joint frame steer_encoder reads",
      close(sent[STEER], [-math.degrees(steer[name]) * 9 / 10 for name in corners]))

print("\n=== episode reset ===")
for _ in range(10):
    drive.manual(0.2, 0.5)
drive.reset_setpoint_window()
transmitter.transmit_joint_setpoints()
check("reset drops the summed steps; the held setpoint is sent",
      close(sent[WHEEL], [speeds[name] for name in order]))

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all setpoint checks passed")
