#!/usr/bin/env python3
"""
End-to-end check of the fault path without Isaac Sim.

    python3 test/test_fault_tc_path.py

Encodes each telecommand exactly as Yamcs would per the generated MDB, decodes it with the rover's
own MdbParsingService, dispatches it the way CommandsHandler does, and confirms the right thing was
broken. Then runs the downlink side and checks what reaches the ground.

This is the check that caught both MDB encoding bugs: arguments silently dropped because a command
container held only its name, and enum arguments with no encoding declared.
"""

import math
import struct
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# src.robots.robot pulls in omni for a type annotation the transmitter never calls. Stub it so the
# TM/TC layer can be exercised on the host.
_stub = types.ModuleType("src.robots.robot")
_stub.Robot = object
sys.modules["src.robots.robot"] = _stub

import yaml

from src.mission_specific.perseverance.faults.fault_injector import (
    CORNERS, UNLIMITED, WHEELS, FaultInjector,
)
from src.mission_specific.perseverance.tmtc.perseverance_commander import PerseveranceCommander
from src.mission_specific.perseverance.tmtc.perseverance_transmitter import PerseveranceTransmitter
from src.subsystems.device import CommonDevice, HealthState
from src.tmtc.mdb_parsing_service import MdbParsingService

TMTC_CFG = yaml.safe_load((ROOT / "cfg/controller/perseverance-controller.yaml").read_text())
TMTC_CFG = TMTC_CFG["robots_settings"]["yamcs_tmtc"]
ROBOT_CFG = yaml.safe_load((ROOT / "cfg/robot/perseverance.yaml").read_text())
FAULT_CFG = ROBOT_CFG["robots_settings"]["parameters"]["fault_injection"]

registry = MdbParsingService.load_mdb_registry([str(ROOT / "cfg/mdb/perseverance.xml")])

failures = []


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


class FakeSubsystems:
    def __init__(self):
        self.health = {
            CommonDevice.MOTOR_CONTROLLER: HealthState.NOMINAL,
            CommonDevice.CAMERA: HealthState.NOMINAL,
            CommonDevice.EPS: HealthState.NOMINAL,
            CommonDevice.RADIO: HealthState.NOMINAL,
            "imu": HealthState.NOMINAL,
        }
        self.parasitic_load = 0.0

    def get_device_health_state(self, device):
        return self.health[device]

    def set_device_health_state(self, device, state):
        self.health[device] = state

    def set_parasitic_load(self, watts):
        self.parasitic_load = watts


class FakeRobot:
    def __init__(self):
        self.wheel_damping = {}
        self.wheel_friction = {}
        self.wheel_sinkage = {}
        self.resistance_calls = []
        self.max_efforts = {}
        self.steer_overrides = {}
        self.steer_positions = [0.0] * 4
        self.subsystems = FakeSubsystems()
        self.imu = {}
        self.camera_noise = 0.0

    def set_joint_max_efforts(self, efforts):
        self.max_efforts.update(efforts)

    def set_steer_override(self, wheel_name, angle=None):
        if angle is None:
            self.steer_overrides.pop(wheel_name, None)
        else:
            self.steer_overrides[wheel_name] = angle

    def clear_steer_overrides(self):
        self.steer_overrides.clear()

    def set_wheel_damping_factor(self, wheel_name, factor=1.0):
        if factor == 1.0:
            self.wheel_damping.pop(wheel_name, None)
        else:
            self.wheel_damping[wheel_name] = factor

    def clear_wheel_damping_factors(self):
        self.wheel_damping.clear()

    def set_wheel_friction(self, wheel_name, friction):
        self.wheel_friction[wheel_name] = friction

    def reset_wheel_friction(self):
        self.wheel_friction.clear()

    def set_wheel_sinkage(self, wheel_name, depth_m):
        self.wheel_sinkage[wheel_name] = depth_m

    def reset_wheel_sinkage(self):
        self.wheel_sinkage.clear()

    def apply_wheel_resistance(self, coefficients, velocity_epsilon):
        self.resistance_calls.append((dict(coefficients), velocity_epsilon))

    def get_wheel_joint_efforts(self, wheel_names):
        return [float(i) for i, _ in enumerate(wheel_names)]

    def get_steer_angles(self):
        return self.steer_positions

    def set_fault_seed(self, seed):
        pass

    def set_imu_corruption(self, **kwargs):
        self.imu = dict(kwargs)

    def set_camera_corruption(self, noise=0.0):
        self.camera_noise = noise


def encode(name, *fields):
    """Build the wire payload the way Yamcs does: ASCII name, then arguments big-endian."""
    payload = name.encode("ascii")
    for kind, value in fields:
        payload += bytes([value]) if kind == "u8" else struct.pack(">f", value)
    return payload


robot = FakeRobot()
injector = FaultInjector(robot, FAULT_CFG, seed=99)
sent = {}
transmitter = PerseveranceTransmitter(
    lambda name, value: sent.__setitem__(name, value),
    None, robot, None, "perseverance", TMTC_CFG["parameters"], fault_injector=injector,
)
commander = PerseveranceCommander(robot, transmitter, None, None, injector)

# Mirrors PerseveranceController.setup_command_callbacks.
CATALOGUE = {
    "inject_wheel_torque_fault": (commander.inject_wheel_torque_fault, ["wheel", "severity"]),
    "inject_wheel_stuck_fault": (commander.inject_wheel_stuck_fault, ["wheel", "severity"]),
    "inject_wheel_slip_fault": (commander.inject_wheel_slip_fault, ["wheel", "severity"]),
    "inject_wheel_sink_fault": (commander.inject_wheel_sink_fault, ["wheel", "severity"]),
    "inject_steer_torque_fault": (commander.inject_steer_torque_fault, ["corner", "severity"]),
    "inject_steer_stuck_fault": (commander.inject_steer_stuck_fault, ["corner", "angle"]),
    "inject_imu_fault": (commander.inject_imu_fault, ["bias", "noise"]),
    "inject_camera_fault": (commander.inject_camera_fault, ["loss", "noise"]),
    "inject_battery_fault": (commander.inject_battery_fault, ["severity"]),
    "inject_comms_fault": (commander.inject_comms_fault, ["tm_loss", "tc_loss"]),
    "clear_faults": (commander.clear_faults, []),
}
BY_PATH = {TMTC_CFG["commands"][key]: (key, *value) for key, value in CATALOGUE.items()}


def uplink(payload):
    """What CommandsHandler._tc_listener_loop plus _execute do with a received datagram."""
    decoded = MdbParsingService.decode_tc_payload(payload, registry)
    assert decoded is not None, f"undecodable payload {payload!r}"
    _key, func, arg_names = BY_PATH[decoded["full_name"]]
    func(*[decoded["arguments"][n] for n in arg_names])
    injector.update()
    return decoded


print("\n=== every fault command survives the wire ===")
cases = [
    ("inject_wheel_torque_fault", [("u8", WHEELS.index("mid_left")), ("f32", 0.75)],
     {"wheel": "MID_LEFT", "severity": 0.75}),
    ("inject_wheel_stuck_fault", [("u8", WHEELS.index("rear_left")), ("f32", 0.5)],
     {"wheel": "REAR_LEFT", "severity": 0.5}),
    ("inject_wheel_slip_fault", [("u8", WHEELS.index("front_right")), ("f32", 0.75)],
     {"wheel": "FRONT_RIGHT", "severity": 0.75}),
    ("inject_wheel_sink_fault", [("u8", WHEELS.index("mid_right")), ("f32", 0.5)],
     {"wheel": "MID_RIGHT", "severity": 0.5}),
    ("inject_steer_torque_fault", [("u8", CORNERS.index("front_left")), ("f32", 0.5)],
     {"corner": "FRONT_LEFT", "severity": 0.5}),
    ("inject_steer_stuck_fault", [("u8", CORNERS.index("rear_right")), ("f32", -20.0)],
     {"corner": "REAR_RIGHT", "angle": -20.0}),
    ("inject_imu_fault", [("f32", 0.5), ("f32", 0.25)], {"bias": 0.5, "noise": 0.25}),
    ("inject_camera_fault", [("f32", 0.0), ("f32", 0.5)], {"loss": 0.0, "noise": 0.5}),
    ("inject_battery_fault", [("f32", 0.5)], {"severity": 0.5}),
    ("inject_comms_fault", [("f32", 0.25), ("f32", 0.0)], {"tm_loss": 0.25, "tc_loss": 0.0}),
]
for name, fields, expected in cases:
    decoded = uplink(encode(name, *fields))
    ok = decoded["arguments"] == expected
    check(f"{name}: {decoded['arguments']}", ok)

print("\n=== the commands actually reached the right subsystem ===")
check("wheel joint capped", abs(robot.max_efforts["drive_joint_mid_left"] - 5.0) < 1e-6)
check("wheel seized", abs(robot.wheel_damping["rear_left"] - 2.0) < 1e-6)
check("wheel grip lowered", abs(robot.wheel_friction["front_right"] - 0.125) < 1e-6)
check("wheel sunk", abs(robot.wheel_sinkage["mid_right"] - 0.015) < 1e-6)
check("sunk wheel dragged every step", robot.resistance_calls and "mid_right" in robot.resistance_calls[-1][0])
check("steer joint capped", abs(robot.max_efforts["steer_joint_front_left"] - 75.0) < 1e-6)
check("corner pinned", abs(robot.steer_overrides["rear_right"] - math.radians(-20.0)) < 1e-9)
check("imu biased", abs(robot.imu["accel_bias"] - 1.0) < 1e-6)
check("camera grained", abs(robot.camera_noise - 40.0) < 1e-6)
check("battery loaded", abs(robot.subsystems.parasitic_load - 20.0) < 1e-6)
check("acknowledgement echoed to the ground", "inject_comms_fault" in transmitter._active_command)
check("float echo is what the operator typed, not the float32 widening",
      "0.25" in transmitter._active_command)

print("\n=== downlink: everything injected reaches the ground ===")
robot.steer_positions = [math.radians(a) for a in (5.0, -20.0, 0.0, 3.0)]
transmitter.transmit_fault_state()
transmitter.transmit_steer_encoder()
check("active summary downlinked", sent["/Rover/faults/active"] != "none")
check("wheel limits are 6 long", len(sent["/Rover/faults/wheel_torque_limit"]) == 6)
check("wheel damping factors downlinked",
      sent["/Rover/faults/wheel_damping_factor"][WHEELS.index("rear_left")] == 2.0)
check("wheel friction downlinked",
      abs(sent["/Rover/faults/wheel_friction"][WHEELS.index("front_right")] - 0.125) < 1e-6)
check("wheel sinkage downlinked",
      abs(sent["/Rover/faults/wheel_sinkage"][WHEELS.index("mid_right")] - 0.015) < 1e-6)
transmitter.transmit_motor_effort()
check("measured drive effort downlinked in wheel order",
      sent["/Rover/motor_effort"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
check("steer limits are 4 long", len(sent["/Rover/faults/steer_torque_limit"]) == 4)
check("steer encoder in degrees", abs(sent["/Rover/steer_encoder"][1] - (-20.0)) < 1e-6)
check("wheel health names the faulted wheel",
      sent["/Rover/faults/wheel_health"][WHEELS.index("mid_left")] == HealthState.DEGRADED)
check("steer health marks the stuck corner FAULT",
      sent["/Rover/faults/steer_health"][CORNERS.index("rear_right")] == HealthState.FAULT)
check("imu health downlinked", sent["/Rover/faults/imu_health"] == HealthState.DEGRADED)
check("camera health downlinked", sent["/Rover/faults/camera_health"] == HealthState.DEGRADED)
check("battery health downlinked", sent["/Rover/faults/battery_health"] == HealthState.DEGRADED)
check("comms health downlinked", sent["/Rover/faults/comms_health"] == HealthState.DEGRADED)
check("parasitic load downlinked", abs(sent["/Rover/faults/parasitic_load"] - 20.0) < 1e-6)
check("drop counters downlinked as an aggregate",
      set(sent["/Rover/faults/dropped"]) == {"tm", "tc", "camera_frames"})

print("\n=== clear_faults is an exact round trip ===")
uplink(encode("clear_faults"))
transmitter.transmit_fault_state()
check("drive joints unlimited again", all(robot.max_efforts[f"drive_joint_{w}"] == UNLIMITED for w in WHEELS))
check("steer joints unlimited again", all(robot.max_efforts[f"steer_joint_{c}"] == UNLIMITED for c in CORNERS))
check("overrides released", robot.steer_overrides == {})
check("seized wheels released", robot.wheel_damping == {})
check("wheel grip restored", robot.wheel_friction == {})
check("sunk wheels lifted", robot.wheel_sinkage == {})
check("ground told no wheel is sunk", sent["/Rover/faults/wheel_sinkage"] == [0.0] * 6)
check("ground told every wheel has nominal friction", sent["/Rover/faults/wheel_friction"] == [0.5] * 6)
check("ground told every wheel damping is nominal", sent["/Rover/faults/wheel_damping_factor"] == [1.0] * 6)
check("sensors clean", not any(robot.imu.values()) and robot.camera_noise == 0.0)
check("battery load removed", robot.subsystems.parasitic_load == 0.0)
check("ground told nothing is active", sent["/Rover/faults/active"] == "none")
check("all health back to nominal",
      sent["/Rover/faults/imu_health"] == sent["/Rover/faults/camera_health"] ==
      sent["/Rover/faults/battery_health"] == sent["/Rover/faults/comms_health"] == HealthState.NOMINAL)

# ── the dataset path must break the rover in exactly the same way ────────────────
print("\n=== a scheduled fault is the same fault ground control sends ===")
from src.mission_specific.perseverance.dataset.episode_runner import INJECTORS
from src.mission_specific.perseverance.dataset.fault_scheduler import KIND_SHAPES


def injector_state(robot, injector):
    """Everything an observer could tell the two paths apart by."""
    return {
        "max_efforts": dict(robot.max_efforts),
        "steer_overrides": dict(robot.steer_overrides),
        "wheel_damping": dict(robot.wheel_damping),
        "wheel_friction": dict(robot.wheel_friction),
        "wheel_sinkage": dict(robot.wheel_sinkage),
        "resistance_calls": list(robot.resistance_calls),
        "imu": dict(robot.imu),
        "camera_noise": robot.camera_noise,
        "parasitic_load": robot.subsystems.parasitic_load,
        "health": dict(robot.subsystems.health),
        "active": injector.active_faults,
        "wheel_limits": injector.wheel_torque_limits,
        "wheel_damping_factors": injector.wheel_damping_factors,
        "wheel_frictions": injector.wheel_frictions,
        "wheel_sinkages": injector.wheel_sinkages,
        "steer_limits": injector.steer_torque_limits,
        "wheel_health": injector.wheel_health,
        "steer_health": injector.steer_health,
        "device_health": [injector.imu_health, injector.camera_health,
                          injector.battery_health, injector.comms_health,
                          injector.motor_controller_health],
    }


def via_ground(command, arg_fields, arg_names):
    """The full uplink: encode as Yamcs, decode with the rover's mdb, dispatch, apply."""
    local_robot = FakeRobot()
    local_injector = FaultInjector(local_robot, FAULT_CFG, seed=99)
    local_transmitter = PerseveranceTransmitter(
        lambda name, value: None, None, local_robot, None, "perseverance",
        TMTC_CFG["parameters"], fault_injector=local_injector,
    )
    local_commander = PerseveranceCommander(local_robot, local_transmitter, None, None, local_injector)

    decoded = MdbParsingService.decode_tc_payload(encode(command, *arg_fields), registry)
    getattr(local_commander, command)(*[decoded["arguments"][n] for n in arg_names])
    local_injector.update()
    return injector_state(local_robot, local_injector)


def via_schedule(kind, target, magnitudes):
    """What EpisodeRunner._inject does with a scheduled event."""
    local_robot = FakeRobot()
    local_injector = FaultInjector(local_robot, FAULT_CFG, seed=99)
    method_name, takes_target = INJECTORS[kind]
    method = getattr(local_injector, method_name)
    method(target, *magnitudes) if takes_target else method(*magnitudes)
    local_injector.update()
    return injector_state(local_robot, local_injector)


EQUIVALENCE = [
    ("wheel_torque", "inject_wheel_torque_fault",
     [("u8", WHEELS.index("mid_left")), ("f32", 0.75)], ["wheel", "severity"],
     "mid_left", (0.75,)),
    ("wheel_stuck", "inject_wheel_stuck_fault",
     [("u8", WHEELS.index("rear_left")), ("f32", 0.75)], ["wheel", "severity"],
     "rear_left", (0.75,)),
    ("wheel_slip", "inject_wheel_slip_fault",
     [("u8", WHEELS.index("front_right")), ("f32", 0.75)], ["wheel", "severity"],
     "front_right", (0.75,)),
    ("wheel_sink", "inject_wheel_sink_fault",
     [("u8", WHEELS.index("mid_right")), ("f32", 0.5)], ["wheel", "severity"],
     "mid_right", (0.5,)),
    ("steer_torque", "inject_steer_torque_fault",
     [("u8", CORNERS.index("front_left")), ("f32", 0.5)], ["corner", "severity"],
     "front_left", (0.5,)),
    ("steer_stuck", "inject_steer_stuck_fault",
     [("u8", CORNERS.index("rear_right")), ("f32", -20.0)], ["corner", "angle"],
     "rear_right", (-20.0,)),
    ("imu", "inject_imu_fault", [("f32", 0.5), ("f32", 0.25)], ["bias", "noise"], None, (0.5, 0.25)),
    ("camera", "inject_camera_fault", [("f32", 0.0), ("f32", 0.5)], ["loss", "noise"], None, (0.0, 0.5)),
    ("battery", "inject_battery_fault", [("f32", 0.5)], ["severity"], None, (0.5,)),
    ("comms", "inject_comms_fault", [("f32", 0.25), ("f32", 0.0)], ["tm_loss", "tc_loss"], None, (0.25, 0.0)),
]

for kind, command, arg_fields, arg_names, target, magnitudes in EQUIVALENCE:
    ground = via_ground(command, arg_fields, arg_names)
    scheduled = via_schedule(kind, target, magnitudes)
    check(f"{kind}: the dataset path and the ground path leave identical state", ground == scheduled)

check("ALL fans out the same way on both paths",
      via_ground("inject_wheel_torque_fault", [("u8", 6), ("f32", 1.0)], ["wheel", "severity"]) ==
      via_schedule("wheel_torque", "ALL", (1.0,)))
check("the scheduler and the runner agree on every kind",
      set(INJECTORS) == set(KIND_SHAPES))
check("the runner knows which kinds take a target",
      all(INJECTORS[kind][1] == (KIND_SHAPES[kind][0] is not None) for kind in INJECTORS))

print("\n=== mdb hygiene ===")
names = sorted(registry, key=len)
collisions = [(a, b) for a in names for b in names if a != b and b.startswith(a)]
check(f"no command name prefixes another ({len(registry)} commands)", not collisions)
check("every configured command exists in the mdb",
      set(TMTC_CFG["commands"].values()) <= {s["full_name"] for s in registry.values()})

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
