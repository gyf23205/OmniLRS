#!/usr/bin/env python3
"""
Host-runnable checks for the perseverance FaultInjector. No Isaac Sim, no omni imports.

    python3 test/test_fault_injector.py

Covers all ten fault kinds, the queue-and-apply threading rule, the ground-truth health
reporting, and the two properties that are easy to break and expensive to debug in simulation:
clearing is an exact round trip, and clear_faults survives a total uplink blackout.
"""

import math
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.faults.fault_injector import (
    CORNERS, UNLIMITED, WHEELS, FaultInjector,
)
from src.subsystems.device import CommonDevice, HealthState

failures = []


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


class FakeSubsystems:
    """The slice of RobotSubsystemsHandler the injector touches."""

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
    """Records what the injector writes, the way Robot would write it to the stage."""

    def __init__(self):
        self.wheel_damping = {}
        self.wheel_friction = {}
        self.wheel_sinkage = {}
        self.resistance_calls = []
        self.max_efforts = {}
        self.steer_overrides = {}
        self.subsystems = FakeSubsystems()
        self.imu = {}
        self.camera_noise = 0.0
        self.seed = None

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

    def set_fault_seed(self, seed):
        self.seed = seed

    def set_imu_corruption(self, **kwargs):
        self.imu = dict(kwargs)

    def set_camera_corruption(self, noise=0.0):
        self.camera_noise = noise


REFS = {
    "nominal_wheel_torque": 20.0,
    "nominal_steer_torque": 150.0,
    "max_imu_accel_bias": 2.0,
    "max_imu_gyro_bias": 0.5,
    "max_imu_orientation_bias_deg": 20.0,
    "max_imu_accel_noise": 1.0,
    "max_imu_gyro_noise": 0.2,
    "max_camera_noise": 80.0,
    "max_parasitic_load_w": 40.0,
}


def build():
    robot = FakeRobot()
    return robot, FaultInjector(robot, REFS, seed=1234)


# ── actuator faults ──────────────────────────────────────────────────────────
print("\n=== wheel torque: ALL fans out, severity is inverse torque ===")
robot, injector = build()
injector.inject_wheel_torque("ALL", 1.0)
injector.update()
check("all six drive joints written", len(robot.max_efforts) == 6)
check("severity 1.0 -> zero torque", all(v == 0.0 for v in robot.max_efforts.values()))
check("limits array all zero", injector.wheel_torque_limits == [0.0] * 6)

robot, injector = build()
injector.inject_wheel_torque("mid_left", 0.25)
injector.update()
light = robot.max_efforts["drive_joint_mid_left"]
injector.inject_wheel_torque("mid_left", 0.75)
injector.update()
check("0.25 -> 15 N*m", abs(light - 15.0) < 1e-9)
check("0.75 -> 5 N*m", abs(robot.max_efforts["drive_joint_mid_left"] - 5.0) < 1e-9)
check("re-injection replaces, not stacks", injector.active_faults.count("mid_left") == 1)

injector.inject_wheel_torque("mid_left", 0.0)
injector.update()
check("severity 0 writes inf, not the nominal", robot.max_efforts["drive_joint_mid_left"] == UNLIMITED)
check("and drops out of active", injector.active_faults == "none")
check("telemetry reports nominal", injector.wheel_torque_limits[WHEELS.index("mid_left")] == 20.0)

robot, injector = build()
injector.inject_wheel_torque("front_left", 5.0)
injector.inject_wheel_torque("front_right", -3.0)
injector.update()
check("severity clamps above 1", robot.max_efforts["drive_joint_front_left"] == 0.0)
check("severity clamps below 0", robot.max_efforts["drive_joint_front_right"] == UNLIMITED)

print("\n=== wheel stuck: severity inflates damping as 1 / (1 - severity) ===")
robot, injector = build()
injector.inject_wheel_stuck("mid_right", 0.5)
injector.update()
check("0.5 -> damping x2", abs(robot.wheel_damping["mid_right"] - 2.0) < 1e-9)
check("only the named wheel touched", list(robot.wheel_damping) == ["mid_right"])
check("torque limits untouched", robot.max_efforts == {})
check("telemetry reports the factor",
      injector.wheel_damping_factors == [1.0, 1.0, 1.0, 2.0, 1.0, 1.0])
check("wheel health DEGRADED", injector.wheel_health[WHEELS.index("mid_right")] == HealthState.DEGRADED.value)
check("mobility DEGRADED", injector.motor_controller_health == HealthState.DEGRADED.value)
check("reported active", "wheel_stuck[mid_right]=0.50" in injector.active_faults)

injector.inject_wheel_stuck("ALL", 1.0)
injector.update()
check("severity 1.0 hits the cap on all six",
      len(robot.wheel_damping) == 6 and all(v == 1000.0 for v in robot.wheel_damping.values()))
check("wheel health FAULT", all(h == HealthState.FAULT.value for h in injector.wheel_health))

injector.inject_wheel_stuck("ALL", 0.9999)
injector.update()
check("near-1 severities stay capped", all(v == 1000.0 for v in robot.wheel_damping.values()))

injector.inject_wheel_stuck("ALL", 0.0)
injector.update()
check("severity 0 releases the brake", robot.wheel_damping == {})
check("and drops out of active", injector.active_faults == "none")
check("health back to NOMINAL", injector.wheel_health == [HealthState.NOMINAL.value] * 6)

robot, injector = build()
injector.inject_wheel_torque("front_left", 0.3)
injector.inject_wheel_stuck("front_left", 0.8)
injector.update()
check("torque and stuck coexist on one wheel",
      "wheel_torque[front_left]" in injector.active_faults and "wheel_stuck[front_left]" in injector.active_faults)
injector.inject_wheel_torque("front_left", 0.0)
injector.update()
check("lifting the torque fault keeps the wheel DEGRADED",
      injector.wheel_health[0] == HealthState.DEGRADED.value)
injector.clear_all()
injector.update()
check("clear_faults releases stuck wheels", robot.wheel_damping == {} and injector.active_faults == "none")

robot, injector = build()
injector.inject_wheel_stuck("front_left", 5.0)
injector.inject_wheel_stuck("front_right", -1.0)
injector.inject_wheel_stuck("left_middle", 0.5)
injector.update()
check("severity clamps; unknown wheel ignored", robot.wheel_damping == {"front_left": 1000.0})

print("\n=== wheel slip: friction falls as (1 - severity) * nominal ===")
installed = []


class InstallRecordingRobot(FakeRobot):
    def install_wheel_materials(self, friction):
        installed.append(friction)


sinkage_installs = []


class SinkInstallRecordingRobot(InstallRecordingRobot):
    def install_wheel_sinkage(self):
        sinkage_installs.append(True)


InstallRecordingRobot = SinkInstallRecordingRobot
FaultInjector.prepare_robot(InstallRecordingRobot(), {"nominal_wheel_friction": 0.7})
FaultInjector.prepare_robot(InstallRecordingRobot())
check("prepare_robot installs materials at the configured nominal, else the PhysX default",
      installed == [0.7, 0.5])
check("prepare_robot also sets up wheel sinkage", sinkage_installs == [True, True])

robot, injector = build()
injector.inject_wheel_slip("ALL", 0.9)
injector.update()
check("all six wheels written", len(robot.wheel_friction) == 6)
check("0.9 of 0.5 -> 0.05", all(abs(v - 0.05) < 1e-9 for v in robot.wheel_friction.values()))
check("joint drives untouched", robot.max_efforts == {} and robot.wheel_damping == {})
check("telemetry reports the friction", all(abs(v - 0.05) < 1e-9 for v in injector.wheel_frictions))
check("wheel health DEGRADED", injector.wheel_health == [HealthState.DEGRADED.value] * 6)
check("mobility DEGRADED", injector.motor_controller_health == HealthState.DEGRADED.value)
check("reported active", injector.active_faults.count("wheel_slip[") == 6)

injector.inject_wheel_slip("mid_left", 1.0)
injector.update()
check("severity 1.0 -> frictionless", robot.wheel_friction["mid_left"] == 0.0)
check("and that wheel reads FAULT", injector.wheel_health[WHEELS.index("mid_left")] == HealthState.FAULT.value)

injector.inject_wheel_slip("mid_left", 0.0)
injector.update()
check("severity 0 writes the nominal back", robot.wheel_friction["mid_left"] == 0.5)
check("telemetry shows it nominal", injector.wheel_frictions[WHEELS.index("mid_left")] == 0.5)
check("and it drops out of active", "wheel_slip[mid_left]" not in injector.active_faults)

injector.inject_wheel_stuck("front_left", 0.5)
injector.update()
injector.clear_all()
injector.update()
check("clear_faults resets friction and releases stuck wheels together",
      robot.wheel_friction == {} and robot.wheel_damping == {} and injector.active_faults == "none")
check("telemetry back to nominal", injector.wheel_frictions == [0.5] * 6)

print("\n=== wheel sink: sinkage and per-step drag both scale with severity ===")
robot, injector = build()
injector.update()
check("no drag is applied while no wheel is sunk", robot.resistance_calls == [])

injector.inject_wheel_sink("mid_left", 0.5)
injector.update()
check("0.5 -> 15 mm of sinkage", abs(robot.wheel_sinkage["mid_left"] - 0.015) < 1e-9)
check("drag applied on the step the fault lands",
      len(robot.resistance_calls) == 1 and abs(robot.resistance_calls[0][0]["mid_left"] - 0.3) < 1e-9)
check("velocity epsilon passed through", robot.resistance_calls[0][1] == 0.05)
check("friction and joint drives untouched",
      robot.wheel_friction == {} and robot.wheel_damping == {} and robot.max_efforts == {})

for _ in range(4):
    injector.update()
check("drag reapplied on every step, not just once", len(robot.resistance_calls) == 5)
check("telemetry reports the sinkage", injector.wheel_sinkages == [0.0, 0.0, 0.015, 0.0, 0.0, 0.0])
check("wheel health DEGRADED", injector.wheel_health[WHEELS.index("mid_left")] == HealthState.DEGRADED.value)
check("mobility DEGRADED", injector.motor_controller_health == HealthState.DEGRADED.value)
check("reported active", "wheel_sink[mid_left]=0.50" in injector.active_faults)

injector.inject_wheel_sink("ALL", 1.0)
injector.update()
check("ALL at 1.0 -> 30 mm and full drag on all six",
      all(abs(robot.wheel_sinkage[w] - 0.03) < 1e-9 for w in WHEELS) and
      all(abs(c - 0.6) < 1e-9 for c in robot.resistance_calls[-1][0].values()) and
      len(robot.resistance_calls[-1][0]) == 6)

injector.inject_wheel_sink("ALL", 0.0)
injector.update()
calls = len(robot.resistance_calls)
injector.update()
check("severity 0 lifts every wheel", all(robot.wheel_sinkage[w] == 0.0 for w in WHEELS))
check("and drag stops", len(robot.resistance_calls) == calls)
check("and it drops out of active", injector.active_faults == "none")

injector.inject_wheel_sink("front_left", 0.8)
injector.update()
injector.clear_all()
injector.update()
calls = len(robot.resistance_calls)
injector.update()
check("clear_faults lifts sunk wheels and stops the drag",
      robot.wheel_sinkage == {} and len(robot.resistance_calls) == calls and injector.active_faults == "none")

print("\n=== a failing drag call does not take the simulation down ===")
failing_calls = []


class FailingResistanceRobot(FakeRobot):
    def apply_wheel_resistance(self, coefficients, velocity_epsilon):
        failing_calls.append(1)
        raise AttributeError("'SingleRigidPrim' object has no attribute 'get_velocities'")


robot = FailingResistanceRobot()
injector = FaultInjector(robot, REFS, seed=1)
injector.inject_wheel_sink("ALL", 0.5)
import io, contextlib
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    for _ in range(30):
        injector.update()
check("update() survives an exception from the drag, every step", len(failing_calls) == 30)
check("the sinkage itself was still applied", len(robot.wheel_sinkage) == 6)
check("the error is reported once, not every step",
      captured.getvalue().count("wheel_sink drag could not be applied") == 1)

print("\n=== steer faults ===")
robot, injector = build()
injector.inject_steer_torque("front_left", 0.8)
injector.inject_steer_stuck("rear_right", -20.0)
injector.update()
check("0.8 of 150 N*m -> 30 N*m", abs(robot.max_efforts["steer_joint_front_left"] - 30.0) < 1e-9)
check("stuck override in radians", abs(robot.steer_overrides["rear_right"] - math.radians(-20.0)) < 1e-9)
check("both reported active", "steer_torque" in injector.active_faults and "steer_stuck" in injector.active_faults)

before = dict(robot.max_efforts)
injector.inject_steer_torque("mid_left", 0.5)
injector.update()
check("mid wheels rejected as non-steerable", robot.max_efforts == before)

# ── sensor faults ────────────────────────────────────────────────────────────
print("\n=== imu fault ===")
robot, injector = build()
injector.inject_imu_fault(0.5, 0.0)
injector.update()
check("accel bias = severity x reference", abs(robot.imu["accel_bias"] - 1.0) < 1e-9)
check("gyro bias scaled independently", abs(robot.imu["gyro_bias"] - 0.25) < 1e-9)
check("orientation bias in degrees", abs(robot.imu["orientation_bias_deg"] - 10.0) < 1e-9)
check("no noise when only bias asked for", robot.imu["accel_noise"] == 0.0)

check("partial severity is DEGRADED", injector.imu_health == HealthState.DEGRADED)

injector.inject_imu_fault(0.0, 1.0)
injector.update()
check("noise without bias", robot.imu["accel_bias"] == 0.0 and robot.imu["accel_noise"] == 1.0)
check("maximum severity is FAULT", injector.imu_health == HealthState.FAULT)
check("imu device flag moved, but only to DEGRADED",
      robot.subsystems.health["imu"] == HealthState.DEGRADED)
check("reported in the active summary", "imu[" in injector.active_faults)

print("\n=== camera fault ===")
robot, injector = build()
injector.inject_camera_fault(0.0, 0.5)
injector.update()
check("noise = severity x reference", abs(robot.camera_noise - 40.0) < 1e-9)
check("no loss means no frames dropped", not any(injector.should_drop_camera_frame() for _ in range(50)))

robot, injector = build()
injector.inject_camera_fault(1.0, 0.0)
injector.update()
check("loss 1.0 drops every frame", all(injector.should_drop_camera_frame() for _ in range(50)))
check("dropped frames counted", injector.dropped_counts["camera_frames"] == 50)
check("camera device flag moved", robot.subsystems.health[CommonDevice.CAMERA] == HealthState.DEGRADED)

robot, injector = build()
injector.inject_camera_fault(0.5, 0.0)
injector.update()
drops = sum(injector.should_drop_camera_frame() for _ in range(10000))
check(f"loss 0.5 drops about half ({drops}/10000)", 4700 < drops < 5300)

# ── battery ──────────────────────────────────────────────────────────────────
print("\n=== battery fault ===")
robot, injector = build()
injector.inject_battery_fault(0.5)
injector.update()
check("parasitic load = severity x reference", abs(injector.parasitic_load - 20.0) < 1e-9)
check("pushed to the power model", abs(robot.subsystems.parasitic_load - 20.0) < 1e-9)
check("eps device flag moved", robot.subsystems.health[CommonDevice.EPS] == HealthState.DEGRADED)
injector.inject_battery_fault(1.0)
injector.update()
check("severity 1.0 -> full reference load", abs(robot.subsystems.parasitic_load - 40.0) < 1e-9)
check("battery health FAULT at 1.0", injector.battery_health == HealthState.FAULT)

# ── comms ────────────────────────────────────────────────────────────────────
print("\n=== comms fault ===")
robot, injector = build()
check("healthy link drops no telemetry", not any(injector.should_drop_tm() for _ in range(200)))
check("healthy link drops no commands", not any(injector.should_drop_tc("goto") for _ in range(200)))

injector.inject_comms_fault(0.3, 0.0)
injector.update()
drops = sum(injector.should_drop_tm() for _ in range(10000))
check(f"tm_loss 0.3 drops about 30% ({drops}/10000)", 2700 < drops < 3300)
check("counted as tm losses", injector.dropped_counts["tm"] == drops)
check("uplink untouched", not any(injector.should_drop_tc("goto") for _ in range(200)))
check("radio device flag moved", robot.subsystems.health[CommonDevice.RADIO] == HealthState.DEGRADED)

print("\n=== a total uplink blackout is still recoverable ===")
# The footgun this guards: at tc_loss 1.0 the only command that lifts the fault is the one being
# dropped, and the sole way out would be restarting the simulator.
robot, injector = build()
injector.inject_comms_fault(0.0, 1.0)
injector.update()
check("every ordinary command is lost", all(injector.should_drop_tc("goto") for _ in range(200)))
check("clear_faults is NEVER lost", not any(injector.should_drop_tc("clear_faults") for _ in range(200)))
check("the exemption is the real command name",
      injector.RECOVERY_COMMAND == "clear_faults")
injector.clear_all()
injector.update()
check("so the rover can be recovered", not any(injector.should_drop_tc("goto") for _ in range(200)))

# ── clearing ─────────────────────────────────────────────────────────────────
print("\n=== clear_all is an exact round trip across every fault family ===")
robot, injector = build()
injector.inject_wheel_torque("ALL", 0.9)
injector.inject_steer_torque("front_left", 0.5)
injector.inject_steer_stuck("rear_left", 15.0)
injector.inject_imu_fault(0.7, 0.7)
injector.inject_camera_fault(0.7, 0.7)
injector.inject_battery_fault(0.7)
injector.inject_comms_fault(0.7, 0.7)
injector.update()
check("all twelve entries are active",
      injector.active_faults.count(";") == 11)

losses_before = sum(injector.should_drop_tm() for _ in range(500))
check("the faulted link is losing telemetry", losses_before > 0)

injector.clear_all()
injector.update()
check("drive joints unlimited", all(robot.max_efforts[f"drive_joint_{w}"] == UNLIMITED for w in WHEELS))
check("steer joints unlimited", all(robot.max_efforts[f"steer_joint_{c}"] == UNLIMITED for c in CORNERS))
check("overrides released", robot.steer_overrides == {})
check("imu corruption zeroed", not any(robot.imu.values()))
check("camera noise zeroed", robot.camera_noise == 0.0)
check("parasitic load zeroed", robot.subsystems.parasitic_load == 0.0)
check("link healthy again", not any(injector.should_drop_tm() for _ in range(200)))
check("nothing active", injector.active_faults == "none")
check("every device flag back to NOMINAL",
      all(v == HealthState.NOMINAL for v in robot.subsystems.health.values()))
check("drop counters are cumulative, not reset by a clear",
      injector.dropped_counts["tm"] == losses_before)

# ── determinism ──────────────────────────────────────────────────────────────
print("\n=== a seeded run replays identically ===")
def loss_sequence(seed):
    robot = FakeRobot()
    inj = FaultInjector(robot, REFS, seed=seed)
    inj.inject_comms_fault(0.5, 0.0)
    inj.update()
    return [inj.should_drop_tm() for _ in range(500)], robot.seed

a, seed_a = loss_sequence(7)
b, _ = loss_sequence(7)
c, _ = loss_sequence(8)
check("same seed, same losses", a == b)
check("different seed, different losses", a != c)
check("the robot is seeded too, for sensor noise", seed_a == 7)

# ── spec parsing ─────────────────────────────────────────────────────────────
print("\n=== --fault spec parsing, all ten kinds ===")
robot, injector = build()
for spec, probe in [
    ("wheel_torque:mid_left:0.8", lambda: abs(robot.max_efforts["drive_joint_mid_left"] - 4.0) < 1e-9),
    ("wheel_stuck:rear_right:0.5", lambda: abs(robot.wheel_damping["rear_right"] - 2.0) < 1e-9),
    ("wheel_slip:front_right:0.9", lambda: abs(robot.wheel_friction["front_right"] - 0.05) < 1e-9),
    ("wheel_sink:rear_left:0.5", lambda: abs(robot.wheel_sinkage["rear_left"] - 0.015) < 1e-9),
    ("steer_torque:front_left:0.8", lambda: abs(robot.max_efforts["steer_joint_front_left"] - 30.0) < 1e-9),
    ("steer_stuck:rear_left:12", lambda: "rear_left" in robot.steer_overrides),
    ("imu:0.5:0.5", lambda: abs(robot.imu["accel_bias"] - 1.0) < 1e-9),
    ("camera:0.0:0.5", lambda: abs(robot.camera_noise - 40.0) < 1e-9),
    ("battery:0.5", lambda: abs(robot.subsystems.parasitic_load - 20.0) < 1e-9),
    ("comms:0.4:0.0", lambda: injector.comms_health == HealthState.DEGRADED),
]:
    injector.apply_spec(spec)
    injector.update()
    check(f"{spec} applied", probe())

before = dict(robot.max_efforts)
for bad in ("nonsense", "bogus_kind:mid_left:0.5", "wheel_torque:mid_left:abc",
            "battery:0.5:0.5", "imu:0.5", "wheel_torque:0.5"):
    injector.apply_spec(bad)
injector.update()
check("malformed specs are ignored, not fatal", robot.max_efforts == before)

# ── threading ────────────────────────────────────────────────────────────────
print("\n=== the telecommand thread only queues ===")
robot, injector = build()
injector.inject_wheel_torque("ALL", 1.0)
injector.inject_imu_fault(1.0, 1.0)
injector.inject_battery_fault(1.0)
check("no joint written before the tick", robot.max_efforts == {})
check("no sensor corruption before the tick", robot.imu == {})
check("no load pushed before the tick", robot.subsystems.parasitic_load == 0.0)
check("no health flag moved before the tick",
      all(v == HealthState.NOMINAL for v in robot.subsystems.health.values()))
injector.update()
check("all of it lands on the tick", len(robot.max_efforts) == 6 and robot.imu != {})
before = dict(robot.max_efforts)
injector.update()
check("an empty tick is a no-op", robot.max_efforts == before)

print("\n=== concurrency: commands from another thread ===")
writer_threads = set()


class ThreadRecordingRobot(FakeRobot):
    def set_joint_max_efforts(self, efforts):
        writer_threads.add(threading.current_thread().name)
        super().set_joint_max_efforts(efforts)

    def set_imu_corruption(self, **kwargs):
        writer_threads.add(threading.current_thread().name)
        super().set_imu_corruption(**kwargs)


robot = ThreadRecordingRobot()
injector = FaultInjector(robot, REFS, seed=1)
stop = threading.Event()


def listener():
    for i in range(200):
        injector.inject_wheel_torque(WHEELS[i % 6], (i % 10) / 10.0)
        injector.inject_imu_fault((i % 5) / 10.0, (i % 3) / 10.0)
        injector.inject_comms_fault((i % 4) / 10.0, 0.0)
    stop.set()


thread = threading.Thread(target=listener, name="tc-listener")
thread.start()
ticks = 0
while not stop.is_set() or ticks < 5:
    injector.update()
    ticks += 1
thread.join()
injector.update()

check("every write landed on the loop thread", writer_threads == {"MainThread"})
check("nothing was written from the listener", "tc-listener" not in writer_threads)
check("the queue drained completely", injector._pending == [])
check("state is coherent afterwards", len(injector.wheel_health) == 6 and len(injector.steer_health) == 4)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
