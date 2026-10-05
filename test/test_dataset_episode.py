#!/usr/bin/env python3
"""
Host-runnable check of the whole episode loop. No Isaac Sim, no omni imports.

    python3 test/test_dataset_episode.py

EpisodeRunner is deliberately free of omni: the world is duck-typed and the pose comes from the
robot group, so the loop that will run inside Isaac can be run here against fakes. What this proves
is the part that is expensive to debug in simulation - that episodes reset, that scheduled faults
reach the injector at the right time, that the labels line up with the telemetry, and that a
nominal episode really is clean.
"""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.control.drive_controller import CommandStatus
from src.mission_specific.perseverance.dataset import fault_scheduler as fs
from src.mission_specific.perseverance.dataset.episode_runner import EpisodeRunner
from src.mission_specific.perseverance.dataset.recorder import DatasetRecorder
from src.mission_specific.perseverance.faults.fault_injector import FaultInjector
from src.subsystems.device import CommonDevice, HealthState, PowerState
from src.subsystems.robot_enums import GoNogoState, ObcState, SolarPanelState

failures = []
ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "cfg/controller/perseverance-controller.yaml").read_text())
PARAMS = PARAMS["robots_settings"]["yamcs_tmtc"]["parameters"]
DATASET_CFG = yaml.safe_load((ROOT / "cfg/robot/perseverance.yaml").read_text())
DATASET_CFG = DATASET_CFG["robots_settings"]["parameters"]["dataset"]


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


# ── fakes ────────────────────────────────────────────────────────────────────
class FakePowerModel:
    def __init__(self):
        self._battery_capacity_wh = 60.0
        self._battery_charge_wh = 60.0

    def drain(self, wh):
        self._battery_charge_wh = max(0.0, self._battery_charge_wh - wh)


class FakeThermalModel:
    def __init__(self):
        self._node_temps = {}
        self.initialized = 0

    def initialize(self):
        self.initialized += 1
        if not self._node_temps:
            self._node_temps = {"+X": -20.0}


class FakeDevice:
    def __init__(self):
        self.power_state = PowerState.ON

    def set_power_state(self, state):
        self.power_state = state


class FakeSubsystems:
    def __init__(self):
        self._devices = {device: FakeDevice() for device in CommonDevice}
        self.solar_panel_state = SolarPanelState.DEPLOYED
        self.health = {
            CommonDevice.MOTOR_CONTROLLER: HealthState.NOMINAL,
            CommonDevice.CAMERA: HealthState.NOMINAL,
            CommonDevice.EPS: HealthState.NOMINAL,
            CommonDevice.RADIO: HealthState.NOMINAL,
            "imu": HealthState.NOMINAL,
        }
        self._power_model = FakePowerModel()
        self._thermal_model = FakeThermalModel()
        self.go_nogo = GoNogoState.NOGO
        self.obc_state = ObcState.IDLE
        self.obc_transitions = []
        self.power_setups = 0
        self.parasitic_load = 0.0
        self.sun = None

    def get_device_power_state(self, device):
        return self._devices[device].power_state

    def set_device_power_state(self, device, state):
        self._devices[device].set_power_state(state)

    def set_solar_panel_state(self, state):
        self.solar_panel_state = state

    def _setup_power_model(self):
        self.power_setups += 1
        self._power_model = FakePowerModel()

    def get_device_health_state(self, device):
        return self.health[device]

    def set_device_health_state(self, device, state):
        self.health[device] = state

    def set_parasitic_load(self, watts):
        self.parasitic_load = watts

    def set_go_nogo_state(self, state):
        self.go_nogo = state

    def get_go_nogo_state(self):
        return self.go_nogo

    def set_obc_state(self, state):
        self.obc_state = state
        self.obc_transitions.append(state)

    def get_obc_state(self):
        return self.obc_state

    def get_obc_status(self):
        return {}

    def set_sun_position(self, position):
        self.sun = position

    def get_power_status(self, position, yaw_deg, dt, obc_state):
        # A real load: the battery has to be seen to drain, and to come back on reset.
        self._power_model.drain(0.01 + self.parasitic_load * dt / 3600.0)
        return {}


class FakeRobot:
    robot_name = "/perseverance"

    def __init__(self):
        self.wheel_damping = {}
        self.wheel_friction = {}
        self.wheel_sinkage = {}
        self.resistance_calls = []
        self.subsystems = FakeSubsystems()
        self.max_efforts = {}
        self.steer_overrides = {}
        self.imu = {}
        self.camera_noise = 0.0
        self.seed = None
        self.reset_poses = []
        self.resets = 0
        self._last_imu = (
            ({"ax": 0.0, "ay": 0.0, "az": 1.62}, {"gx": 0.0}, {"roll": 0.0, "yaw": 10.0}),
            ({"ax": 0.0, "ay": 0.0, "az": 1.62}, {"gx": 0.0}, {"roll": 0.0, "yaw": 10.0}),
        )

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

    def set_reset_pose(self, position, orientation):
        self.reset_poses.append((list(position), list(orientation)))

    def reset(self):
        self.resets += 1

    def get_rgba_camera_view(self, resolution):
        import numpy as np
        return np.zeros((4, 4, 4), dtype=np.uint8)


class FakeRobotGroup:
    def __init__(self):
        self.position = [10.0, 10.0, 0.3]

    def get_pose_of_base_link(self):
        return list(self.position), [1.0, 0.0, 0.0, 0.0]


class FakeRobotManager:
    def __init__(self):
        self.robot = FakeRobot()
        self.robot_RG = FakeRobotGroup()


class FakeDrive:
    """Accepts commands and completes them after a fixed number of update() calls."""

    LEG_STEPS = 40

    def __init__(self, robot_group):
        self._rg = robot_group
        self.status = CommandStatus.IDLE
        self._remaining = 0
        self.commands = []

    def _start(self, label):
        self.commands.append(label)
        self.status = CommandStatus.EXECUTING
        self._remaining = self.LEG_STEPS

    def command_goto(self, x, y):
        self._target = (x, y)
        self._start("goto")

    def command_straight(self, v, d):
        self._start("drive_straight")

    def command_turn(self, r, a):
        self._start("drive_turn")

    def command_stop(self):
        self.status = CommandStatus.IDLE
        self._remaining = 0

    def abort(self, reason=""):
        self.status = CommandStatus.ABORTED
        self._remaining = 0

    def update(self):
        if self.status != CommandStatus.EXECUTING:
            return
        self._remaining -= 1
        # Drift so waypoints are not all drawn from the same place.
        self._rg.position[0] += 0.01
        if self._remaining <= 0:
            self.status = CommandStatus.COMPLETE


class FakeWorld:
    def __init__(self):
        self.steps = 0

    def step(self, render=True):
        self.steps += 1

    def is_playing(self):
        return True


class FakeEnvironment:
    def __init__(self):
        self.rock_calls = 0
        self.terrain_calls = 0

    def randomize_rocks(self, num):
        self.rock_calls += 1

    def switch_terrain(self, flag=-1):
        self.terrain_calls += 1


class FakeTransmitter:
    """A handful of real parameter paths - enough to prove the tick wiring."""

    def __init__(self, transmit, robot, robot_group):
        self._transmit = transmit
        self._robot = robot
        self._rg = robot_group

    def transmit_active_command(self):
        self._transmit(PARAMS["active_command"], "goto")

    def transmit_pose_of_base_link(self):
        position, orientation = self._rg.get_pose_of_base_link()
        self._transmit(PARAMS["pose_of_base_link"], {
            "position": {"x": position[0], "y": position[1], "z": position[2]},
            "orientation": {"w": orientation[0], "x": orientation[1],
                            "y": orientation[2], "z": orientation[3]},
        })

    def transmit_imu_readings(self):
        self._transmit(PARAMS["imu_accelerometer"], dict(self._robot._last_imu[1][0]))

    def transmit_power_info(self, interval_s):
        model = self._robot.subsystems._power_model
        self._transmit(PARAMS["battery_charge"],
                       round(100.0 * model._battery_charge_wh / model._battery_capacity_wh))
        self._transmit(PARAMS["net_power"], -3.0)

    def transmit_steer_encoder(self):
        self._transmit(PARAMS["steer_encoder"], [0.0, 0.0, 0.0, 0.0])


REFS = {
    "nominal_wheel_torque": 20.0, "nominal_steer_torque": 150.0,
    "max_imu_accel_bias": 2.0, "max_imu_gyro_bias": 0.5,
    "max_imu_orientation_bias_deg": 20.0, "max_imu_accel_noise": 1.0,
    "max_imu_gyro_noise": 0.2, "max_camera_noise": 80.0, "max_parasitic_load_w": 40.0,
}

STEPS = 900          # 30 s of simulation at 33 ms
PHYSICS_DT = 0.0333


def build_run(tmp, episodes=6, seed=0, scheduler_config=None, record_images=False,
              crater_config=None, drive_cls=None, environment=None, terrain_resolution=None):
    world = FakeWorld()
    RM = FakeRobotManager()
    drive = (drive_cls or FakeDrive)(RM.robot_RG)
    injector = FaultInjector(RM.robot, REFS, seed=seed)
    recorder = DatasetRecorder(tmp, injector, robot=RM.robot, subsystems=RM.robot.subsystems,
                               record_images=record_images, min_free_gb=0.0)
    recorder._transmitter = FakeTransmitter(recorder.transmit, RM.robot, RM.robot_RG)
    environment = environment or FakeEnvironment()

    runner = EpisodeRunner(
        world, RM, drive, injector, recorder,
        sun_fn=lambda: (1000.0, 0.0, 500.0),
        physics_dt=PHYSICS_DT,
        spawn=(10.0, 10.0, 3.0),
        episodes=episodes,
        seed=seed,
        config={**DATASET_CFG.get("episode", {}), "episode_steps": STEPS, "settle_steps": 5},
        scheduler_config=scheduler_config or DATASET_CFG.get("faults"),
        mission_config=DATASET_CFG.get("mission"),
        environment=environment,
        record_images=record_images,
        crater_config=crater_config,
        terrain_resolution=terrain_resolution,
    )
    return world, RM, drive, injector, recorder, runner, environment


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


# ── a whole run ──────────────────────────────────────────────────────────────
print("\n=== a six-episode run completes and writes the layout ===")
tmp = tempfile.mkdtemp(prefix="dsep-")
world, RM, drive, injector, recorder, runner, environment = build_run(tmp, episodes=6)
manifest = runner.run()

check("every episode was recorded", manifest["episodes"] == 6)
check("the world stepped for every episode", world.steps >= 6 * STEPS)
episodes = sorted((Path(tmp) / "episodes").iterdir())
check("one directory per episode", len(episodes) == 6)
check("each episode has both tiers",
      all((d / "observable/telemetry.csv").exists() and (d / "oracle/truth.csv").exists()
          for d in episodes))
check("each episode has its labels",
      all((d / "oracle/faults.jsonl").exists() and (d / "oracle/schedule.json").exists()
          for d in episodes))
check("the manifest reports both class levels",
      "sample_level_faulted_fraction" in manifest["class_balance"] and
      manifest["class_balance"]["episodes"])

print("\n=== the rover really is reset between episodes ===")
check("the battery is re-initialised once per episode", RM.robot.subsystems.power_setups == 6)
check("the rover is teleported once per episode", RM.robot.resets == 6)
check("each episode starts somewhere different",
      len({tuple(pose[0]) for pose in RM.robot.reset_poses}) == 6)
check("the spawn stays within the configured jitter",
      all(abs(pose[0][0] - 10.0) <= 3.0 and abs(pose[0][1] - 10.0) <= 3.0
          for pose in RM.robot.reset_poses))
check("go/nogo is set to GO, or every command would be rejected",
      RM.robot.subsystems.go_nogo == GoNogoState.GO)
check("obc uptime is restarted through an OFF transition",
      RM.robot.subsystems.obc_transitions.count(ObcState.OFF) == 6)
check("thermal state is cleared before re-initialising",
      RM.robot.subsystems._thermal_model.initialized == 6)
check("the scene is re-randomised per episode", environment.rock_calls == 6)
check("terrain is left alone unless asked", environment.terrain_calls == 0)
check("each episode reseeds the robot's noise", RM.robot.seed is not None)

print("\n=== the rover is actually driven ===")
check("commands were issued", len(drive.commands) > 5)
check("more than one kind of command was used", len(set(drive.commands)) > 1)
first = read_csv(episodes[0] / "observable/telemetry.csv")
check("telemetry ticks at the downlink period, not the physics rate",
      abs(len(first) - STEPS * PHYSICS_DT) <= 2)
check("the battery drains within an episode",
      float(read_csv(episodes[0] / "oracle/truth.csv")[0]["oracle.battery_charge_wh"]) >
      float(read_csv(episodes[0] / "oracle/truth.csv")[-1]["oracle.battery_charge_wh"]))
check("and comes back full for the next one",
      abs(float(read_csv(episodes[1] / "oracle/truth.csv")[0]["oracle.battery_charge_wh"]) - 60.0) < 0.5)

# ── labels line up with telemetry ────────────────────────────────────────────
print("\n=== labels line up with the telemetry ===")
for directory in episodes:
    meta = json.loads((directory / "meta.json").read_text())
    schedule = json.loads((directory / "oracle/schedule.json").read_text())
    events = [json.loads(line) for line in
              (directory / "oracle/faults.jsonl").read_text().splitlines() if line]
    truth = read_csv(directory / "oracle/truth.csv")

    if meta["is_nominal"]:
        check(f"ep{meta['index']} nominal: no fault event was applied", not events)
        check(f"ep{meta['index']} nominal: the label is clear for the whole episode",
              all(row["oracle.fault_active"] == "0" for row in truth))
        check(f"ep{meta['index']} nominal: no health flag ever left NOMINAL",
              all(row["oracle.imu_health"] == str(HealthState.NOMINAL.value) for row in truth))
    else:
        check(f"ep{meta['index']} faulted: every scheduled onset was applied",
              len(events) >= len(schedule["faults"]))
        onset = min(fault["onset_s"] for fault in schedule["faults"])
        before = [row for row in truth if float(row["time_s"]) < onset]
        after = [row for row in truth if float(row["time_s"]) > onset + 2.0]
        check(f"ep{meta['index']} faulted: clean before the onset",
              bool(before) and all(row["oracle.fault_active"] == "0" for row in before))
        check(f"ep{meta['index']} faulted: labelled after the onset",
              bool(after) and any(row["oracle.fault_active"] == "1" for row in after))

print("\n=== a scheduled fault reaches the rover, not just the label ===")
tmp_forced = tempfile.mkdtemp(prefix="dsep-")
forced = {"class_weights": {"nominal": 0.0, "single": 1.0, "concurrent": 0.0, "sequential": 0.0},
          "kind_weights": {kind: (1.0 if kind == "wheel_torque" else 0.0)
                           for kind in fs.KIND_SHAPES},
          "recovery_probability": 0.0, "all_targets_probability": 1.0}
world, RM, drive, injector, recorder, runner, environment = build_run(
    tmp_forced, episodes=1, seed=4, scheduler_config=forced)
runner.run()
directory = next((Path(tmp_forced) / "episodes").iterdir())
truth = read_csv(directory / "oracle/truth.csv")
schedule = json.loads((directory / "oracle/schedule.json").read_text())
severity = schedule["faults"][0]["magnitudes"][0]

# clear_all writes UNLIMITED to every joint at reset, so only the drive joints should be capped;
# the steer joints are still inf and that is what a healthy joint looks like.
_drive_efforts = {name: effort for name, effort in RM.robot.max_efforts.items()
                  if name.startswith("drive_joint_")}
check("every drive joint was actually capped in the simulation",
      len(_drive_efforts) == 6 and all(effort < 20.0 for effort in _drive_efforts.values()))
check("the steer joints were left unlimited",
      all(effort == float("inf") for name, effort in RM.robot.max_efforts.items()
          if name.startswith("steer_joint_")))
check("the recorded torque limit matches the injected severity",
      abs(float(truth[-1]["oracle.wheel_torque_limit.0"]) - (1.0 - severity) * 20.0) < 1e-4)
check("the health flag followed", truth[-1]["oracle.wheel_health.0"] != str(HealthState.NOMINAL.value))
check("the fault family is named in the label", "wheel" in truth[-1]["oracle.fault_families"])

print("\n=== a tc_loss fault makes commands disappear on their way to the rover ===")
tmp_comms = tempfile.mkdtemp(prefix="dsep-")
world, RM, drive, injector, recorder, runner, environment = build_run(tmp_comms, episodes=1, seed=11)
recorder.begin_episode(0, 11, fs.FaultSchedule("nominal", 30.0, []))
injector.inject_comms_fault(0.0, 1.0)
injector.update()
issued = len(drive.commands)
for step in range(200):
    record = runner._scripter.update(step * PHYSICS_DT, (10.0, 10.0))
    if record is not None:
        recorder.log_command(step * PHYSICS_DT, step, record["command"],
                             record["arguments"], record["delivered"], record["note"])
summary = recorder.end_episode()
commands = [json.loads(line) for line in
            (Path(tmp_comms) / "episodes/ep_0000/observable/commands.jsonl").read_text().splitlines()]

check("the ground still logged the commands it sent", len(commands) > 0)
check("none of them reached the rover", all(not c["delivered"] for c in commands))
check("the rover really did not move", len(drive.commands) == issued)
check("the drop counter agrees", summary["dropped"]["tc"] == len(commands))

print("\n=== config precedence: an explicit override beats the config file ===")
from src.mission_specific.perseverance.dataset.episode_runner import merged_config as episode_config

check("the config file supplies the default",
      episode_config(DATASET_CFG["episode"])["episode_steps"] == 18000)
check("an explicit value overrides it",
      episode_config({**DATASET_CFG["episode"], "episode_steps": 1500})["episode_steps"] == 1500)
check("unknown keys are ignored rather than smuggled in",
      "nonsense" not in episode_config({"nonsense": 1}))

print("\n=== waypoints stay inside the terrain ===")
from src.mission_specific.perseverance.dataset.mission_scripter import MissionScripter

_bounded = MissionScripter(
    FakeDrive(FakeRobotGroup()), None,
    {"bounds": [[2.0, 18.0], [2.0, 18.0]], "command_weights": {"goto": 1.0}},
    seed=1, origin=(17.0, 17.0),
)
_points = [_bounded._waypoint((17.0, 17.0)) for _ in range(2000)]
check("every waypoint is inside the bounds even from a corner start",
      all(2.0 <= x <= 18.0 and 2.0 <= y <= 18.0 for x, y in _points))
_unbounded = MissionScripter(FakeDrive(FakeRobotGroup()), None,
                             {"command_weights": {"goto": 1.0}}, seed=1, origin=(17.0, 17.0))
check("without bounds the draw is unrestricted",
      any(x > 18.0 or y > 18.0 for x, y in
          [_unbounded._waypoint((17.0, 17.0)) for _ in range(2000)]))

print("\n=== a physics divergence ends the episode instead of recording garbage ===")


class DivergingRobotGroup(FakeRobotGroup):
    """Blows up the way PhysX did in simulation: a position around 1e13 after N reads."""

    def __init__(self, after=200):
        super().__init__()
        self.reads = 0
        self.resets = 0
        self.after = after

    def get_pose_of_base_link(self):
        self.reads += 1
        if self.reads > self.after:
            return [-3.8e12, 1.2e13, -5.3e12], [1.0, 0.0, 0.0, 0.0]
        return list(self.position), [1.0, 0.0, 0.0, 0.0]

    def recover(self):
        self.reads = 0
        self.resets += 1
        if self.resets >= 2:      # the first reset is the one before episode 0
            self.after = float("inf")


tmp_diverge = tempfile.mkdtemp(prefix="dsep-")
world, RM, drive, injector, recorder, runner, environment = build_run(tmp_diverge, episodes=2)
RM.robot_RG = DivergingRobotGroup(after=200)
runner._robot_RG = RM.robot_RG
# The teleport is what brings PhysX back; model that by clearing the fault on reset.
_original_reset = RM.robot.reset
RM.robot.reset = lambda: (RM.robot_RG.recover(), _original_reset())[1]
manifest = runner.run()

check("the diverged episode was still written", manifest["episodes"] >= 1)
_diverged_meta = json.loads((Path(tmp_diverge) / "episodes/ep_0000/meta.json").read_text())
check("and is marked as aborted, not complete", "diverged" in _diverged_meta["outcome"])
check("the rows captured before the divergence were kept",
      _diverged_meta["downlink_ticks"] > 0)
_rows = read_csv(Path(tmp_diverge) / "episodes/ep_0000/oracle/truth.csv")
check("no recorded pose is the blown-up one",
      all(abs(float(row["pose_ground_truth.position.x"])) < 500.0 for row in _rows))
check("the run recovered and went on to the next episode", manifest["episodes"] == 2)
check("the recovered episode is complete",
      json.loads((Path(tmp_diverge) / "episodes/ep_0001/meta.json").read_text())["outcome"] == "complete")

print("\n=== a rover that cannot be recovered stops the run ===")
tmp_dead = tempfile.mkdtemp(prefix="dsep-")
world, RM, drive, injector, recorder, runner, environment = build_run(tmp_dead, episodes=4)
runner._robot_RG = DivergingRobotGroup(after=100)   # never recovers: reset does not clear it
manifest = runner.run()
check("the run stopped instead of writing episode after episode of garbage",
      manifest["episodes"] < 4)
check("and says why", "diverg" in manifest["stopped_early"])

print("\n=== the disk guard stops the run instead of filling the volume ===")
tmp_guard = tempfile.mkdtemp(prefix="dsep-")
world, RM, drive, injector, recorder, runner, environment = build_run(tmp_guard, episodes=3)
recorder._min_free_gb = 10_000_000.0  # more free space than any disk has
manifest = runner.run()
check("no episode was written", manifest["episodes"] == 0)
check("and the run says why", "free" in manifest["stopped_early"])

print("\n=== crater episodes: stamped, restored, spawned from, and labelled from the true pose ===")
import numpy as np
from src.mission_specific.perseverance.dataset import crater as crater_module


class TerrainEnvironment(FakeEnvironment):
    """A 20 x 20 m flat terrain at 2.5 cm that records every edit."""

    def __init__(self):
        super().__init__()
        self.dem = np.full((800, 800), 0.2, dtype=np.float32)
        self.mask = np.ones_like(self.dem)
        self.pristine = self.dem.copy()
        self.sets = []

    def get_terrain(self):
        return self.dem.copy(), self.mask.copy()

    def set_terrain(self, dem, mask):
        self.dem, self.mask = np.array(dem), np.array(mask)
        self.sets.append("pristine" if np.array_equal(self.dem, self.pristine) else "crater")


class CraterDrive(FakeDrive):
    """gotos actually move the rover, at 5 cm a step, so a dash really reaches the crater."""

    def command_goto(self, x, y):
        super().command_goto(x, y)
        self._remaining = 10 ** 9

    def update(self):
        if self.status != CommandStatus.EXECUTING:
            return
        if getattr(self, "_target", None) is None or self._remaining < 10 ** 8:
            return super().update()
        dx, dy = self._target[0] - self._rg.position[0], self._target[1] - self._rg.position[1]
        distance = (dx * dx + dy * dy) ** 0.5
        step = min(0.05, distance)
        if distance > 0:
            self._rg.position[0] += dx / distance * step
            self._rg.position[1] += dy / distance * step
        if distance <= 0.05:
            self.status = CommandStatus.COMPLETE

    def command_straight(self, v, d):
        self._target = None
        super().command_straight(v, d)

    def command_turn(self, r, a):
        self._target = None
        super().command_turn(r, a)


crater_cfg = {**DATASET_CFG["crater"], "bounds": [[0.0, 20.0], [0.0, 20.0]], "resolution": 0.025,
              "outcome_weights": {"trapped": 1.0, "avoid": 0.0, "skirt": 0.0}, "fault_probability": 0.0}
mix_cfg = {**DATASET_CFG["faults"],
           "class_weights": {"nominal": 1.0, "single": 0.0, "concurrent": 0.0, "sequential": 0.0, "crater": 1.0}}
tmp_crater = tempfile.mkdtemp(prefix="dsep-")
terrain = TerrainEnvironment()
world, RM, drive, injector, recorder, runner, environment = build_run(
    tmp_crater, episodes=6, seed=4, scheduler_config=mix_cfg, crater_config=crater_cfg,
    drive_cls=CraterDrive, environment=terrain,
)
manifest = runner.run()
episode_meta = [json.loads((Path(tmp_crater) / f"episodes/ep_{i:04d}/meta.json").read_text())
                for i in range(manifest["episodes"])]
classes = [meta["episode_class"] for meta in episode_meta]
check(f"the run mixes crater and nominal episodes {classes}", "crater" in classes and "nominal" in classes)

expected_sets = []
on_terrain = False
for meta in episode_meta:
    if meta["episode_class"] == "crater":
        expected_sets.append("crater")
        on_terrain = True
    elif on_terrain:
        expected_sets.append("pristine")
        on_terrain = False
check(f"a crater is stamped for each crater episode and removed before the next plain one {terrain.sets}",
      terrain.sets == expected_sets)
check("every crater episode records that its crater was stamped",
      all(meta["crater_stamped"] for meta in episode_meta if meta["episode_class"] == "crater"))

columns = None
same_columns = True
for index, meta in enumerate(episode_meta):
    with open(Path(tmp_crater) / f"episodes/ep_{index:04d}/oracle/truth.csv", newline="") as handle:
        header = next(csv.reader(handle))
    columns = columns or header
    same_columns &= header == columns
check("every episode's truth.csv has the same columns, crater or not", same_columns)
check("crater columns are present", "oracle.crater.in_crater" in columns and "oracle.crater.rim_distance_m" in columns)

crater_index = classes.index("crater")
crater_meta = episode_meta[crater_index]
check("the crater episode spawned where its scenario said",
      [round(v, 3) for v in crater_meta["spawn"][:2]] == [round(v, 3) for v in crater_meta["crater"]["spawn"]])
rows = read_csv(Path(tmp_crater) / f"episodes/ep_{crater_index:04d}/oracle/truth.csv")
dash_s = crater_meta["crater"]["dash_s"]
before = [row for row in rows if float(row["time_s"]) < dash_s]
inside = [row for row in rows if row["oracle.crater.in_crater"] == "1"]
check("clean before the dash", all(row["oracle.fault_active"] == "0" for row in before))
check(f"the rover ends up in the crater ({len(inside)} ticks)", len(inside) > 0)
check("entry is after the dash and matches meta",
      crater_meta["crater_entered_s"] is not None and crater_meta["crater_entered_s"] >= dash_s and
      abs(float(inside[0]["time_s"]) - crater_meta["crater_entered_s"]) < 1.0)
check("in the crater the label is on, under the crater family",
      all(row["oracle.fault_active"] == "1" and "crater" in row["oracle.fault_families"] for row in inside))
check("the plan is recorded alongside the truth",
      all("crater" in row["oracle.scheduled_kinds"] for row in rows if float(row["time_s"]) >= dash_s))
check("no health flag moves: nothing on the rover is broken",
      all(row["oracle.motor_controller_health"] == str(HealthState.NOMINAL.value) for row in inside))
nominal_index = classes.index("nominal")
nominal_rows = read_csv(Path(tmp_crater) / f"episodes/ep_{nominal_index:04d}/oracle/truth.csv")
check("a plain episode's crater columns are blank and never in",
      all(row["oracle.crater.in_crater"] == "0" and row["oracle.crater.center_x"] == "" for row in nominal_rows))

print("\n=== the rover lands before anything commands it ===")
import math as _math
tmp_hold = tempfile.mkdtemp(prefix="dsep-")
world, RM, drive, injector, recorder, runner, environment = build_run(tmp_hold, episodes=1, seed=2)
runner._config["landing_hold_s"] = 2.0
_commands_at_reset = []
_original_reset_episode = runner._reset_episode


def _reset_and_count(*args, **kwargs):
    before = world.steps
    result = _original_reset_episode(*args, **kwargs)
    _commands_at_reset.append((world.steps - before, len(drive.commands)))
    return result


runner._reset_episode = _reset_and_count
runner.run()
hold_steps = _math.ceil(2.0 / PHYSICS_DT)
check(f"the settle lasts the landing hold, not just settle_steps ({_commands_at_reset[0][0]} >= {hold_steps})",
      _commands_at_reset[0][0] >= hold_steps)
check("and no command was issued during it", _commands_at_reset[0][1] == 0)
shutil.rmtree(tmp_hold, ignore_errors=True)

print("\n=== the rover is placed just above the ground, not dropped from the spawn height ===")
tmp_ground = tempfile.mkdtemp(prefix="dsep-")
ground_terrain = TerrainEnvironment()        # flat at 0.2 m
world, RM, drive, injector, recorder, runner, environment = build_run(
    tmp_ground, episodes=2, seed=4, environment=ground_terrain, terrain_resolution=0.025,
)
runner.run()
heights = [pose[0][2] for pose in RM.robot.reset_poses]
check(f"every reset places the rover at ground + clearance {heights}",
      heights and all(abs(z - (0.2 + 0.15)) < 1e-6 for z in heights))
world, RM, drive, injector, recorder, runner, environment = build_run(tmp_ground, episodes=1, seed=4)
runner.run()
check("without a known terrain it falls back to the configured spawn height",
      RM.robot.reset_poses[-1][0][2] == 3.0)
shutil.rmtree(tmp_ground, ignore_errors=True)

print("\n=== a drive_straight never ends outside the bounds ===")


class HeadingDrive(FakeDrive):
    """Reports a fixed heading, the way PerseveranceDriveController.pose() does."""

    heading = 0.0

    def pose(self):
        return self._rg.position[0], self._rg.position[1], self.heading

    def command_straight(self, v, d):
        self.straight_distance = d
        super().command_straight(v, d)


_straights = MissionScripter(
    HeadingDrive(FakeRobotGroup()), None,
    {"bounds": [[2.0, 18.0], [2.0, 18.0]],
     "command_weights": {"goto": 0.0, "drive_turn": 0.0, "drive_straight": 1.0}},
    seed=3, origin=(10.0, 10.0),
)
_ends, _gotos = [], 0
_rng = np.random.default_rng(0)
for _ in range(2000):
    x, y = _rng.uniform(1.0, 19.0, 2)            # some starts already outside the bounds
    heading = _rng.uniform(-np.pi, np.pi)
    _straights._drive.heading = heading
    name, arguments, _handler = _straights._draw((x, y))
    if name == "goto":
        _gotos += 1
        continue
    d = arguments["distance"]
    _ends.append(((x, y), (x + d * np.cos(heading), y + d * np.sin(heading))))


def _outside(p):
    """How far a point is outside the [2, 18] bounds, 0 inside."""
    return max(0.0, 2.0 - min(p), max(p) - 18.0)


_inside = [(a, b) for a, b in _ends if _outside(a) == 0.0]
check(f"every straight from inside the bounds stops inside them ({len(_inside)} straights)",
      _inside and all(_outside(b) == 0.0 for _, b in _inside))
check("a straight from outside the bounds never takes the rover further out",
      all(_outside(b) <= _outside(a) + 1e-9 for a, b in _ends))
check(f"a straight with no room left becomes a goto ({_gotos} of 2000)", _gotos > 0)
_edge = MissionScripter(
    HeadingDrive(FakeRobotGroup()), None,
    {"bounds": [[2.0, 18.0], [2.0, 18.0]],
     "command_weights": {"goto": 0.0, "drive_turn": 0.0, "drive_straight": 1.0}}, seed=3,
)
_edge._drive.heading = 0.0                      # facing +x, 1 m from the bound
check("facing the edge from 1 m away is never a straight",
      all(_edge._draw((17.0, 10.0))[0] == "goto" for _ in range(200)))
_blind = MissionScripter(
    FakeDrive(FakeRobotGroup()), None,
    {"bounds": [[2.0, 18.0], [2.0, 18.0]], "command_weights": {"drive_straight": 1.0, "goto": 1.0}}, seed=3,
)
check("without a heading, bounds rule straights out entirely",
      all(_blind._draw((10.0, 10.0))[0] != "drive_straight" for _ in range(200)))

print("\n=== a rover that drives off the terrain ends the episode at once, without the fall ===")


class FallingRobotGroup(FakeRobotGroup):
    """Drives over the +x edge of a 20 m terrain and falls under lunar gravity, once."""

    def __init__(self, fall_after_s=20.0):
        super().__init__()
        self.reads = 0
        self.fall_after = int(fall_after_s / PHYSICS_DT)
        self.falls = True

    def get_pose_of_base_link(self):
        self.reads += 1
        if not self.falls or self.reads <= self.fall_after:
            return [10.0, 10.0, 0.5], [1.0, 0.0, 0.0, 0.0]
        t = (self.reads - self.fall_after) * PHYSICS_DT
        return [19.9 + 0.2 * t, 10.0, 0.5 - 0.5 * 1.62 * t * t], [1.0, 0.0, 0.0, 0.0]


tmp_fall = tempfile.mkdtemp(prefix="dsep-")
world, RM, drive, injector, recorder, runner, environment = build_run(
    tmp_fall, episodes=2, seed=5, environment=TerrainEnvironment(), terrain_resolution=0.025,
    record_images=True,
)
RM.robot_RG = runner._robot_RG = FallingRobotGroup(fall_after_s=20.0)
_original_fall_reset = RM.robot.reset
RM.robot.reset = lambda: (setattr(RM.robot_RG, "falls", RM.robot.resets == 0), _original_fall_reset())[1]
manifest = runner.run()

_fell = json.loads((Path(tmp_fall) / "episodes/ep_0000/meta.json").read_text())
check(f"the episode is marked as a fall, not a divergence ({_fell['outcome']})",
      _fell["outcome"] == "aborted: fell off terrain")
_truth = read_csv(Path(tmp_fall) / "episodes/ep_0000/oracle/truth.csv")
_last = float(_truth[-1]["time_s"])
check(f"it ended within a second or two of leaving, not after 25 s of falling (last row {_last} s)",
      _truth and _last < 20.0)
check("no recorded row has the rover below the ground",
      all(float(row["pose_ground_truth.position.z"]) > 0.0 for row in _truth))
check(f"the rows before the fall were cut back by fall_trim_s (cut at {_fell.get('discarded_after_s')} s)",
      _fell.get("discarded_after_s") is not None and _last <= _fell["discarded_after_s"])
_frames = sorted(p.name for p in (Path(tmp_fall) / "episodes/ep_0000/observable/images").iterdir())
_indexed = [row["filename"] for row in read_csv(Path(tmp_fall) / "episodes/ep_0000/observable/images_index.csv")
            if row["filename"]]
check("frames from the discarded seconds are deleted, not left unindexed", _frames == sorted(_indexed))
check("the fall does not stop the run", manifest["episodes"] == 2 and not manifest["stopped_early"])
check("the next episode is complete",
      json.loads((Path(tmp_fall) / "episodes/ep_0001/meta.json").read_text())["outcome"] == "complete")
shutil.rmtree(tmp_fall, ignore_errors=True)

print("\n=== without --crater nothing changes ===")
tmp_plain = tempfile.mkdtemp(prefix="dsep-")
plain_terrain = TerrainEnvironment()
world, RM, drive, injector, recorder, runner, environment = build_run(
    tmp_plain, episodes=2, seed=4, environment=plain_terrain,
)
manifest = runner.run()
check("the terrain is never touched", plain_terrain.sets == [])
check("and no crater columns are written", not any(c.startswith("oracle.crater.") for c in manifest["columns"]["oracle"]))

for directory in (tmp, tmp_forced, tmp_comms, tmp_guard, tmp_diverge, tmp_dead, tmp_crater, tmp_plain):
    shutil.rmtree(directory, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for label in failures:
        print(f"  - {label}")
    sys.exit(1)

print("all episode checks passed")
