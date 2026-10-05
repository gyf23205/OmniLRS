#!/usr/bin/env python3
"""
Host-runnable checks for the dataset recorder. No Isaac Sim, no omni imports.

    python3 test/test_dataset_recorder.py

The recorder is driven with a fake transmitter that emits the real parameter paths taken from
cfg/controller/perseverance-controller.yaml, and with the real FaultInjector on a fake robot. So the
checks below are about the contract, not about a mock: that the observable columns are exactly the
downlink surface minus the oracle ones, that a comms fault appears as missing cells and nothing
else, and that the oracle row is never gated because it is not a downlink.
"""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.dataset import fault_scheduler as fs
from src.mission_specific.perseverance.dataset.recorder import DatasetRecorder, _is_oracle
from src.mission_specific.perseverance.faults.fault_injector import FaultInjector
from src.subsystems.device import CommonDevice, HealthState

failures = []
ROOT = Path(__file__).resolve().parents[1]


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


# ── fakes ────────────────────────────────────────────────────────────────────
class FakeSubsystemsHealth:
    def __init__(self):
        self.health = {
            CommonDevice.MOTOR_CONTROLLER: HealthState.NOMINAL,
            CommonDevice.CAMERA: HealthState.NOMINAL,
            CommonDevice.EPS: HealthState.NOMINAL,
            CommonDevice.RADIO: HealthState.NOMINAL,
            "imu": HealthState.NOMINAL,
        }
        self._power_model = FakePowerModel()

    def get_device_health_state(self, device):
        return self.health[device]

    def set_device_health_state(self, device, state):
        self.health[device] = state

    def set_parasitic_load(self, watts):
        pass


class FakePowerModel:
    _battery_capacity_wh = 60.0
    _battery_charge_wh = 45.0


class FakeRobot:
    def __init__(self):
        self.wheel_damping = {}
        self.wheel_friction = {}
        self.wheel_sinkage = {}
        self.resistance_calls = []
        self.subsystems = FakeSubsystemsHealth()
        self.max_efforts = {}
        self.steer_overrides = {}
        self._last_imu = None

    def set_joint_max_efforts(self, efforts):
        self.max_efforts.update(efforts)

    def set_steer_override(self, wheel_name, angle=None):
        pass

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
        pass

    def set_imu_corruption(self, **kwargs):
        pass

    def set_camera_corruption(self, noise=0.0):
        pass


def load_parameters_conf():
    text = (ROOT / "cfg" / "controller" / "perseverance-controller.yaml").read_text()
    conf = yaml.safe_load(text)
    return conf["robots_settings"]["yamcs_tmtc"]["parameters"]


PARAMS = load_parameters_conf()


class FakeTransmitter:
    """
    Emits the real parameter paths with representative shapes.

    Grouped by the transmitter method that owns each one, so the recorder's TICK_SEQUENCE is
    exercised exactly as it will be against PerseveranceTransmitter.
    """

    def __init__(self, transmit, robot):
        self._robot = robot
        self.imu_calls = 0
        self.observable_sends = 0

        def counting(path, value):
            if not _is_oracle(path):
                self.observable_sends += 1
            transmit(path, value)

        self._transmit = counting

    def transmit_active_command(self):
        self._transmit(PARAMS["active_command"], "goto(x=1.0, y=2.0)")

    def transmit_command_progress(self):
        self._transmit(PARAMS["command_status"], 1)
        self._transmit(PARAMS["command_distance_remaining"], 3.25)
        self._transmit(PARAMS["command_heading_error"], -1.5)
        self._transmit(PARAMS["command_target"], {"x": 1.0, "y": 2.0})

    def transmit_pose_of_base_link(self):
        self._transmit(PARAMS["pose_of_base_link"], {
            "position": {"x": 1.0, "y": 2.0, "z": 0.3},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        })

    def transmit_wheels_joint_angles(self):
        self._transmit(PARAMS["motor_encoder"], [1, 2, 3, 4, 5, 6])

    def transmit_motor_effort(self):
        self._transmit(PARAMS["motor_effort"], [2.5] * 6)

    def transmit_steer_encoder(self):
        self._transmit(PARAMS["steer_encoder"], [0.1, 0.2, 0.3, 0.4])

    def transmit_contact_forces(self):
        for name in ("front_left", "front_right", "mid_left", "mid_right", "rear_left", "rear_right"):
            self._transmit(PARAMS[f"contact_force_{name}"], 12.5)

    def transmit_imu_readings(self):
        self.imu_calls += 1
        clean = ({"ax": 0.0, "ay": 0.0, "az": 1.62},
                 {"gx": 0.0, "gy": 0.0, "gz": 0.0},
                 {"roll": 0.0, "pitch": 0.0, "yaw": 30.0})
        measured = ({"ax": 0.5, "ay": 0.5, "az": 2.12},
                    {"gx": 0.1, "gy": 0.1, "gz": 0.1},
                    {"roll": 2.0, "pitch": 2.0, "yaw": 32.0})
        self._robot._last_imu = (clean, measured)
        self._transmit(PARAMS["imu_accelerometer"], dict(measured[0]))
        self._transmit(PARAMS["imu_gyroscope"], dict(measured[1]))
        self._transmit(PARAMS["imu_orientation"], dict(measured[2]))

    def transmit_estimator(self):
        wheels = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2]
        snapshot = {
            "pose": {"x": 1.0, "y": 2.0, "yaw": 30.0},
            "motion": {"speed": 0.3, "lateral_speed": 0.0, "yaw_rate": 0.05},
            "bias": {"gyro": 0.001, "accel": 0.01},
            "imu_residual": {f"{k}_{f}": 0.1 for k in ("gyro", "heading", "lateral_force") for f in ("mean", "max")},
            "nis": {"wheels": 1.0, "gyro": 0.9, "heading": 1.1},
            "cusum_alarm": 0,
            "reacquisitions": 0,
        }
        for prefix in ("wheel_rolling", "wheel_sideslip", "wheel_effort", "wheel_tracking"):
            snapshot[f"{prefix}_mean"] = wheels
            snapshot[f"{prefix}_max"] = wheels
        for field, value in snapshot.items():
            self._transmit(PARAMS[f"estimator_{field}"], value)

    def transmit_power_info(self, interval_s):
        # Mirrors the real method overwriting the imu snapshot with its own read.
        self._robot._last_imu = (({"ax": 9.9}, {"gx": 9.9}, {"roll": 9.9}),
                                 ({"ax": 9.9}, {"gx": 9.9}, {"roll": 9.9}))
        for key, value in (("battery_charge", 88), ("battery_voltage", 16.4),
                           ("total_current_in", 1.1), ("total_current_out", 1.3),
                           ("net_power", -3.2), ("current_draw_obc", 0.2),
                           ("current_draw_motor_controller", 0.3), ("current_draw_camera", 0.1),
                           ("current_draw_radio", 0.1), ("current_draw_eps", 0.1)):
            self._transmit(PARAMS[key], value)
        self._transmit(PARAMS["motor_current"], [0.1] * 6)

    def transmit_thermal_info(self, interval_s):
        for face in ("front", "back", "left", "right", "top", "bottom"):
            self._transmit(PARAMS[f"temperature_{face}"], -20.0)

    def transmit_radio_signal_info(self):
        self._transmit(PARAMS["rssi"], -78)

    def transmit_obc_metrics(self):
        for key in ("obc_cpu_usage", "obc_ram_usage", "obc_disk_usage", "obc_uptime"):
            self._transmit(PARAMS[key], 42)

    def transmit_obc_state(self):
        self._transmit(PARAMS["obc_state"], 0)

    def transmit_go_nogo(self):
        self._transmit(PARAMS["go_nogo"], 1)

    def transmit_solar_panel_state(self):
        self._transmit(PARAMS["solar_panel_state"], 1)


REFS = {
    "nominal_wheel_torque": 20.0, "nominal_steer_torque": 150.0,
    "max_imu_accel_bias": 2.0, "max_imu_gyro_bias": 0.5,
    "max_imu_orientation_bias_deg": 20.0, "max_imu_accel_noise": 1.0,
    "max_imu_gyro_noise": 0.2, "max_camera_noise": 80.0, "max_parasitic_load_w": 40.0,
}


def build(tmp, record_images=True, seed=5):
    robot = FakeRobot()
    injector = FaultInjector(robot, REFS, seed=seed)
    recorder = DatasetRecorder(tmp, injector, robot=robot, subsystems=robot.subsystems,
                               record_images=record_images, min_free_gb=0.0)
    recorder._transmitter = FakeTransmitter(recorder.transmit, robot)
    return robot, injector, recorder


def run_episode(recorder, injector, ticks=10, index=0, schedule=None):
    schedule = schedule or fs.FaultSchedule("nominal", float(ticks), [])
    recorder.begin_episode(index, 1234, schedule)
    for step in range(ticks):
        injector.update()
        recorder.tick(float(step), step, 1.0)
    return recorder.end_episode()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


# ── the observable / oracle split ────────────────────────────────────────────
print("\n=== the split follows the parameter paths, not a hand-written list ===")
tmp = tempfile.mkdtemp(prefix="dsrec-")
robot, injector, recorder = build(tmp)
summary = run_episode(recorder, injector, ticks=5)

observable = read_csv(Path(tmp) / "episodes/ep_0000/observable/telemetry.csv")
oracle = read_csv(Path(tmp) / "episodes/ep_0000/oracle/truth.csv")

check("one observable row per tick", len(observable) == 5)
check("one oracle row per tick", len(oracle) == 5)

observable_columns = set(observable[0])
oracle_columns = set(oracle[0])

expected_observable = {
    path[len("/Rover/"):].replace("/", ".")
    for path in PARAMS.values() if not _is_oracle(path)
}
# The fake emits every parameter except the fault ones, which the recorder assembles itself.
missing = {name for name in expected_observable if not any(
    column == name or column.startswith(name + ".") for column in observable_columns)}
check("every non-oracle parameter reaches the observable file", not missing)
check("no fault parameter leaks into the observable file",
      not any(column.startswith("faults.") for column in observable_columns))
check("ground-truth pose is not observable",
      not any(column.startswith("pose_ground_truth") for column in observable_columns))
check("ground-truth pose is in the oracle file",
      "pose_ground_truth.position.x" in oracle_columns)
check("aggregates are flattened one scalar per column",
      {"imu_accelerometer.ax", "imu_accelerometer.ay", "imu_accelerometer.az"} <= observable_columns)
check("arrays are flattened by index",
      {"motor_encoder.0", "motor_encoder.5"} <= observable_columns and
      "motor_encoder.6" not in observable_columns)
check("measured drive effort is an observable column, one per wheel",
      {"motor_effort.0", "motor_effort.5"} <= observable_columns and
      "motor_effort.6" not in observable_columns)
check("sinkage is an oracle column", "oracle.wheel_sinkage.5" in oracle_columns)

print("\n=== the oracle carries what a mission never could ===")
check("ground-truth health is present",
      {"oracle.wheel_health.0", "oracle.steer_health.3", "oracle.imu_health",
       "oracle.battery_health", "oracle.comms_health"} <= oracle_columns)
check("the direct label is present",
      {"oracle.fault_active", "oracle.n_active_faults", "oracle.fault_families",
       "oracle.scheduled_kinds"} <= oracle_columns)
check("true battery charge is present, not just the measured percentage",
      oracle[0]["oracle.battery_charge_wh"] == "45.0")
check("torque limits are present per actuator",
      {"oracle.wheel_torque_limit.0", "oracle.steer_torque_limit.3"} <= oracle_columns)

print("\n=== the imu residual is exact, and taken before the later reads clobber it ===")
check("the clean reading is recorded", float(oracle[0]["oracle.imu_clean.az"]) == 1.62)
check("the residual is measured minus clean", abs(float(oracle[0]["oracle.imu_error.az"]) - 0.5) < 1e-9)
check("the yaw residual survives transmit_power_info reading the imu again",
      abs(float(oracle[0]["oracle.imu_error.yaw"]) - 2.0) < 1e-9)

# ── the comms fault is missing data ──────────────────────────────────────────
print("\n=== a comms fault is missing cells, and nothing else ===")
tmp_blackout = tempfile.mkdtemp(prefix="dsrec-")
robot, injector, recorder = build(tmp_blackout)
injector.inject_comms_fault(1.0, 0.0)
summary = run_episode(recorder, injector, ticks=5)

observable = read_csv(Path(tmp_blackout) / "episodes/ep_0000/observable/telemetry.csv")
oracle = read_csv(Path(tmp_blackout) / "episodes/ep_0000/oracle/truth.csv")
payload = [{k: v for k, v in row.items() if k not in ("time_s", "step")} for row in observable]

check("a total blackout empties every observable cell",
      all(value == "" for row in payload for value in row.values()))
check("the timestamp still marks the tick that was lost",
      [row["time_s"] for row in observable] == ["0.0", "1.0", "2.0", "3.0", "4.0"])
# The label comes from the injector's ground-truth health, so it is right even though this fault
# was injected directly rather than through the episode's schedule.
check("the oracle labels the fault even though the schedule does not know about it",
      all(row["oracle.fault_active"] == "1" for row in oracle) and
      all(row["oracle.fault_families"] == "comms" for row in oracle))
check("the oracle row is complete during a blackout",
      all(row["pose_ground_truth.position.x"] != "" for row in oracle))
check("the oracle knows the link is faulted", oracle[-1]["oracle.comms_health"] == str(HealthState.FAULT.value))
check("the drop counter matches the parameters that were offered",
      summary["dropped"]["tm"] == recorder._transmitter.observable_sends)

print("\n=== a healthy link drops nothing ===")
tmp_clean = tempfile.mkdtemp(prefix="dsrec-")
robot, injector, recorder = build(tmp_clean)
summary = run_episode(recorder, injector, ticks=5)
observable = read_csv(Path(tmp_clean) / "episodes/ep_0000/observable/telemetry.csv")
check("no cell is empty on a healthy link",
      not any(value == "" for row in observable for value in row.values()))
check("the drop counter is zero", summary["dropped"]["tm"] == 0)

print("\n=== a partial loss lands between the two, per parameter ===")
tmp_partial = tempfile.mkdtemp(prefix="dsrec-")
robot, injector, recorder = build(tmp_partial, seed=99)
injector.inject_comms_fault(0.5, 0.0)
summary = run_episode(recorder, injector, ticks=40)
observable = read_csv(Path(tmp_partial) / "episodes/ep_0000/observable/telemetry.csv")
payload = [{k: v for k, v in row.items() if k not in ("time_s", "step")} for row in observable]
cells = sum(len(row) for row in payload)
empty = sum(1 for row in payload for value in row.values() if value == "")
check(f"about half the cells are lost ({empty}/{cells})", 0.4 < empty / cells < 0.6)
check("the loss is per parameter, not per tick",
      not any(all(v == "" for v in row.values()) or all(v != "" for v in row.values())
              for row in payload))

# ── event logs ───────────────────────────────────────────────────────────────
print("\n=== event logs ===")
tmp_events = tempfile.mkdtemp(prefix="dsrec-")
robot, injector, recorder = build(tmp_events)
schedule = fs.FaultSchedule("single", 10.0, [
    fs.ScheduledFault("battery", None, (0.8,), 4.0, None),
])
recorder.begin_episode(0, 77, schedule)
injector.set_on_apply(lambda kind, targets, mags: recorder.log_fault_event(3.0, 3, kind, targets, mags))
recorder.log_command(1.0, 1, "goto", {"x": 1.0, "y": 2.0}, delivered=True)
recorder.log_command(2.0, 2, "drive_straight", {"linear_velocity": 0.3, "distance": 2.0},
                     delivered=False, note="lost in transit to the rover")
injector.inject_battery_fault(0.8)
injector.update()
recorder.log_image(5.0, 5, 0, "000000.png")
recorder.log_image(10.0, 10, 1, None)
for step in range(10):
    recorder.tick(float(step), step, 1.0)
summary = recorder.end_episode()

commands = [json.loads(line) for line in
            (Path(tmp_events) / "episodes/ep_0000/observable/commands.jsonl").read_text().splitlines()]
faults = [json.loads(line) for line in
          (Path(tmp_events) / "episodes/ep_0000/oracle/faults.jsonl").read_text().splitlines()]
images = read_csv(Path(tmp_events) / "episodes/ep_0000/observable/images_index.csv")

check("both commands are logged", len(commands) == 2)
check("a lost command is still in the ground's log", commands[1]["delivered"] is False)
check("a delivered command says so", commands[0]["delivered"] is True)
check("the fault event records the applied magnitude",
      len(faults) == 1 and faults[0]["kind"] == "battery" and faults[0]["magnitudes"] == [0.8])
check("a saved frame is indexed", images[0]["status"] == "saved" and images[0]["filename"] == "000000.png")
check("a lost frame is indexed as lost", images[1]["status"] == "lost" and images[1]["filename"] == "")
check("the summary counts frames both ways",
      summary["images_saved"] == 1 and summary["images_lost"] == 1)
check("the schedule is written next to the labels",
      json.loads((Path(tmp_events) / "episodes/ep_0000/oracle/schedule.json").read_text())["episode_class"] == "single")

# ── the manifest ─────────────────────────────────────────────────────────────
print("\n=== a re-run does not inherit the previous run's files ===")
tmp_stale = tempfile.mkdtemp(prefix="dsrec-")
robot, injector, recorder = build(tmp_stale, record_images=True)
recorder.begin_episode(0, 1, fs.FaultSchedule("nominal", 5.0, []))
_images = Path(tmp_stale) / "episodes/ep_0000/observable/images"
for index in range(9):                      # a long previous run wrote nine frames
    (_images / f"{index:06d}.png").write_bytes(b"stale")
    recorder.log_image(float(index), index, index, f"{index:06d}.png")
for step in range(5):
    recorder.tick(float(step), step, 1.0)
recorder.end_episode()
check("the first run wrote nine frames", len(list(_images.iterdir())) == 9)

# A shorter re-run into the same directory would otherwise leave six orphans behind, indexed
# nowhere, looking exactly like this episode's data.
robot, injector, recorder = build(tmp_stale, record_images=True)
recorder.begin_episode(0, 1, fs.FaultSchedule("nominal", 5.0, []))
for index in range(3):
    (_images / f"{index:06d}.png").write_bytes(b"fresh")
    recorder.log_image(float(index), index, index, f"{index:06d}.png")
for step in range(5):
    recorder.tick(float(step), step, 1.0)
summary = recorder.end_episode()

check("the re-run cleared the stale frames", len(list(_images.iterdir())) == 3)
check("no orphan is left that the index does not mention",
      {path.name for path in _images.iterdir()} ==
      {row["filename"] for row in read_csv(Path(tmp_stale) / "episodes/ep_0000/observable/images_index.csv")})
check("the summary counts only this run's frames", summary["images_saved"] == 3)

print("\n=== the manifest describes the dataset, both class levels ===")
tmp_manifest = tempfile.mkdtemp(prefix="dsrec-")
robot, injector, recorder = build(tmp_manifest)
run_episode(recorder, injector, ticks=10, index=0,
            schedule=fs.FaultSchedule("nominal", 10.0, []))
run_episode(recorder, injector, ticks=10, index=1,
            schedule=fs.FaultSchedule("single", 10.0, [fs.ScheduledFault("imu", None, (0.5, 0.0), 5.0)]))
manifest = recorder.finalize({"run_seed": 0})

check("both episodes are indexed", manifest["episodes"] == 2)
check("episode-level balance is reported", manifest["class_balance"]["episodes"]["nominal"] == 0.5)
check("sample-level balance is reported and lower than the episode level",
      0.0 < manifest["class_balance"]["sample_level_faulted_fraction"] < 0.5)
check("columns are listed for both tiers",
      manifest["columns"]["observable"] and manifest["columns"]["oracle"])
check("the derived_from_truth columns are tagged",
      "command_heading_error" in manifest["columns"]["notes"] and
      "contact_force_mid_left" in manifest["columns"]["notes"])
check("every tagged column actually exists in the observable file",
      set(manifest["columns"]["notes"]) <= set(manifest["columns"]["observable"]))
check("the determinism caveat is stated", any("PhysX" in note for note in manifest["caveats"]))

# ── header stability ─────────────────────────────────────────────────────────
print("\n=== the header is the union, so a parameter lost on tick 1 is not lost forever ===")
tmp_union = tempfile.mkdtemp(prefix="dsrec-")
robot, injector, recorder = build(tmp_union, seed=3)
injector.inject_comms_fault(0.9, 0.0)
run_episode(recorder, injector, ticks=30)
observable = read_csv(Path(tmp_union) / "episodes/ep_0000/observable/telemetry.csv")
check("a parameter dropped on every single tick still has its column",
      not {name for name in expected_observable if not any(
          column == name or column.startswith(name + ".") for column in observable[0])})
check("time_s and step lead the header", list(observable[0])[:2] == ["time_s", "step"])

for directory in (tmp, tmp_blackout, tmp_clean, tmp_partial, tmp_events, tmp_manifest, tmp_union, tmp_stale):
    shutil.rmtree(directory, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for label in failures:
        print(f"  - {label}")
    sys.exit(1)

print("all recorder checks passed")
