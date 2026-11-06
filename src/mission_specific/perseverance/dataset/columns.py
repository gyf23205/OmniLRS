__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
Column groups of a fault-detection dataset, and a reader that returns one group at a time.

observable/ and oracle/ say where a column is stored. The groups below say what a model should do
with it. An action-conditioned world model is conditioned on the two action groups and predicts the
two observation groups:

    control_action          what the joints are driven with: wheel speed and steer angle setpoints
    other_action            the ground's commands: drive goals, device power, solar panel
    dynamics_observation    the rover's motion and the forces on it: joints, IMU, wheel loads
    other_observation       power, thermal, comms, avionics and software state; the nav-cam
                            frames (observable/images/) belong here too, outside the tables

and never reads estimator or oracle. The groups are recorded in manifest.json -> columns.groups and
applied here by column name.

The command.* action columns are not in any CSV. read_group expands observable/commands.jsonl into
them, one row per downlink tick, so the events line up with the telemetry they caused.

A column matching no rule is reported as "unassigned" rather than guessed into a group: a parameter
newly added to the rover has to be placed here on purpose.

Pure python + numpy, no pandas: the reader has to run in the Isaac Sim python and on a host whose
pandas may not import.
"""

import bisect
import csv
import fnmatch
import json
import os
from typing import Dict, List

import numpy as np

UNASSIGNED = "unassigned"

# Group -> fnmatch patterns on the full column name ("motor_effort.*" covers motor_effort.0-5).
# Order is the order columns.groups lists them in. Rules must not overlap; test_dataset_recorder
# checks that.
GROUPS = (
    # Row alignment. time_s and step are in both telemetry.csv and truth.csv. obc_uptime counts
    # seconds since boot, so it is a clock too. It starts at a random uptime, so it no longer gives
    # the episode time, but it still says nothing about the state.
    ("index", ("time_s", "step", "obc_uptime")),
    # The joint setpoints the onboard controller drove with, mean over the downlink interval. The
    # value at row t covers (t-1, t]: pair it with the transition from row t-1 to row t. Torque is
    # not here: the drive joints are velocity-controlled, so torque (motor_effort) is a measured
    # response.
    ("control_action", ("wheel_speed_setpoint.*", "steer_angle_setpoint.*")),
    # The ground's commands, expanded per tick. None of them reaches a joint directly: a drive goal
    # acts through the onboard controller, and powering off the motor controller stops the rover only
    # by aborting it, so their effect on the motion is already in the setpoints.
    ("other_action", (
        "command.type.*", "command.drive_straight.*", "command.drive_turn.*",
        "command.power.*", "command.solar_panel_deployed",
    )),
    # The rover's motion and the forces acting on it. contact_force_* is the solver's load per wheel,
    # averaged over the downlink interval, standing in for whichever load sensor the real rover
    # carries (no noise model).
    ("dynamics_observation", (
        "motor_encoder.*", "motor_effort.*", "steer_encoder.*", "contact_force_*",
        "imu_accelerometer.*", "imu_gyroscope.*", "imu_orientation.*",
    )),
    # Measured, but not the motion. motor_current is here because the power model draws a fixed
    # current whenever the rover is in motor mode, whatever the load on the wheel.
    ("other_observation", (
        "motor_current.*", "battery_charge", "battery_voltage", "total_current_in", "total_current_out",
        "net_power", "current_draw_*", "temperature_*", "radio_rssi",
        "obc_cpu_usage", "obc_ram_usage", "obc_disk_usage", "obc_state", "solar_panel_state",
        "command_status",
    )),
    # The onboard navigation filter's outputs: functions of the observation history.
    ("estimator", ("estimator.*",)),
    # What no rover could know: true pose, injected fault state, labels. Also the commanded goto
    # target (world frame) and the progress fields computed from the true pose.
    ("oracle", (
        "pose_ground_truth.*", "oracle.*", "faults.*",
        "command_target.*", "command_distance_remaining", "command_heading_error",
    )),
)

# Downlinked, but never written to a dataset (fnmatch patterns on the column name), because they are
# constant by construction in a dataset run. Both still reach a live ground station. active_command
# is only set by a Yamcs telecommand, and the scripter sends none; go_nogo only changes when the
# ground sends NOGO, which the scripter never does. Columns that are constant only because a run did
# not happen to exercise them (a wheel no fault hit) are not listed: they vary in another run.
DROPPED_COLUMNS = ("active_command", "go_nogo")


def is_dropped(column: str) -> bool:
    """True for a column that DROPPED_COLUMNS keeps out of a dataset."""
    return any(fnmatch.fnmatchcase(column, pattern) for pattern in DROPPED_COLUMNS)

GROUP_NAMES = tuple(name for name, _ in GROUPS) + (UNASSIGNED,)

# ── commands as per-tick columns ─────────────────────────────────────────────────
# Drive commands replace one another: the latest one is the rover's current goal. Each maps to the
# arguments that become action columns, in the units the ground sends them: drive_straight in m/s
# and m, drive_turn in deg/s and deg. goto and stop have no argument columns (a goto target is in the
# world frame, which no observation shares).
MOTION_COMMANDS = {
    "goto": (),
    "drive_straight": ("linear_velocity", "distance"),
    "drive_turn": ("angular_velocity", "angle"),
    "stop": (),
}
# Drive commands given a one-hot command.type.<name>. stop has none: it ends the goal, leaving every
# command.type at zero as before the first command - the scripter never sends it outside crater runs.
TYPED_COMMANDS = ("goto", "drive_straight", "drive_turn")
# Housekeeping commands set a state that persists alongside the drive goal. Defaults are the state
# every episode starts in (EpisodeRunner._reset_subsystems): all devices on, panel deployed. Only the
# devices the scripter powers off (dataset.mission.power_off_devices) get a column; a command to any
# other device is refused rather than dropped.
DEVICES = ("MOTOR_CONTROLLER", "CAMERA")

COMMAND_COLUMNS = (
    tuple(f"command.type.{name}" for name in TYPED_COMMANDS)
    + tuple(f"command.{name}.{arg}" for name, args in MOTION_COMMANDS.items() for arg in args)
    + tuple(f"command.power.{device}" for device in DEVICES)
    + ("command.solar_panel_deployed",)
)


def group_of(column: str) -> str:
    """The group a column belongs to, or UNASSIGNED."""
    for name, patterns in GROUPS:
        if any(fnmatch.fnmatchcase(column, pattern) for pattern in patterns):
            return name
    return UNASSIGNED


def group_columns(columns: List[str]) -> Dict[str, List[str]]:
    """{group: columns in their given order}, every group listed, duplicates dropped."""
    groups: Dict[str, List[str]] = {name: [] for name in GROUP_NAMES}
    for column in dict.fromkeys(columns):
        groups[group_of(column)].append(column)
    return groups


# ── reader ───────────────────────────────────────────────────────────────────────
def _read_csv(path: str) -> Dict[str, List[str]]:
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {}
    header, body = rows[0], rows[1:]
    return {name: [row[i] if i < len(row) else "" for row in body] for i, name in enumerate(header)}


def _to_array(cells: List[str]) -> np.ndarray:
    """float64 with NaN for an empty cell (a value lost to a comms fault); str if any cell is not a number."""
    try:
        return np.array([float(cell) if cell != "" else np.nan for cell in cells], dtype=np.float64)
    except ValueError:
        return np.array(cells, dtype=object)


def command_columns(commands: List[Dict], steps: List[int]) -> Dict[str, np.ndarray]:
    """
    Expand command records into COMMAND_COLUMNS, one entry per downlink tick.

    A command sent at step s belongs to the first tick at or after s, the same (t-1, t] interval the
    setpoint means cover. Each column holds the state set by the latest command sent at or before
    the tick: a one-hot of the current drive command (all zero before the first and after a stop)
    with its arguments (zero for the other drive commands), each device's commanded power (1 = ON),
    and the commanded panel deployment (1 = DEPLOYED).

    Every command the ground sent counts, delivered or not. The ground cannot know a command was
    lost in transit, and the mismatch with what follows is what a tc_loss fault looks like.
    """
    ticks = len(steps)
    out = {name: np.zeros(ticks) for name in COMMAND_COLUMNS}
    for device in DEVICES:
        out[f"command.power.{device}"][:] = 1.0
    out["command.solar_panel_deployed"][:] = 1.0

    for record in sorted((r for r in commands if r.get("command")), key=lambda r: r["step"]):
        name, arguments = record["command"], record.get("arguments") or {}
        tick = bisect.bisect_left(steps, int(record["step"]))
        if tick >= ticks:
            break
        held = slice(tick, None)

        if name in MOTION_COMMANDS:
            for other, args in MOTION_COMMANDS.items():
                if other in TYPED_COMMANDS:
                    out[f"command.type.{other}"][held] = 1.0 if other == name else 0.0
                for arg in args:
                    out[f"command.{other}.{arg}"][held] = float(arguments[arg]) if other == name else 0.0
        elif name == "power_electronics":
            device = arguments["subsystem_id"]
            if device not in DEVICES:
                raise ValueError(f"power command to {device!r} has no action column; add it to DEVICES")
            out[f"command.power.{device}"][held] = 1.0 if arguments["power_state"] == "ON" else 0.0
        elif name == "deploy_solar_panel":
            out["command.solar_panel_deployed"][held] = 1.0 if arguments["deployment"] == "DEPLOYED" else 0.0
        else:
            # Same rule as an unassigned column: a new command must be given columns on purpose.
            raise ValueError(f"command {name!r} has no action columns; add it to dataset/columns.py")
    return out


def read_group(episode_dir: str, group: str) -> Dict[str, np.ndarray]:
    """
    One episode's columns in `group`, as {column: array with one entry per downlink tick}.

    Reads observable/telemetry.csv and oracle/truth.csv, whose rows are the same downlink ticks.
    Numeric columns come back as float64 with NaN for a lost value; text columns as str arrays.
    The two action groups also carry their share of COMMAND_COLUMNS, built from commands.jsonl.
    Stack an all-numeric group into a (ticks, columns) matrix with np.column_stack(list(d.values())).
    """
    if group not in GROUP_NAMES:
        raise ValueError(f"unknown group {group!r}; expected one of {GROUP_NAMES}")

    observable = _read_csv(os.path.join(episode_dir, "observable", "telemetry.csv"))
    truth_path = os.path.join(episode_dir, "oracle", "truth.csv")
    truth = _read_csv(truth_path) if os.path.exists(truth_path) else {}
    if truth and truth.get("step") != observable.get("step"):
        raise ValueError(f"{episode_dir}: telemetry.csv and truth.csv rows are not the same ticks")

    columns = dict(observable)
    for name, cells in truth.items():
        columns.setdefault(name, cells)
    selected = {name: _to_array(cells) for name, cells in columns.items() if group_of(name) == group}

    if group in ("control_action", "other_action"):
        with open(os.path.join(episode_dir, "observable", "commands.jsonl")) as handle:
            commands = [json.loads(line) for line in handle if line.strip()]
        expanded = command_columns(commands, [int(float(s)) for s in observable.get("step", [])])
        selected.update({name: values for name, values in expanded.items() if group_of(name) == group})
    return selected
