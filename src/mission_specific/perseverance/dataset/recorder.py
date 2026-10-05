__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
Recording for fault-detection datasets.

The observable stream is not re-derived here. PerseveranceTransmitter already defines what crosses
the link - 62 parameters, their names and their shapes - and its transmit_* methods never touch the
intervals handler, so this recorder drives the real transmitter with a recording transmit_func
instead of the Yamcs one. The dataset's observable surface is therefore the downlink surface by
construction, and a parameter added to the rover appears in the dataset without a change here.

OBSERVABLE VS ORACLE. Two files per episode. observable/ holds only what a real mission could see;
oracle/ holds what the simulator knows and a rover never would. Three columns sit in between and are
tagged rather than hidden - see COLUMN_NOTES.

THE COMMS FAULT IS MISSING DATA. A dropped parameter is written as an empty cell, per parameter, the
way PerseveranceController.transmit_to_yamcs drops it. That is the only form in which a link fault
could ever reach a detector, so it has to survive into the file rather than being smoothed over.

The oracle row is never gated: it is not a downlink. It is assembled directly from the injector and
the robot rather than from the transmitted stream, which is also what makes the two capture modes
(own transmitter, or tapping a live controller) produce identical oracle data.

Rows are buffered and written when the episode ends, so the header can be the union of every column
seen - otherwise a parameter first dropped by a comms fault would be missing from the header and
every later value for it would be lost.

No omni/pxr imports: the transmitter is injected, never imported, so this module is testable with
plain python3 against a fake.
"""

import csv
import json
import os
import shutil
import time
from typing import Dict, List, Optional

from src.subsystems.device import HealthState

# Downlink paths that are ground truth despite being downlinked today. Recorded as oracle.
ORACLE_PATH_PREFIXES = ("/Rover/faults/",)
ORACLE_PATHS = ("/Rover/pose_ground_truth",)

# Observable, but computed from ground truth rather than measured. A real rover would downlink the
# same fields from its own navigation filter and would have no wheel force sensors at all, so they
# are kept and tagged instead of being quietly promoted to oracle.
COLUMN_NOTES = {
    "command_distance_remaining": "derived_from_truth",
    "command_heading_error": "derived_from_truth",
    "contact_force_front_left": "derived_from_truth",
    "contact_force_front_right": "derived_from_truth",
    "contact_force_mid_left": "derived_from_truth",
    "contact_force_mid_right": "derived_from_truth",
    "contact_force_rear_left": "derived_from_truth",
    "contact_force_rear_right": "derived_from_truth",
}

# The transmitter methods that make up one downlink tick, in order. transmit_fault_state is
# deliberately absent: fault telemetry is assembled straight from the injector so that it is
# identical in both capture modes.
TICK_SEQUENCE = (
    "transmit_active_command",
    "transmit_command_progress",
    "transmit_pose_of_base_link",
    "transmit_wheels_joint_angles",
    "transmit_motor_effort",
    "transmit_steer_encoder",
    "transmit_contact_forces",
    "transmit_imu_readings",
    "transmit_estimator",
    "transmit_power_info",
    "transmit_thermal_info",
    "transmit_radio_signal_info",
    "transmit_obc_metrics",
    "transmit_obc_state",
    "transmit_go_nogo",
    "transmit_solar_panel_state",
)

# Methods that take the downlink interval; the rest take nothing.
INTERVAL_METHODS = {"transmit_power_info", "transmit_thermal_info"}

IMU_FIELDS = ("ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw")


def _column_name(path: str) -> str:
    """/Rover/faults/wheel_health -> faults.wheel_health. Stable, and unambiguous across groups."""
    trimmed = path[len("/Rover/"):] if path.startswith("/Rover/") else path.lstrip("/")
    return trimmed.replace("/", ".")


def _flatten(name: str, value, into: Dict[str, object]) -> None:
    """Aggregates become name.key, arrays become name.0 - one scalar per column, recursively."""
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{name}.{key}", item, into)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten(f"{name}.{index}", item, into)
    elif hasattr(value, "tolist") and getattr(value, "shape", ()) != ():
        _flatten(name, value.tolist(), into)
    else:
        into[name] = value.item() if hasattr(value, "item") else value


def _is_oracle(path: str) -> bool:
    return path in ORACLE_PATHS or path.startswith(ORACLE_PATH_PREFIXES)


def _ensure_dir(path: str) -> None:
    """
    Create a directory the host user can also manage.

    The simulator container runs as root, so everything written here lands root-owned on the host
    and a plain `rm -rf` of an old dataset fails with permission denied. Making the directories
    world-writable is what lets the user delete their own data without sudo.
    """
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass


def _relax(path: str) -> None:
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass


def _write_csv(path: str, rows: List[Dict[str, object]], leading: List[str]) -> List[str]:
    """
    Write rows with the union of their columns as the header.

    A missing key is written as an empty cell, which is what a comms fault produces and what pandas
    reads back as NaN.
    """
    columns = list(leading)
    seen = set(columns)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _relax(path)

    return columns


class DatasetRecorder:
    """
    Collects one run's episodes into a dataset directory.

    Lifetime: begin_episode / many tick / end_episode, repeated, then finalize once.
    """

    def __init__(
        self,
        root_dir: str,
        fault_injector,
        robot=None,
        subsystems=None,
        transmitter=None,
        record_images: bool = True,
        min_free_gb: float = 2.0,
        max_disk_gb: Optional[float] = None,
    ):
        self._root = root_dir
        self._faults = fault_injector
        self._robot = robot
        self._subsystems = subsystems
        self._transmitter = transmitter
        self._record_images = record_images
        self._min_free_gb = float(min_free_gb)
        self._max_disk_gb = max_disk_gb

        _ensure_dir(self._root)
        episodes_dir = os.path.join(self._root, "episodes")
        _ensure_dir(episodes_dir)

        stale = [name for name in os.listdir(episodes_dir) if name.startswith("ep_")]
        if stale:
            # Episode indices restart at 0, so a previous run's episodes are not a continuation of
            # this one. Each episode directory is cleared as it is written, but a longer previous
            # run leaves orphans behind. manifest.json lists exactly what this run produced, which
            # is why a consumer should iterate episode_index rather than globbing the directory.
            print(f"[dataset] {len(stale)} episode directories already exist in {self._root}. "
                  f"They are not part of this run; read manifest.json's episode_index, or start "
                  f"from an empty directory.", flush=True)

        self._episode: Optional[Dict] = None
        # Column order, accumulated across every episode and registered BEFORE the comms-fault gate.
        # Without this a parameter that happened to be dropped on every tick of a heavily faulted
        # episode would have no column at all, and that episode's file could not be concatenated
        # with the others. Registering pre-gate makes the header the downlink surface rather than
        # whatever survived it.
        self._observable_columns: List[str] = []
        self._oracle_columns: List[str] = []
        self._episode_summaries: List[Dict] = []
        self._bytes_written = 0

    # ── episode lifetime ─────────────────────────────────────────────────────────
    def begin_episode(self, index: int, seed: int, schedule, meta: Optional[Dict] = None) -> str:
        directory = os.path.join(self._root, "episodes", f"ep_{index:04d}")
        # Clear it rather than write into it. A re-run writes fewer frames than a longer previous
        # run left behind, and the leftovers would sit in the directory looking like this episode's
        # data while appearing nowhere in images_index.csv.
        if os.path.isdir(directory):
            shutil.rmtree(directory, ignore_errors=True)
        _ensure_dir(os.path.join(directory, "observable"))
        _ensure_dir(os.path.join(directory, "oracle"))
        if self._record_images:
            _ensure_dir(os.path.join(directory, "observable", "images"))

        self._episode = {
            "index": index,
            "seed": seed,
            "dir": directory,
            "schedule": schedule,
            "meta": dict(meta or {}),
            "observable_rows": [],
            "oracle_rows": [],
            "commands": [],
            "fault_events": [],
            "images": [],
            "dropped_at_start": dict(self._faults.dropped_counts),
            "started_wall": time.time(),
            # Filled by the recording transmit_func between tick() boundaries.
            "pending_observable": {},
            "pending_oracle": {},
        }

        schedule_path = os.path.join(directory, "oracle", "schedule.json")
        with open(schedule_path, "w") as handle:
            json.dump(schedule.as_dict(), handle, indent=2)
        _relax(schedule_path)

        return directory

    def end_episode(self, outcome: str = "complete") -> Dict:
        """Write the episode out. Safe to call twice; the second call is a no-op."""
        if self._episode is None:
            return {}

        episode = self._episode
        self._episode = None
        directory = episode["dir"]

        _write_csv(
            os.path.join(directory, "observable", "telemetry.csv"),
            episode["observable_rows"],
            ["time_s", "step"] + self._observable_columns,
        )
        _write_csv(
            os.path.join(directory, "oracle", "truth.csv"),
            episode["oracle_rows"],
            ["time_s", "step"] + self._oracle_columns,
        )
        self._write_jsonl(os.path.join(directory, "observable", "commands.jsonl"), episode["commands"])
        self._write_jsonl(os.path.join(directory, "oracle", "faults.jsonl"), episode["fault_events"])

        if self._record_images:
            _write_csv(
                os.path.join(directory, "observable", "images_index.csv"),
                episode["images"],
                ["time_s", "step", "index", "filename", "status"],
            )

        schedule = episode["schedule"]
        dropped_now = dict(self._faults.dropped_counts)
        summary = {
            "index": episode["index"],
            "seed": episode["seed"],
            "outcome": outcome,
            "episode_class": schedule.episode_class,
            "is_nominal": schedule.is_nominal,
            "duration_s": round(episode["observable_rows"][-1]["time_s"], 3) if episode["observable_rows"] else 0.0,
            "downlink_ticks": len(episode["observable_rows"]),
            "commands": len(episode["commands"]),
            "fault_events": len(episode["fault_events"]),
            "images_saved": sum(1 for row in episode["images"] if row["status"] == "saved"),
            "images_lost": sum(1 for row in episode["images"] if row["status"] == "lost"),
            "faulted_fraction": round(schedule.faulted_fraction(), 6),
            "onset_steps": [fault.as_dict() for fault in schedule.faults],
            "crater": None if getattr(schedule, "crater", None) is None else schedule.crater.as_dict(),
            "dropped": {
                key: dropped_now.get(key, 0) - episode["dropped_at_start"].get(key, 0)
                for key in dropped_now
            },
            "wall_seconds": round(time.time() - episode["started_wall"], 1),
        }
        summary.update(episode["meta"])

        meta_path = os.path.join(directory, "meta.json")
        with open(meta_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        _relax(meta_path)

        self._episode_summaries.append(summary)
        self._bytes_written += self._directory_size(directory)

        return summary

    def discard_after(self, time_s: float) -> None:
        """
        Drop everything the open episode recorded after time_s, frames on disk included.

        For an episode that ends in a way its rows must not describe - a rover tipping over the edge of
        the terrain would otherwise go into the dataset as seconds of ordinary driving. The cut is
        written into meta.json so the shortened episode says why it is short.
        """
        if self._episode is None:
            return

        episode = self._episode
        keep = lambda row: row["time_s"] <= time_s  # noqa: E731
        for key in ("observable_rows", "oracle_rows", "commands", "fault_events"):
            episode[key] = [row for row in episode[key] if keep(row)]

        for row in episode["images"]:
            if not keep(row) and row["filename"]:
                path = os.path.join(episode["dir"], "observable", "images", row["filename"])
                if os.path.exists(path):
                    os.remove(path)
        episode["images"] = [row for row in episode["images"] if keep(row)]

        episode["pending_observable"] = {}
        episode["pending_oracle"] = {}
        episode["meta"]["discarded_after_s"] = round(float(time_s), 3)

    # ── the downlink tick ────────────────────────────────────────────────────────
    def transmit(self, param_name: str, param_value) -> None:
        """
        The recording transmit_func, and also the sink for tapping a live controller.

        Oracle paths are stored ungated; everything else is offered to the comms fault exactly as
        PerseveranceController.transmit_to_yamcs offers it, one draw per parameter. The column is
        registered either way - a dropped parameter is an empty cell, never an absent column.
        """
        if self._episode is None:
            return

        flat: Dict[str, object] = {}
        _flatten(_column_name(param_name), param_value, flat)

        if _is_oracle(param_name):
            self._register(self._oracle_columns, flat)
            self._episode["pending_oracle"].update(flat)
            return

        self._register(self._observable_columns, flat)
        if self._faults is not None and self._faults.should_drop_tm():
            return

        self._episode["pending_observable"].update(flat)

    @staticmethod
    def _register(columns: List[str], flat: Dict[str, object]) -> None:
        for name in flat:
            if name not in columns:
                columns.append(name)

    def update_episode_meta(self, **values) -> None:
        """Add to the current episode's meta.json, for facts only known once it has run."""
        if self._episode is not None:
            self._episode["meta"].update(values)

    def tick(self, sim_time_s: float, step: int, interval_s: float,
             extra_oracle: Optional[Dict[str, object]] = None) -> None:
        """
        Record one downlink tick.

        In own-transmitter mode this drives the transmitter; when a transmitter was not supplied the
        values are expected to have arrived through transmit() already, from a tapped controller.

        extra_oracle carries ground truth the injector does not know, such as the crater columns. A
        truthy oracle.crater.in_crater joins the label as the "crater" family.
        """
        if self._episode is None:
            return

        if self._transmitter is not None:
            self._drive_transmitter(interval_s)

        observable = self._episode["pending_observable"]
        oracle = self._episode["pending_oracle"]
        self._episode["pending_observable"] = {}
        self._episode["pending_oracle"] = {}

        stamp = {"time_s": round(float(sim_time_s), 3), "step": int(step)}
        observable.update(stamp)
        oracle.update(stamp)
        oracle.update(extra_oracle or {})
        oracle.update(self._oracle_fault_state(sim_time_s, extra_oracle or {}))
        oracle.update(self._oracle_imu())
        oracle.update(self._oracle_power())

        self._register(self._oracle_columns, oracle)
        self._episode["observable_rows"].append(observable)
        self._episode["oracle_rows"].append(oracle)

    def _drive_transmitter(self, interval_s: float) -> None:
        for name in TICK_SEQUENCE:
            method = getattr(self._transmitter, name, None)
            if method is None:
                continue
            try:
                method(interval_s) if name in INTERVAL_METHODS else method()
            except Exception as exc:  # a sensor that is not ready must not end the episode
                print(f"[dataset] {name} failed: {exc}", flush=True)

            if name == "transmit_imu_readings":
                # transmit_power_info and transmit_thermal_info each call get_imu_readings() again,
                # so the snapshot has to be taken here, before they overwrite it.
                self._episode["imu_snapshot"] = getattr(self._robot, "_last_imu", None)

    # ── oracle assembly ──────────────────────────────────────────────────────────
    def _oracle_fault_state(self, sim_time_s: float, extra_oracle: Optional[Dict] = None) -> Dict[str, object]:
        """
        The label, taken from the injector rather than from the schedule.

        The schedule says what was meant to happen; the injector says what did. They agree in a
        dataset run, but a fault set at startup with --fault, or one an operator sends over a live
        link, is in the injector and not in the schedule - and a label that missed those would be
        wrong in exactly the runs someone is most likely to inspect by hand. The scheduled view is
        kept alongside so the two can be checked against each other.
        """
        faults = self._faults
        schedule = self._episode["schedule"]
        scheduled = schedule.active_at(sim_time_s)

        nominal = HealthState.NOMINAL.value
        families = {
            "wheel": any(state != nominal for state in faults.wheel_health),
            "steer": any(state != nominal for state in faults.steer_health),
            "imu": faults.imu_health != nominal,
            "camera": faults.camera_health != nominal,
            "battery": faults.battery_health != nominal,
            "comms": faults.comms_health != nominal,
        }
        if extra_oracle and "oracle.crater.in_crater" in extra_oracle:
            # Not an injected fault: nothing on the rover is broken. It is still what a detector has
            # to flag, so it joins the label - from the true pose, never from the plan.
            families["crater"] = bool(extra_oracle["oracle.crater.in_crater"])
        live = sorted(name for name, faulted in families.items() if faulted)

        row: Dict[str, object] = {
            "oracle.fault_active": int(bool(live)),
            "oracle.n_active_faults": len(live),
            "oracle.fault_families": ";".join(live),
            # The scheduler's finer-grained names: steer_torque and steer_stuck both show up as the
            # steer family above (wheel_torque/stuck/slip/sink as wheel), and only the schedule can
            # tell them apart.
            "oracle.scheduled_kinds": ";".join(sorted(
                [fault.kind for fault in scheduled]
                + (["crater"] if getattr(schedule, "crater_planned_at", lambda _t: False)(sim_time_s) else [])
            )),
            "oracle.active_faults": faults.active_faults,
            "oracle.parasitic_load_w": faults.parasitic_load,
            "oracle.motor_controller_health": faults.motor_controller_health,
            "oracle.imu_health": faults.imu_health,
            "oracle.camera_health": faults.camera_health,
            "oracle.battery_health": faults.battery_health,
            "oracle.comms_health": faults.comms_health,
        }
        _flatten("oracle.wheel_torque_limit", faults.wheel_torque_limits, row)
        _flatten("oracle.wheel_damping_factor", faults.wheel_damping_factors, row)
        _flatten("oracle.wheel_friction", faults.wheel_frictions, row)
        _flatten("oracle.wheel_sinkage", faults.wheel_sinkages, row)
        _flatten("oracle.steer_torque_limit", faults.steer_torque_limits, row)
        _flatten("oracle.wheel_health", faults.wheel_health, row)
        _flatten("oracle.steer_health", faults.steer_health, row)

        start = self._episode["dropped_at_start"]
        for key, value in faults.dropped_counts.items():
            row[f"oracle.dropped.{key}"] = value - start.get(key, 0)

        return row

    def _oracle_imu(self) -> Dict[str, object]:
        """
        Clean reading and the corruption it picked up.

        Robot.get_imu_readings snapshots both halves, so the residual is exact rather than inferred -
        which is the point of recording it at all: it supervises on how wrong the sensor is, not on
        whether a label says it is broken.
        """
        snapshot = self._episode.get("imu_snapshot") or getattr(self._robot, "_last_imu", None)
        if not snapshot:
            return {}

        clean, measured = snapshot
        row: Dict[str, object] = {}
        for group_clean, group_measured in zip(clean, measured):
            for field in group_clean:
                row[f"oracle.imu_clean.{field}"] = group_clean[field]
                row[f"oracle.imu_error.{field}"] = group_measured[field] - group_clean[field]

        return row

    def _oracle_power(self) -> Dict[str, object]:
        """True battery state, ahead of the measurement noise the downlinked values carry."""
        model = getattr(self._subsystems, "_power_model", None)
        if model is None:
            return {}

        charge = getattr(model, "_battery_charge_wh", None)
        capacity = getattr(model, "_battery_capacity_wh", None)
        row: Dict[str, object] = {}
        if charge is not None:
            row["oracle.battery_charge_wh"] = charge
        if charge is not None and capacity:
            row["oracle.battery_charge_pct"] = 100.0 * charge / capacity

        return row

    # ── event logs ───────────────────────────────────────────────────────────────
    def log_command(self, sim_time_s: float, step: int, name: str, arguments: Dict,
                    delivered: bool = True, note: str = "") -> None:
        """
        One command as the ground would have logged it.

        delivered=False is a command the ground sent and the rover never executed - the observable
        shape of a tc_loss fault. It stays in the observable log on purpose: the ground station's
        own record is a mission product, and the discrepancy is exactly what a detector should see.
        """
        if self._episode is None:
            return

        self._episode["commands"].append({
            "time_s": round(float(sim_time_s), 3),
            "step": int(step),
            "command": name,
            "arguments": arguments,
            "delivered": bool(delivered),
            "note": note,
        })

    def log_status(self, sim_time_s: float, step: int, status: str, detail: str = "") -> None:
        if self._episode is None:
            return

        self._episode["commands"].append({
            "time_s": round(float(sim_time_s), 3),
            "step": int(step),
            "command": None,
            "status": status,
            "detail": detail,
        })

    def log_fault_event(self, sim_time_s: float, step: int, kind: str, targets, magnitudes) -> None:
        """Ground truth, called from FaultInjector._apply - the instant the fault took effect."""
        if self._episode is None:
            return

        self._episode["fault_events"].append({
            "time_s": round(float(sim_time_s), 3),
            "step": int(step),
            "kind": kind,
            "targets": list(targets) if targets else [],
            "magnitudes": [round(float(m), 6) for m in magnitudes],
        })

    def log_image(self, sim_time_s: float, step: int, index: int, filename: Optional[str]) -> None:
        if self._episode is None or not self._record_images:
            return

        self._episode["images"].append({
            "time_s": round(float(sim_time_s), 3),
            "step": int(step),
            "index": index,
            "filename": filename or "",
            "status": "saved" if filename else "lost",
        })

    def image_path(self, index: int) -> str:
        return os.path.join(self._episode["dir"], "observable", "images", f"{index:06d}.png")

    # ── disk guard ───────────────────────────────────────────────────────────────
    def disk_stop_reason(self) -> str:
        """Why the run should stop now, or '' to keep going."""
        free_gb = shutil.disk_usage(self._root).free / 1e9
        if free_gb < self._min_free_gb:
            return f"only {free_gb:.1f} GB free, below the {self._min_free_gb:.1f} GB floor"
        if self._max_disk_gb is not None and self._bytes_written / 1e9 >= float(self._max_disk_gb):
            return f"dataset reached {self._bytes_written / 1e9:.1f} GB, the configured budget"

        return ""

    # ── manifest ─────────────────────────────────────────────────────────────────
    def finalize(self, extra: Optional[Dict] = None) -> Dict:
        episodes = self._episode_summaries
        total_ticks = sum(item["downlink_ticks"] for item in episodes) or 1
        faulted_ticks = sum(
            item["downlink_ticks"] * item["faulted_fraction"] for item in episodes
        )
        classes: Dict[str, int] = {}
        for item in episodes:
            classes[item["episode_class"]] = classes.get(item["episode_class"], 0) + 1

        manifest = {
            "version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "episodes": len(episodes),
            "class_balance": {
                "episodes": {name: count / max(1, len(episodes)) for name, count in classes.items()},
                # What a per-timestep detector actually trains on. Much lower than the episode-level
                # number, because every faulted episode opens with a clean baseline.
                "sample_level_faulted_fraction": round(faulted_ticks / total_ticks, 6),
            },
            "columns": {
                "observable": self._observable_columns,
                "oracle": self._oracle_columns,
                "notes": COLUMN_NOTES,
            },
            "layout": {
                "observable/telemetry.csv": "one row per downlink tick; empty cell = lost to a comms fault",
                "observable/commands.jsonl": "commands as the ground logged them; delivered=false was lost in transit",
                "observable/images/": "downlinked nav-cam frames, possibly corrupted by a camera fault",
                "oracle/truth.csv": "clean sensors, true pose and battery, ground-truth health",
                "oracle/faults.jsonl": "the instant each fault took effect",
                "oracle/schedule.json": "the sampled schedule, including faults the episode never reached",
            },
            "caveats": [
                "Seeding covers the fault RNG and the measurement-noise models, but PhysX "
                "determinism is not configured, so episodes are reproducible in distribution "
                "rather than bit-exactly.",
                "Columns tagged derived_from_truth are downlinked by this rover but computed from "
                "ground truth; drop them for a strict observability setting.",
            ],
            "episode_index": episodes,
        }
        manifest.update(extra or {})

        manifest_path = os.path.join(self._root, "manifest.json")
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
        _relax(manifest_path)

        return manifest

    # ── helpers ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _write_jsonl(path: str, records: List[Dict]) -> None:
        _ensure_dir(os.path.dirname(path))
        with open(path, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        _relax(path)

    @staticmethod
    def _directory_size(directory: str) -> int:
        total = 0
        for root, _, files in os.walk(directory):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass

        return total
