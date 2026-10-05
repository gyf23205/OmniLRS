__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
The operator, replaced by a seeded random number generator.

An episode needs the rover to be doing something, and doing something different each time: a fault's
signature depends on what the rover was in the middle of when it hit. So this chains randomized
goto / drive_straight / drive_turn commands, one at a time, issuing the next when the last finishes.

Completion is read from PerseveranceDriveController.status, which decides on measured pose rather
than elapsed time - so a weakened wheel that makes a leg take twice as long simply makes it take
twice as long, instead of the mission moving on without it.

STRAIGHTS STAY ON THE TERRAIN. A goto is clamped into the bounds and pivots in place before it
drives, but a drive_straight goes wherever the rover happens to face. So its distance is cut at the
point where the current heading leaves the bounds, and a straight with no room for its minimum
length becomes a goto instead. Without a heading (a drive controller with no pose()) and with bounds
set, no drive_straight is issued at all.

COMMANDS GO THROUGH THE UPLINK GATE. Without a ground station there is no CommandsHandler and no
_uplinked wrapper, so a tc_loss fault would have nothing to drop and would be invisible in the
dataset. This calls fault_injector.should_drop_tc(name) itself, exactly as
PerseveranceController._uplinked does, and reports the command either way: the ground's log shows a
command it sent, and the rover never moves. That discrepancy is the observable form of the fault.

CRATER EPISODES. Given a CraterScenario, the operator has three phases:

    approach    goto and drive_turn only, and only gotos whose straight path stays outside the
                keep-out circle - drive_straight is never issued, because without the heading the
                scripter cannot tell where it ends
    dash        at dash_s the current command is stopped and one goto is issued: over the rim for a
                trapped episode, past the crater on a tangent for a skirt. An avoid episode has no
                dash and stays in approach
    escape      a trapped rover keeps being commanded - gotos back out, straights, turns - each
                abandoned with a stop after escape_timeout_s. Wheels turning against a wall while the
                pose goes nowhere is the observable form of being trapped

The plan is not the label. A skirt that clips the rim and falls in is recorded as in the crater,
because the recorder labels from the true pose.

HOUSEKEEPING. Rarely - each planned in about one episode in 20 - a free slot goes to a subsystem
command instead of a drive: power_electronics (a device OFF) or deploy_solar_panel (STOWED). Each is undone by the matching ON / DEPLOYED after a
sampled hold, sent even mid-drive and retried until delivered, so a lost restore means a longer
outage. While the motor controller is off the operator does not drive, as a real one would not.
The motor controller is never cycled in a crater episode: the dash would be rejected and the
trapped scenario would never happen. These draws come off their own rng, so the drive commands
an episode issues are the same sequence with housekeeping on or off, only shifted in time.

No omni/pxr imports - drive control and the injector are both reachable without them.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from src.mission_specific.perseverance.control.drive_controller import CommandStatus
from src.mission_specific.perseverance.dataset import crater as crater_module
from src.subsystems.device import CommonDevice, PowerState
from src.subsystems.robot_enums import SolarPanelState

# Once a command ends, in any way, the next one may be issued.
TERMINAL = (CommandStatus.IDLE, CommandStatus.COMPLETE, CommandStatus.REJECTED, CommandStatus.ABORTED)

DEFAULTS = {
    # Waypoints are drawn inside this radius of the episode's start, so the rover works a bounded
    # patch of terrain instead of wandering off the DEM.
    "waypoint_radius_m": 8.0,
    # A leg shorter than this is not worth commanding: the rover would arrive before a fault
    # injected mid-leg had time to show.
    "min_leg_m": 2.5,
    "command_weights": {"goto": 0.6, "drive_straight": 0.25, "drive_turn": 0.15},
    "straight_speed_range": [0.15, 0.35],
    "straight_distance_range": [2.0, 6.0],
    # A drive_straight is shortened so it ends this far inside the bounds, for the coast after the
    # stop and the drift of a leg that is never perfectly straight.
    "straight_end_margin_m": 0.5,
    "turn_rate_range_deg": [8.0, 20.0],
    "turn_angle_range_deg": [30.0, 180.0],
    # Simulation seconds to wait after a command ends before issuing the next, so the telemetry
    # shows the rover at rest between legs.
    "inter_command_pause_s": 2.0,
    # A rejected command means an interlock, not a finished leg; back off further before retrying.
    "reject_backoff_s": 5.0,
    # How many waypoint draws to make before accepting one closer than min_leg_m.
    "waypoint_attempts": 8,
    # [[xmin, xmax], [ymin, ymax]] in world metres, or None for unbounded. The launch script fills
    # this in from the environment size. Waypoints are clamped into it and every drive_straight is
    # cut short at it: run01 lost 13 of 20 episodes to straights that drove off the terrain edge.
    "bounds": None,
    # Chance, per episode, that each housekeeping command is planned: 0.05 is once every 20 episodes.
    # Decided per episode rather than per drive slot, so the rate does not move with leg length or
    # episode_steps. 0 turns that command off.
    "housekeeping_episode_probability": {"power_electronics": 0.05, "deploy_solar_panel": 0.05},
    # When in the episode a planned command goes out (at the first free slot after this time). The
    # upper end leaves room for the longest hold to be undone inside a 600 s episode.
    "housekeeping_window_s": [30.0, 420.0],
    # Devices power_electronics may switch off. RADIO, OBC and EPS are left out: the downlink keeps
    # being recorded regardless, so switching those off would describe a rover that could not exist.
    "power_off_devices": {"MOTOR_CONTROLLER": 0.6, "CAMERA": 0.4},
    "power_off_range_s": [15.0, 60.0],
    "stow_range_s": [30.0, 120.0],
}


def merged_config(config: Optional[Dict] = None) -> Dict:
    merged = {key: (dict(value) if isinstance(value, dict) else value) for key, value in DEFAULTS.items()}
    for key, value in dict(config or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        elif key in merged:
            merged[key] = value

    return merged


def _housekeeping_seed(seed: int) -> int:
    # Any fixed offset works; it only has to differ from the drive rng's seed.
    return int(seed) * 7919 + 1


class MissionScripter:
    """Issues one command at a time and reports what it issued, for the observable command log."""

    def __init__(self, drive_controller, fault_injector=None, config: Optional[Dict] = None,
                 seed: int = 0, origin: Tuple[float, float] = (0.0, 0.0),
                 crater_config: Optional[Dict] = None, subsystems=None):
        self._drive = drive_controller
        self._faults = fault_injector
        # None disables housekeeping: there is nothing to switch.
        self._subsystems = subsystems
        self._config = merged_config(config)
        self._crater_config = crater_module.merged_config(crater_config)
        self._rng = random.Random(seed)
        self._housekeeping_rng = random.Random(_housekeeping_seed(seed))
        self._origin = (float(origin[0]), float(origin[1]))
        # (due_s, name, arguments, handler, note) of the command that undoes the last housekeeping one
        self._pending_restore = None
        self._housekeeping_plan = self._plan_housekeeping()

        self._next_allowed_s = 0.0
        self._issued = 0
        self._lost = 0
        self._last_status: Optional[CommandStatus] = None
        self._crater = None
        self._crater_phase = "approach"
        self._command_started_s = 0.0

    # ── properties ───────────────────────────────────────────────────────────────
    @property
    def issued(self) -> int:
        return self._issued

    @property
    def lost(self) -> int:
        """Commands dropped by an injected tc_loss fault."""
        return self._lost

    @property
    def crater_phase(self) -> str:
        return self._crater_phase

    def reset(self, seed: int, origin: Tuple[float, float], crater=None) -> None:
        self._rng = random.Random(seed)
        self._housekeeping_rng = random.Random(_housekeeping_seed(seed))
        self._pending_restore = None
        self._housekeeping_plan = self._plan_housekeeping()
        self._origin = (float(origin[0]), float(origin[1]))
        self._next_allowed_s = 0.0
        self._issued = 0
        self._lost = 0
        self._last_status = None
        self._crater = crater
        self._crater_phase = "approach"
        self._command_started_s = 0.0

    # ── the per-step hook ────────────────────────────────────────────────────────
    def update(self, sim_time_s: float, position: Tuple[float, float]) -> Optional[Dict]:
        """
        Issue the next command if the rover is free to take one.

        Returns a record for the command log, or None if nothing was issued this step. The record is
        returned whether or not the command actually reached the rover.
        """
        status = self._drive.status
        if status != self._last_status:
            # A command that just ended sets the earliest time the next one may go out.
            if self._last_status == CommandStatus.EXECUTING or status in TERMINAL:
                pause = self._config["reject_backoff_s"] if status == CommandStatus.REJECTED \
                    else self._config["inter_command_pause_s"]
                self._next_allowed_s = max(self._next_allowed_s, sim_time_s + float(pause))
            self._last_status = status

        # A restore is due on time whatever the rover is doing - it does not wait for a leg to end.
        if self._pending_restore is not None and sim_time_s >= self._pending_restore[0]:
            _, name, arguments, handler, note = self._pending_restore
            record = self._send(name, arguments, handler, sim_time_s, note=note, drive=False)
            if record["delivered"]:
                self._pending_restore = None
            else:
                self._pending_restore = (sim_time_s + float(self._config["inter_command_pause_s"]),
                                         name, arguments, handler, note)
            return record

        if self._crater is not None:
            record = self._crater_update(sim_time_s, position, status)
            if record is not None or status == CommandStatus.EXECUTING:
                return record

        if status == CommandStatus.EXECUTING or sim_time_s < self._next_allowed_s:
            return None

        if self._motors_held():
            return None

        if self._housekeeping_due(sim_time_s):
            return self._issue_housekeeping(sim_time_s)

        return self._issue(sim_time_s, position)

    # ── crater episodes ──────────────────────────────────────────────────────────
    def _crater_update(self, sim_time_s: float, position: Tuple[float, float], status) -> Optional[Dict]:
        """
        Phase changes and timeouts for a crater episode. Returns a record when it issued something.

        Ordinary issuing (_issue -> _draw) still happens in update(); _draw reads the phase to decide
        which commands are allowed.
        """
        scenario = self._crater

        if self._crater_phase == "approach" and scenario.dash_s is not None and sim_time_s >= scenario.dash_s:
            if status == CommandStatus.EXECUTING:
                # Whatever leg was in flight gives way to the dash. Gated like any command, so a
                # lost stop means the dash waits for the next step's retry.
                record = self._send("stop", {}, self._drive.command_stop, sim_time_s,
                                    note="crater dash: cancelling the current leg")
                return record

            target = None
            if scenario.outcome == "trapped":
                target = crater_module.dash_target(scenario.spec, position)
                note = "crater dash: over the rim"
            else:
                target = crater_module.skirt_target(
                    scenario.spec, position, scenario.skirt_margin, scenario.skirt_overshoot,
                    side=self._rng.choice((-1.0, 1.0)),
                )
                note = "crater skirt: passing outside the rim"

            if target is None:
                # Already too close to skirt safely; treat it as an avoid episode from here on.
                self._crater_phase = "done"
                return None

            x, y = self._clamp(*target)
            self._crater_phase = "dash" if scenario.outcome == "trapped" else "done"
            return self._send("goto", {"x": round(x, 4), "y": round(y, 4)},
                              lambda: self._drive.command_goto(x, y), sim_time_s, note=note)

        if self._crater_phase == "dash" and status != CommandStatus.EXECUTING:
            self._crater_phase = "escape"

        # A trapped rover's commands never finish on their own; abandon them after the timeout.
        if (self._crater_phase in ("dash", "escape") and status == CommandStatus.EXECUTING
                and sim_time_s - self._command_started_s >= float(scenario.escape_timeout_s)):
            self._crater_phase = "escape"
            return self._send("stop", {}, self._drive.command_stop, sim_time_s,
                              note="no progress: abandoning the attempt")

        return None

    # ── issuing ──────────────────────────────────────────────────────────────────
    def _issue(self, sim_time_s: float, position: Tuple[float, float]) -> Dict:
        name, arguments, handler = self._draw(position)
        return self._send(name, arguments, handler, sim_time_s)

    def _send(self, name: str, arguments: Dict, handler, sim_time_s: float, note: str = "",
              drive: bool = True) -> Dict:
        """drive=False for subsystem commands, which must not restart the crater escape timeout."""
        self._issued += 1

        # The same gate the Yamcs uplink applies, so a tc_loss fault means the same thing here.
        if self._faults is not None and self._faults.should_drop_tc(name):
            self._lost += 1
            self._next_allowed_s = sim_time_s + float(self._config["inter_command_pause_s"])
            return {
                "command": name, "arguments": arguments, "delivered": False,
                "note": "lost in transit to the rover" + (f" ({note})" if note else ""),
            }

        handler()
        if drive:
            self._command_started_s = sim_time_s

        return {"command": name, "arguments": arguments, "delivered": True, "note": note}

    # ── housekeeping ─────────────────────────────────────────────────────────────
    def _motors_held(self) -> bool:
        """True while the operator has the motor controller off and is waiting to switch it back on."""
        return (self._pending_restore is not None
                and self._pending_restore[2].get("subsystem_id") == CommonDevice.MOTOR_CONTROLLER.value)

    def _plan_housekeeping(self) -> List[Tuple[float, str]]:
        """This episode's housekeeping commands as (earliest time, name), in time order."""
        rng = self._housekeeping_rng
        low, high = (float(v) for v in self._config["housekeeping_window_s"])
        plan = []
        for name, probability in self._config["housekeeping_episode_probability"].items():
            if rng.random() < float(probability):
                plan.append((rng.uniform(low, high), name))
        return sorted(plan)

    def _housekeeping_due(self, sim_time_s: float) -> bool:
        # One at a time: a second planned command waits until the first has been undone.
        return (self._subsystems is not None and self._pending_restore is None
                and bool(self._housekeeping_plan) and sim_time_s >= self._housekeeping_plan[0][0])

    def _issue_housekeeping(self, sim_time_s: float) -> Optional[Dict]:
        rng = self._housekeeping_rng
        _, choice = self._housekeeping_plan.pop(0)

        if choice == "power_electronics":
            devices = dict(self._config["power_off_devices"])
            if self._crater is not None:
                devices.pop(CommonDevice.MOTOR_CONTROLLER.value, None)
            devices = {name: weight for name, weight in devices.items() if float(weight) > 0}
            if not devices:
                return None

            device = rng.choices(list(devices), weights=[float(w) for w in devices.values()], k=1)[0]
            hold = rng.uniform(*[float(v) for v in self._config["power_off_range_s"]])
            record = self._send(
                "power_electronics", {"subsystem_id": device, "power_state": PowerState.OFF.value},
                lambda: self._set_power(device, PowerState.OFF), sim_time_s,
                note=f"housekeeping: {device} off for {hold:.0f} s", drive=False,
            )
            restore = ("power_electronics", {"subsystem_id": device, "power_state": PowerState.ON.value},
                       lambda: self._set_power(device, PowerState.ON), f"housekeeping: {device} back on")
        else:
            hold = rng.uniform(*[float(v) for v in self._config["stow_range_s"]])
            record = self._send(
                "deploy_solar_panel", {"deployment": SolarPanelState.STOWED.name},
                lambda: self._subsystems.set_solar_panel_state(SolarPanelState.STOWED), sim_time_s,
                note=f"housekeeping: panel stowed for {hold:.0f} s", drive=False,
            )
            restore = ("deploy_solar_panel", {"deployment": SolarPanelState.DEPLOYED.name},
                       lambda: self._subsystems.set_solar_panel_state(SolarPanelState.DEPLOYED),
                       "housekeeping: panel redeployed")

        # A lost command changed nothing, so there is nothing to undo.
        if record["delivered"]:
            self._pending_restore = (sim_time_s + hold, *restore)
        # Leave the rover at rest a moment either side of the switch, as between legs.
        self._next_allowed_s = sim_time_s + float(self._config["inter_command_pause_s"])
        return record

    def _set_power(self, device_name: str, state: PowerState) -> None:
        """What PerseveranceCommander.handle_electronics_on_off does, minus the Yamcs acknowledgement."""
        device = CommonDevice(device_name)
        self._subsystems.set_device_power_state(device, state)
        if device == CommonDevice.MOTOR_CONTROLLER and state == PowerState.OFF:
            self._drive.abort("motor controller powered off")

    def _draw(self, position: Tuple[float, float]):
        weights = dict(self._config["command_weights"])
        keeping_out = self._crater is not None and self._crater_phase in ("approach", "done")
        if keeping_out or (self._config.get("bounds") and self._heading() is None):
            # Without the heading the end of a drive_straight is unknown, so it could run into the
            # crater or off the terrain. gotos are checked against the keep-out circle and clamped
            # into the bounds; turns in place are always safe.
            weights.pop("drive_straight", None)
        names = [name for name, weight in weights.items() if weight > 0]
        choice = self._rng.choices(names, weights=[float(weights[name]) for name in names], k=1)[0]

        if choice == "drive_straight":
            speed = self._uniform("straight_speed_range")
            distance = min(self._uniform("straight_distance_range"), self._room_ahead(position))
            if distance >= float(self._config["straight_distance_range"][0]):
                return ("drive_straight", {"linear_velocity": round(speed, 4), "distance": round(distance, 4)},
                        lambda: self._drive.command_straight(speed, distance))
            # Facing the edge: drive back into the patch instead.

        if choice == "drive_turn":
            rate = self._uniform("turn_rate_range_deg") * self._rng.choice((-1.0, 1.0))
            angle = self._uniform("turn_angle_range_deg")
            return ("drive_turn", {"angular_velocity": round(rate, 4), "angle": round(angle, 4)},
                    lambda: self._drive.command_turn(rate, angle))

        if keeping_out:
            waypoint = self._clear_waypoint(position)
            if waypoint is None:
                rate = self._uniform("turn_rate_range_deg") * self._rng.choice((-1.0, 1.0))
                angle = self._uniform("turn_angle_range_deg")
                return ("drive_turn", {"angular_velocity": round(rate, 4), "angle": round(angle, 4)},
                        lambda: self._drive.command_turn(rate, angle))
            x, y = waypoint
        else:
            x, y = self._waypoint(position)
        return ("goto", {"x": round(x, 4), "y": round(y, 4)}, lambda: self._drive.command_goto(x, y))

    def _clear_waypoint(self, position: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """A waypoint whose straight path from here stays outside the crater's keep-out circle."""
        spec = self._crater.spec
        radius = crater_module.keep_out_radius(spec, self._crater_config)
        for _ in range(int(self._config["waypoint_attempts"]) * 4):
            candidate = self._waypoint(position)
            if crater_module.path_clear(spec, position, candidate, radius) and \
                    math.hypot(candidate[0] - spec.center_x, candidate[1] - spec.center_y) >= radius:
                return candidate
        return None

    def _waypoint(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """A point inside the episode's patch, preferring one far enough to be worth driving to."""
        radius = float(self._config["waypoint_radius_m"])
        min_leg = float(self._config["min_leg_m"])
        fallback = None

        for _ in range(int(self._config["waypoint_attempts"])):
            angle = self._rng.uniform(-math.pi, math.pi)
            # sqrt keeps the draw uniform over the disc rather than clustered at the centre.
            distance = radius * math.sqrt(self._rng.random())
            x, y = self._clamp(self._origin[0] + distance * math.cos(angle),
                               self._origin[1] + distance * math.sin(angle))
            fallback = fallback or (x, y)
            if math.hypot(x - position[0], y - position[1]) >= min_leg:
                return x, y

        return fallback

    def _heading(self) -> Optional[float]:
        pose = getattr(self._drive, "pose", None)
        return None if pose is None else float(pose()[2])

    def _room_ahead(self, position: Tuple[float, float]) -> float:
        """How far the rover can drive along its current heading and still stop inside the bounds."""
        bounds = self._config.get("bounds")
        if not bounds:
            return math.inf

        heading = self._heading()
        direction = (math.cos(heading), math.sin(heading))
        room = math.inf
        for start, step, (low, high) in zip(position, direction, bounds):
            if step > 1e-9:
                room = min(room, (float(high) - start) / step)
            elif step < -1e-9:
                room = min(room, (float(low) - start) / step)

        # Negative when the rover is already outside the bounds: no room at all.
        return max(0.0, room - float(self._config["straight_end_margin_m"]))

    def _clamp(self, x: float, y: float) -> Tuple[float, float]:
        bounds = self._config.get("bounds")
        if not bounds:
            return x, y

        (x_min, x_max), (y_min, y_max) = bounds
        return (min(max(x, float(x_min)), float(x_max)),
                min(max(y, float(y_min)), float(y_max)))

    def _uniform(self, key: str) -> float:
        low, high = self._config[key]
        return self._rng.uniform(float(low), float(high))
