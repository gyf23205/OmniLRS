__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
Onboard closed-loop drive controller for the perseverance rover.

This is the rover's flight software, not ground software. Commands arriving from Yamcs are goals
("drive 2 m", "go to x,y"); this converts them into wheel motion by running at the physics rate and
steering on measured pose. The ground link is 1 Hz and light-time-delayed on a real mission, so it
cannot close this loop — which is exactly why rovers are commanded with goals rather than joint
targets.

Closed-loop on ground-truth pose rather than dead reckoning: wheel slip on regolith and pitch
changes on slopes make a time-based "distance = velocity x duration" estimate drift badly, and
goto() needs absolute position feedback to mean anything at all.

Deliberately does not use src/tmtc/drive_handler.py: that is open-loop (schedules a stop after
distance/velocity seconds) and skid-steer. It stays as-is for pragyaan; only its interlock
condition is mirrored here, in _check_interlocks.
"""

import math
from enum import Enum
from typing import Optional, Tuple

from src.mission_specific.perseverance.control.ackermann_model import AckermannModel
from src.subsystems.device import CommonDevice, HealthState, PowerState
from src.subsystems.robot_enums import GoNogoState, ObcState


class CommandStatus(Enum):
    """Mirrors the /Rover/command_status enum in cfg/mdb/perseverance.xml."""

    IDLE = 0
    EXECUTING = 1
    COMPLETE = 2
    REJECTED = 3
    ABORTED = 4


class _Phase(Enum):
    NONE = 0
    TURN_IN_PLACE = 1
    DRIVE_STRAIGHT = 2
    DRIVE_TO_TARGET = 3


def _wrap_angle(angle: float) -> float:
    """Wrap radians into (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class PerseveranceDriveController:
    """
    Executes one high-level drive command at a time, closed-loop on measured pose.

    update() must be called every physics step. It reads the rover pose, decides how far there is
    left to go, asks the Ackermann model for wheel steer angles and speeds, and writes them to the
    joints. Command completion is judged on measured distance/heading, never on elapsed time.

    Progress fields (status, distance_remaining, heading_error, target) are read by
    PerseveranceTransmitter and downlinked at 1 Hz.
    """

    def __init__(self, robot, robot_RG, ackermann: AckermannModel, config: dict = None):
        config = config or {}
        self._robot = robot
        self._robot_RG = robot_RG
        self._ackermann = ackermann

        # Tolerances. Position tolerance is generous because the rover is ~1 m long and lunar
        # regolith slip makes tighter targets meaningless.
        self._position_tolerance = float(config.get("position_tolerance", 0.15))       # m
        self._heading_tolerance = math.radians(float(config.get("heading_tolerance_deg", 2.0)))
        self._distance_tolerance = float(config.get("distance_tolerance", 0.05))       # m
        # Heading error beyond which goto turns in place first: no arc reaches a target behind you.
        self._turn_first_threshold = math.radians(float(config.get("turn_first_threshold_deg", 90.0)))
        # Proportional steering gain, curvature (1/m) per radian of heading error.
        self._steering_gain = float(config.get("steering_gain", 1.2))
        self._max_curvature = float(config.get("max_curvature", 1.0))                  # 1/m
        self._default_speed = float(config.get("default_speed", 0.3))                  # m/s
        self._default_yaw_rate = math.radians(float(config.get("default_yaw_rate_deg", 15.0)))
        # Sign conventions resolved by the --calibrate pass; see cfg/robot/perseverance.yaml.
        self._forward_sign = float(config.get("forward_axis_sign", 1.0))
        self._steer_sign = float(config.get("steer_sign", 1.0))

        self._status = CommandStatus.IDLE
        self._phase = _Phase.NONE
        self._reject_reason = ""
        self._label = "none"

        self._start_position: Optional[Tuple[float, float]] = None
        self._start_yaw: float = 0.0
        self._last_yaw: float = 0.0
        self._turned: float = 0.0
        self._target_position: Optional[Tuple[float, float]] = None
        self._target_distance: float = 0.0
        # None means "turn until the goto bearing is acquired"; a float means a fixed angle.
        self._target_yaw_delta: Optional[float] = None
        self._speed: float = 0.0
        self._yaw_rate: float = 0.0

        self._distance_remaining: float = 0.0
        self._heading_error: float = 0.0

    # ── telemetry accessors ──────────────────────────────────────────────────────
    @property
    def status(self) -> CommandStatus:
        return self._status

    @property
    def distance_remaining(self) -> float:
        return self._distance_remaining

    @property
    def heading_error_deg(self) -> float:
        return math.degrees(self._heading_error)

    @property
    def target(self) -> Tuple[float, float]:
        return self._target_position if self._target_position is not None else (0.0, 0.0)

    @property
    def is_executing(self) -> bool:
        return self._status == CommandStatus.EXECUTING

    # ── pose ─────────────────────────────────────────────────────────────────────
    def _pose(self) -> Tuple[float, float, float]:
        """Rover (x, y, yaw) in world coordinates. yaw is the heading of the forward axis."""
        position, orientation = self._robot_RG.get_pose_of_base_link()
        w, x, y, z = (float(v) for v in orientation)

        # Yaw of the body +Y axis (the rover's forward axis) expressed in the world frame.
        # Rotating (0,1,0) by the quaternion gives (2(xy - wz), 1 - 2(x^2 + z^2), 2(yz + wx)).
        # forward_axis_sign flips the *vector* when +Y turns out to point aft; negating the
        # resulting angle instead would mirror the heading rather than reverse it.
        forward_x = 2.0 * (x * y - w * z) * self._forward_sign
        forward_y = (1.0 - 2.0 * (x * x + z * z)) * self._forward_sign
        yaw = math.atan2(forward_y, forward_x)

        return float(position[0]), float(position[1]), yaw

    # ── interlocks ───────────────────────────────────────────────────────────────
    def _check_interlocks(self) -> Optional[str]:
        """
        Returns a rejection reason, or None when the rover is cleared to drive.

        Same three conditions as DriveHandler._is_robot_able_to_drive, but reported rather than
        swallowed: an operator on the ground has no way to tell "driving" from "silently refused"
        unless the reason is downlinked.
        """
        subsystems = self._robot.subsystems

        if subsystems.get_go_nogo_state() != GoNogoState.GO:
            return "go_nogo is NOGO"
        if subsystems.get_device_power_state(CommonDevice.MOTOR_CONTROLLER) != PowerState.ON:
            return "motor controller is OFF"
        if subsystems.get_device_health_state(CommonDevice.MOTOR_CONTROLLER) != HealthState.NOMINAL:
            return "motor controller health is FAULT"
        return None

    def _begin(self, label: str) -> bool:
        reason = self._check_interlocks()
        if reason is not None:
            self._status = CommandStatus.REJECTED
            self._reject_reason = reason
            self._label = label
            self._halt()
            print(f"[ctrl] {label} REJECTED: {reason}", flush=True)
            return False

        x, y, yaw = self._pose()
        self._start_position = (x, y)
        self._start_yaw = yaw
        self._last_yaw = yaw
        self._turned = 0.0
        self._label = label
        self._status = CommandStatus.EXECUTING
        self._reject_reason = ""
        self._robot.subsystems.set_obc_state(ObcState.MOTOR)
        return True

    # ── commands ─────────────────────────────────────────────────────────────────
    def command_straight(self, linear_velocity: float, distance: float) -> None:
        if linear_velocity == 0.0 or distance == 0.0:
            self.command_stop()
            return
        if not self._begin(f"drive_straight({linear_velocity}, {distance})"):
            return

        self._phase = _Phase.DRIVE_STRAIGHT
        self._speed = math.copysign(abs(linear_velocity), distance)
        self._target_distance = abs(distance)
        self._target_position = None
        self._distance_remaining = self._target_distance

    def command_turn(self, angular_velocity_deg: float, angle_deg: float) -> None:
        if angular_velocity_deg == 0.0 or angle_deg == 0.0:
            self.command_stop()
            return
        if not self._begin(f"drive_turn({angular_velocity_deg}, {angle_deg})"):
            return

        self._phase = _Phase.TURN_IN_PLACE
        self._yaw_rate = math.copysign(math.radians(abs(angular_velocity_deg)), angle_deg)
        self._target_yaw_delta = math.radians(abs(angle_deg))
        self._target_position = None
        self._heading_error = self._target_yaw_delta

    def command_goto(self, x: float, y: float) -> None:
        if not self._begin(f"goto({x}, {y})"):
            return

        self._target_position = (float(x), float(y))
        self._speed = self._default_speed
        self._yaw_rate = self._default_yaw_rate

        # Only pivot first when the target is far enough off-heading that no arc would reach it.
        # Otherwise steer onto it continuously, which is the point of having the corner wheels.
        _, _, yaw = self._pose()
        bearing = math.atan2(y - self._start_position[1], x - self._start_position[0])
        if abs(_wrap_angle(bearing - yaw)) > self._turn_first_threshold:
            self._phase = _Phase.TURN_IN_PLACE
            self._target_yaw_delta = None    # turn until the bearing is acquired, not a fixed angle
        else:
            self._phase = _Phase.DRIVE_TO_TARGET

    def command_stop(self) -> None:
        was_executing = self._status == CommandStatus.EXECUTING
        self._halt()
        self._status = CommandStatus.COMPLETE if was_executing else CommandStatus.IDLE
        self._label = "stop"
        print("[ctrl] stop", flush=True)

    def abort(self, reason: str = "") -> None:
        """Cancel any command in flight. Used by manual takeover, NOGO and motor power-off."""
        if self._status != CommandStatus.EXECUTING:
            return
        self._halt()
        self._status = CommandStatus.ABORTED
        print(f"[ctrl] {self._label} ABORTED{': ' + reason if reason else ''}", flush=True)

    def manual(self, linear_velocity: float, curvature: float, point_turn_rate: float = 0.0) -> bool:
        """
        Direct operator control from the keyboard.

        Called every physics step, including when no key is held, so it must distinguish "operator
        is steering" from "operator is idle". Only actual input takes over: an idle call while a
        command is running is ignored so the command keeps driving, and an idle call otherwise
        leaves the wheels stopped.

        Returns True when manual control wrote to the joints this step, so the caller knows not to
        also run the command update.
        """
        has_input = abs(linear_velocity) > 1e-6 or abs(point_turn_rate) > 1e-6

        if not has_input:
            if self._status == CommandStatus.EXECUTING:
                return False        # let the active command keep driving
            self._apply(*self._ackermann.stop())
            self._robot.subsystems.set_obc_state(ObcState.IDLE)
            return True

        # Real input: take over, cancelling anything in flight so the two never fight over the
        # same joint targets.
        if self._status == CommandStatus.EXECUTING:
            self.abort("manual input")

        if abs(linear_velocity) < 1e-6:
            steer, speeds = self._ackermann.solve_point_turn(point_turn_rate)
        else:
            steer, speeds = self._ackermann.solve(linear_velocity, curvature)

        self._apply(steer, speeds)
        self._robot.subsystems.set_obc_state(ObcState.MOTOR)
        return True

    # ── per-step update ──────────────────────────────────────────────────────────
    def update(self) -> None:
        """Advance the active command. Call once per physics step."""
        if self._status != CommandStatus.EXECUTING:
            return

        # An interlock can drop mid-drive (operator sends NOGO, or a fault is injected).
        reason = self._check_interlocks()
        if reason is not None:
            self._halt()
            self._status = CommandStatus.REJECTED
            self._reject_reason = reason
            print(f"[ctrl] {self._label} halted mid-drive: {reason}", flush=True)
            return

        x, y, yaw = self._pose()

        if self._phase == _Phase.TURN_IN_PLACE:
            self._update_turn(x, y, yaw)
        elif self._phase == _Phase.DRIVE_STRAIGHT:
            self._update_straight(x, y)
        elif self._phase == _Phase.DRIVE_TO_TARGET:
            self._update_goto(x, y, yaw)

    def _update_turn(self, x: float, y: float, yaw: float) -> None:
        if self._target_yaw_delta is None:
            # goto pre-turn: rotate until the rover points at the target.
            bearing = math.atan2(self._target_position[1] - y, self._target_position[0] - x)
            error = _wrap_angle(bearing - yaw)
            self._heading_error = error
            self._distance_remaining = math.hypot(
                self._target_position[0] - x, self._target_position[1] - y
            )
            if abs(error) <= self._heading_tolerance:
                self._phase = _Phase.DRIVE_TO_TARGET
                return
            rate = math.copysign(self._default_yaw_rate, error)
        else:
            # drive_turn: rotate by a fixed angle from the starting heading.
            # Accumulate per-step deltas rather than comparing against the start heading: a single
            # wrapped difference folds at +/-180 degrees, so a commanded 270 degree turn would
            # otherwise report "done" after 90.
            self._turned += abs(_wrap_angle(yaw - self._last_yaw))
            self._last_yaw = yaw
            remaining = self._target_yaw_delta - self._turned
            self._heading_error = remaining
            self._distance_remaining = 0.0
            if remaining <= self._heading_tolerance:
                self._finish()
                return
            rate = self._yaw_rate

        steer, speeds = self._ackermann.solve_point_turn(rate)
        self._apply(steer, speeds)

    def _update_straight(self, x: float, y: float) -> None:
        travelled = math.hypot(x - self._start_position[0], y - self._start_position[1])
        remaining = self._target_distance - travelled
        self._distance_remaining = max(0.0, remaining)
        self._heading_error = 0.0

        if remaining <= self._distance_tolerance:
            self._finish()
            return

        steer, speeds = self._ackermann.solve(self._speed, 0.0)
        self._apply(steer, speeds)

    def _update_goto(self, x: float, y: float, yaw: float) -> None:
        target_x, target_y = self._target_position
        remaining = math.hypot(target_x - x, target_y - y)
        self._distance_remaining = remaining

        if remaining <= self._position_tolerance:
            self._finish()
            return

        bearing = math.atan2(target_y - y, target_x - x)
        error = _wrap_angle(bearing - yaw)
        self._heading_error = error

        # If the rover has been pushed badly off-heading (slip on a slope, an obstacle), pivot
        # again rather than driving a long arc back.
        if abs(error) > self._turn_first_threshold:
            self._phase = _Phase.TURN_IN_PLACE
            self._target_yaw_delta = None
            return

        curvature = max(-self._max_curvature, min(self._max_curvature, self._steering_gain * error))
        steer, speeds = self._ackermann.solve(self._speed, curvature)
        self._apply(steer, speeds)

    # ── plumbing ─────────────────────────────────────────────────────────────────
    def _apply(self, steer_angles: dict, wheel_speeds: dict) -> None:
        if self._steer_sign != 1.0:
            steer_angles = {name: angle * self._steer_sign for name, angle in steer_angles.items()}
        self._robot.set_steer_angles(steer_angles)
        self._robot.set_wheel_velocities(wheel_speeds)

    def _halt(self) -> None:
        steer, speeds = self._ackermann.stop()
        self._apply(steer, speeds)
        self._phase = _Phase.NONE
        self._distance_remaining = 0.0
        self._heading_error = 0.0
        self._robot.subsystems.set_obc_state(ObcState.IDLE)

    def _finish(self) -> None:
        self._halt()
        self._status = CommandStatus.COMPLETE
        print(f"[ctrl] {self._label} COMPLETE", flush=True)
