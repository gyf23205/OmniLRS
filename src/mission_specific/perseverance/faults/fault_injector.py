__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
Fault injection for the perseverance rover.

This is not rover software and it is not ground software: it is a simulation backdoor that happens
to ride the ground-control transport, because that transport is already built and the Yamcs command
form is a convenient interface. Everything it touches is reported under /Rover/faults so an
injected value can never be mistaken for a real rover reading.

Two families, four injection points:

    ACTUATION   wheel torque, wheel stuck, steer torque,   -> joint drive limits, damping
                steer stuck
    TRACTION    wheel slip, wheel sink                     -> wheel physics materials,
                                                              collider offsets, per-step drag
    SENSING     imu bias/noise, camera loss/noise          -> the sensor accessors
    POWER       parasitic load                             -> the power model
    LINK        telemetry loss, telecommand loss           -> the tm/tc chokepoints

Every magnitude is a severity in [0, 1], healthy to dead, scaling a reference from the robot
config. One rule for every fault, so an operator never has to know sensible units per subsystem.

Health is reported from ground truth. The injector knows exactly what it broke, so it says so:
each actuator gets its own NOMINAL / DEGRADED / FAULT state, and the affected device carries a
coarse summary. Nothing is inferred from dynamics - there is no detection logic here and none is
pretended.

DEGRADED is not an interlock. A weakened actuator has to keep driving, because watching the closed
loop fight it is the entire point; the rover refusing to move would hide exactly what a fault is
worth showing. So an injected fault never reaches HealthState.FAULT on a device - that stays
reserved for the hardware itself being dead, which power_electronics(OFF) already covers.

THREADING. Commands arrive on the CommandsHandler listener thread, and applying an actuator fault
ends in a usd write. Authoring usd off the simulation thread deadlocks against Kit - the simulation
freezes mid-command, which also stops the downlink, so the ground keeps displaying the last healthy
telemetry it got and the fault looks like it was never received. So the injection methods only
queue; update(), called from the simulation loop, is the only thing that applies anything. The
sensor and link faults would survive being applied in place, but they go through the same queue so
there is exactly one path from command to effect.

DETERMINISM. Noise and packet loss need randomness, so the injector owns one seeded stream and
hands it to the robot for sensor corruption. A faulted run replays identically given the same seed.

No omni/pxr imports here on purpose - the rover is reached only through Robot methods, so this
module is testable with plain python3, like control/ackermann_model.py.
"""

import math
import random
import threading
from typing import Dict, List, Tuple

from src.subsystems.device import CommonDevice, HealthState

# Order matches PerseveranceTransmitter.WHEEL_NAMES and the target_links list in the robot config,
# so the telemetry arrays line up with the contact-force parameters.
WHEELS = ("front_left", "front_right", "mid_left", "mid_right", "rear_left", "rear_right")
# The two mid wheels are not steerable; see the steer_joints list in cfg/robot/perseverance.yaml.
CORNERS = ("front_left", "front_right", "rear_left", "rear_right")

ALL = "ALL"

# The usd authors no maxForce on any joint, so an unfaulted joint is unlimited. Restoring means
# writing this back, not writing the configured nominal - otherwise injecting and clearing a fault
# would quietly leave the rover weaker than it started.
UNLIMITED = float("inf")

# Reference magnitudes, overridable from cfg/robot/perseverance.yaml. Each is what severity 1.0
# means for that fault. They size a fault and nothing else: a healthy subsystem never sees them.
DEFAULTS = {
    "nominal_wheel_torque": 20.0,          # N*m at the drive joint
    "nominal_steer_torque": 150.0,         # N*m at the steer joint
    "max_wheel_stuck_damping_factor": 1000.0,  # drive damping multiplier a seized wheel tops out at
    "nominal_wheel_friction": 0.5,         # wheel-ground friction of a healthy wheel (PhysX default)
    "max_wheel_sinkage_m": 0.03,           # how far a wheel settles into the ground at severity 1.0
    "max_sink_resistance_coeff": 0.6,      # drag per unit normal load at severity 1.0
    "sink_resistance_velocity_eps": 0.05,  # m/s below which the drag fades to zero
    "max_imu_accel_bias": 2.0,             # m/s^2 offset
    "max_imu_gyro_bias": 0.5,              # rad/s offset
    "max_imu_orientation_bias_deg": 20.0,  # deg offset on roll/pitch/yaw
    "max_imu_accel_noise": 1.0,            # m/s^2, one sigma
    "max_imu_gyro_noise": 0.2,             # rad/s, one sigma
    "max_camera_noise": 80.0,              # counts of 255, one sigma
    "max_parasitic_load_w": 40.0,          # W drawn straight off the battery
}


def _clamp_severity(severity: float) -> float:
    return max(0.0, min(1.0, float(severity)))


class FaultInjector:
    """
    Registry of injected faults, and the bookkeeping that undoes them.

    Magnitudes are fractions of a configured reference rather than absolute units. For the joints
    that is forced on us - the usd leaves maxForce unauthored, so nominal is infinite and there is
    no absolute scale to be a fraction of. For the rest it is a choice: one convention across every
    fault is worth more than per-sensor precision, and the references live in one config block.
    """

    # Never dropped by a comms fault; see should_drop_tc.
    RECOVERY_COMMAND = "clear_faults"

    def __init__(self, robot, config: Dict = None, seed: int = 0, on_apply=None):
        config = dict(config or {})
        self._robot = robot
        # Called from _apply, on the simulation thread, the instant a fault takes effect. That is
        # the only moment worth labelling: the dataset recorder uses it as ground truth, and it is
        # necessarily later than the moment the command was received.
        self._on_apply = on_apply
        self._ref = {name: float(config.get(name, default)) for name, default in DEFAULTS.items()}

        self._rng = random.Random(seed)
        # The robot draws its own sensor noise; seed it from here so one seed replays the whole run.
        if hasattr(robot, "set_fault_seed"):
            robot.set_fault_seed(seed)

        # Only faulted things appear in the per-target registries. Absence means healthy, which is
        # what keeps the "clear is an exact round trip" property easy to hold onto.
        self._wheel_torque: Dict[str, float] = {}   # wheel  -> severity
        self._wheel_stuck: Dict[str, float] = {}    # wheel  -> severity
        self._wheel_slip: Dict[str, float] = {}     # wheel  -> severity
        self._wheel_sink: Dict[str, float] = {}     # wheel  -> severity
        self._steer_torque: Dict[str, float] = {}   # corner -> severity
        self._steer_stuck: Dict[str, float] = {}    # corner -> angle, degrees

        # Whole-subsystem faults: a single severity each, 0.0 meaning healthy.
        self._imu_bias = 0.0
        self._imu_noise = 0.0
        self._camera_loss = 0.0
        self._camera_noise = 0.0
        self._battery = 0.0
        self._tm_loss = 0.0
        self._tc_loss = 0.0

        # Cumulative for the whole run, deliberately NOT reset by clear_faults: a counter that only
        # ever rises is the one an operator can plot without wondering where the steps came from.
        self._dropped = {"tm": 0, "tc": 0, "camera_frames": 0}

        # Requests arriving from the telecommand thread, drained by update() on the simulation
        # thread. See the module docstring for why nothing may be applied where it arrives.
        self._pending: List[tuple] = []
        self._lock = threading.Lock()
        # The last per-step drag error, so a broken physics call is reported once rather than 30
        # times a second. See update().
        self._resistance_error = None

        self._appliers = {
            "clear": self._apply_clear,
            "wheel_torque": self._apply_wheel_torque,
            "wheel_stuck": self._apply_wheel_stuck,
            "wheel_slip": self._apply_wheel_slip,
            "wheel_sink": self._apply_wheel_sink,
            "steer_torque": self._apply_steer_torque,
            "steer_stuck": self._apply_steer_stuck,
            "imu": self._apply_imu,
            "camera": self._apply_camera,
            "battery": self._apply_battery,
            "comms": self._apply_comms,
        }

    @staticmethod
    def prepare_robot(robot, config: Dict = None) -> None:
        """
        Stage setup some faults need before physics first steps. Call right after loading the robot.

        Wheel slip changes friction on a per-wheel physics material, and wheel sink changes each
        wheel collider's rest offset. Both have to be authored when PhysX parses the rover, so that a
        fault only ever changes a value - see Robot.install_wheel_materials. This is the only thing
        here that is not queued: it runs on the simulation thread, before the simulation runs.
        """
        config = dict(config or {})
        robot.install_wheel_materials(
            float(config.get("nominal_wheel_friction", DEFAULTS["nominal_wheel_friction"]))
        )
        robot.install_wheel_sinkage()

    def set_on_apply(self, callback) -> None:
        """Register the _apply hook after construction, for a recorder built later than this."""
        self._on_apply = callback

    # ── injection ────────────────────────────────────────────────────────────────
    # These only queue. They are called from the telecommand listener thread, where an actuator
    # fault's usd write would deadlock Kit.

    def inject_wheel_torque(self, wheel: str, severity: float) -> None:
        """Weaken one drive motor, or ALL of them. severity 0 restores it, 1 kills it."""
        targets = self._resolve(wheel, WHEELS, "wheel")
        if targets is not None:
            self._enqueue("wheel_torque", targets, _clamp_severity(severity))

    def inject_wheel_stuck(self, wheel: str, severity: float) -> None:
        """
        Seize one wheel, or ALL of them, by raising its joint damping. severity 0 restores it.

        Distinct from a weak motor: a torque-limited wheel is dragged along freely by the other five,
        while a seized one actively resists turning, so the rover pulls toward it and the wheel
        skids. severity is the fraction of free-running speed lost - the damping rises as
        1 / (1 - severity), capped at max_wheel_stuck_damping_factor, which is what 1.0 reaches.
        """
        targets = self._resolve(wheel, WHEELS, "wheel")
        if targets is not None:
            self._enqueue("wheel_stuck", targets, _clamp_severity(severity))

    def inject_wheel_slip(self, wheel: str, severity: float) -> None:
        """
        Take the grip away from one wheel, or ALL of them. severity 0 restores it.

        Friction becomes (1 - severity) * nominal_wheel_friction, so the motor still spins the wheel
        at the commanded speed but it cannot push: the encoder keeps counting while the rover
        barely moves. Needs FaultInjector.prepare_robot to have run at load.
        """
        targets = self._resolve(wheel, WHEELS, "wheel")
        if targets is not None:
            self._enqueue("wheel_slip", targets, _clamp_severity(severity))

    def inject_wheel_sink(self, wheel: str, severity: float) -> None:
        """
        Sink one wheel, or ALL of them, into very soft ground. severity 0 restores it.

        Two effects, both scaled by severity. The wheel settles severity * max_wheel_sinkage_m into
        the ground, and every physics step it is pushed against its horizontal motion with
        severity * max_sink_resistance_coeff times its normal load. Unlike wheel slip, grip is
        untouched: the wheel has to work against the soil, so drive effort rises and the rover stops
        quickly instead of coasting. Needs FaultInjector.prepare_robot to have run at load.
        """
        targets = self._resolve(wheel, WHEELS, "wheel")
        if targets is not None:
            self._enqueue("wheel_sink", targets, _clamp_severity(severity))

    def inject_steer_torque(self, corner: str, severity: float) -> None:
        """
        Weaken one steer actuator, or ALL of them.

        Distinct from a stuck joint: a weak corner still follows the controller, just late and
        short, so it lags the commanded angle instead of holding a constant one.
        """
        targets = self._resolve(corner, CORNERS, "corner")
        if targets is not None:
            self._enqueue("steer_torque", targets, _clamp_severity(severity))

    def inject_steer_stuck(self, corner: str, angle_deg: float) -> None:
        """
        Pin one corner at a fixed angle, ignoring the controller.

        Robot.set_steer_angles re-applies the override on every write, so this holds against a
        controller recomputing its steer targets at the physics rate.
        """
        targets = self._resolve(corner, CORNERS, "corner")
        if targets is not None:
            self._enqueue("steer_stuck", targets, float(angle_deg))

    def inject_imu_fault(self, bias: float, noise: float) -> None:
        """
        Corrupt the inertial measurement unit: a constant offset, a random jitter, or both.

        Applied at the sensor, so every consumer sees the same broken imu - including the power and
        thermal models, which read imu yaw for the sun angle. The drive controller closes its loop
        on ground-truth pose, so driving is unaffected: this fault misleads the operator, not the
        rover's feet.
        """
        self._enqueue("imu", [], _clamp_severity(bias), _clamp_severity(noise))

    def inject_camera_fault(self, loss: float, noise: float) -> None:
        """
        Lose frames, corrupt frames, or both.

        loss is a per-frame drop probability applied at the downlink; noise is sensor grain applied
        to the frames that survive. They are separable on purpose - a blind camera and a noisy one
        look nothing alike from the ground.
        """
        self._enqueue("camera", [], _clamp_severity(loss), _clamp_severity(noise))

    def inject_battery_fault(self, severity: float) -> None:
        """
        Drain the pack with a parasitic load - a short, a stuck heater.

        Modelled as a real load rather than by writing the charge directly, so it shows up
        everywhere it should: net_power, total_current_out, and a battery_charge curve that bends.
        """
        self._enqueue("battery", [], _clamp_severity(severity))

    def inject_comms_fault(self, tm_loss: float, tc_loss: float) -> None:
        """
        Lose telemetry on the way down, telecommands on the way up, or both.

        clear_faults is exempt from tc_loss - see should_drop_tc.
        """
        self._enqueue("comms", [], _clamp_severity(tm_loss), _clamp_severity(tc_loss))

    def clear_all(self) -> None:
        """Restore every subsystem to the state it started in."""
        self._enqueue("clear", [])

    # ── per-step update ──────────────────────────────────────────────────────────
    def update(self) -> None:
        """
        Apply whatever the ground asked for. Call once per physics step, from the simulation loop.

        This is the only place the injector changes anything. Everything upstream of it runs on the
        telecommand listener thread, and authoring usd from there deadlocks against Kit: the
        simulation freezes mid-command, which also stops the downlink, so the ground goes on showing
        the last healthy telemetry it received. Both halves of that failure have one cause, and this
        is the fix - the same split the drive controller already uses, where a command thread records
        a goal and the simulation thread acts on it.
        """
        with self._lock:
            pending, self._pending = self._pending, []

        if pending:
            for kind, targets, magnitudes in pending:
                self._apply(kind, targets, magnitudes)
            self._refresh_health()

        # The one fault that has to act every step rather than once: the drag of soft soil depends
        # on how fast each wheel is moving now, and an external force only lasts one physics step.
        if self._wheel_sink:
            try:
                self._robot.apply_wheel_resistance(
                    {name: self._sink_resistance(severity) for name, severity in self._wheel_sink.items()},
                    self._ref["sink_resistance_velocity_eps"],
                )
                self._resistance_error = None
            except Exception as exc:
                # Runs every physics step inside the simulation loop, so an exception here would take
                # the whole simulator down. The sinkage still applies; only the drag is missing, and
                # the error is printed once so the missing drag is not silent.
                message = f"{type(exc).__name__}: {exc}"
                if message != self._resistance_error:
                    print(f"[fault] wheel_sink drag could not be applied, the rover will not feel it: "
                          f"{message}", flush=True)
                    self._resistance_error = message

    def _enqueue(self, kind: str, targets: List[str], *magnitudes: float) -> None:
        with self._lock:
            self._pending.append((kind, targets, tuple(magnitudes)))

    def _apply(self, kind: str, targets: List[str], magnitudes: Tuple[float, ...]) -> None:
        """Perform one queued request. Simulation thread only."""
        applier = self._appliers.get(kind)
        if applier is None:
            print(f"[fault] unknown queued fault kind {kind!r}", flush=True)
            return

        applier(targets, magnitudes)

        if self._on_apply is not None:
            try:
                self._on_apply(kind, list(targets), tuple(magnitudes))
            except Exception as exc:
                # A recorder that fails must not take the simulation down with it.
                print(f"[fault] on_apply hook failed for {kind}: {exc}", flush=True)

    # ── appliers, one per fault kind ─────────────────────────────────────────────
    def _apply_clear(self, targets, magnitudes) -> None:
        self._robot.set_joint_max_efforts({
            **{f"drive_joint_{name}": UNLIMITED for name in WHEELS},
            **{f"steer_joint_{name}": UNLIMITED for name in CORNERS},
        })
        self._robot.clear_steer_overrides()
        self._robot.clear_wheel_damping_factors()
        self._robot.reset_wheel_friction()
        self._robot.reset_wheel_sinkage()
        self._wheel_torque.clear()
        self._wheel_stuck.clear()
        self._wheel_slip.clear()
        self._wheel_sink.clear()
        self._steer_torque.clear()
        self._steer_stuck.clear()

        self._imu_bias = self._imu_noise = 0.0
        self._camera_loss = self._camera_noise = 0.0
        self._battery = 0.0
        self._tm_loss = self._tc_loss = 0.0
        self._push_sensor_corruption()
        self._push_parasitic_load()

        print("[fault] cleared: all subsystems nominal", flush=True)

    def _apply_wheel_torque(self, targets, magnitudes) -> None:
        severity = magnitudes[0]
        for name in targets:
            self._record(self._wheel_torque, name, severity)
        self._robot.set_joint_max_efforts({
            f"drive_joint_{name}": self._limit(severity, self._ref["nominal_wheel_torque"])
            for name in targets
        })
        self._log("wheel_torque", targets, f"severity={severity:.2f}")

    def _apply_wheel_stuck(self, targets, magnitudes) -> None:
        severity = magnitudes[0]
        factor = self._damping_factor(severity)
        for name in targets:
            self._record(self._wheel_stuck, name, severity)
            self._robot.set_wheel_damping_factor(name, factor)
        self._log("wheel_stuck", targets, f"severity={severity:.2f} (damping x{factor:.1f})")

    def _apply_wheel_slip(self, targets, magnitudes) -> None:
        severity = magnitudes[0]
        friction = self._friction(severity)
        for name in targets:
            self._record(self._wheel_slip, name, severity)
            self._robot.set_wheel_friction(name, friction)
        self._log("wheel_slip", targets, f"severity={severity:.2f} (friction {friction:.3f})")

    def _apply_wheel_sink(self, targets, magnitudes) -> None:
        severity = magnitudes[0]
        depth = self._sinkage(severity)
        for name in targets:
            self._record(self._wheel_sink, name, severity)
            self._robot.set_wheel_sinkage(name, depth)
        self._log("wheel_sink", targets, f"severity={severity:.2f} (sinkage {depth * 1000:.1f} mm, "
                                         f"drag {self._sink_resistance(severity):.2f} x load)")

    def _apply_steer_torque(self, targets, magnitudes) -> None:
        severity = magnitudes[0]
        for name in targets:
            self._record(self._steer_torque, name, severity)
        self._robot.set_joint_max_efforts({
            f"steer_joint_{name}": self._limit(severity, self._ref["nominal_steer_torque"])
            for name in targets
        })
        self._log("steer_torque", targets, f"severity={severity:.2f}")

    def _apply_steer_stuck(self, targets, magnitudes) -> None:
        angle_deg = magnitudes[0]
        for name in targets:
            self._steer_stuck[name] = angle_deg
            self._robot.set_steer_override(name, math.radians(angle_deg))
        self._log("steer_stuck", targets, f"angle={angle_deg:.1f}deg")

    def _apply_imu(self, targets, magnitudes) -> None:
        self._imu_bias, self._imu_noise = magnitudes
        self._push_sensor_corruption()
        self._log("imu", ["sensor"], f"bias={self._imu_bias:.2f} noise={self._imu_noise:.2f}")

    def _apply_camera(self, targets, magnitudes) -> None:
        self._camera_loss, self._camera_noise = magnitudes
        self._push_sensor_corruption()
        self._log("camera", ["sensor"], f"loss={self._camera_loss:.2f} noise={self._camera_noise:.2f}")

    def _apply_battery(self, targets, magnitudes) -> None:
        self._battery = magnitudes[0]
        self._push_parasitic_load()
        self._log("battery", ["pack"], f"severity={self._battery:.2f} ({self.parasitic_load:.1f} W)")

    def _apply_comms(self, targets, magnitudes) -> None:
        self._tm_loss, self._tc_loss = magnitudes
        self._log("comms", ["link"], f"tm_loss={self._tm_loss:.2f} tc_loss={self._tc_loss:.2f}")

    def _push_sensor_corruption(self) -> None:
        """Hand the sensor magnitudes to the robot, which owns the accessors that apply them."""
        self._robot.set_imu_corruption(
            accel_bias=self._imu_bias * self._ref["max_imu_accel_bias"],
            gyro_bias=self._imu_bias * self._ref["max_imu_gyro_bias"],
            orientation_bias_deg=self._imu_bias * self._ref["max_imu_orientation_bias_deg"],
            accel_noise=self._imu_noise * self._ref["max_imu_accel_noise"],
            gyro_noise=self._imu_noise * self._ref["max_imu_gyro_noise"],
        )
        self._robot.set_camera_corruption(
            noise=self._camera_noise * self._ref["max_camera_noise"],
        )

    def _push_parasitic_load(self) -> None:
        subsystems = getattr(self._robot, "subsystems", None)
        if subsystems is not None:
            subsystems.set_parasitic_load(self.parasitic_load)

    # ── link gates, read from the tm/tc paths ────────────────────────────────────
    def should_drop_tm(self) -> bool:
        """True when this downlinked parameter is lost. Called on the telemetry interval."""
        if self._tm_loss <= 0.0 or self._rng.random() >= self._tm_loss:
            return False
        self._dropped["tm"] += 1
        return True

    def should_drop_tc(self, command_name: str = "") -> bool:
        """
        True when this uplinked command is lost.

        clear_faults is never dropped. Without that exemption inject_comms_fault(tc_loss=1.0) is
        unrecoverable: the one command that lifts the fault is the one being discarded, and the only
        way out is to restart the simulator.
        """
        if command_name == self.RECOVERY_COMMAND:
            return False
        if self._tc_loss <= 0.0 or self._rng.random() >= self._tc_loss:
            return False
        self._dropped["tc"] += 1
        return True

    def should_drop_camera_frame(self) -> bool:
        """True when this frame never makes it to the ground. Called on the image interval."""
        if self._camera_loss <= 0.0 or self._rng.random() >= self._camera_loss:
            return False
        self._dropped["camera_frames"] += 1
        return True

    def apply_spec(self, spec: str) -> None:
        """
        Apply one fault from a colon-separated string, for the --fault launch flag.

        Lets a run be reproduced without a ground station, which matters because the interesting
        faults only show up over a whole drive. Shapes, mirroring each command's arguments:

            wheel_torque:<wheel|ALL>:<severity>      steer_stuck:<corner|ALL>:<angle deg>
            wheel_stuck:<wheel|ALL>:<severity>       imu:<bias>:<noise>
            wheel_slip:<wheel|ALL>:<severity>        comms:<tm_loss>:<tc_loss>
            wheel_sink:<wheel|ALL>:<severity>        battery:<severity>
            steer_torque:<corner|ALL>:<severity>     camera:<loss>:<noise>
        """
        # kind -> (handler, takes a target, how many magnitudes)
        shapes = {
            "wheel_torque": (self.inject_wheel_torque, True, 1),
            "wheel_stuck": (self.inject_wheel_stuck, True, 1),
            "wheel_slip": (self.inject_wheel_slip, True, 1),
            "wheel_sink": (self.inject_wheel_sink, True, 1),
            "steer_torque": (self.inject_steer_torque, True, 1),
            "steer_stuck": (self.inject_steer_stuck, True, 1),
            "imu": (self.inject_imu_fault, False, 2),
            "camera": (self.inject_camera_fault, False, 2),
            "battery": (self.inject_battery_fault, False, 1),
            "comms": (self.inject_comms_fault, False, 2),
        }

        parts = [part.strip() for part in spec.split(":")]
        shape = shapes.get(parts[0]) if parts else None
        if shape is None:
            print(f"[fault] unknown fault kind in {spec!r}; expected one of {sorted(shapes)}", flush=True)
            return

        handler, takes_target, magnitude_count = shape
        expected = 1 + int(takes_target) + magnitude_count
        if len(parts) != expected:
            print(f"[fault] bad --fault spec {spec!r}: expected {expected} colon-separated fields, "
                  f"got {len(parts)}", flush=True)
            return

        try:
            magnitudes = [float(part) for part in parts[1 + int(takes_target):]]
        except ValueError:
            print(f"[fault] non-numeric magnitude in {spec!r}", flush=True)
            return

        handler(*([parts[1]] if takes_target else []), *magnitudes)

    # ── telemetry ────────────────────────────────────────────────────────────────
    @property
    def active_faults(self) -> str:
        """One-line summary of everything injected, for downlink as /Rover/faults/active."""
        entries = []
        entries += [f"wheel_torque[{n}]={s:.2f}" for n, s in self._wheel_torque.items()]
        entries += [f"wheel_stuck[{n}]={s:.2f}" for n, s in self._wheel_stuck.items()]
        entries += [f"wheel_slip[{n}]={s:.2f}" for n, s in self._wheel_slip.items()]
        entries += [f"wheel_sink[{n}]={s:.2f}" for n, s in self._wheel_sink.items()]
        entries += [f"steer_torque[{n}]={s:.2f}" for n, s in self._steer_torque.items()]
        entries += [f"steer_stuck[{n}]={a:.1f}deg" for n, a in self._steer_stuck.items()]
        if self._imu_bias or self._imu_noise:
            entries.append(f"imu[bias={self._imu_bias:.2f},noise={self._imu_noise:.2f}]")
        if self._camera_loss or self._camera_noise:
            entries.append(f"camera[loss={self._camera_loss:.2f},noise={self._camera_noise:.2f}]")
        if self._battery:
            entries.append(f"battery[{self._battery:.2f}]")
        if self._tm_loss or self._tc_loss:
            entries.append(f"comms[tm={self._tm_loss:.2f},tc={self._tc_loss:.2f}]")

        return "; ".join(entries) if entries else "none"

    @property
    def wheel_torque_limits(self) -> List[float]:
        """
        Per-wheel torque limit in N*m, in WHEELS order.

        A healthy wheel reports the nominal rather than inf: the archive has to be able to plot it.
        """
        return [
            (1.0 - self._wheel_torque.get(name, 0.0)) * self._ref["nominal_wheel_torque"]
            for name in WHEELS
        ]

    @property
    def wheel_damping_factors(self) -> List[float]:
        """Per-wheel drive damping multiplier, in WHEELS order. A healthy wheel reads 1.0."""
        return [self._damping_factor(self._wheel_stuck.get(name, 0.0)) for name in WHEELS]

    @property
    def wheel_frictions(self) -> List[float]:
        """Per-wheel ground friction coefficient, in WHEELS order. A healthy wheel reads the nominal."""
        return [self._friction(self._wheel_slip.get(name, 0.0)) for name in WHEELS]

    @property
    def wheel_sinkages(self) -> List[float]:
        """Per-wheel sinkage into the ground in metres, in WHEELS order. A healthy wheel reads 0."""
        return [self._sinkage(self._wheel_sink.get(name, 0.0)) for name in WHEELS]

    @property
    def steer_torque_limits(self) -> List[float]:
        """Per-corner torque limit in N*m, in CORNERS order."""
        return [
            (1.0 - self._steer_torque.get(name, 0.0)) * self._ref["nominal_steer_torque"]
            for name in CORNERS
        ]

    @property
    def parasitic_load(self) -> float:
        """Extra load on the battery in watts. The battery fault's observable."""
        return self._battery * self._ref["max_parasitic_load_w"]

    @property
    def dropped_counts(self) -> Dict[str, int]:
        """Cumulative losses for the run: telemetry, telecommands, camera frames."""
        return dict(self._dropped)

    @property
    def wheel_health(self) -> List[int]:
        """
        Ground-truth health of each drive actuator, in WHEELS order.

        A wheel carrying more than one fault reports the worst of them.
        """
        return [
            self._health_of(max(
                self._wheel_torque.get(name, 0.0),
                self._wheel_stuck.get(name, 0.0),
                self._wheel_slip.get(name, 0.0),
                self._wheel_sink.get(name, 0.0),
            )).value
            for name in WHEELS
        ]

    @property
    def steer_health(self) -> List[int]:
        """
        Ground-truth health of each steer actuator, in CORNERS order.

        A stuck corner is a hard failure however much torque it still has, so it outranks any
        torque fault on the same corner.
        """
        health = []
        for name in CORNERS:
            if name in self._steer_stuck:
                health.append(HealthState.FAULT.value)
            else:
                health.append(self._health_of(self._steer_torque.get(name)).value)

        return health

    @property
    def imu_health(self) -> int:
        return self._health_of(max(self._imu_bias, self._imu_noise)).value

    @property
    def camera_health(self) -> int:
        return self._health_of(max(self._camera_loss, self._camera_noise)).value

    @property
    def battery_health(self) -> int:
        return self._health_of(self._battery).value

    @property
    def comms_health(self) -> int:
        return self._health_of(max(self._tm_loss, self._tc_loss)).value

    @property
    def motor_controller_health(self) -> int:
        """
        Coarse summary of the mobility system, matching what the device flag carries.

        Never FAULT from an injected fault, however dead the actuators are: FAULT is an interlock
        condition and would stop the rover, hiding the behaviour the fault exists to produce.
        """
        return self._mobility_health().value

    # ── plumbing ─────────────────────────────────────────────────────────────────
    def _resolve(self, target: str, valid: tuple, label: str):
        """Expand ALL, or validate a single name. Returns None when the name is unknown."""
        name = str(target).strip().lower()
        if name == ALL.lower():
            return list(valid)
        if name in valid:
            return [name]

        print(f"[fault] unknown {label} {target!r}; expected one of {list(valid) + [ALL]}", flush=True)
        return None

    @staticmethod
    def _health_of(severity) -> HealthState:
        """Map a severity onto a health state. None means the subsystem was never touched."""
        if severity is None or severity <= 0.0:
            return HealthState.NOMINAL
        return HealthState.FAULT if severity >= 1.0 else HealthState.DEGRADED

    def _mobility_health(self) -> HealthState:
        any_fault = bool(self._wheel_torque or self._wheel_stuck or self._wheel_slip or self._wheel_sink
                         or self._steer_torque or self._steer_stuck)
        return HealthState.DEGRADED if any_fault else HealthState.NOMINAL

    def _device_states(self) -> Dict[str, HealthState]:
        """
        Which device flag each fault family drives, from ground truth.

        Capped at DEGRADED for the same reason everywhere: FAULT means dead hardware, and on the
        motor controller it is an interlock. "imu" is a plain string because that is how
        PerseveranceSubsystemsHandler._setup_devices registers it.
        """
        degraded = HealthState.DEGRADED
        nominal = HealthState.NOMINAL
        return {
            CommonDevice.MOTOR_CONTROLLER: self._mobility_health(),
            "imu": degraded if (self._imu_bias or self._imu_noise) else nominal,
            CommonDevice.CAMERA: degraded if (self._camera_loss or self._camera_noise) else nominal,
            CommonDevice.EPS: degraded if self._battery else nominal,
            CommonDevice.RADIO: degraded if (self._tm_loss or self._tc_loss) else nominal,
        }

    def _refresh_health(self) -> None:
        """
        Push the coarse states onto the device flags, from ground truth.

        Only ever moves a flag between NOMINAL and DEGRADED. A FAULT set by something else - a
        powered-off device, a hard fault injected elsewhere - is left alone, because clearing a
        simulated wheel fault has no business declaring unrelated hardware healthy.
        """
        subsystems = getattr(self._robot, "subsystems", None)
        if subsystems is None:
            return

        for device, target in self._device_states().items():
            current = subsystems.get_device_health_state(device)
            if current != HealthState.FAULT and target != current:
                subsystems.set_device_health_state(device, target)

    @staticmethod
    def _limit(severity: float, nominal: float) -> float:
        # A cleared fault has to restore the unauthored default, not the nominal reference.
        return UNLIMITED if severity <= 0.0 else (1.0 - severity) * nominal

    def _damping_factor(self, severity: float) -> float:
        """
        Damping multiplier for a stuck-wheel severity: 1 / (1 - severity), capped.

        That shape makes severity the fraction of free-running speed the wheel loses, since the
        robot divides the velocity target by the same factor. The cap is what 1.0 means, and it
        keeps the damping finite.
        """
        cap = max(1.0, self._ref["max_wheel_stuck_damping_factor"])
        if severity >= 1.0:
            return cap
        return min(cap, 1.0 / (1.0 - max(0.0, severity)))

    def _friction(self, severity: float) -> float:
        return (1.0 - severity) * self._ref["nominal_wheel_friction"]

    def _sinkage(self, severity: float) -> float:
        return severity * self._ref["max_wheel_sinkage_m"]

    def _sink_resistance(self, severity: float) -> float:
        return severity * self._ref["max_sink_resistance_coeff"]

    @staticmethod
    def _record(registry: Dict[str, float], name: str, severity: float) -> None:
        if severity <= 0.0:
            registry.pop(name, None)
        else:
            registry[name] = severity

    @staticmethod
    def _log(kind: str, targets: List[str], detail: str) -> None:
        print(f"[fault] {kind} {','.join(targets)}: {detail}", flush=True)
