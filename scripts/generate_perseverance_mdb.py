#!/usr/bin/env python3
"""
Generates cfg/mdb/perseverance.xml, the Yamcs mission database for the perseverance rover.

The MDB is the contract shared by three parties:
  - Yamcs, which uses it to render the parameter/command UI and to encode outgoing telecommands;
  - PerseveranceTransmitter, which downlinks the parameters named here (via the name map in
    cfg/controller/perseverance-controller.yaml);
  - MdbParsingService, which decodes the telecommand packets Yamcs puts on the wire.

Telemetry parameters are declared with DataSource.LOCAL ("software parameters"). That is what makes
processor.set_parameter_value() — the call behind YamcsTMTC.transmit_to_yamcs — legal for them.

Telecommands are identified on the wire by their name in plain ASCII, followed by their arguments
in declaration order, big-endian. That layout is what MdbParsingService.decode_tc_payload expects.

This is an offline build step; the generated XML is committed. Regenerate with:

    pip install yamcs-pymdb
    python3 scripts/generate_perseverance_mdb.py
"""

import os

from yamcs.pymdb import (
    ArgumentEntry,
    Command,
    DataSource,
    EnumeratedArgument,
    EnumeratedDataType,
    EnumeratedParameter,
    FixedValueEntry,
    FloatArgument,
    FloatDataType,
    FloatMember,
    FloatParameter,
    IntegerDataType,
    IntegerParameter,
    StringParameter,
    Subsystem,
    System,
    AggregateMember,
    AggregateParameter,
    ArrayParameter,
    float32_t,
    uint8_t,
)

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "cfg", "mdb", "perseverance.xml")

LOCAL = DataSource.LOCAL

# Keep in sync with PerseveranceSubsystemsHandler._setup_devices.
DEVICE_CHOICES = [
    (0, "OBC", "On-board computer"),
    (1, "MOTOR_CONTROLLER", "Motor controller"),
    (2, "CAMERA", "Camera system"),
    (3, "RADIO", "Radio"),
    (4, "EPS", "Electric power system"),
]
POWER_CHOICES = [(0, "OFF", "Powered off"), (1, "ON", "Powered on")]
SOLAR_PANEL_CHOICES = [(0, "STOWED", "Panel stowed"), (1, "DEPLOYED", "Panel deployed")]
GO_NOGO_CHOICES = [(0, "NOGO", "Rover held"), (1, "GO", "Rover cleared to operate")]
# Must stay in sync with src/subsystems/device.py HealthState. Listed worst-last for the ui; the
# numbers are identifiers, not a ranking - DEGRADED was appended to the enum rather than inserted so
# no already-archived value changed meaning.
HEALTH_CHOICES = [
    (0, "NOMINAL", "Working to spec"),
    (2, "DEGRADED", "Still working, but not to spec"),
    (1, "FAULT", "Failed"),
]
# Must stay in sync with control/drive_controller.py CommandStatus.
COMMAND_STATUS_CHOICES = [
    (0, "IDLE", "No command active"),
    (1, "EXECUTING", "Command in progress"),
    (2, "COMPLETE", "Command finished successfully"),
    (3, "REJECTED", "Command refused, or halted by an interlock"),
    (4, "ABORTED", "Command cancelled by manual takeover or NOGO"),
]
OBC_STATE_CHOICES = [
    (0, "OFF", "Off"), (1, "BOOT", "Booting"), (2, "IDLE", "Idle"),
    (3, "CAMERA", "Camera activity"), (4, "MOTOR", "Driving"),
    (5, "SAFE", "Safe mode"), (6, "ERROR", "Error"),
]

WHEELS = ["front_left", "front_right", "mid_left", "mid_right", "rear_left", "rear_right"]
# The two mid wheels are not steerable; keep in sync with steer_joints in cfg/robot/perseverance.yaml.
CORNERS = ["front_left", "front_right", "rear_left", "rear_right"]

# Fault targets. ALL is last so adding a wheel would not renumber it.
WHEEL_CHOICES = [(i, wheel.upper(), f"{wheel.replace('_', ' ')} wheel") for i, wheel in enumerate(WHEELS)]
WHEEL_CHOICES.append((len(WHEELS), "ALL", "Every wheel"))
CORNER_CHOICES = [(i, corner.upper(), f"{corner.replace('_', ' ')} corner") for i, corner in enumerate(CORNERS)]
CORNER_CHOICES.append((len(CORNERS), "ALL", "Every steerable corner"))

THERMAL_FACES = [
    ("front", "+X face"), ("back", "-X face"), ("left", "+Y face"),
    ("right", "-Y face"), ("top", "+Z face"), ("bottom", "-Z face"),
]


def build_telemetry(rover: System, faults: Subsystem) -> None:
    # ── pose and attitude ────────────────────────────────────────────────────
    AggregateParameter(
        system=rover, name="pose_ground_truth", data_source=LOCAL,
        short_description="Ground-truth pose of the rover base link",
        members=[
            AggregateMember(name="position", members=[
                FloatMember(name="x", units="m"),
                FloatMember(name="y", units="m"),
                FloatMember(name="z", units="m"),
            ]),
            AggregateMember(name="orientation", members=[
                FloatMember(name="w"), FloatMember(name="x"),
                FloatMember(name="y"), FloatMember(name="z"),
            ]),
        ],
    )
    AggregateParameter(
        system=rover, name="imu_accelerometer", data_source=LOCAL,
        short_description="IMU linear acceleration",
        members=[FloatMember(name=axis, units="m/s^2") for axis in ("ax", "ay", "az")],
    )
    AggregateParameter(
        system=rover, name="imu_gyroscope", data_source=LOCAL,
        short_description="IMU angular velocity",
        members=[FloatMember(name=axis, units="rad/s") for axis in ("gx", "gy", "gz")],
    )
    AggregateParameter(
        system=rover, name="imu_orientation", data_source=LOCAL,
        short_description="IMU-derived attitude",
        members=[FloatMember(name=angle, units="deg") for angle in ("roll", "pitch", "yaw")],
    )

    # ── mobility ─────────────────────────────────────────────────────────────
    ArrayParameter(
        system=rover, name="motor_encoder", data_source=LOCAL, length=6,
        data_type=IntegerDataType(signed=False, bits=16),
        short_description="Drive joint angles, 1024 counts per revolution",
    )
    # Measured by the physics solver, not computed from the command. Encoder against pose shows a
    # wheel slipping; effort says why - near zero on ice, high in soft soil.
    ArrayParameter(
        system=rover, name="motor_effort", data_source=LOCAL, length=6,
        data_type=FloatDataType(units="N*m"),
        short_description="Measured drive joint torque per wheel: front, mid, rear; left then right",
    )
    # Measured, not commanded. This is what makes a stuck or weakened steer actuator visible from
    # the ground: compare it against the arc the rover was asked to drive.
    ArrayParameter(
        system=rover, name="steer_encoder", data_source=LOCAL, length=4,
        data_type=FloatDataType(units="deg"),
        short_description="Measured steer angle of the four corner joints",
    )
    for wheel in WHEELS:
        FloatParameter(
            system=rover, name=f"contact_force_{wheel}", data_source=LOCAL, units="N",
            minimum=0.0, short_description=f"Net contact force on the {wheel.replace('_', ' ')} wheel",
        )

    # ── power ────────────────────────────────────────────────────────────────
    IntegerParameter(
        system=rover, name="battery_charge", signed=False, bits=8, data_source=LOCAL,
        units="%", minimum=0, maximum=100, short_description="Battery state of charge",
    )
    for name, units, description in [
        ("battery_voltage", "V", "Battery voltage"),
        ("total_current_in", "A", "Solar input current"),
        ("total_current_out", "A", "Total current draw"),
        ("net_power", "W", "Solar input power minus total load"),
        ("current_draw_obc", "A", "Current consumed by the on-board computer"),
        ("current_draw_motor_controller", "A", "Current consumed by the motor controller"),
        ("current_draw_camera", "A", "Current consumed by the camera system"),
        ("current_draw_radio", "A", "Current consumed by the radio"),
        ("current_draw_eps", "A", "Current consumed by the electric power system"),
    ]:
        FloatParameter(
            system=rover, name=name, data_source=LOCAL, units=units, short_description=description,
        )
    ArrayParameter(
        system=rover, name="motor_current", data_source=LOCAL, length=6,
        data_type=FloatDataType(units="A"),
        short_description="Per-motor current draw",
    )

    # ── thermal and comms ────────────────────────────────────────────────────
    for face, description in THERMAL_FACES:
        FloatParameter(
            system=rover, name=f"temperature_{face}", data_source=LOCAL, units="degC",
            short_description=f"Temperature of the {description}",
        )
    IntegerParameter(
        system=rover, name="radio_rssi", bits=16, data_source=LOCAL, units="dBm",
        short_description="Received signal strength at the relay",
    )

    # ── on-board computer ────────────────────────────────────────────────────
    EnumeratedParameter(
        system=rover, name="obc_state", data_source=LOCAL,
        choices=OBC_STATE_CHOICES, short_description="On-board computer state",
    )
    for name, units, description in [
        ("obc_cpu_usage", "%", "CPU usage"),
        ("obc_ram_usage", "%", "RAM usage"),
        ("obc_disk_usage", "%", "Disk usage"),
    ]:
        IntegerParameter(
            system=rover, name=name, signed=False, bits=8, data_source=LOCAL,
            units=units, minimum=0, maximum=100, short_description=description,
        )
    IntegerParameter(
        system=rover, name="obc_uptime", signed=False, bits=32, data_source=LOCAL,
        units="s", short_description="Seconds since the last on-board computer boot",
    )

    # ── mission state ────────────────────────────────────────────────────────
    EnumeratedParameter(
        system=rover, name="go_nogo", data_source=LOCAL,
        choices=GO_NOGO_CHOICES, short_description="Mission GO/NOGO status",
    )
    EnumeratedParameter(
        system=rover, name="solar_panel_state", data_source=LOCAL,
        choices=SOLAR_PANEL_CHOICES, short_description="Solar panel deployment state",
    )
    StringParameter(
        system=rover, name="active_command", data_source=LOCAL, initial_value="none",
        short_description="Last high-level command the rover acknowledged",
    )

    # ── command execution progress ───────────────────────────────────────────
    # Acknowledging a command is not the same as executing it. These let an operator tell
    # "still driving" from "refused because NOGO" without waiting to see the rover stop.
    EnumeratedParameter(
        system=rover, name="command_status", data_source=LOCAL,
        choices=COMMAND_STATUS_CHOICES,
        short_description="Execution state of the active high-level command",
    )
    FloatParameter(
        system=rover, name="command_distance_remaining", data_source=LOCAL, units="m",
        minimum=0.0, short_description="Distance still to travel on the active command",
    )
    FloatParameter(
        system=rover, name="command_heading_error", data_source=LOCAL, units="deg",
        short_description="Heading error to the commanded bearing",
    )
    AggregateParameter(
        system=rover, name="command_target", data_source=LOCAL,
        short_description="Commanded goto waypoint",
        members=[FloatMember(name="x", units="m"), FloatMember(name="y", units="m")],
    )

    # ── injected faults ──────────────────────────────────────────────────────
    # Everything artificial lives under /Rover/faults, so no injected value can be read as a real
    # rover measurement. Note steer_encoder above deliberately stays on the plain /Rover path: it is
    # a genuine reading off the joint, and filing it here would misrepresent it.
    StringParameter(
        system=faults, name="active", data_source=LOCAL, initial_value="none",
        short_description="Summary of every fault currently injected, or 'none'",
    )
    ArrayParameter(
        system=faults, name="wheel_torque_limit", data_source=LOCAL, length=6,
        data_type=FloatDataType(units="N*m"),
        short_description="Per-wheel drive torque limit; a healthy wheel reads the configured nominal",
    )
    ArrayParameter(
        system=faults, name="wheel_damping_factor", data_source=LOCAL, length=6,
        data_type=FloatDataType(),
        short_description="Per-wheel drive damping multiplier from a stuck-wheel fault; a healthy wheel reads 1.0",
    )
    ArrayParameter(
        system=faults, name="wheel_friction", data_source=LOCAL, length=6,
        data_type=FloatDataType(),
        short_description="Per-wheel ground friction coefficient; a slip fault lowers it from the configured nominal",
    )
    ArrayParameter(
        system=faults, name="wheel_sinkage", data_source=LOCAL, length=6,
        data_type=FloatDataType(units="m"),
        short_description="Per-wheel depth a sink fault has settled the wheel into the ground; 0 when healthy",
    )
    ArrayParameter(
        system=faults, name="steer_torque_limit", data_source=LOCAL, length=4,
        data_type=FloatDataType(units="N*m"),
        short_description="Per-corner steer torque limit; a healthy corner reads the configured nominal",
    )
    # Ground truth, not a diagnosis. The injector knows precisely what it broke, so the health it
    # reports is exact - which is the whole reason these can name the individual actuator rather
    # than just flagging the subsystem.
    ArrayParameter(
        system=faults, name="wheel_health", data_source=LOCAL, length=6,
        data_type=EnumeratedDataType(choices=HEALTH_CHOICES, encoding=uint8_t),
        short_description="Ground-truth health of each drive actuator",
    )
    ArrayParameter(
        system=faults, name="steer_health", data_source=LOCAL, length=4,
        data_type=EnumeratedDataType(choices=HEALTH_CHOICES, encoding=uint8_t),
        short_description="Ground-truth health of each steer actuator; a stuck corner reads FAULT",
    )
    EnumeratedParameter(
        system=faults, name="motor_controller_health", data_source=LOCAL, choices=HEALTH_CHOICES,
        short_description="Coarse mobility health; DEGRADED whenever any actuator fault is injected",
    )
    for name, description in [
        ("imu_health", "Inertial measurement unit"),
        ("camera_health", "Camera"),
        ("battery_health", "Battery / power system"),
        ("comms_health", "Ground link"),
    ]:
        EnumeratedParameter(
            system=faults, name=name, data_source=LOCAL, choices=HEALTH_CHOICES,
            short_description=f"{description} health, from ground truth",
        )
    FloatParameter(
        system=faults, name="parasitic_load", data_source=LOCAL, units="W", minimum=0.0,
        short_description="Injected extra load on the battery; the battery fault's observable",
    )
    # Cumulative for the run and never reset by clear_faults: a counter that only rises is the one
    # an operator can plot without wondering where the steps came from.
    AggregateParameter(
        system=faults, name="dropped", data_source=LOCAL,
        short_description="Cumulative losses caused by injected faults",
        members=[
            FloatMember(name="tm", short_description="Telemetry parameters lost on the downlink"),
            FloatMember(name="tc", short_description="Telecommands lost on the uplink"),
            FloatMember(name="camera_frames", short_description="Camera frames never sent"),
        ],
    )


def build_estimator(estimator: Subsystem) -> None:
    """
    Onboard navigation filter output (src/mission_specific/perseverance/estimation/nav_filter.py).

    Computed on the rover from its own sensors only, so it is observable telemetry. Residuals are
    normalized (units of sigma, nominally N(0, 1)) and summarized over each 1 s window: "mean" for
    the persistent part a fault leaves behind, "max" (signed extreme) for transients. Wheel arrays
    are in ALL_WHEELS order: front_left, front_right, mid_left, mid_right, rear_left, rear_right.
    """
    AggregateParameter(
        system=estimator, name="pose", data_source=LOCAL,
        short_description="Estimated pose, episode-local position and world heading of the forward axis",
        members=[FloatMember(name="x", units="m"), FloatMember(name="y", units="m"), FloatMember(name="yaw", units="deg")],
    )
    AggregateParameter(
        system=estimator, name="motion", data_source=LOCAL,
        short_description="Estimated forward speed, sideways sliding speed and yaw rate",
        members=[
            FloatMember(name="speed", units="m/s"),
            FloatMember(name="lateral_speed", units="m/s"),
            FloatMember(name="yaw_rate", units="rad/s"),
        ],
    )
    AggregateParameter(
        system=estimator, name="bias", data_source=LOCAL,
        short_description="Estimated IMU biases",
        members=[FloatMember(name="gyro", units="rad/s"), FloatMember(name="accel", units="m/s^2")],
    )
    AggregateParameter(
        system=estimator, name="imu_residual", data_source=LOCAL,
        short_description="Normalized gyro and heading innovations, and unexplained lateral force (sigma)",
        members=[FloatMember(name=f"{kind}_{field}") for kind in ("gyro", "heading", "lateral_force")
                 for field in ("mean", "max")],
    )
    AggregateParameter(
        system=estimator, name="nis", data_source=LOCAL,
        short_description="Mean normalized innovation squared over the window (nominally 1)",
        members=[FloatMember(name=name) for name in ("wheels", "gyro", "heading")],
    )
    for name, description in [
        ("wheel_rolling", "Wheel speed innovation against the EKF prediction"),
        ("wheel_sideslip", "Wheel sliding along its axle (no-side-slip innovation)"),
        ("wheel_effort", "Drive torque in excess of the torque model, in the direction of motion"),
        ("wheel_tracking", "Commanded wheel rate the wheel failed to reach"),
    ]:
        for field in ("mean", "max"):
            ArrayParameter(
                system=estimator, name=f"{name}_{field}", data_source=LOCAL, length=6,
                data_type=FloatDataType(),
                short_description=f"{description}, window {field}, sigma",
            )
    IntegerParameter(
        system=estimator, name="cusum_alarm", signed=False, bits=32, data_source=LOCAL,
        short_description="Persistent-shift alarms, one bit per residual in nav_filter.RESIDUAL_KEYS order",
    )
    IntegerParameter(
        system=estimator, name="reacquisitions", signed=False, bits=16, data_source=LOCAL,
        short_description="Times the gyro or heading channel was re-accepted after being rejected for reacquire_s (episode total)",
    )


def build_camera_metadata(camera: Subsystem) -> None:
    """
    Declares what ImagesHandler._inform_yamcs publishes for each downlinked frame.

    The image itself lives in a Yamcs bucket; these parameters are how the ground finds it. Real
    telemetry, not fault bookkeeping, so it sits on /Rover/camera rather than /Rover/faults.
    """
    images = Subsystem(camera, "images_navcam")
    IntegerParameter(
        system=images, name="number", signed=False, bits=32, data_source=LOCAL,
        short_description="Sequence number of the latest nav-cam frame",
    )
    StringParameter(
        system=images, name="name", data_source=LOCAL, initial_value="none",
        short_description="File name of the latest nav-cam frame",
    )
    for name, description in [
        ("url_storage", "Bucket-relative path of the latest frame"),
        ("url_full", "Absolute Yamcs URL of the latest frame"),
        ("url_full_nginx", "Reverse-proxy URL of the latest frame"),
    ]:
        StringParameter(
            system=images, name=name, data_source=LOCAL, initial_value="none",
            short_description=description,
        )


def add_command(system: System, name: str, description: str, arguments=None) -> None:
    """
    Declares a telecommand whose wire form is its ASCII name followed by its arguments.

    MdbParsingService.decode_tc_payload matches the leading bytes of the UDP payload against the
    command name, so the FixedValueEntry has to carry exactly that name and nothing else.

    Every argument then needs its own explicit ArgumentEntry. Passing `entries` at all turns off
    pymdb's automatic entry generation, so without these the container would hold only the name and
    Yamcs would put the command on the wire with its arguments silently dropped.
    """
    arguments = arguments or []
    Command(
        system=system,
        name=name,
        short_description=description,
        arguments=arguments,
        entries=[
            FixedValueEntry(binary=name.encode("ascii"), name="command_name"),
            *(ArgumentEntry(argument) for argument in arguments),
        ],
    )


def build_commands(motor: Subsystem, rover_system: Subsystem, faults: Subsystem) -> None:
    # Argument names and order match DriveHandler.drive_robot_straight / drive_robot_turn, so the
    # motion controller can delegate to them unchanged once it exists.
    add_command(
        motor, "drive_straight", "Drive a set distance in a straight line",
        arguments=[
            FloatArgument(name="linear_velocity", encoding=float32_t, units="m/s"),
            FloatArgument(name="distance", encoding=float32_t, units="m"),
        ],
    )
    add_command(
        motor, "drive_turn", "Turn in place by a set angle",
        arguments=[
            FloatArgument(name="angular_velocity", encoding=float32_t, units="deg/s"),
            FloatArgument(name="angle", encoding=float32_t, units="deg"),
        ],
    )
    add_command(motor, "stop", "Stop all rover motion")
    add_command(
        motor, "goto", "Drive to a waypoint in environment coordinates",
        arguments=[
            FloatArgument(name="x", encoding=float32_t, units="m"),
            FloatArgument(name="y", encoding=float32_t, units="m"),
        ],
    )
    add_command(
        rover_system, "power_electronics", "Switch a subsystem on or off",
        arguments=[
            EnumeratedArgument(name="subsystem_id", choices=DEVICE_CHOICES, encoding=uint8_t),
            EnumeratedArgument(name="power_state", choices=POWER_CHOICES, encoding=uint8_t),
        ],
    )
    add_command(
        rover_system, "deploy_solar_panel", "Deploy or stow the solar panel",
        arguments=[EnumeratedArgument(name="deployment", choices=SOLAR_PANEL_CHOICES, encoding=uint8_t)],
    )
    add_command(
        rover_system, "go_nogo", "Set the mission GO/NOGO status",
        arguments=[EnumeratedArgument(name="decision", choices=GO_NOGO_CHOICES, encoding=uint8_t)],
    )

    # Fault injection. Not flight commands — these reach into the simulation and break things on
    # purpose, which is why they sit in their own subsystem rather than beside the drive commands.
    # Severity runs healthy (0.0) to dead (1.0), so a larger number is always a worse fault, and
    # 0.0 is how a fault is lifted.
    add_command(
        faults, "inject_wheel_torque_fault", "[SIM] Weaken a drive motor by limiting its torque",
        arguments=[
            EnumeratedArgument(name="wheel", choices=WHEEL_CHOICES, encoding=uint8_t),
            FloatArgument(name="severity", encoding=float32_t, minimum=0.0, maximum=1.0),
        ],
    )
    add_command(
        faults, "inject_wheel_stuck_fault", "[SIM] Seize a wheel by raising its joint damping",
        arguments=[
            EnumeratedArgument(name="wheel", choices=WHEEL_CHOICES, encoding=uint8_t),
            FloatArgument(name="severity", encoding=float32_t, minimum=0.0, maximum=1.0),
        ],
    )
    add_command(
        faults, "inject_wheel_slip_fault", "[SIM] Take a wheel's grip away by lowering its ground friction",
        arguments=[
            EnumeratedArgument(name="wheel", choices=WHEEL_CHOICES, encoding=uint8_t),
            FloatArgument(name="severity", encoding=float32_t, minimum=0.0, maximum=1.0),
        ],
    )
    add_command(
        faults, "inject_wheel_sink_fault", "[SIM] Sink a wheel into very soft ground",
        arguments=[
            EnumeratedArgument(name="wheel", choices=WHEEL_CHOICES, encoding=uint8_t),
            FloatArgument(name="severity", encoding=float32_t, minimum=0.0, maximum=1.0),
        ],
    )
    add_command(
        faults, "inject_steer_torque_fault", "[SIM] Weaken a steer actuator by limiting its torque",
        arguments=[
            EnumeratedArgument(name="corner", choices=CORNER_CHOICES, encoding=uint8_t),
            FloatArgument(name="severity", encoding=float32_t, minimum=0.0, maximum=1.0),
        ],
    )
    add_command(
        faults, "inject_steer_stuck_fault", "[SIM] Freeze a corner at a fixed steer angle",
        arguments=[
            EnumeratedArgument(name="corner", choices=CORNER_CHOICES, encoding=uint8_t),
            FloatArgument(name="angle", encoding=float32_t, units="deg"),
        ],
    )
    add_command(
        faults, "inject_imu_fault", "[SIM] Bias and/or roughen the inertial measurement unit",
        arguments=[
            FloatArgument(name="bias", encoding=float32_t, minimum=0.0, maximum=1.0),
            FloatArgument(name="noise", encoding=float32_t, minimum=0.0, maximum=1.0),
        ],
    )
    add_command(
        faults, "inject_camera_fault", "[SIM] Lose and/or corrupt camera frames",
        arguments=[
            FloatArgument(name="loss", encoding=float32_t, minimum=0.0, maximum=1.0),
            FloatArgument(name="noise", encoding=float32_t, minimum=0.0, maximum=1.0),
        ],
    )
    add_command(
        faults, "inject_battery_fault", "[SIM] Drain the battery with a parasitic load",
        arguments=[FloatArgument(name="severity", encoding=float32_t, minimum=0.0, maximum=1.0)],
    )
    # clear_faults is never dropped however high tc_loss goes - see FaultInjector.should_drop_tc.
    add_command(
        faults, "inject_comms_fault", "[SIM] Lose telemetry and/or telecommands",
        arguments=[
            FloatArgument(name="tm_loss", encoding=float32_t, minimum=0.0, maximum=1.0),
            FloatArgument(name="tc_loss", encoding=float32_t, minimum=0.0, maximum=1.0),
        ],
    )
    add_command(faults, "clear_faults", "[SIM] Restore every subsystem to nominal")


def main() -> None:
    rover = System("Rover")
    faults = Subsystem(rover, "faults")
    build_telemetry(rover, faults)
    build_camera_metadata(Subsystem(rover, "camera"))
    build_estimator(Subsystem(rover, "estimator"))
    build_commands(Subsystem(rover, "motor"), Subsystem(rover, "system"), faults)

    output = os.path.abspath(OUTPUT)
    with open(output, "wt") as handle:
        rover.dump(handle)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
