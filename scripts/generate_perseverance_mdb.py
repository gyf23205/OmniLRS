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
THERMAL_FACES = [
    ("front", "+X face"), ("back", "-X face"), ("left", "+Y face"),
    ("right", "-Y face"), ("top", "+Z face"), ("bottom", "-Z face"),
]


def build_telemetry(rover: System) -> None:
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


def build_commands(motor: Subsystem, rover_system: Subsystem) -> None:
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


def main() -> None:
    rover = System("Rover")
    build_telemetry(rover)
    build_commands(Subsystem(rover, "motor"), Subsystem(rover, "system"))

    output = os.path.abspath(OUTPUT)
    with open(output, "wt") as handle:
        rover.dump(handle)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
