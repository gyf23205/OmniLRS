__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

from src.mission_specific.perseverance.tmtc.perseverance_commander import PerseveranceCommander
from src.mission_specific.perseverance.tmtc.perseverance_transmitter import PerseveranceTransmitter
from src.tmtc.yamcs_TMTC import YamcsTMTC


class PerseveranceController(YamcsTMTC):
    """
    Ground-station controller for the perseverance rover.

    Follows the architecture of PragyaanController: YamcsTMTC supplies the generic handlers
    (intervals, obc, drive, commands, images) and this class provides the two mission-specific
    pieces — the command mapping and the telemetry streaming schedule.

    Telemetry groups are registered as repeating IntervalsHandler intervals at
    yamcs_tmtc.intervals.robot_stats seconds, so the rover state reaches the ground at a low,
    configurable rate independent of the physics step rate.
    """

    def __init__(
        self,
        yamcs_instance_conf,
        yamcs_conf,
        robot_name,
        robot_RG,
        robot,
        drive_controller,
    ):
        super().__init__(yamcs_instance_conf, yamcs_conf, robot_name, robot_RG, robot)
        self._intervals = yamcs_conf["intervals"]
        self._transmit_errors = 0
        # The onboard closed-loop executor. Built by the launch script, which owns the simulation
        # loop that has to tick it every physics step.
        self._drive_controller = drive_controller
        self._transmitter: PerseveranceTransmitter = PerseveranceTransmitter(
            self.transmit_to_yamcs,
            self._intervals_handler,
            self._robot,
            self._robot_RG,
            robot_name,
            self._yamcs_conf["parameters"],
            drive_controller=drive_controller,
        )
        self._commander: PerseveranceCommander = PerseveranceCommander(
            self._robot,
            self._transmitter,
            drive_controller,
            self._obc_handler,
        )

    # Rate-limit the "Yamcs is unreachable" complaint: the downlink fires once per second per
    # parameter group, and a dead server would otherwise flood the log.
    _TRANSMIT_ERROR_EVERY = 60

    def transmit_to_yamcs(self, param_name, param_value):
        """
        Downlink one parameter, tolerating a ground station that has gone away.

        Overridden because the transmit calls originate from IntervalsHandler callbacks, which run
        on Kit's update event stream. An exception escaping there would take down the subscription
        and, with it, the simulation's update loop — losing the ground station must not stop the
        rover.
        """
        try:
            super().transmit_to_yamcs(param_name, param_value)
            self._transmit_errors = 0
        except Exception as exc:
            if self._transmit_errors % self._TRANSMIT_ERROR_EVERY == 0:
                print(f"[gs] downlink failed ({param_name}): {exc}", flush=True)
            self._transmit_errors += 1

    def shutdown(self):
        """Stop the telecommand listener thread and release its UDP socket."""
        handler = self._commands_handler
        stop_event = getattr(handler, "_tc_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        thread = getattr(handler, "_tc_thread", None)
        if thread is not None:
            thread.join(timeout=getattr(handler, "SOCKET_TIMEOUT_SEC", 2.0) + 1.0)
        socket_ = getattr(handler, "_tc_socket", None)
        if socket_ is not None:
            socket_.close()

    def setup_command_callbacks(self, commands_conf):
        """Maps the Yamcs commands onto the functions that act on the simulation."""
        self._commands_handler.add_command(
            commands_conf["drive_straight"], self._commander.drive_straight,
            args=["linear_velocity", "distance"],
        )
        self._commands_handler.add_command(
            commands_conf["drive_turn"], self._commander.drive_turn,
            args=["angular_velocity", "angle"],
        )
        self._commands_handler.add_command(commands_conf["stop"], self._commander.stop)
        self._commands_handler.add_command(
            commands_conf["goto"], self._commander.goto, args=["x", "y"],
        )
        self._commands_handler.add_command(
            commands_conf["power_electronics"], self._commander.handle_electronics_on_off,
            args=["subsystem_id", "power_state"],
        )
        self._commands_handler.add_command(
            commands_conf["solar_panel"], self._commander.handle_solar_panel,
            args=["deployment"],
        )
        self._commands_handler.add_command(
            commands_conf["go_nogo"], self._commander.handle_go_nogo, args=["decision"],
        )

    def start_streaming_data(self):
        """Creates the intervals that periodically downlink the rover state to Yamcs."""
        period = self._intervals["robot_stats"]

        streams = [
            ("Pose of base link", self._transmitter.transmit_pose_of_base_link, ()),
            ("IMU readings", self._transmitter.transmit_imu_readings, ()),
            ("Wheel contact forces", self._transmitter.transmit_contact_forces, ()),
            ("Motor encoder", self._transmitter.transmit_wheels_joint_angles, ()),
            ("Power status", self._transmitter.transmit_power_info, [period]),
            ("Thermal info", self._transmitter.transmit_thermal_info, [period]),
            ("Radio rssi", self._transmitter.transmit_radio_signal_info, ()),
            ("OBC state", self._transmitter.transmit_obc_state, ()),
            ("OBC metrics", self._transmitter.transmit_obc_metrics, ()),
            ("GO_NOGO", self._transmitter.transmit_go_nogo, ()),
            ("Solar panel state", self._transmitter.transmit_solar_panel_state, ()),
            ("Active command", self._transmitter.transmit_active_command, ()),
            ("Command progress", self._transmitter.transmit_command_progress, ()),
        ]

        for name, function, f_args in streams:
            self._intervals_handler.add_new_interval(
                name=name, seconds=period, is_repeating=True, execute_immediately=True,
                function=function, f_args=f_args,
            )
