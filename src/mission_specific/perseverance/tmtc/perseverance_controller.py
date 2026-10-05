__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

from yamcs.client import YamcsClient

from src.mission_specific.perseverance.tmtc.perseverance_camera_handler import PerseveranceCameraHandler
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
        fault_injector=None,
        nav_filter=None,
    ):
        super().__init__(yamcs_instance_conf, yamcs_conf, robot_name, robot_RG, robot)
        self._intervals = yamcs_conf["intervals"]
        self._transmit_errors = 0
        # The onboard closed-loop executor. Built by the launch script, which owns the simulation
        # loop that has to tick it every physics step.
        self._drive_controller = drive_controller
        # The simulation backdoor. Optional: without it the /Rover/faults commands report that they
        # cannot act, rather than the link failing to come up.
        self._fault_injector = fault_injector
        self._transmitter: PerseveranceTransmitter = PerseveranceTransmitter(
            self.transmit_to_yamcs,
            self._intervals_handler,
            self._robot,
            self._robot_RG,
            robot_name,
            self._yamcs_conf["parameters"],
            drive_controller=drive_controller,
            fault_injector=fault_injector,
            nav_filter=nav_filter,
        )
        self._commander: PerseveranceCommander = PerseveranceCommander(
            self._robot,
            self._transmitter,
            drive_controller,
            self._obc_handler,
            fault_injector,
        )

        # Nav-cam downlink. The generic ImagesHandler writes the file and publishes where it lives;
        # giving it a storage client is what actually puts the bytes in the bucket, so the published
        # url resolves. Only perseverance does this - pragyaan's handler is left as it was.
        self._camera_conf = yamcs_conf.get("camera_downlink", {})
        self._images_handler.set_storage_client(
            YamcsClient(yamcs_instance_conf["address"]).get_storage_client(),
            instance=yamcs_instance_conf.get("instance"),
        )
        self._camera_handler = PerseveranceCameraHandler(
            self._images_handler, self._robot, self._camera_conf, fault_injector,
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
        # An injected comms fault loses this parameter on the way down. Checked here because this is
        # the single point every downlinked value passes through.
        if self._fault_injector is not None and self._fault_injector.should_drop_tm():
            return

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

    def _uplinked(self, func, name=""):
        """
        Wrap a command callback so an injected comms fault can lose it.

        Dropping here rather than inside CommandsHandler keeps the shared uplink path untouched and
        discards the command BEFORE its handler runs, which is what "the command never arrived"
        actually means.
        """
        def gated(*args, **kwargs):
            if self._fault_injector is not None and self._fault_injector.should_drop_tc(name):
                print(f"[TC] lost in transit: {name}", flush=True)
                return
            return func(*args, **kwargs)

        return gated

    def setup_command_callbacks(self, commands_conf):
        """
        Maps the Yamcs commands onto the functions that act on the simulation.

        Table-driven because every entry is now wrapped for the comms fault, and one list of
        (config key, handler, argument names) is far easier to keep in step with the mdb than
        eighteen near-identical calls.
        """
        commands = [
            # flight commands
            ("drive_straight", self._commander.drive_straight, ["linear_velocity", "distance"]),
            ("drive_turn", self._commander.drive_turn, ["angular_velocity", "angle"]),
            ("stop", self._commander.stop, []),
            ("goto", self._commander.goto, ["x", "y"]),
            ("power_electronics", self._commander.handle_electronics_on_off, ["subsystem_id", "power_state"]),
            ("solar_panel", self._commander.handle_solar_panel, ["deployment"]),
            ("go_nogo", self._commander.handle_go_nogo, ["decision"]),
            # fault injection - a simulation backdoor, hence the separate /Rover/faults mdb path
            ("inject_wheel_torque_fault", self._commander.inject_wheel_torque_fault, ["wheel", "severity"]),
            ("inject_wheel_stuck_fault", self._commander.inject_wheel_stuck_fault, ["wheel", "severity"]),
            ("inject_wheel_slip_fault", self._commander.inject_wheel_slip_fault, ["wheel", "severity"]),
            ("inject_wheel_sink_fault", self._commander.inject_wheel_sink_fault, ["wheel", "severity"]),
            ("inject_steer_torque_fault", self._commander.inject_steer_torque_fault, ["corner", "severity"]),
            ("inject_steer_stuck_fault", self._commander.inject_steer_stuck_fault, ["corner", "angle"]),
            ("inject_imu_fault", self._commander.inject_imu_fault, ["bias", "noise"]),
            ("inject_camera_fault", self._commander.inject_camera_fault, ["loss", "noise"]),
            ("inject_battery_fault", self._commander.inject_battery_fault, ["severity"]),
            ("inject_comms_fault", self._commander.inject_comms_fault, ["tm_loss", "tc_loss"]),
            ("clear_faults", self._commander.clear_faults, []),
        ]

        for key, handler, args in commands:
            self._commands_handler.add_command(
                commands_conf[key], self._uplinked(handler, key), args=args,
            )

    def start_streaming_data(self):
        """Creates the intervals that periodically downlink the rover state to Yamcs."""
        period = self._intervals["robot_stats"]

        streams = [
            ("Pose of base link", self._transmitter.transmit_pose_of_base_link, ()),
            ("IMU readings", self._transmitter.transmit_imu_readings, ()),
            ("Wheel contact forces", self._transmitter.transmit_contact_forces, ()),
            ("Motor encoder", self._transmitter.transmit_wheels_joint_angles, ()),
            ("Motor effort", self._transmitter.transmit_motor_effort, ()),
            ("Steer encoder", self._transmitter.transmit_steer_encoder, ()),
            ("Power status", self._transmitter.transmit_power_info, [period]),
            ("Thermal info", self._transmitter.transmit_thermal_info, [period]),
            ("Radio rssi", self._transmitter.transmit_radio_signal_info, ()),
            ("OBC state", self._transmitter.transmit_obc_state, ()),
            ("OBC metrics", self._transmitter.transmit_obc_metrics, ()),
            ("GO_NOGO", self._transmitter.transmit_go_nogo, ()),
            ("Solar panel state", self._transmitter.transmit_solar_panel_state, ()),
            ("Active command", self._transmitter.transmit_active_command, ()),
            ("Command progress", self._transmitter.transmit_command_progress, ()),
            ("Injected faults", self._transmitter.transmit_fault_state, ()),
            ("Navigation filter", self._transmitter.transmit_estimator, ()),
        ]

        for name, function, f_args in streams:
            self._intervals_handler.add_new_interval(
                name=name, seconds=period, is_repeating=True, execute_immediately=True,
                function=function, f_args=f_args,
            )

        # Images ride their own, much slower interval. execute_immediately is off because the very
        # first frame lands before the scene has rendered and would be empty.
        image_period = self._camera_conf.get("period", 0)
        if image_period:
            self._intervals_handler.add_new_interval(
                name="Nav camera", seconds=image_period, is_repeating=True,
                execute_immediately=False, function=self._camera_handler.transmit_camera_view,
                f_args=(),
            )
