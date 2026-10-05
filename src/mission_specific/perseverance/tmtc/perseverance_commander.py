__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

from src.subsystems.device import CommonDevice, PowerState
from src.subsystems.robot_enums import GoNogoState, SolarPanelState


class PerseveranceCommander:
    """
    Executes the high-level commands uplinked from Yamcs.

    Commands are goals, not joint targets: the ground says "drive 2 m" and
    PerseveranceDriveController turns that into wheel motion onboard, closed-loop on measured pose.
    The link is 1 Hz (and light-time delayed on a real mission), so low-level control cannot live
    on the ground.

    Every command is acknowledged the moment it arrives — logged and echoed back as the
    /Rover/active_command telemetry parameter — so an operator can see the rover received exactly
    what was sent. Execution progress then follows on /Rover/command_status and the
    command_distance_remaining / command_heading_error parameters.
    """

    def __init__(self, robot, transmitter, drive_controller, obc_handler, fault_injector=None):
        self._robot = robot
        self._transmitter = transmitter
        self._drive = drive_controller
        self._obc_handler = obc_handler
        self._faults = fault_injector

    @staticmethod
    def _format(value):
        """
        Render one argument for the acknowledgement.

        Arguments cross the wire as float32 and come back widened to float64, so an operator who
        typed 0.8 would otherwise see it echoed as 0.800000011920929. Rounding to float32's real
        precision restores what was actually sent - and this string is archived as
        /Rover/active_command, so it is worth being readable months later.
        """
        return round(value, 6) if isinstance(value, float) else value

    def _acknowledge(self, name, **kwargs):
        args = ", ".join(f"{key}={self._format(value)}" for key, value in kwargs.items())
        description = f"{name}({args})"
        print(f"[TC] received: {description}", flush=True)
        self._transmitter.set_active_command(description)
        return description

    # ── drive commands ───────────────────────────────────────────────────────────
    # The controller enforces the go_nogo / motor-power / motor-health interlocks and reports a
    # rejection reason, so these hand the goal straight over.
    def drive_straight(self, linear_velocity, distance):
        self._acknowledge("drive_straight", linear_velocity=linear_velocity, distance=distance)
        self._drive.command_straight(float(linear_velocity), float(distance))

    def drive_turn(self, angular_velocity, angle):
        self._acknowledge("drive_turn", angular_velocity=angular_velocity, angle=angle)
        self._drive.command_turn(float(angular_velocity), float(angle))

    def stop(self):
        self._acknowledge("stop")
        self._drive.command_stop()

    def goto(self, x, y):
        self._acknowledge("goto", x=x, y=y)
        self._drive.command_goto(float(x), float(y))

    # ── subsystem commands ───────────────────────────────────────────────────────
    def handle_electronics_on_off(self, subsystem_id, power_state):
        self._acknowledge("power_electronics", subsystem_id=subsystem_id, power_state=power_state)

        if subsystem_id not in list(CommonDevice):
            print(f"[TC] unknown subsystem_id: {subsystem_id}", flush=True)
            return

        device = CommonDevice(subsystem_id)
        state = PowerState(power_state)
        self._robot.subsystems.set_device_power_state(device, state)

        # Cutting motor controller power must stop the rover, not leave a command running against
        # dead motors. PragyaanCommander does the same.
        if device == CommonDevice.MOTOR_CONTROLLER and state == PowerState.OFF:
            self._drive.abort("motor controller powered off")

    def handle_solar_panel(self, deployment):
        self._acknowledge("deploy_solar_panel", deployment=deployment)

        state = SolarPanelState[deployment] if isinstance(deployment, str) else SolarPanelState(deployment)
        self._robot.subsystems.set_solar_panel_state(state)

    def handle_go_nogo(self, decision):
        self._acknowledge("go_nogo", decision=decision)

        state = GoNogoState[decision] if isinstance(decision, str) else GoNogoState(decision)
        self._robot.subsystems.set_go_nogo_state(state)

        # NOGO is the ground pulling clearance: anything in flight stops immediately.
        if state == GoNogoState.NOGO:
            self._drive.abort("go_nogo set to NOGO")

    # ── fault injection ──────────────────────────────────────────────────────────
    # Not flight commands. These reach past the rover software into the simulation to break an
    # actuator, a sensor, the battery or the link, so the closed loop can be watched compensating —
    # or failing to. None of them aborts a command: a fault degrades the rover, it does not stop it.
    # Health flags do move, but only to DEGRADED, which is deliberately not an interlock.
    def inject_wheel_torque_fault(self, wheel, severity):
        self._acknowledge("inject_wheel_torque_fault", wheel=wheel, severity=severity)
        if self._require_injector():
            self._faults.inject_wheel_torque(wheel, float(severity))

    def inject_wheel_stuck_fault(self, wheel, severity):
        self._acknowledge("inject_wheel_stuck_fault", wheel=wheel, severity=severity)
        if self._require_injector():
            self._faults.inject_wheel_stuck(wheel, float(severity))

    def inject_wheel_slip_fault(self, wheel, severity):
        self._acknowledge("inject_wheel_slip_fault", wheel=wheel, severity=severity)
        if self._require_injector():
            self._faults.inject_wheel_slip(wheel, float(severity))

    def inject_wheel_sink_fault(self, wheel, severity):
        self._acknowledge("inject_wheel_sink_fault", wheel=wheel, severity=severity)
        if self._require_injector():
            self._faults.inject_wheel_sink(wheel, float(severity))

    def inject_steer_torque_fault(self, corner, severity):
        self._acknowledge("inject_steer_torque_fault", corner=corner, severity=severity)
        if self._require_injector():
            self._faults.inject_steer_torque(corner, float(severity))

    def inject_steer_stuck_fault(self, corner, angle):
        self._acknowledge("inject_steer_stuck_fault", corner=corner, angle=angle)
        if self._require_injector():
            self._faults.inject_steer_stuck(corner, float(angle))

    def inject_imu_fault(self, bias, noise):
        self._acknowledge("inject_imu_fault", bias=bias, noise=noise)
        if self._require_injector():
            self._faults.inject_imu_fault(float(bias), float(noise))

    def inject_camera_fault(self, loss, noise):
        self._acknowledge("inject_camera_fault", loss=loss, noise=noise)
        if self._require_injector():
            self._faults.inject_camera_fault(float(loss), float(noise))

    def inject_battery_fault(self, severity):
        self._acknowledge("inject_battery_fault", severity=severity)
        if self._require_injector():
            self._faults.inject_battery_fault(float(severity))

    def inject_comms_fault(self, tm_loss, tc_loss):
        self._acknowledge("inject_comms_fault", tm_loss=tm_loss, tc_loss=tc_loss)
        if self._require_injector():
            self._faults.inject_comms_fault(float(tm_loss), float(tc_loss))

    def clear_faults(self):
        self._acknowledge("clear_faults")
        if self._require_injector():
            self._faults.clear_all()

    def _require_injector(self) -> bool:
        if self._faults is None:
            print("[TC] fault injection unavailable: no injector was built", flush=True)
            return False
        return True
