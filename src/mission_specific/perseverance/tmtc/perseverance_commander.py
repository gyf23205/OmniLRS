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

    def __init__(self, robot, transmitter, drive_controller, obc_handler):
        self._robot = robot
        self._transmitter = transmitter
        self._drive = drive_controller
        self._obc_handler = obc_handler

    def _acknowledge(self, name, **kwargs):
        args = ", ".join(f"{key}={value}" for key, value in kwargs.items())
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
