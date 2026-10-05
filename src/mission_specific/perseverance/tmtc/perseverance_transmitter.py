__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

import math

import numpy as np

from src.robots.robot import Robot
from src.subsystems.device import CommonDevice
from src.subsystems.robot_enums import SolarPanelState


class PerseveranceTransmitter:
    """
    Transmitter for the perseverance rover, modeled on PragyaanTransmitter.

    Each method arranges one group of simulation data into the shape the transmit_func expects and
    hands it over. transmit_func comes from YamcsTMTC and pushes the value into Yamcs, where the
    parameter archive stores it. Methods here are registered as repeating IntervalsHandler
    intervals by PerseveranceController.start_streaming_data(), which is what makes the downlink
    low-rate (1 Hz) relative to the 30 Hz physics loop.

    Differences from PragyaanTransmitter: no neutron spectrometer / APXS payload, no lander-relative
    pose, and it adds wheel contact forces plus an echo of the last high-level command received.
    """

    # Order matches the target_links list in cfg/robot/perseverance.yaml.
    WHEEL_NAMES = (
        "front_left", "front_right",
        "mid_left", "mid_right",
        "rear_left", "rear_right",
    )

    def __init__(
        self,
        transmit_func,
        intervals_handler,
        robot,
        robot_RG,
        robot_name,
        parameters_conf,
        drive_controller=None,
        fault_injector=None,
        nav_filter=None,
    ):
        self._transmit = transmit_func
        self._robot: Robot = robot
        self._robot_RG = robot_RG
        self._intervals_handler = intervals_handler
        self._parameters_conf = parameters_conf
        self._robot_name = robot_name
        self._drive = drive_controller
        self._faults = fault_injector
        self._nav = nav_filter
        self._active_command = "none"

    # ── command echo and execution progress ──────────────────────────────────────
    def set_active_command(self, description: str):
        """Record the last high-level command accepted by the rover, for downlink."""
        self._active_command = description

    def transmit_active_command(self):
        self._transmit(self._parameters_conf["active_command"], self._active_command)

    def transmit_command_progress(self):
        """
        Downlink how the active command is going.

        Without this an operator sees only that a command was received, with no way to tell
        "still driving" from "refused because NOGO" until the rover visibly stops.
        """
        if self._drive is None:
            return

        self._transmit(self._parameters_conf["command_status"], self._drive.status.value)
        self._transmit(self._parameters_conf["command_distance_remaining"], float(self._drive.distance_remaining))
        self._transmit(self._parameters_conf["command_heading_error"], float(self._drive.heading_error_deg))

        target_x, target_y = self._drive.target
        self._transmit(
            self._parameters_conf["command_target"],
            {"x": float(target_x), "y": float(target_y)},
        )

    # ── injected faults ──────────────────────────────────────────────────────────
    def transmit_fault_state(self):
        """
        Downlink what has been broken on purpose.

        Sent so that a run in the archive can be read months later without guessing why the rover
        crabbed: the injected state is recorded alongside its consequences.
        """
        if self._faults is None:
            return

        self._transmit(self._parameters_conf["faults_active"], self._faults.active_faults)
        self._transmit(self._parameters_conf["wheel_torque_limit"], self._faults.wheel_torque_limits)
        self._transmit(self._parameters_conf["wheel_damping_factor"], self._faults.wheel_damping_factors)
        self._transmit(self._parameters_conf["wheel_friction"], self._faults.wheel_frictions)
        self._transmit(self._parameters_conf["wheel_sinkage"], self._faults.wheel_sinkages)
        self._transmit(self._parameters_conf["steer_torque_limit"], self._faults.steer_torque_limits)

        # Health is ground truth here: the injector knows exactly what it broke, so it can name the
        # individual actuator instead of flagging the whole subsystem. Nothing is inferred from
        # dynamics - see the fault injector for why no detection logic is pretended.
        self._transmit(self._parameters_conf["wheel_health"], self._faults.wheel_health)
        self._transmit(self._parameters_conf["steer_health"], self._faults.steer_health)
        self._transmit(self._parameters_conf["motor_controller_health"], self._faults.motor_controller_health)
        self._transmit(self._parameters_conf["imu_health"], self._faults.imu_health)
        self._transmit(self._parameters_conf["camera_health"], self._faults.camera_health)
        self._transmit(self._parameters_conf["battery_health"], self._faults.battery_health)
        self._transmit(self._parameters_conf["comms_health"], self._faults.comms_health)

        self._transmit(self._parameters_conf["parasitic_load"], float(self._faults.parasitic_load))
        dropped = self._faults.dropped_counts
        self._transmit(self._parameters_conf["dropped"], {
            "tm": int(dropped["tm"]),
            "tc": int(dropped["tc"]),
            "camera_frames": int(dropped["camera_frames"]),
        })

    # ── pose / motion ────────────────────────────────────────────────────────────
    def transmit_pose_of_base_link(self):
        position, orientation = self._robot_RG.get_pose_of_base_link()
        position = np.asarray(position).tolist()
        orientation = np.asarray(orientation).tolist()
        pose_of_base_link = {
            "position": {"x": position[0], "y": position[1], "z": position[2]},
            "orientation": {"w": orientation[0], "x": orientation[1], "y": orientation[2], "z": orientation[3]},
        }
        self._transmit(self._parameters_conf["pose_of_base_link"], pose_of_base_link)

    def transmit_wheels_joint_angles(self):
        angles = self._robot.get_wheels_joint_angles()
        self._transform_joint_angles(angles)
        self._transmit(self._parameters_conf["motor_encoder"], angles)

    def transmit_motor_effort(self):
        """
        Downlink the measured torque on each drive joint, in WHEEL_NAMES order.

        A genuine reading, so it sits beside the encoders rather than under /Rover/faults. It is what
        tells a wheel spinning on ice (low effort) from one grinding through soft soil (high effort):
        both show the encoder running ahead of the pose.
        """
        efforts = self._robot.get_wheel_joint_efforts(list(self.WHEEL_NAMES))
        self._transmit(self._parameters_conf["motor_effort"], efforts)

    def transmit_steer_encoder(self):
        """
        Downlink the measured corner steer angles in degrees.

        Sent as a real angle rather than encoder counts because the sign matters: a corner stuck at
        -20 deg and one stuck at +20 deg send the rover opposite ways.
        """
        angles = self._robot.get_steer_angles()
        self._transmit(self._parameters_conf["steer_encoder"], [math.degrees(a) for a in angles])

    def _transform_joint_angles(self, angles):
        """Wrap to one revolution and scale to the 10-bit encoder counts the MDB declares."""
        for i in range(len(angles)):
            modulo = angles[i] % (2 * math.pi)
            angles[i] = int(modulo * (1024.0 / (2 * math.pi)))

    def transmit_contact_forces(self):
        forces = self._robot_RG.get_net_contact_forces()
        magnitudes = np.linalg.norm(forces, axis=1)
        for name, magnitude in zip(self.WHEEL_NAMES, magnitudes):
            self._transmit(self._parameters_conf[f"contact_force_{name}"], float(magnitude))

    def transmit_imu_readings(self):
        imu_accelerometer, imu_gyroscope, orientation = self._robot.get_imu_readings()
        self._transmit(self._parameters_conf["imu_accelerometer"], imu_accelerometer)
        self._transmit(self._parameters_conf["imu_gyroscope"], imu_gyroscope)
        self._transmit(self._parameters_conf["imu_orientation"], orientation)

    # ── onboard navigation filter ────────────────────────────────────────────────
    ESTIMATOR_FIELDS = (
        "pose", "motion", "bias", "imu_residual", "nis",
        "wheel_rolling_mean", "wheel_rolling_max", "wheel_sideslip_mean", "wheel_sideslip_max",
        "wheel_effort_mean", "wheel_effort_max", "wheel_tracking_mean", "wheel_tracking_max",
        "cusum_alarm", "reacquisitions",
    )

    def transmit_estimator(self):
        """
        Downlink the navigation filter's last closed window.

        A field the filter did not produce is skipped rather than zero-filled - the torque residual
        before the torque model is calibrated, or anything in the first second of an episode - so a
        missing value reads as missing, not as a healthy zero.
        """
        if self._nav is None:
            return
        snapshot = self._nav.telemetry()
        for field in self.ESTIMATOR_FIELDS:
            if field in snapshot:
                self._transmit(self._parameters_conf[f"estimator_{field}"], snapshot[field])

    # ── subsystems ───────────────────────────────────────────────────────────────
    def transmit_power_info(self, interval_s):
        robot_position, _ = self._robot_RG.get_pose_of_base_link()
        _, _, imu_orientation = self._robot.get_imu_readings()
        obc_state = self._robot.subsystems.get_obc_state()
        power_status = self._robot.subsystems.get_power_status(
            robot_position,
            imu_orientation["yaw"],
            interval_s,
            obc_state,
        )

        self._transmit(self._parameters_conf["battery_charge"], int(power_status["battery_percentage_measured"]))
        self._transmit(self._parameters_conf["battery_voltage"], power_status["battery_voltage_measured"])
        self._transmit(self._parameters_conf["total_current_in"], power_status["solar_input_current_measured"])
        self._transmit(self._parameters_conf["total_current_out"], power_status["total_current_out_measured"])
        self._transmit(self._parameters_conf["net_power"], power_status["net_power"])
        self._transmit(self._parameters_conf["motor_current"], power_status["motor_currents_measured"])

        device_currents = power_status["device_currents_measured"]
        self._transmit(self._parameters_conf["current_draw_obc"], device_currents[CommonDevice.OBC])
        self._transmit(self._parameters_conf["current_draw_motor_controller"], device_currents[CommonDevice.MOTOR_CONTROLLER])
        self._transmit(self._parameters_conf["current_draw_camera"], device_currents[CommonDevice.CAMERA])
        self._transmit(self._parameters_conf["current_draw_radio"], device_currents[CommonDevice.RADIO])
        self._transmit(self._parameters_conf["current_draw_eps"], device_currents[CommonDevice.EPS])

    def transmit_thermal_info(self, interval_s):
        robot_position, _ = self._robot_RG.get_pose_of_base_link()
        _, _, imu_orientation = self._robot.get_imu_readings()
        temperatures = self._robot.subsystems.get_thermal_status(
            robot_position,
            imu_orientation["yaw"],
            interval_s,
        )
        self._transmit(self._parameters_conf["temperature_front"], temperatures["+X"])
        self._transmit(self._parameters_conf["temperature_back"], temperatures["-X"])
        self._transmit(self._parameters_conf["temperature_left"], temperatures["+Y"])
        self._transmit(self._parameters_conf["temperature_right"], temperatures["-Y"])
        self._transmit(self._parameters_conf["temperature_top"], temperatures["+Z"])
        self._transmit(self._parameters_conf["temperature_bottom"], temperatures["-Z"])

    def transmit_radio_signal_info(self):
        robot_position, _ = self._robot_RG.get_pose_of_base_link()
        rssi = self._robot.subsystems.get_radio_status(robot_position)
        self._transmit(self._parameters_conf["rssi"], int(rssi))

    def transmit_obc_metrics(self):
        obc_metrics = self._robot.subsystems.get_obc_status()
        self._transmit(self._parameters_conf["obc_cpu_usage"], int(obc_metrics["cpu_usage"]))
        self._transmit(self._parameters_conf["obc_ram_usage"], int(obc_metrics["ram_usage"]))
        self._transmit(self._parameters_conf["obc_disk_usage"], int(obc_metrics["disk_usage"]))
        self._transmit(self._parameters_conf["obc_uptime"], obc_metrics["uptime"])

    def transmit_obc_state(self):
        self._transmit(self._parameters_conf["obc_state"], self._robot.subsystems.get_obc_state().value)

    def transmit_go_nogo(self):
        self._transmit(self._parameters_conf["go_nogo"], self._robot.subsystems.get_go_nogo_state().value)

    def transmit_solar_panel_state(self):
        state: SolarPanelState = self._robot.subsystems.get_solar_panel_state()
        self._transmit(self._parameters_conf["solar_panel_state"], state.value)
