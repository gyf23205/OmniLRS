__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

import math

from src.subsystems.device import CommonDevice, Device, HealthState, PowerState
from src.subsystems.robot_enums import ObcState
from src.subsystems.robot_physics_models.obc_metrics_model import ObcMetricsModel
from src.subsystems.robot_physics_models.power_model import PowerModel
from src.subsystems.robot_physics_models.radio_model import RadioModel
from src.subsystems.robot_physics_models.thermal_model import ThermalModel
from src.subsystems.robot_subsystems_handler import RobotSubsystemsHandler


class PerseveranceSubsystemsHandler(RobotSubsystemsHandler):
    """
    Subsystems handler for the perseverance rover.

    Unlike PragyaanSubsystemsHandler, this one needs no mission-specific physics models: the stock
    PowerModel / ThermalModel / RadioModel / ObcMetricsModel cover everything the perseverance
    telemetry set reports. Only the device list and the battery/motor sizing are rover-specific.

    Sun position: PragyaanSubsystemsHandler hardcodes a fixed azimuth. Here the sun is settable via
    set_sun_position() so the simulation can feed the live stellar engine position each step. Until
    it does, SUN_FALLBACK (the static azimuth/elevation the launch scripts use) applies.
    """

    # Sizing of the perseverance power system.
    BATTERY_CAPACITY_WH = 60.0  # Watt-hours
    SOLAR_PANEL_MAX_POWER = 30.0  # Watts
    MOTOR_COUNT = 6
    MOTOR_POWER_W = 10.0

    # Static sun direction, matching the SunConf defaults of the launch scripts. Used until the
    # simulation starts pushing the stellar engine position through set_sun_position().
    SUN_DISTANCE = 1000.0  # m
    SUN_AZIMUTH_DEG = 180.0
    SUN_ELEVATION_DEG = 45.0
    SUN_FALLBACK = (
        SUN_DISTANCE * math.cos(math.radians(SUN_ELEVATION_DEG)) * math.sin(math.radians(SUN_AZIMUTH_DEG)),
        SUN_DISTANCE * math.cos(math.radians(SUN_ELEVATION_DEG)) * math.cos(math.radians(SUN_AZIMUTH_DEG)),
        SUN_DISTANCE * math.sin(math.radians(SUN_ELEVATION_DEG)),
    )

    # The rover talks to a fixed ground relay for the RSSI model. No lander in this scene, so the
    # relay sits at the origin of the environment.
    RELAY_POSITION = (0.0, 0.0, 0.0)

    def __init__(self, relay_position=None):
        super().__init__(
            power_model=PowerModel(),
            thermal_model=ThermalModel(),
            radio_model=RadioModel(),
            obc_metrics_model=ObcMetricsModel(),
        )
        self._setup_devices()
        self._setup_power_model()
        self._sun_pos = self.SUN_FALLBACK
        self._relay_pos = relay_position if relay_position is not None else self.RELAY_POSITION

    def _setup_devices(self):
        self._devices[CommonDevice.OBC] = Device(CommonDevice.OBC, current_draw=(0.0, 7.5), power_state=PowerState.ON)
        self._devices[CommonDevice.MOTOR_CONTROLLER] = Device(CommonDevice.MOTOR_CONTROLLER, current_draw=(0.0, 2.0), power_state=PowerState.ON)
        self._devices[CommonDevice.CAMERA] = Device(CommonDevice.CAMERA, current_draw=(0.0, 5.0), power_state=PowerState.ON)
        self._devices[CommonDevice.RADIO] = Device(CommonDevice.RADIO, current_draw=(0.0, 5.0), power_state=PowerState.ON)
        self._devices[CommonDevice.EPS] = Device(CommonDevice.EPS, current_draw=(0.0, 1.0), power_state=PowerState.ON)
        self._devices["imu"] = Device("imu", current_draw=(0.0, 0.5), power_state=PowerState.ON)

    def _setup_power_model(self):
        self._power_model.initialize(
            battery_capacity_wh=self.BATTERY_CAPACITY_WH,
            battery_charge_wh=self.BATTERY_CAPACITY_WH,
            solar_panel_max_power=self.SOLAR_PANEL_MAX_POWER,
            solar_panel_state=self._solar_panel_state,
            motor_count=self.MOTOR_COUNT,
            motor_power_w=self.MOTOR_POWER_W,
            devices=self._devices,
        )

    def set_sun_position(self, sun_position):
        """Feed the live stellar engine sun position. Called from the simulation loop."""
        self._sun_pos = sun_position

    def get_sun_position(self):
        return self._sun_pos

    def get_radio_status(self, robot_position):
        self._radio_model.set_inputs(self._relay_pos, robot_position)
        return self._radio_model.get_rssi()

    def get_thermal_status(self, robot_position, robot_yaw_deg, interval_s):
        self._thermal_model.set_inputs(robot_position, self._sun_pos, robot_yaw_deg)
        self._thermal_model.compute(interval_s)
        return self._thermal_model.temperatures()

    def get_power_status(self, robot_position, robot_yaw_deg, interval_s, obc_state):
        # device states are reflected between the handler and power model, as they use the same dict
        self._power_model.set_inputs(
            rover_position=robot_position,
            sun_position=self._sun_pos,
            rover_yaw_deg=robot_yaw_deg,
            solar_panel_state=self._solar_panel_state,
            is_in_motor_state=(obc_state == ObcState.MOTOR),
        )
        self._power_model.compute(interval_s)
        return self._power_model.get_outputs()

    def set_device_health_state(self, device_name: str, state: HealthState):
        if device_name not in self._devices:
            print("Invalid electronics naming: ", device_name)
            return

        self._devices[device_name].set_health_state(state)
