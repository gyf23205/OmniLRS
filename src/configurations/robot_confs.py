__author__ = "Antoine Richard, Aleksa Stanivuk"
__copyright__ = "Copyright 2023-26, JAOPS, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

import os
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Pose:
    position: List[float] = field(default_factory=list)
    orientation: List[float] = field(default_factory=list)


@dataclass
class RobotParameters:
    robot_name: str = field(default_factory=str)
    usd_path: str = field(default_factory=str)
    pose: Pose = field(default_factory=dict)
    domain_id: int = field(default_factory=int)
    target_links: List[str] = field(default_factory=list)
    wheel_joints: Dict = field(default_factory=dict)
    base_link: str = field(default_factory=str)
    pos_relative_to_prim: str = field(default_factory=str)
    camera: Dict = field(default_factory=dict)
    imu_sensor_path: str = field(default_factory=str)
    dimensions: dict = field(default_factory=dict)
    turn_speed_coef: float = 1
    solar_panel_joint: str = field(default_factory=str)
    scale: float = 1.0
    usd_prim_path: str = field(default_factory=str)
    # Corner steer joints, for robots that steer rather than skid. Empty means skid steering.
    steer_joints: List[str] = field(default_factory=list)
    # Rover dimensions in unscaled usd units, consumed by the Ackermann model. Scaled by `scale`.
    geometry: Dict = field(default_factory=dict)
    # Closed-loop drive controller gains and tolerances.
    drive_control: Dict = field(default_factory=dict)
    # Reference torques used only to size injected faults; never written to a healthy joint.
    fault_injection: Dict = field(default_factory=dict)
    dataset: Dict = field(default_factory=dict)
    # Onboard navigation EKF: imu axis map, noise, torque model, cusum. See estimation/nav_filter.py.
    estimator: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.usd_path = os.path.join(os.getcwd(), self.usd_path)
        self.pose = Pose(**self.pose)

@dataclass
class RobotManagerConf:
    robots_root: str = "/Robots"
    parameters: RobotParameters = None
    yamcs_tmtc: Dict = field(default_factory=dict) #TODO for v4: separate yamcs_tmtc and robot_controller into separate dataclass (and don't merge with environment configs)
    robot_controller: str = field(default_factory=str)

    def __post_init__(self):
        if self.parameters:
            self.parameters = RobotParameters(**self.parameters)
        else:
            self.parameters = None
