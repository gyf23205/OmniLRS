__author__ = "Antoine Richard, Junnosuke Kamohara, Aleksa Stanivuk"
__copyright__ = "Copyright 2023-26, JAOPS, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

import math
import random
import threading
import time
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R
import numpy as np
import warnings
import os

import omni
from isaacsim.core.api.world import World
import omni.graph.core as og
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.dynamic_control import _dynamic_control
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim, RigidPrim
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from WorldBuilders.pxr_utils import addDefaultOps, createXform, createObject, setDefaultOpsTyped
from src.configurations.robot_confs import RobotManagerConf
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.configurations.simulator_mode_enum import SimulatorMode
from src.environments.utils import transform_orientation_from_xyzw_into_xyz, transform_orientation_into_xyz
from src.subsystems.robot_subsystems_handler import RobotSubsystemsHandler
# from src.robots.subsystems_manager import RobotSubsystemsManager
from omni.isaac.sensor import Camera

from isaacsim.sensors.physics import _sensor

#TODO for v4: rethink which methods should be in Manager, RRG, what should be in Robot
#TODO for v4: separate into a different file (very complex and lengthy classes)
class RobotManager:
    """
    RobotManager class.
    It allows to spawn, reset, teleport robots. It also allows to automatically add namespaces to topics,
    and tfs to enable multi-robot operation."""

    def __init__(
        self,
        RM_conf: RobotManagerConf,
        mode:SimulatorMode = SimulatorMode.ROS2,
    ) -> None:
        """
        Args:
            RM_conf (RobotManagerConf): The configuration of the robot manager.
        """

        self.stage = omni.usd.get_context().get_stage() # TODO for v4: logcally an instance of a robot, should not have access to the instance of stage... change?
        self.RM_conf = RobotManagerConf(**RM_conf)
        self.is_ROS2 = mode == SimulatorMode.ROS2
        self.robot_parameters = self.RM_conf.parameters
        self.robots_root = self.RM_conf.robots_root
        createXform(self.stage, self.robots_root)
        self.robot: Robot = None   # TODO for v4: if only 1 robot, no need for a dict, just an instance
        self.robot_RG: RobotRigidGroup = None # TODO for v4: if only 1 robot, no need for a dict, just an instance

    def preload_robot(
        self,
        world: World,
    ) -> None:
        """
        Preload the robot in the scene.
        Args:
            world (Usd.Stage): The usd stage scene.
        """
        print("self.robot_parameters")

        print(self.robot_parameters)


        self.add_robot(
            self.robot_parameters.usd_path,
            self.robot_parameters.robot_name,
            self.robot_parameters.pose.position,
            self.robot_parameters.pose.orientation,
            self.robot_parameters.domain_id,
            self.robot_parameters.wheel_joints,
            self.robot_parameters.camera,
            self.robot_parameters.imu_sensor_path,
            self.robot_parameters.dimensions,
            self.robot_parameters.turn_speed_coef,
            self.robot_parameters.pos_relative_to_prim,
            self.robot_parameters.solar_panel_joint,
            self.robot_parameters.scale,
            self.robot_parameters.usd_prim_path,
            self.robot_parameters.steer_joints,
        )
        self.add_RRG(
            self.robot_parameters.robot_name,
            self.robot_parameters.target_links,
            self.robot_parameters.base_link,
            world,
        )

    def preload_robot_at_pose(
        self,
        world: World,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ) -> None:
        """
        Preload the robot in the scene.
        Args:
            world (Usd.Stage): The usd stage scene.
            position (Tuple[float, float, float]): The position of the robot. (x, y, z)
            orientation (Tuple[float, float, float, float]): The orientation of the robot. (w, x, y, z)
        """
        self.add_robot(
            self.robot_parameters.usd_path,
            self.robot_parameters.robot_name,
            position,
            orientation,
            self.robot_parameters.domain_id,
            self.robot_parameters.wheel_joints,
            self.robot_parameters.camera,
            self.robot_parameters.imu_sensor_path,
            self.robot_parameters.dimensions,
            self.robot_parameters.turn_speed_coef,
            self.robot_parameters.pos_relative_to_prim,
            self.robot_parameters.solar_panel_joint,
            self.robot_parameters.scale,
            self.robot_parameters.usd_prim_path,
            self.robot_parameters.steer_joints,
        )
        self.add_RRG(
            self.robot_parameters.robot_name,
            self.robot_parameters.target_links,
            self.robot_parameters.base_link,
            world,
        )

    def add_robot(
        self,
        usd_path: str = None,
        robot_name: str = None,
        p: Tuple[float, float, float] = [0, 0, 0],
        q: Tuple[float, float, float, float] = [0, 0, 0, 1],
        domain_id: int = None,
        wheel_joints: dict = {},
        camera_conf :dict={},
        imu_sensor_path:str="",
        dimensions:dict={},
        turn_speed_coef:float=1,
        pos_relative_to_prim:str="",
        solar_panel_joint:str="",
        scale:float=1.0,
        usd_prim_path:str="",
        steer_joints:List[str]=None,
    ) -> None:
        """
        Add a robot to the scene.

        Args:
            usd_path (str): The path of the robot's usd file.
            robot_name (str): The name of the robot.
            p (Tuple[float, float, float]): The position of the robot. (x, y, z)
            q (Tuple[float, float, float, float]): The orientation of the robot. (w, x, y, z)
            domain_id (int): The domain id of the robot. Not required if the robot is not ROS2 enabled.
        """

        if robot_name[0] != "/":
            print(robot_name)
            robot_name = "/" + robot_name

        self.robot = Robot(
            usd_path,
            robot_name,
            is_ROS2=self.is_ROS2,
            domain_id=domain_id,
            robots_root=self.robots_root,
            wheel_joints=wheel_joints,
            camera_conf=camera_conf,
            imu_sensor_path=imu_sensor_path,
            dimensions=dimensions,
            turn_speed_coef=turn_speed_coef,
            pos_relative_to_prim=pos_relative_to_prim,
            solar_panel_joint=solar_panel_joint,
            scale=scale,
            usd_prim_path=usd_prim_path,
            steer_joints=steer_joints,
        )
        self.robot.load(p, q)

    def add_RRG(
        self,
        robot_name: str = None,
        target_links: List[str] = None,
        pose_base_link: str = None,
        world = None,
    ) -> None:
        """
        Add a robot rigid group to the scene.

        Args:
            robot_name (str): The name of the robot.
            target_links (List[str]): List of link names.
            world (Usd.Stage): usd stage scene.
        """
        rrg = RobotRigidGroup(
            self.robots_root,
            robot_name,
            target_links,
            pose_base_link,
        )
        rrg.initialize(world)
        self.robot_RG = rrg
        # The robot reaches its own links' velocities, contact forces and applied forces through
        # this, so a fault injector can keep talking to the Robot alone.
        self.robot.rigid_group = rrg

    def reset_robot(self) -> None:
        """
        Reset the robot to its original position.
        """
        self.robot.reset()

    def teleport_robot(
        self, position: np.ndarray = None, orientation: np.ndarray = None
    ) -> None:
        """
        Teleport the robot to a specific position and orientation.
        """
        self.robot.teleport(position, orientation)

class Robot:
    """
    Robot class.
    It allows to spawn, reset, teleport a robot. It also allows to automatically add namespaces to topics,
    and tfs to enable multi-robot operation.
    """

    #TODO for v4: simplify and refactor the initialization, put some inits into a separate init methods, 
    #TODO for v4: lower the number of arguments, simplify
    def __init__(
        self,
        usd_path: str,
        robot_name: str,
        robots_root: str = "/Robots",
        is_ROS2: bool = False,
        domain_id: int = 0,
        wheel_joints: Dict = {},
        camera_conf:Dict = {},
        imu_sensor_path:str = "",
        dimensions:dict = {},
        turn_speed_coef:float=1,
        pos_relative_to_prim:str = "",
        solar_panel_joint:str = "",
        scale:float = 1.0,
        usd_prim_path:str = "",
        steer_joints:List[str] = None,

    ) -> None:
        """
        Args:
            usd_path (str): The path of the robot's usd file.
            robot_name (str): The name of the robot.
            robots_root (str, optional): The root path of the robots. Defaults to "/Robots".
            is_ROS2 (bool, optional): Whether the robots are ROS2 enabled or not. Defaults to False.
            domain_id (int, optional): The domain id of the robot. Defaults to 0.
            scale (float, optional): Uniform scale applied to the robot. Defaults to 1.0.
            usd_prim_path (str, optional): Prim to reference inside the usd file. Required for usd
                files that declare no default prim. Defaults to "" (use the default prim)."""

        self.stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.usd_path = str(usd_path)
        self.scale = float(scale)
        self._usd_prim_path = usd_prim_path
        self.robots_root = robots_root
        self.robot_name = robot_name
        self.robot_path = os.path.join(self.robots_root, self.robot_name.strip("/"))
        self.is_ROS2 = is_ROS2
        self.domain_id = int(domain_id)
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.root_body_id = None
        self._wheel_joint_names = wheel_joints
        self._steer_joint_names = steer_joints or []
        self._dofs = {} # dof = Degree of Freedom
        # Name-keyed dofs for per-wheel drive and corner steering; see _init_named_dofs.
        self._wheel_dofs = None
        self._steer_dofs = None
        # Tensor-api articulation view for measured joint forces, built lazily; see get_wheel_joint_efforts.
        self._force_view = None
        self._force_view_failed = False
        self._force_dof_index: Dict[str, int] = {}
        # Joint prims by name, for drive limits; and forced steer angles. See _init_joint_prims.
        self._joint_prims = None
        self._steer_overrides: Dict[str, float] = {}
        # Seized wheels: wheel name -> damping factor, and the damping the usd authored for each
        # drive joint before anything touched it. See set_wheel_damping_factor.
        self._wheel_damping_factors: Dict[str, float] = {}
        self._nominal_wheel_damping: Dict[str, float] = {}
        # Per-wheel physics materials for slip faults, by wheel name. See install_wheel_materials.
        self._wheel_materials: Dict[str, UsdPhysics.MaterialAPI] = {}
        self._nominal_wheel_friction = None
        # Per-wheel collider rest offsets for sinkage faults, and what they were before any fault.
        # See install_wheel_sinkage.
        self._wheel_rest_offsets: Dict[str, list] = {}
        self._nominal_wheel_rest_offsets: Dict[str, float] = {}
        # The RobotRigidGroup for this robot's links, set by RobotManager.add_RRG.
        self.rigid_group = None
        # Simulated sensor degradation, set by the mission's fault injector. Zero means untouched,
        # and the accessors below skip the whole path in that case so a healthy run is unaffected.
        self._imu_corruption = {
            "accel_bias": 0.0, "gyro_bias": 0.0, "orientation_bias_deg": 0.0,
            "accel_noise": 0.0, "gyro_noise": 0.0,
        }
        self._camera_noise = 0.0
        # (clean, measured) from the last get_imu_readings call; see there.
        self._last_imu = None
        # Seeded from the injector so a faulted run replays identically. random for scalars,
        # numpy for whole image arrays - drawing a per-pixel gaussian in python would be far too slow.
        self._fault_rng = random.Random(0)
        self._fault_np_rng = np.random.default_rng(0)
        self._camera_conf = camera_conf
        self._cameras = {}
        self._depth_cameras = {}
        self.dimensions = dimensions
        self.turn_speed_coef = turn_speed_coef
        self._imu_sensor_interface = _sensor.acquire_imu_sensor_interface()
        self._imu_sensor_path:str = imu_sensor_path
        self._solar_panel_joint = solar_panel_joint
        self._solar_panel_dof = None
        self._setup_subsystems_handler(pos_relative_to_prim)

    def _setup_subsystems_handler(self, pos_relative_to_prim):
        self.subsystems:RobotSubsystemsHandler = None
        robot_name = self.robot_name.strip("/")
        
        if robot_name == "pragyaan":
            from src.mission_specific.pragyaan.subsystems.pragyaan_subsystems_handler import PragyaanSubsystemsHandler
            self.subsystems = PragyaanSubsystemsHandler(pos_relative_to_prim)
        elif robot_name == "perseverance":
            from src.mission_specific.perseverance.subsystems.perseverance_subsystems_handler import PerseveranceSubsystemsHandler
            self.subsystems = PerseveranceSubsystemsHandler()

    def get_root_rigid_body_path(self) -> None:
        """
        Get the root rigid body path of the robot.
        """

        art = self.dc.get_articulation(self.robot_path)
        self.root_body_id = self.dc.get_articulation_root_body(art)

    def _get_art(self):
        return self.dc.get_articulation(self.robot_path)

    def edit_graphs(self) -> None:
        """
        Edit the graphs of the robot to add namespaces to topics and tfs.
        """

        selected_paths = []
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.robot_path)):
            l = [attr for attr in prim.GetAttributes() if attr.GetName().split(":")[0] == "graph"]
            if l:
                selected_paths.append(prim.GetPath())

        for path in selected_paths:
            prim = self.stage.GetPrimAtPath(path)
            prim.GetAttribute("graph:variable:Namespace").Set(self.robot_name)
            if self.is_ROS2:
                prim.GetAttribute("graph:variable:Context").Set(self.domain_id)

    def load(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """
        Load the robot in the scene, and automatically edit its graphs.

        Args:
            position (np.ndarray): The position of the robot.
            orientation (np.ndarray): The orientation of the robot.
        """

        self.stage = omni.usd.get_context().get_stage()
        self.set_reset_pose(position, orientation)
        if self._usd_prim_path:
            # createObject() references the usd file's default prim. Robots whose usd declares no
            # default prim (or that nest the robot under an absolute path their joints refer to)
            # need the reference targeted explicitly, so build the xform here instead.
            obj_prim, _ = createXform(self.stage, self.robot_path)
            xform = UsdGeom.Xformable(obj_prim)
            addDefaultOps(xform)
            setDefaultOpsTyped(
                xform,
                Gf.Vec3d(*position),
                Gf.Quatd(*orientation),
                Gf.Vec3d(self.scale, self.scale, self.scale),
            )
            obj_prim.GetReferences().AddReference(
                assetPath=self.usd_path,
                primPath=Sdf.Path(self._usd_prim_path),
            )
        else:
            createObject(
                self.robot_path,
                self.stage,
                self.usd_path,
                is_instance=False,
                position=Gf.Vec3d(*position),
                rotation=Gf.Quatd(*orientation),
                scale=Gf.Vec3d(self.scale, self.scale, self.scale),
            )
        # Before anything steps the simulation. PhysX validates a collider's rest and contact
        # offsets as they are authored, and rejects a finite rest offset next to an auto (-inf)
        # contact offset - which is exactly the state a later write passes through, one error per
        # wheel. Authoring both here, while the rover is still just usd, means PhysX parses the
        # wheels once, already valid. _initialize_cameras() below pumps the app, which is what
        # triggers that first parse.
        self.install_wheel_sinkage()

        self.edit_graphs()
        self._initialize_cameras()

    def get_streaming_cam_resolution(self):
        return (self._camera_conf["resolutions"]["low"][0], self._camera_conf["resolutions"]["low"][1])
    
    def get_high_cam_resolution(self):
        return (self._camera_conf["resolutions"]["high"][0], self._camera_conf["resolutions"]["high"][1])
        
    def _initialize_cameras(self) -> None:
        # Camera is a wrapper, therefore it just wraps around the camera instance if it already exists
        # otherwise it creates a new camera instance on the provided prim_path
        if "resolutions" not in self._camera_conf:
            return
        
        resolutions = list(self._camera_conf.get("resolutions").keys())

        for res in resolutions:
            self._cameras[res] = Camera(self._camera_conf["prim_path"], 
                                resolution=(self._camera_conf["resolutions"][res][0], self._camera_conf["resolutions"][res][1]))
            self._cameras[res].initialize()

        for res in resolutions:
            self._depth_cameras[res] = Camera(self._camera_conf["prim_path"], 
                                resolution=(self._camera_conf["resolutions"][res][0], self._camera_conf["resolutions"][res][1]))
            self._depth_cameras[res].initialize()
            self._depth_cameras[res].add_distance_to_image_plane_to_frame()

    def get_rgba_camera_view(self, resolution) -> np.ndarray:
        frame = self._cameras[resolution].get_rgba()

        # An injected camera fault grains the image. Only the colour channels: noise on alpha would
        # make the frame translucent rather than noisy. The camera returns an empty array until it
        # has rendered once, so guard on size rather than assuming a shape.
        if self._camera_noise and getattr(frame, "size", 0):
            noise = self._fault_np_rng.normal(0.0, self._camera_noise, size=frame[..., :3].shape)
            frame = np.array(frame, dtype=float)
            frame[..., :3] = np.clip(frame[..., :3] + noise, 0.0, 255.0)

        return frame
    
    def get_depth_camera_view(self, resolution) -> np.ndarray:
        """Returns depth image in meters as (H, W) float32 array."""
        depth = self._depth_cameras[resolution].get_depth()
        print("depth")
        print(depth)
        return depth
    
    def get_imu_readings(self):
        if (self._imu_sensor_path == ""):
            raise Exception("Path to imu sensor is not defined. Please check your .yaml configuration file. 'imu_sensor_path' should be defined on the same level as 'robot_name'.")
        
        # https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_physics_imu.html#reading-sensor-output
        sensor_reading = self._imu_sensor_interface.get_sensor_reading(self._imu_sensor_path, use_latest_data = True, read_gravity = True)
        linear_acceleration = {"ax": sensor_reading.lin_acc_x, "ay": sensor_reading.lin_acc_y, "az": sensor_reading.lin_acc_z}
        angular_velocity = {"gx":sensor_reading.ang_vel_x, "gy":sensor_reading.ang_vel_y, "gz":sensor_reading.ang_vel_z} 
        
        # orientation = sensor_reading.orientation # w, x, y, z 
        orientation = sensor_reading.orientation # x, y, z, w
        xyz_orientation = transform_orientation_from_xyzw_into_xyz(orientation) 
        orientation = {"roll":-float(xyz_orientation[0]), "pitch":-float(xyz_orientation[1]), "yaw":float(xyz_orientation[2])}

        # Snapshot both halves before and after corruption. The dataset recorder reads this to
        # supervise on how wrong the sensor is, not merely on a label saying that it is broken; on a
        # healthy imu the two are equal and the residual is zero.
        clean = (dict(linear_acceleration), dict(angular_velocity), dict(orientation))
        self._corrupt_imu(linear_acceleration, angular_velocity, orientation)
        self._last_imu = (clean, (dict(linear_acceleration), dict(angular_velocity), dict(orientation)))

        # print(linear_acceleration, angular_velocity, orientation)
        return linear_acceleration, angular_velocity, orientation


    def get_pose(self) -> List[float]:
        """
        Get the pose of the robot.
        Returns:
            List[float]: The pose of the robot. (x, y, z), (qx, qy, qz, qw)
        """
        if self.root_body_id is None:
            self.get_root_rigid_body_path()
        pose = self.dc.get_rigid_body_pose(self.root_body_id)
        return pose.p, pose.r
    

    def set_reset_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """
        Set the reset pose of the robot.

        Args:
            position (np.ndarray): The position of the robot.
            orientation (np.ndarray): The orientation of the robot.
        """

        self.reset_position = position
        self.reset_orientation = orientation

    def teleport(self, p: List[float], q: List[float]) -> None:
        """
        Teleport the robot to a specific position and orientation.

        Args:
            p (list): The position of the robot.
            q (list): The orientation of the robot. (x, y, z, w)
        """

        self.get_root_rigid_body_path()
        transform = _dynamic_control.Transform(p, q)
        self.dc.set_rigid_body_pose(self.root_body_id, transform)
        self.dc.set_rigid_body_linear_velocity(self.root_body_id, [0, 0, 0])
        self.dc.set_rigid_body_angular_velocity(self.root_body_id, [0, 0, 0])

    def reset(self) -> None:
        """
        Reset the robot to its original position and orientation.
        """

        # w = self.reset_orientation.GetReal()
        # xyz = self.reset_orientation.GetImaginary()
        self.root_body_id = None
        self.teleport(
            [self.reset_position[0], self.reset_position[1], self.reset_position[2]],
            [
                self.reset_orientation[1],
                self.reset_orientation[2],
                self.reset_orientation[3],
                self.reset_orientation[0],
            ],
        )

    def drive_straight(self, linear_velocity):
        self._set_wheels_velocity(linear_velocity, "left")
        self._set_wheels_velocity(linear_velocity, "right")

    def drive_turn(self, wheel_speed):
        print(wheel_speed)
        if (wheel_speed > 0):
            print("turns left")
        else:
            print("turns right")
        self._set_wheels_velocity(-wheel_speed, "left")
        self._set_wheels_velocity(wheel_speed, "right")

    def stop_drive(self):
        self._set_wheels_velocity(0, "left")
        self._set_wheels_velocity(0, "right")

    def _set_wheels_velocity(self, velocity, side:str):
        self._init_dofs()

        if side not in ["left","right"]:
            print("Wrong side param:", side, "Side can only be [left] or [right].")
            return

        for dof in self._dofs[side]:
            self.dc.set_dof_velocity_target(dof, velocity)

    def get_wheels_joint_angles(self):
        self._init_dofs()

        joint_angles = []
        for side in ["left","right"]:
            for dof in self._dofs[side]:
                joint_angle = self.dc.get_dof_position(dof)
                joint_angles.append(joint_angle)

        return joint_angles

    # ── per-wheel / steering control ─────────────────────────────────────────────
    # drive_straight/drive_turn above set a whole side to one speed, which is enough for skid
    # steering. Ackermann needs each wheel driven at its own speed and the corner wheels steered
    # individually, so these address joints by name instead.

    def set_wheel_velocities(self, velocities: Dict[str, float]) -> None:
        """
        Set drive joint velocity targets (rad/s), keyed by wheel name.

        Wheel names are the drive joint names with the "drive_joint_" prefix stripped, e.g.
        "front_left". Names with no matching dof are ignored so a partial dict is safe.
        """
        self._init_named_dofs()

        for wheel_name, velocity in velocities.items():
            dof = self._wheel_dofs.get(wheel_name)
            if dof is not None:
                # A seized wheel's target is pulled toward zero on every write, for the same reason
                # the steer override is: the controller rewrites every target each physics step.
                velocity = float(velocity) / self._wheel_damping_factors.get(wheel_name, 1.0)
                self.dc.set_dof_velocity_target(dof, velocity)

    def set_steer_angles(self, angles: Dict[str, float]) -> None:
        """
        Set steer joint position targets (radians), keyed by wheel name.

        The steer joints in rover_with_sensors.usd are position drives (stiffness 500,
        damping 200), so a position target is the right control mode here — unlike the drive
        joints, which are velocity drives.
        """
        self._init_named_dofs()

        for wheel_name, angle in angles.items():
            dof = self._steer_dofs.get(wheel_name)
            if dof is None:
                continue
            # A stuck joint ignores whoever is steering. The override has to be applied here, on
            # every write, because the drive controller rewrites all four targets each physics step.
            angle = self._steer_overrides.get(wheel_name, angle)
            self.dc.set_dof_position_target(dof, float(angle))

    def get_steer_angles(self) -> List[float]:
        """
        Measured steer joint angles in radians, in the order steer_joints lists them.

        The steering counterpart of get_wheels_joint_angles. Measured, not commanded: it is what
        makes a stuck or weakened steer joint visible from the ground.

        Always one entry per configured steer joint, padded with 0.0 for any that failed to resolve.
        The mdb declares a fixed-length array, so a short list would be rejected at downlink and
        cost the whole parameter rather than the one joint (_init_named_dofs already warns).
        """
        self._init_named_dofs()

        angles = []
        for joint_name in self._steer_joint_names:
            dof = self._steer_dofs.get(joint_name.replace("steer_joint_", ""))
            angles.append(float(self.dc.get_dof_position(dof)) if dof is not None else 0.0)

        return angles

    def has_steering(self) -> bool:
        """True when the robot config declared steer joints and they resolved to dofs."""
        self._init_named_dofs()
        return bool(self._steer_dofs)

    # ── joint drive limits and overrides ─────────────────────────────────────────
    # These model a degraded actuator: a torque limit makes a motor too weak to do its job, an
    # override pins a joint where it should not be. Both are deliberately generic — which joint is
    # faulted, and what a fault "severity" means, belongs to the mission-specific fault injector.

    def set_joint_max_efforts(self, efforts: Dict[str, float]) -> None:
        """
        Set the angular drive force limit on named joints, keyed by joint name.

        Names are the joint prim names as authored, e.g. "drive_joint_mid_left" or
        "steer_joint_front_left" — not the wheel names set_wheel_velocities uses, because both
        joint families share the wheel names and the limits apply to either.

        Written through the usd PhysicsDriveAPI rather than dynamic_control: the attribute is
        stage-level, so it survives world.reset() and shows up in the property panel. Passing
        float("inf") restores the unlimited default the usd ships with.
        """
        self._init_joint_prims()

        for joint_name, max_effort in efforts.items():
            drive = self._angular_drive(joint_name)
            if drive is not None:
                drive.CreateMaxForceAttr().Set(float(max_effort))

    def get_joint_max_efforts(self, joint_names: List[str]) -> Dict[str, float]:
        """
        Read back the angular drive force limits.

        An unauthored limit reads as inf, which is what the usd means by leaving maxForce out.
        """
        self._init_joint_prims()

        efforts = {}
        for joint_name in joint_names:
            drive = self._angular_drive(joint_name)
            if drive is None:
                continue
            attribute = drive.GetMaxForceAttr()
            if attribute and attribute.HasAuthoredValue():
                efforts[joint_name] = float(attribute.Get())
            else:
                efforts[joint_name] = float("inf")

        return efforts

    def set_steer_override(self, wheel_name: str, angle: float = None) -> None:
        """
        Pin one steer joint to a fixed angle in radians, or pass None to release it.

        See set_steer_angles: the override is re-applied on every write, so it holds against a
        controller that recomputes its targets at the physics rate.
        """
        if angle is None:
            self._steer_overrides.pop(wheel_name, None)
        else:
            self._steer_overrides[wheel_name] = float(angle)

    def clear_steer_overrides(self) -> None:
        self._steer_overrides.clear()

    def set_wheel_damping_factor(self, wheel_name: str, factor: float = 1.0) -> None:
        """
        Seize one drive joint by multiplying its drive damping, keyed by wheel name. 1.0 releases it.

        The drive joints are velocity drives (stiffness 0), so their damping D is the gain pulling
        the wheel toward its commanded speed, not a resistance: raising it alone would make the
        wheel track the command more stiffly, the opposite of stuck. A seized bearing is a passive
        viscous brake B on top of the motor, and the two sum into one drive exactly:

            D (w_cmd - w) - B w  =  (D + B) (w_cmd * D / (D + B) - w)

        So with factor = (D + B) / D the damping becomes D * factor and every velocity target is
        divided by factor (applied in set_wheel_velocities). At a large factor the wheel is held
        near zero and resists being dragged by the other five.

        The nominal damping is read from the usd on first touch and written back on release, so a
        seize-then-release is an exact round trip. Note a drive torque limit (set_joint_max_efforts)
        caps the combined drive, brake included.
        """
        self._init_joint_prims()
        joint_name = f"drive_joint_{wheel_name}"
        drive = self._angular_drive(joint_name)
        if drive is None:
            return

        if joint_name not in self._nominal_wheel_damping:
            self._nominal_wheel_damping[joint_name] = float(drive.GetDampingAttr().Get() or 0.0)

        factor = max(1.0, float(factor))
        drive.GetDampingAttr().Set(self._nominal_wheel_damping[joint_name] * factor)
        if factor == 1.0:
            self._wheel_damping_factors.pop(wheel_name, None)
        else:
            self._wheel_damping_factors[wheel_name] = factor

    def install_wheel_materials(self, friction: float) -> None:
        """
        Give every wheel's colliders its own physics material, at the given friction.

        Call once right after the robot is loaded, before physics has stepped. The usd binds no
        physics material anywhere, so contacts use the PhysX default; pass that default here and a
        healthy run is unchanged. Binding up front matters: PhysX picks up a new binding when it
        parses the stage, but may not re-parse a running shape, whereas changing the friction
        values on an already-bound material is applied live - which is all set_wheel_friction does.

        The combine mode is "min", so the contact uses the lower of the wheel and ground friction.
        PhysX resolves the combine mode by priority (max > multiply > min > average) and unbound
        ground uses average, so the wheel's setting wins: a slippery wheel stays slippery rather
        than being averaged back up. With a healthy wheel equal to the ground's, min and average
        give the same number.
        """
        from pxr import PhysxSchema  # Kit-only schema; imported here like the other runtime deps

        self.stage = omni.usd.get_context().get_stage()
        self._nominal_wheel_friction = float(friction)
        materials_root = os.path.join(self.robots_root, "FaultMaterials", self.robot_name.strip("/"))

        for wheel_name in self._drive_wheel_names():
            colliders = self._wheel_colliders(wheel_name)
            if not colliders:
                print(f"[warn] no colliders under wheel_{wheel_name}; slip faults cannot reach it")
                continue

            material = UsdShade.Material.Define(self.stage, f"{materials_root}/wheel_{wheel_name}")
            prim = material.GetPrim()
            physics = UsdPhysics.MaterialAPI.Apply(prim)
            physics.CreateStaticFrictionAttr().Set(self._nominal_wheel_friction)
            physics.CreateDynamicFrictionAttr().Set(self._nominal_wheel_friction)
            physics.CreateRestitutionAttr().Set(0.0)
            physx = PhysxSchema.PhysxMaterialAPI.Apply(prim)
            physx.CreateFrictionCombineModeAttr().Set("min")

            for collider in colliders:
                UsdShade.MaterialBindingAPI.Apply(collider).Bind(
                    material, UsdShade.Tokens.weakerThanDescendants, "physics",
                )
            self._wheel_materials[wheel_name] = physics

    def set_wheel_friction(self, wheel_name: str, friction: float) -> None:
        """
        Set static and dynamic friction on one wheel's material, keyed by wheel name.

        Only changes values on the material install_wheel_materials bound, so it is safe mid-run.
        """
        physics = self._wheel_materials.get(wheel_name)
        if physics is None:
            print(f"[warn] wheel '{wheel_name}' has no fault material; call install_wheel_materials "
                  f"after loading the robot")
            return

        friction = max(0.0, float(friction))
        physics.GetStaticFrictionAttr().Set(friction)
        physics.GetDynamicFrictionAttr().Set(friction)

    def reset_wheel_friction(self) -> None:
        """Put every installed wheel material back to the friction it was installed with."""
        for wheel_name in self._wheel_materials:
            self.set_wheel_friction(wheel_name, self._nominal_wheel_friction)

    def install_wheel_sinkage(self, contact_margin_m: float = 0.02) -> None:
        """
        Make each wheel collider's rest offset explicit, so a sinkage fault can change it mid-run.

        Called from load() before the wheels are first parsed, and again by a mission's fault
        injector at startup, which is a no-op re-read for a robot that came through load().
        The rest offset is the distance at which PhysX lets two shapes come to rest: 0 means
        touching, and a negative value lets the wheel settle that far INTO the ground, which is
        how a sunk wheel is modelled geometrically. An authored finite value is kept as the nominal;
        an unauthored one (auto, which is 0 for rigid bodies) is written as 0.

        PhysX rejects a finite rest offset next to an auto (-inf) contact offset, and requires the
        rest offset to be the smaller of the two. So a contact offset that is unauthored, or not
        above the nominal rest offset, is written as nominal + contact_margin_m. Sinkage only lowers
        the rest offset, so the pair stays valid mid-run.
        """
        from pxr import PhysxSchema

        self.stage = omni.usd.get_context().get_stage()
        for wheel_name in self._drive_wheel_names():
            colliders = self._wheel_colliders(wheel_name)
            if not colliders:
                print(f"[warn] no colliders under wheel_{wheel_name}; sinkage faults cannot reach it")
                continue

            attributes = []
            nominal = 0.0
            for collider in colliders:
                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider)
                attribute = collision_api.CreateRestOffsetAttr()
                value = attribute.Get() if attribute.HasAuthoredValue() else None
                if value is not None and math.isfinite(value):
                    nominal = float(value)
                attribute.Set(nominal)
                attributes.append(attribute)

                contact = collision_api.CreateContactOffsetAttr()
                contact_value = contact.Get() if contact.HasAuthoredValue() else None
                if contact_value is None or not math.isfinite(contact_value) or contact_value <= nominal:
                    contact.Set(nominal + contact_margin_m)

            self._wheel_rest_offsets[wheel_name] = attributes
            self._nominal_wheel_rest_offsets[wheel_name] = nominal

    def set_wheel_sinkage(self, wheel_name: str, depth_m: float) -> None:
        """
        Let one wheel settle depth_m into the ground (0 restores it), keyed by wheel name.

        Only changes the rest offset install_wheel_sinkage authored, so it is safe mid-run. Geometry
        only: the drag a sunk wheel feels is apply_wheel_resistance, applied every step.
        """
        attributes = self._wheel_rest_offsets.get(wheel_name)
        if attributes is None:
            print(f"[warn] wheel '{wheel_name}' has no sinkage setup; call install_wheel_sinkage "
                  f"after loading the robot")
            return

        offset = self._nominal_wheel_rest_offsets[wheel_name] - max(0.0, float(depth_m))
        for attribute in attributes:
            attribute.Set(offset)

    def reset_wheel_sinkage(self) -> None:
        """Lift every wheel back to its nominal rest offset."""
        for wheel_name in self._wheel_rest_offsets:
            self.set_wheel_sinkage(wheel_name, 0.0)

    def apply_wheel_resistance(self, coefficients: Dict[str, float], velocity_epsilon: float) -> None:
        """
        Push against each named wheel's horizontal motion with coefficient x its normal load.

        This is the drag of a wheel ploughing through soft soil: rolling resistance proportional to
        the load the wheel carries, acting at the hub against the direction it travels - forward,
        backward or sideways alike, so a sunk wheel also resists being steered through the soil. A
        wheel off the ground carries no load and feels nothing.

        External forces last one physics step, so call this every step while a sinkage fault is
        active, before the world steps. The normal load comes from the previous step's contact
        forces. Below velocity_epsilon the force fades linearly to zero instead of flipping sign
        every step, which is what an ideal Coulomb drag would do at a 33 ms step.
        """
        group = self.rigid_group
        if group is None or not coefficients:
            return

        velocities, _ = group.get_velocities()
        contacts = group.get_net_contact_forces()
        forces = np.zeros((len(group.target_links), 3))
        epsilon = max(1e-6, float(velocity_epsilon))

        for i, link in enumerate(group.target_links):
            coefficient = coefficients.get(link.replace("wheel_", "", 1), 0.0)
            if coefficient <= 0.0:
                continue
            horizontal = np.array([velocities[i, 0], velocities[i, 1], 0.0])
            speed = float(np.linalg.norm(horizontal))
            normal_load = max(0.0, float(contacts[i, 2]))
            if speed <= 0.0 or normal_load <= 0.0:
                continue
            # Full magnitude above epsilon, fading linearly to zero below it.
            forces[i] = -coefficient * normal_load * horizontal / max(speed, epsilon)

        group.apply_global_forces(forces)

    def get_wheel_joint_efforts(self, wheel_names: List[str]) -> List[float]:
        """
        Measured drive joint torques in N*m, in the order given, as solved by PhysX.

        Read from the tensor api's projected joint forces (Articulation.get_measured_joint_efforts):
        the force the joint actually transmitted along its axis this step. dynamic_control's
        STATE_EFFORT, used here before, reads about 1e-4 N*m and does not follow slope or
        acceleration at all (correlation ~0 in the estimator calibration run, where the projected
        force is 3-9 N*m and correlates with the load), so every motor_effort downlinked before this
        change is meaningless. Falls back to STATE_EFFORT only if the view cannot be built.
        Padded with 0.0 for a wheel that failed to resolve, so a fixed-length mdb array is never short.
        """
        view = self._joint_force_view()
        if view is not None:
            values = np.asarray(view.get_measured_joint_efforts()).reshape(-1)
            return [float(values[self._force_dof_index[w]]) if w in self._force_dof_index else 0.0
                    for w in wheel_names]

        self._init_named_dofs()
        efforts = []
        for wheel_name in wheel_names:
            dof = self._wheel_dofs.get(wheel_name)
            if dof is None:
                efforts.append(0.0)
                continue
            state = self.dc.get_dof_state(dof, _dynamic_control.STATE_EFFORT)
            efforts.append(float(state.effort))
        return efforts

    def _joint_force_view(self):
        """Tensor-api articulation view over the rover, built on first use; None if unavailable."""
        if self._force_view is None and not self._force_view_failed:
            try:
                from isaacsim.core.prims import SingleArticulation

                root = None
                for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.robot_path)):
                    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                        root = str(prim.GetPath())
                        break
                view = SingleArticulation(prim_path=root or self.robot_path)
                view.initialize()
                self._force_dof_index = {
                    name.replace("drive_joint_", ""): i for i, name in enumerate(view.dof_names)
                    if name.startswith("drive_joint_")
                }
                self._force_view = view
                print(f"[robot] joint force view on {root or self.robot_path}: "
                      f"{len(self._force_dof_index)} drive dofs", flush=True)
            except Exception as exc:
                self._force_view_failed = True
                print(f"[warn] joint force view unavailable, motor effort falls back to "
                      f"dynamic_control (not a real torque): {exc}", flush=True)
        return self._force_view

    def get_wheel_joint_velocities(self, wheel_names: List[str]) -> List[float]:
        """
        Measured drive joint rates in rad/s, in the order given.

        The unwrapped rate the onboard navigation filter needs at physics rate. The downlinked
        motor_encoder is an angle wrapped to one revolution and sampled at 1 Hz, which aliases at
        normal driving speeds. Padded with 0.0 for a wheel that failed to resolve, like the efforts.
        """
        self._init_named_dofs()

        velocities = []
        for wheel_name in wheel_names:
            dof = self._wheel_dofs.get(wheel_name)
            if dof is None:
                velocities.append(0.0)
                continue
            state = self.dc.get_dof_state(dof, _dynamic_control.STATE_VEL)
            velocities.append(float(state.vel))

        return velocities

    def _drive_wheel_names(self) -> List[str]:
        return [
            joint_name.replace("drive_joint_", "")
            for side in ("left", "right")
            for joint_name in self._wheel_joint_names.get(side, [])
        ]

    def _wheel_colliders(self, wheel_name: str) -> list:
        """The collision prims of one wheel's link, found by traversal like _init_joint_prims."""
        link = self.stage.GetPrimAtPath(os.path.join(self.robot_path, f"wheel_{wheel_name}"))
        if not link:
            return []
        return [prim for prim in Usd.PrimRange(link) if prim.HasAPI(UsdPhysics.CollisionAPI)]

    def clear_wheel_damping_factors(self) -> None:
        """Release every seized wheel, restoring the damping the usd authored."""
        for joint_name in list(self._nominal_wheel_damping):
            self.set_wheel_damping_factor(joint_name.replace("drive_joint_", ""), 1.0)
        self._wheel_damping_factors.clear()

    # ── simulated sensor degradation ─────────────────────────────────────────────
    # Applied inside the accessors, so every consumer sees the same broken sensor rather than only
    # whatever happens to be downlinked. As with the joint limits, these are generic knobs: what a
    # "severity" means, and which reference scales it, belongs to the mission's fault injector.

    def set_fault_seed(self, seed: int) -> None:
        """Reseed the sensor-noise streams so a faulted run replays identically."""
        self._fault_rng = random.Random(seed)
        self._fault_np_rng = np.random.default_rng(seed)

    def set_imu_corruption(self, accel_bias=0.0, gyro_bias=0.0, orientation_bias_deg=0.0,
                           accel_noise=0.0, gyro_noise=0.0) -> None:
        """
        Degrade the imu. Bias is a constant offset, noise a zero-mean gaussian per sample.

        The bias is applied identically on every axis rather than as a random per-axis vector: it
        makes the fault legible from the ground as a clean constant offset, which is what an
        operator would actually be asked to spot.
        """
        self._imu_corruption = {
            "accel_bias": float(accel_bias),
            "gyro_bias": float(gyro_bias),
            "orientation_bias_deg": float(orientation_bias_deg),
            "accel_noise": float(accel_noise),
            "gyro_noise": float(gyro_noise),
        }

    def set_camera_corruption(self, noise: float = 0.0) -> None:
        """Sensor grain added to captured frames, as one sigma in counts of 255."""
        self._camera_noise = float(noise)

    def _corrupt_imu(self, linear_acceleration, angular_velocity, orientation):
        """Apply the injected bias and noise in place. A no-op on a healthy imu."""
        corruption = self._imu_corruption
        if not any(corruption.values()):
            return

        for axis in linear_acceleration:
            linear_acceleration[axis] += corruption["accel_bias"]
            if corruption["accel_noise"]:
                linear_acceleration[axis] += self._fault_rng.gauss(0.0, corruption["accel_noise"])

        for axis in angular_velocity:
            angular_velocity[axis] += corruption["gyro_bias"]
            if corruption["gyro_noise"]:
                angular_velocity[axis] += self._fault_rng.gauss(0.0, corruption["gyro_noise"])

        # Degrees: the imu orientation is reported in degrees, and the mdb declares it that way.
        # A yaw offset here is what makes the fault reach the power and thermal models, which use
        # imu yaw for the sun angle.
        for angle in orientation:
            orientation[angle] += corruption["orientation_bias_deg"]


    def _angular_drive(self, joint_name: str):
        """The UsdPhysics angular DriveAPI of a named joint, or None with a warning."""
        prim = self._joint_prims.get(joint_name)
        if prim is None:
            print(f"[warn] joint '{joint_name}' not found under {self.robot_path}")
            return None

        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            print(f"[warn] joint '{joint_name}' has no angular drive")
            return None

        return drive

    def _init_joint_prims(self) -> None:
        """
        Resolve the robot's joint prims, keyed by joint name.

        A joint prim is authored under its child body rather than at a path derivable from its own
        name — drive_joint_mid_left sits under wheel_mid_left, steer_joint_front_left under
        steer_front_left — so find them by traversing, the way edit_graphs does.
        """
        if self._joint_prims is not None:
            return

        self.stage = omni.usd.get_context().get_stage()
        self._joint_prims = {
            prim.GetName(): prim
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.robot_path))
            if prim.IsA(UsdPhysics.Joint)
        }

    def _init_named_dofs(self) -> None:
        """
        Resolve drive and steer joints to dofs, keyed by wheel name.

        Lazily initialized for the same reason as _init_dofs: the articulation is not queryable
        from load(), so this runs on first use instead.
        """
        if self._wheel_dofs is not None:
            return

        self._wheel_dofs = {}
        self._steer_dofs = {}
        art = self._get_art()

        for side in ("left", "right"):
            for joint_name in self._wheel_joint_names.get(side, []):
                dof = self.dc.find_articulation_dof(art, joint_name)
                if dof != _dynamic_control.INVALID_HANDLE:
                    self._wheel_dofs[joint_name.replace("drive_joint_", "")] = dof

        for joint_name in self._steer_joint_names:
            dof = self.dc.find_articulation_dof(art, joint_name)
            if dof != _dynamic_control.INVALID_HANDLE:
                self._steer_dofs[joint_name.replace("steer_joint_", "")] = dof
            else:
                print(f"[warn] steer joint '{joint_name}' not found in articulation {self.robot_path}")

    def _init_dofs(self):
        #NOTE idealy, this would be initialized inside load(),
        # however, for an unknown reason art, and dc do not work well when invoked there
        # thus not populating dofs correctly
        # therefore, it was implemented as singleton, and should be called at the begging of every commanding function
        #
        # more about the use of dofs for robot movement can be read on: 
        # https://docs.isaacsim.omniverse.nvidia.com/5.0.0/python_scripting/robots_simulation.html#velocity-control
        if not self._wheel_joint_names:
            return

        if "left" in list(self._dofs.keys()):
            return # it means it is already initialized
        
        self._dofs = {
            "left": [],
            "right": []
        }
        art = self._get_art()

        for rover_side in ["left", "right"]:
            for joint_name in self._wheel_joint_names[rover_side]:
                dof = self.dc.find_articulation_dof(art, joint_name)
                self._dofs[rover_side].append(dof)

    def _init_solar_panel_dof(self):
        if self._solar_panel_dof == None and self._solar_panel_joint != "":
            art = self._get_art()
            self._solar_panel_dof = self.dc.find_articulation_dof(art, self._solar_panel_joint)
            return
        elif self._solar_panel_dof != None:
            return

        if self._solar_panel_joint == "":
            raise Exception("Solar panel joint is not specified. Please check your .yaml configuration file. 'solar_panel_joint' should be defined on the same level as 'robot_name'.")

    def deploy_solar_panel(self):
        self._init_solar_panel_dof()
        self.dc.set_dof_position_target(self._solar_panel_dof, math.radians(0))

    def stow_solar_panel(self):
        self._init_solar_panel_dof()
        self.dc.set_dof_position_target(self._solar_panel_dof, math.radians(-80))

#TODO for v4: rethink which methods should be in RRG, what should be in Robot
#TODO for v4: separate into a different file (very complex and lengthy classes)
class RobotRigidGroup:
    """
    Class which deals with rigidprims and rigidprimview of a single robot.
    It is used to retrieve world pose, and contact forces, or apply force/torque.
    """

    def __init__(self, root_path: str = "/Robots", robot_name: str = None, target_links: List[str] = None, base_link:str=None):
        """
        Args:
            root_path (str): The root path of the robots.
            robot_name (str): The name of the robot.
            target_links (List[str]): List of link names.
        """

        self.root_path = root_path
        self.robot_name = robot_name
        self.target_links = target_links
        self.prims = []
        self.prim_views = []
        self.base_link = base_link
        self.base_prim = None

    def initialize(self, world: World) -> None:
        """
        Initialize the rigidprims and rigidprimviews of the robot.

        Args:
            world (World): A Omni.isaac.core.world.World object.
        """

        self.dt = world.get_physics_dt()
        world.reset()
        self._initialize_target_links()
        self._initialize_base_link()
        world.reset()

        print("initialized")

    def _initialize_target_links(self):
        if len(self.target_links) > 0:
            for target_link in self.target_links:
                print(target_link)
                rigid_prim, rigid_prim_view = self._initialize_link(target_link)
                self.prims.append(rigid_prim)
                self.prim_views.append(rigid_prim_view)

    def _initialize_base_link(self):
        if not self.base_link:
            raise ValueError(
                f"Robot '{self.robot_name}' is missing required 'base_link' in its YAML configuration. "
                "Please add e.g. base_link: \"base_link\" under the robot's parameters."
            )
        # Use SingleXFormPrim instead of SingleRigidPrim for the base link.
        # SingleRigidPrim eagerly queries physics velocities in its constructor,
        # which fails when the physics tensor simulation view has been invalidated
        # by prior RigidPrim view creations. The base link only needs get_world_pose(),
        # so a transform-only prim is sufficient and avoids the tensor dependency.
        self.base_prim = SingleXFormPrim(
            prim_path=os.path.join(self.root_path, self.robot_name, self.base_link),
            name=f"{self.robot_name}/{self.base_link}",
        )
        print("initialized base link")

    def _initialize_link(self, link):
        rigid_prim = SingleRigidPrim(
            prim_path=os.path.join(self.root_path, self.robot_name, link),
            name=f"{self.robot_name}/{link}",
        )
        rigid_prim_view = RigidPrim(
            prim_paths_expr=os.path.join(self.root_path, self.robot_name, link),
            name=f"{self.robot_name}/{link}_view",
            track_contact_forces=True,
        )
        rigid_prim_view.initialize()

        return rigid_prim, rigid_prim_view

    def get_world_poses(self) -> np.ndarray:
        """
        Returns the world pose matrix of target links.

        Returns:
            pose (np.ndarray): The world pose matrix of target links.
        """

        n_links = len(self.target_links)
        pose = np.zeros((n_links, 4, 4))
        for i, prim in enumerate(self.prims):
            position, orientation = prim.get_world_pose()
            orientation = quat_to_rot_matrix(orientation)
            pose[i, :3, 3] = 1
            pose[i, :3, :3] = orientation
            pose[i, :3, 3] = position
        return pose

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the pose (position and orientation) of target links in the global frame.

        Notes:
        - Orientations are quaternions in (w, x, y, z) format.
        - The local coordinate system of each wheel rotates as the wheels rotate.
          To ensure consistent orientations in the global frame, the pitch rotation
          is removed. This aligns each wheel's local coordinate system with the global frame.

        Returns:
            positions (np.ndarray): The position of target links. (x, y, z)
            orientations (np.ndarray): The orientation of target links. (w, x, y, z)
        """

        n_links = len(self.target_links)
        positions = np.zeros((n_links, 3))
        orientations = np.zeros((n_links, 4))
        for i, prim in enumerate(self.prims):
            position, orientation = prim.get_world_pose()

            # Rearrange quaternion from (w, x, y, z) to (x, y, z, w) for scipy
            quaternion = [orientation[1], orientation[2], orientation[3], orientation[0]]
            rotation = R.from_quat(quaternion)

            # Remove pitch rotation to align wheel's local frame with global frame
            pitch_angle = 2 * np.arctan2(rotation.as_quat()[1], rotation.as_quat()[3])
            pitch_correction_quat = [0, -np.sin(pitch_angle / 2), 0, np.cos(pitch_angle / 2)]
            inverse_pitch_rotation = R.from_quat(pitch_correction_quat)
            rotation_corrected = rotation * inverse_pitch_rotation

            # Convert back to (w, x, y, z) and store results
            quaternion_corrected = rotation_corrected.as_quat()
            orientation_corrected = [quaternion_corrected[3], quaternion_corrected[0], quaternion_corrected[1], quaternion_corrected[2]]
            positions[i, :] = position
            orientations[i, :] = orientation_corrected
        return positions, orientations
    
    def get_pose_of_base_link(self) -> Tuple[list, list]:
        """
        Returns a pair of value representing the robot's pose, and orientation respectively, based on the base_link.

        Returns:
            position (np.ndarray): The position of base link. (x, y, z)
            orientation (np.ndarray): The orientation of base link. (x, y, z, w)
        """
        position, orientation = self.base_prim.get_world_pose()

        return position, orientation

    def get_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the linear/angular velocity of target links.

        Returns:
            linear_velocities (np.ndarray): The linear velocity of target links.
            angular_velocities (np.ndarray): The angular velocity of target links.
        """

        n_links = len(self.target_links)
        linear_velocities = np.zeros((n_links, 3))
        angular_velocities = np.zeros((n_links, 3))
        # Read through the RigidPrim views, like get_net_contact_forces. SingleRigidPrim has no
        # get_velocities() in Isaac Sim 5.x (only get_linear_velocity / get_angular_velocity), and the
        # view returns both halves in one (1, 6) read.
        for i, prim_view in enumerate(self.prim_views):
            velocities = np.asarray(prim_view.get_velocities()).reshape(-1)
            linear_velocities[i, :] = velocities[:3]
            angular_velocities[i, :] = velocities[3:6]
        return linear_velocities, angular_velocities

    def get_net_contact_forces(self) -> np.ndarray:
        """
        Returns net contact forces on each target link.

        Returns:
            contact_forces (np.ndarray): The net contact forces on each target link.
        """

        n_links = len(self.target_links)
        contact_forces = np.zeros((n_links, 3))
        for i, prim_view in enumerate(self.prim_views):
            contact_force = prim_view.get_net_contact_forces(dt = self.dt).squeeze()
            contact_forces[i, :] = contact_force
        return contact_forces

    def apply_global_forces(self, forces: np.ndarray) -> None:
        """
        Apply a world-frame force at the origin of each target link, for one physics step.

        Rows of zeros are skipped. Unlike apply_force_torque, the frame is global: a resistance
        against the direction of travel is naturally expressed in world coordinates.
        """
        assert forces.shape[0] == len(self.target_links), "given force does not have matching shape."
        for i, prim_view in enumerate(self.prim_views):
            if np.any(forces[i]):
                prim_view.apply_forces_and_torques_at_pos(
                    forces=np.asarray(forces[i], dtype=np.float32).reshape(1, 3), is_global=True,
                )

    def apply_force_torque(self, forces: np.ndarray, torques: np.ndarray) -> None:
        """
        Apply force and torque (defined in local body frame) to body frame of the four wheels.

        Args:
            forces (np.ndarray): The forces to apply to the body origin of the four wheels.
                                 (Fx, Fy, Fz) = (F_DP, F_S, F_N)
            torques (np.ndarray): The torques to apply to the body origin of the four wheels.
                                 (Mx, My, Mz0 = (M_O,-M_R, M_S)
        """

        n_links = len(self.target_links)
        assert forces.shape[0] == n_links, "given force does not have matching shape."
        assert torques.shape[0] == n_links, "given torque does not have matching shape."
        for i, prim_view in enumerate(self.prim_views):
            prim_view.apply_forces_and_torques_at_pos(forces=forces[i], torques=torques[i], is_global=False)
