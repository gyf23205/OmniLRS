__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
The episode loop for dataset generation.

Replaces the interactive loop rather than living inside it: no keyboard, no ui panels, no ground
station. What it keeps from that loop is the ordering, because the ordering is load-bearing -
commands, then drive, then fault_injector.update(), then the power integration, all between
world.step() calls. Nothing here writes usd off the simulation thread.

RESETTING IS THE HARD PART. Nothing in the subsystems stack has a reset; battery charge, face
temperatures, obc uptime, go/nogo and every injected fault persist for the life of the process. An
episode that inherited the previous one's flat battery would be labelled nominal and look nothing
like one. _reset_episode below is that list, and it is deliberately explicit about which of the
resets are reaching past a public interface.

GO/NOGO. The interlock defaults to NOGO and the only thing that ever sets it is the Yamcs command
handler, so a run without a ground station has every command rejected. Episode setup sets GO.

Determinism is bounded and the manifest says so: seeding covers the fault rng and the measurement
noise (which all comes off the global random module), but PhysX determinism is not configured, so
episodes replay in distribution rather than bit-exactly.

No omni/pxr imports - the world is duck-typed as something with step()/is_playing(), and the pose
comes from the robot group rather than an xform cache.
"""

import math
import os
import random
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from src.mission_specific.perseverance.dataset import crater as crater_module
from src.mission_specific.perseverance.dataset import fault_scheduler
from src.mission_specific.perseverance.dataset.mission_scripter import MissionScripter
from src.subsystems.device import CommonDevice, PowerState
from src.subsystems.robot_enums import GoNogoState, ObcState, SolarPanelState

# kind -> (injector method name, whether it takes a target)
INJECTORS = {
    "wheel_torque": ("inject_wheel_torque", True),
    "wheel_stuck": ("inject_wheel_stuck", True),
    "wheel_slip": ("inject_wheel_slip", True),
    "wheel_sink": ("inject_wheel_sink", True),
    "steer_torque": ("inject_steer_torque", True),
    "steer_stuck": ("inject_steer_stuck", True),
    "imu": ("inject_imu_fault", False),
    "camera": ("inject_camera_fault", False),
    "battery": ("inject_battery_fault", False),
    "comms": ("inject_comms_fault", False),
}

DEFAULTS = {
    "episode_steps": 18000,        # physics steps; 600 s of simulation at 33 ms
    "downlink_period_s": 1.0,      # matches yamcs_tmtc.intervals.robot_stats
    "image_period_s": 5.0,         # matches camera_downlink.period
    "settle_steps": 30,            # after a teleport, before the episode starts recording
    # Rendering is most of the cost of a step, and only the nav-cam needs it: telemetry, the imu and
    # the faults all read PhysX directly. So an episode steps physics alone and renders only to take a
    # frame - world.render(), which updates the renderer WITHOUT stepping physics, repeated
    # render_warmup_frames times at the capture pose.
    #
    # Why more than once: the renderer accumulates across frames, and after a stretch of physics-only
    # steps its history is of the scene as it was at the previous capture. One render returns a blend
    # of the two. Warming up on the spot converges that history at the pose the frame is meant to show,
    # which stepping cannot do - every step of warm-up would move the rover further.
    # render_every_step restores the old render-every-step behaviour for comparison or for the GUI.
    "render_every_step": False,
    "render_warmup_frames": 8,
    # The rover is dropped from spawn height (3 m falls for ~1.8 s under lunar gravity) and must land
    # and settle before the operator's first command, or driving it mid-landing can flip it. The settle
    # lasts at least this long; nothing is commanded or recorded during it.
    "landing_hold_s": 3.0,
    # The rover is placed this far above the highest ground under it, rather than at the fixed
    # spawn height. A 3 m drop flipped the rover while the launch script still left Isaac at Earth
    # gravity, and a low spawn is gentler at any gravity. Needs an environment with get_terrain() and terrain_resolution.
    "spawn_clearance_m": 0.15,
    "spawn_footprint_radius_m": 0.75,  # covers the rover's ~0.8 x 0.8 m footprint at scale 0.35
    "spawn_jitter_m": 3.0,         # how far an episode's start may wander from the configured spawn
    "rocks_per_episode": 8,
    # A driven rover never gets this far from the origin, so a pose beyond it means the solver blew
    # up - PhysX reports positions around 1e13 when it does. Unattended generation has to notice that
    # and move on rather than record hours of garbage. A rover driving off the terrain is caught much
    # earlier by the fall check below; without it, it free-falls for ~25 s before reaching this.
    "max_position_m": 500.0,
    # The rover has left the terrain once its base is outside the DEM, or this far below the ground
    # under it. Needs an environment with get_terrain() and terrain_resolution; otherwise only the
    # max_position_m check applies.
    "fall_depth_m": 0.5,
    # Rows from this long before a fall is detected are discarded: the rover was already tipping
    # over the edge, and those rows would be labelled as ordinary driving.
    "fall_trim_s": 3.0,
    "recovery_settle_steps": 120,  # extra steps to let PhysX settle after a divergence
}


def merged_config(config: Optional[Dict] = None) -> Dict:
    merged = dict(DEFAULTS)
    for key, value in dict(config or {}).items():
        if key in merged:
            merged[key] = value

    return merged


def _is_sane(position, limit: float) -> bool:
    """True while the pose is one a driven rover could actually have."""
    return all(math.isfinite(float(v)) for v in position) and \
        max(abs(float(v)) for v in position) <= limit


def _yaw_deg(orientation) -> float:
    """Yaw from a (w, x, y, z) quaternion, matching the convention the power model expects."""
    w, x, y, z = (float(v) for v in orientation)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class EpisodeRunner:
    """Runs episodes back to back in one simulator process, recording each."""

    def __init__(
        self,
        world,
        robot_manager,
        drive_controller,
        fault_injector,
        recorder,
        sun_fn: Callable[[], Tuple[float, float, float]],
        physics_dt: float,
        spawn: Tuple[float, float, float],
        episodes: int = 1,
        seed: int = 0,
        config: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        mission_config: Optional[Dict] = None,
        environment=None,
        randomize_terrain: bool = False,
        camera_resolution: str = "low",
        record_images: bool = True,
        is_running: Optional[Callable[[], bool]] = None,
        crater_config: Optional[Dict] = None,
        terrain_resolution: Optional[float] = None,
        fault_specs: Optional[Sequence[str]] = None,
        nav_filter=None,
    ):
        self._world = world
        self._RM = robot_manager
        self._robot = robot_manager.robot
        self._robot_RG = robot_manager.robot_RG
        self._drive = drive_controller
        self._faults = fault_injector
        self._recorder = recorder
        # Onboard navigation filter. Optional; ticked every step after the faults, reset per episode.
        self._nav = nav_filter
        self._sun_fn = sun_fn
        self._dt = float(physics_dt)
        self._spawn = tuple(float(v) for v in spawn)
        self._episodes = int(episodes)
        self._seed = int(seed)
        self._config = merged_config(config)
        self._scheduler_config = scheduler_config
        # --fault specs. Given, they replace the sampled schedule in every episode.
        self._fault_specs = list(fault_specs or [])
        # Fail at launch, not after the scene has loaded and the first episode is half recorded.
        for spec in self._fault_specs:
            fault_scheduler.parse_spec(spec)
        self._environment = environment
        self._randomize_terrain = bool(randomize_terrain)
        self._terrain_resolution = terrain_resolution
        self._camera_resolution = camera_resolution
        self._record_images = bool(record_images)
        self._is_running = is_running or (lambda: True)

        # None disables crater episodes entirely, including their oracle columns, so a run without
        # the flag writes exactly the dataset it wrote before craters existed.
        self._crater_config = None if crater_config is None else crater_module.merged_config(crater_config)
        # The terrain as it was before any crater was cut into it, and whether one is there now.
        self._pristine_terrain = None
        self._crater_on_terrain = False
        self._crater_entered_s = None
        # The crater actually on the terrain this episode. Labels follow this, not the schedule, so
        # a crater that failed to stamp can never produce an "in crater" label.
        self._stamped_crater = None

        self._subsystems = self._robot.subsystems
        self._scripter = MissionScripter(drive_controller, fault_injector, mission_config, seed=seed,
                                         crater_config=crater_config, subsystems=self._subsystems)

        # Where the fault labels come from: the injector calls this the instant a fault takes
        # effect, on the simulation thread, which is later than when the schedule asked for it.
        self._faults.set_on_apply(self._on_fault_applied)

        self._sim_time = 0.0
        self._step = 0
        self._images_saved = 0
        self._diverged = False
        # This episode's height map, read once after the scene is set up. None disables the fall check.
        self._dem = None

    # ── the run ──────────────────────────────────────────────────────────────────
    def run(self) -> Dict:
        stop_reason = ""
        for index in range(self._episodes):
            reason = self._recorder.disk_stop_reason()
            if reason:
                stop_reason = reason
                print(f"[dataset] stopping before episode {index}: {reason}", flush=True)
                break
            if not self._is_running():
                stop_reason = "simulator closed"
                break

            episode_seed = self._seed * 100003 + index
            duration_s = self._config["episode_steps"] * self._dt
            if self._fault_specs:
                schedule = fault_scheduler.from_specs(self._fault_specs, episode_seed, duration_s,
                                                      self._scheduler_config)
            else:
                schedule = fault_scheduler.generate(episode_seed, duration_s, self._scheduler_config,
                                                    crater_config=self._crater_config)

            spawn = self._reset_episode(episode_seed, index, schedule)
            if self._diverged:
                # _reset_episode clears the flag when the teleport brought the rover back. Still
                # set means the articulation is gone, and every further episode would be garbage.
                stop_reason = "the rover could not be recovered after a physics divergence"
                print(f"[dataset] stopping: {stop_reason}", flush=True)
                break

            self._recorder.begin_episode(index, episode_seed, schedule, meta={
                "spawn": [round(v, 3) for v in spawn],
                "physics_dt": self._dt,
                "downlink_period_s": self._config["downlink_period_s"],
                "image_period_s": self._config["image_period_s"],
            })

            outcome = "complete"
            try:
                outcome = self._run_episode(schedule)
            finally:
                if schedule.crater is not None:
                    self._recorder.update_episode_meta(
                        crater_stamped=self._stamped_crater is not None,
                        crater_entered_s=(
                            None if self._crater_entered_s is None else round(self._crater_entered_s, 3)
                        ),
                    )
                summary = self._recorder.end_episode(outcome)
                if self._nav is not None:
                    dump = self._nav.save_dump(f"episode_{index:04d}")
                    if dump:
                        print(f"[dataset] estimator calibration dump: {dump}", flush=True)

            print(f"[dataset] episode {index:04d} {schedule.episode_class:<10} "
                  f"{summary.get('downlink_ticks', 0)} ticks  "
                  f"{summary.get('images_saved', 0)} frames  "
                  f"{summary.get('fault_events', 0)} fault events  "
                  f"({summary.get('wall_seconds', 0)} s wall)", flush=True)

        manifest = self._recorder.finalize({
            "run_seed": self._seed,
            "requested_episodes": self._episodes,
            "stopped_early": stop_reason,
            "scheduler_config": fault_scheduler.merged_config(self._scheduler_config),
            "fault_specs": self._fault_specs,
        })
        print(f"[dataset] wrote {manifest['episodes']} episodes; "
              f"sample-level faulted fraction "
              f"{manifest['class_balance']['sample_level_faulted_fraction']:.3f}", flush=True)

        return manifest

    def _run_episode(self, schedule) -> str:
        steps = int(self._config["episode_steps"])
        downlink = float(self._config["downlink_period_s"])
        image_period = float(self._config["image_period_s"])

        render_every_step = bool(self._config["render_every_step"])

        next_downlink, next_image = 0.0, image_period
        attempt = 0
        self._step = 0

        while self._step < steps:
            if not self._is_running():
                return "aborted: simulator closed"

            # Physics only; _capture_image renders in place when a frame is due.
            self._world.step(render=render_every_step)
            if not self._world.is_playing():
                continue

            self._sim_time = self._step * self._dt

            # 1. faults whose time has come, through the same public methods the ground uses
            for event in schedule.pop_due(self._sim_time):
                self._inject(event)

            # 2. the operator
            position, orientation = self._robot_RG.get_pose_of_base_link()
            position = np.asarray(position).tolist()

            if not _is_sane(position, float(self._config["max_position_m"])):
                print(f"[dataset] episode abandoned at step {self._step}: the rover diverged "
                      f"(position {position}). Recording what was captured so far.", flush=True)
                self._diverged = True
                return "aborted: physics diverged"

            if self._off_terrain(position):
                cut = self._sim_time - float(self._config["fall_trim_s"])
                print(f"[dataset] episode abandoned at step {self._step}: the rover left the terrain "
                      f"(position {[round(v, 2) for v in position]}). Keeping rows up to {cut:.1f} s.",
                      flush=True)
                self._recorder.discard_after(cut)
                return "aborted: fell off terrain"

            record = self._scripter.update(self._sim_time, (position[0], position[1]))
            if record is not None:
                self._recorder.log_command(
                    self._sim_time, self._step, record["command"], record["arguments"],
                    delivered=record["delivered"], note=record["note"],
                )

            # 3. closed-loop control, then 4. the queued faults, in that order - the same ordering
            # the interactive loop uses.
            self._drive.update()
            self._faults.update()
            if self._nav is not None:
                self._nav.update(self._dt)

            # 5. power and thermal integrate here or not at all
            self._subsystems.set_sun_position(self._sun_fn())
            try:
                self._subsystems.get_power_status(
                    (position[0], position[1], position[2]),
                    _yaw_deg(np.asarray(orientation).tolist()),
                    self._dt,
                    self._subsystems.get_obc_state(),
                )
            except Exception as exc:
                print(f"[dataset] power status failed: {exc}", flush=True)

            # 6. the downlink, with the crater's ground truth beside the injector's
            if self._sim_time >= next_downlink:
                extra = None
                if self._crater_config is not None:
                    extra = crater_module.oracle_row(self._stamped_crater, position, self._crater_config)
                    if extra["oracle.crater.in_crater"] and self._crater_entered_s is None:
                        self._crater_entered_s = self._sim_time
                self._recorder.tick(self._sim_time, self._step, downlink, extra_oracle=extra)
                next_downlink += downlink

            # 7. the camera, far slower than the parameter downlink
            if self._record_images and self._sim_time >= next_image:
                self._capture_image(attempt)
                attempt += 1
                next_image += image_period

            self._step += 1

        return "complete"

    def _off_terrain(self, position) -> bool:
        """True once the rover is outside the height map or has dropped below the ground under it."""
        if self._dem is None:
            return False

        resolution = float(self._terrain_resolution)
        rows, cols = self._dem.shape
        x, y, z = (float(v) for v in position[:3])
        if not (-resolution <= x <= cols * resolution and -resolution <= y <= rows * resolution):
            return True

        # The ground under the base itself, not the highest point around it: on a crater wall the
        # highest point nearby is well above a rover that is sitting where it should.
        ground = crater_module.ground_height(self._dem, resolution, x, y, 0.0)
        return z < ground - float(self._config["fall_depth_m"])

    def _read_dem(self):
        if self._terrain_resolution is None or not hasattr(self._environment, "get_terrain"):
            return None
        try:
            return np.asarray(self._environment.get_terrain()[0])
        except Exception as exc:
            print(f"[dataset] could not read the terrain, the fall check is off this episode: {exc}",
                  flush=True)
            return None

    # ── faults ───────────────────────────────────────────────────────────────────
    def _inject(self, event) -> None:
        name, takes_target = INJECTORS[event.kind]
        method = getattr(self._faults, name)
        if takes_target:
            method(event.target, *event.magnitudes)
        else:
            method(*event.magnitudes)

    def _on_fault_applied(self, kind: str, targets, magnitudes) -> None:
        self._recorder.log_fault_event(self._sim_time, self._step, kind, targets, magnitudes)

    # ── camera ───────────────────────────────────────────────────────────────────
    def _capture_image(self, attempt: int) -> None:
        """
        One nav-cam frame, dropped or corrupted exactly as the real downlink would have it.

        The loss check comes first, so a lost frame is never rendered at all - which is what makes
        the counter in /Rover/faults/dropped mean what it says.
        """
        # A camera the operator switched off takes no frame at all. Not logged as lost: nothing was
        # sent, and a "lost" row would read as a comms fault.
        if self._subsystems.get_device_power_state(CommonDevice.CAMERA) != PowerState.ON:
            return

        if self._faults.should_drop_camera_frame():
            self._recorder.log_image(self._sim_time, self._step, attempt, None)
            return

        try:
            from PIL import Image

            # Warm the renderer up at this exact pose. world.render() updates the renderer without
            # stepping physics, so the frame shows where the rover is now, not where it was.
            if not self._config["render_every_step"]:
                for _ in range(max(1, int(self._config["render_warmup_frames"]))):
                    self._world.render()

            frame = self._robot.get_rgba_camera_view(self._camera_resolution)
            if frame is None or getattr(frame, "size", 0) == 0:
                # The scene has not rendered yet; not a fault, so not logged as a loss.
                return

            path = self._recorder.image_path(self._images_saved)
            Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGBA").save(path)
            self._recorder.log_image(self._sim_time, self._step, attempt, os.path.basename(path))
            self._images_saved += 1
        except Exception as exc:
            print(f"[dataset] camera capture failed: {exc}", flush=True)

    # ── reset ────────────────────────────────────────────────────────────────────
    def _reset_episode(self, episode_seed: int, index: int, schedule=None) -> Tuple[float, float, float]:
        """
        Put the rover and every stateful model back to a clean start.

        The private reaches are deliberate and each is the only way in: the power model's reset is
        its own initialize(), the thermal model's initialize() preserves existing temperatures by
        design, and obc uptime only restarts through an OFF transition.
        """
        # One seed for everything. The measurement noise in the power, thermal and obc models all
        # comes off the bare global random module, so this covers all three at once.
        random.seed(episode_seed)
        rng = random.Random(episode_seed)
        if hasattr(self._robot, "set_fault_seed"):
            self._robot.set_fault_seed(episode_seed)

        self._faults.clear_all()
        self._faults.update()
        self._drive.command_stop()

        jitter = float(self._config["spawn_jitter_m"])
        spawn = (
            self._spawn[0] + rng.uniform(-jitter, jitter),
            self._spawn[1] + rng.uniform(-jitter, jitter),
            self._spawn[2],
        )
        scenario = getattr(schedule, "crater", None)
        if scenario is not None:
            # The crater decided where the rover starts: on flat ground, a sampled distance from the rim.
            spawn = (scenario.spawn[0], scenario.spawn[1], self._spawn[2])
        self._crater_entered_s = None
        heading = rng.uniform(-math.pi, math.pi)
        orientation = (math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0))  # w, x, y, z

        # The scene first, so the spawn height is read from the terrain this episode will actually
        # have (a --randomize-terrain switch or a crater edit changes it).
        self._stamped_crater = None
        if self._environment is not None:
            self._randomize_scene(rng, scenario)
        spawn = (spawn[0], spawn[1], self._spawn_height(spawn[0], spawn[1], spawn[2]))
        # After the scene, so a crater stamped this episode is part of the ground the check reads.
        self._dem = self._read_dem()

        self._robot.set_reset_pose(np.array(spawn), np.array(orientation))
        self._robot.reset()

        self._reset_subsystems()
        self._scripter.reset(episode_seed, (spawn[0], spawn[1]), scenario)
        self._images_saved = 0

        # A teleport moves the root body only; the wheels need a few steps to settle before the
        # first telemetry row is worth keeping.
        settle = max(int(self._config["settle_steps"]),
                     int(math.ceil(float(self._config["landing_hold_s"]) / self._dt)))
        if self._diverged:
            # Coming back from a solver blow-up takes more than a teleport: give PhysX room to
            # settle, and check it actually recovered rather than recording another bad episode.
            settle += int(self._config["recovery_settle_steps"])
        for _ in range(settle):
            self._world.step(render=True)

        if self._diverged:
            position, _ = self._robot_RG.get_pose_of_base_link()
            if _is_sane(np.asarray(position).tolist(), float(self._config["max_position_m"])):
                print(f"[dataset] recovered from the divergence; episode {index} starting clean",
                      flush=True)
                self._diverged = False
            else:
                print("[dataset] the rover did not recover from the divergence", flush=True)

        # After the settle: the filter's local origin and heading are the pose the episode starts from.
        if self._nav is not None:
            self._nav.reset()

        return spawn

    def _spawn_height(self, x: float, y: float, fallback: float) -> float:
        """Just above the highest ground under the rover, or the configured height if the terrain is unknown."""
        environment = self._environment
        if self._terrain_resolution is None or not hasattr(environment, "get_terrain"):
            return fallback
        try:
            dem, _ = environment.get_terrain()
            ground = crater_module.ground_height(dem, float(self._terrain_resolution), x, y,
                                                 float(self._config["spawn_footprint_radius_m"]))
            return ground + float(self._config["spawn_clearance_m"])
        except Exception as exc:
            print(f"[dataset] could not read the ground height at the spawn, dropping from "
                  f"{fallback:.2f} m: {exc}", flush=True)
            return fallback

    def _reset_subsystems(self) -> None:
        subsystems = self._subsystems

        # An episode can end with the operator's housekeeping still in effect - a device off, the
        # panel stowed. Every device starts ON (see _setup_devices), and the launch script deploys
        # the panel. Done before the power model re-initializes, since it reads both.
        for device in subsystems._devices.values():
            device.set_power_state(PowerState.ON)
        subsystems.set_solar_panel_state(SolarPanelState.DEPLOYED)

        try:
            subsystems._setup_power_model()  # re-initialize() puts the pack back to full
        except Exception as exc:
            print(f"[dataset] power model reset failed: {exc}", flush=True)

        thermal = getattr(subsystems, "_thermal_model", None)
        if thermal is not None:
            # initialize() preserves whatever temperatures are already there, so clear first.
            thermal._node_temps = {}
            thermal.initialize()

        # Uptime restarts only through an OFF transition.
        subsystems.set_obc_state(ObcState.OFF)
        subsystems.get_obc_status()
        subsystems.set_obc_state(ObcState.IDLE)
        subsystems.get_obc_status()

        # Without this every command is REJECTED: the interlock defaults to NOGO and only the Yamcs
        # command handler ever clears it.
        subsystems.set_go_nogo_state(GoNogoState.GO)

    def _randomize_scene(self, rng: random.Random, scenario=None) -> None:
        environment = self._environment
        try:
            if self._randomize_terrain and hasattr(environment, "switch_terrain"):
                environment.switch_terrain(-1)
                # A fresh DEM has no crater in it, and is the new thing to restore to.
                self._pristine_terrain = None
                self._crater_on_terrain = False
        except Exception as exc:
            print(f"[dataset] terrain randomisation failed: {exc}", flush=True)

        # Before the rocks, so they are placed against the crater's cleared mask.
        if self._crater_config is not None:
            self._apply_crater_safely(scenario)

        try:
            if hasattr(environment, "randomize_rocks"):
                environment.randomize_rocks(int(self._config["rocks_per_episode"]))
        except Exception as exc:
            print(f"[dataset] scene randomisation failed: {exc}", flush=True)

    def _apply_crater_safely(self, scenario) -> None:
        try:
            self._apply_crater(scenario)
        except Exception as exc:
            self._stamped_crater = None
            print(f"[dataset] crater stamping failed, the episode runs without one: {exc}", flush=True)

    def _apply_crater(self, scenario) -> None:
        """
        Cut this episode's crater into the terrain, or put back the terrain a previous one changed.

        Always from the pristine copy, so craters never accumulate. The terrain collider is rebuilt
        only when the height map actually changes: a nominal episode after a nominal episode costs
        nothing.
        """
        environment = self._environment
        if not (hasattr(environment, "get_terrain") and hasattr(environment, "set_terrain")):
            if scenario is not None:
                print("[dataset] this environment cannot edit its terrain; the crater is not stamped",
                      flush=True)
            return

        if self._pristine_terrain is None:
            self._pristine_terrain = environment.get_terrain()
        dem, mask = self._pristine_terrain

        if scenario is not None:
            dem, mask = crater_module.stamp(dem, mask, scenario.spec, float(self._crater_config["resolution"]))
            environment.set_terrain(dem, mask)
            self._crater_on_terrain = True
            self._stamped_crater = scenario.spec
        elif self._crater_on_terrain:
            environment.set_terrain(dem, mask)
            self._crater_on_terrain = False
