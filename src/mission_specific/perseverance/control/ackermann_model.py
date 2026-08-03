__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
Four-corner Ackermann steering geometry for the perseverance rover.

Pure geometry: no omni / pxr / Isaac Sim imports, no state carried between calls. That keeps it
unit-testable with a plain python3 interpreter, which matters because sign errors here are
invisible in the simulator until the rover drives off in the wrong direction.

The rover has six driven wheels and four steered corners (the two mid wheels are not steerable),
which is the real rocker-bogie layout. Steering them properly lets the wheels roll through a turn
instead of skidding sideways, which is both smoother and far less sensitive to terrain.

Body frame convention used throughout:
    x = lateral, positive to the rover's LEFT
    y = longitudinal, positive FORWARD
    yaw = positive counter-clockwise (turning left)

The turn centre is always placed on the lateral line through the MID axle. That choice is what
makes the model consistent for the unsteered mid wheels: their longitudinal offset from the centre
is zero by construction, so the geometry never asks them to steer (see _solve_wheel).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

# Wheel names, matching the drive joint suffixes in rover_with_sensors.usd.
STEERED_WHEELS = ("front_left", "front_right", "rear_left", "rear_right")
FIXED_WHEELS = ("mid_left", "mid_right")
ALL_WHEELS = ("front_left", "front_right", "mid_left", "mid_right", "rear_left", "rear_right")

# Below this |curvature| (1/m) a turn is treated as a straight line, avoiding a 1/0 turn radius.
STRAIGHT_CURVATURE_EPS = 1e-6


@dataclass
class RoverGeometry:
    """
    Rover dimensions in metres.

    Lengths default to the values measured directly from rover_with_sensors.usd, i.e. UNSCALED.
    Build via from_config(geometry, scale) to apply the usd scale factor; constructing this class
    directly gives you the scale-1.0 rover.
    """

    wheel_radius: float = 0.2625
    corner_half_track: float = 1.062       # lateral offset of the four steered wheels
    mid_half_track: float = 1.184          # lateral offset of the two fixed wheels
    front_offset: float = -1.185           # longitudinal offset of the front axle from the mid axle
    rear_offset: float = 1.075             # longitudinal offset of the rear axle from the mid axle
    max_steer_angle: float = math.radians(75.0)
    max_wheel_speed: float = 20.0          # rad/s at the drive joint, not a length: never scaled

    @classmethod
    def from_config(cls, geometry: Dict, scale: float = 1.0) -> "RoverGeometry":
        """
        Build from the unscaled `geometry` block in cfg/robot/perseverance.yaml.

        Lengths are stored unscaled and multiplied by the usd scale here, so changing --scale
        cannot silently leave the controller working from the wrong dimensions. Angles and
        rate limits are not lengths and pass through untouched.
        """
        geometry = dict(geometry or {})
        lengths = ("wheel_radius", "corner_half_track", "mid_half_track", "front_offset", "rear_offset")
        defaults = cls()

        kwargs = {name: float(geometry.get(name, getattr(defaults, name))) * scale for name in lengths}
        if "max_steer_angle_deg" in geometry:
            kwargs["max_steer_angle"] = math.radians(float(geometry["max_steer_angle_deg"]))
        if "max_wheel_speed" in geometry:
            kwargs["max_wheel_speed"] = float(geometry["max_wheel_speed"])
        return cls(**kwargs)

    def wheel_positions(self) -> Dict[str, Tuple[float, float]]:
        """(x, y) of each wheel in the body frame, relative to the mid axle."""
        return {
            "front_left":  (self.corner_half_track, self.front_offset),
            "front_right": (-self.corner_half_track, self.front_offset),
            "mid_left":    (self.mid_half_track, 0.0),
            "mid_right":   (-self.mid_half_track, 0.0),
            "rear_left":   (self.corner_half_track, self.rear_offset),
            "rear_right":  (-self.corner_half_track, self.rear_offset),
        }


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class AckermannModel:
    """
    Converts a body-level motion request into per-wheel steer angles and drive speeds.

    Two entry points:
        solve(v, curvature)      - drive along an arc; curvature 0 is a straight line
        solve_point_turn(yaw_rate) - rotate in place about the rover centre

    Both return (steer_angles, wheel_speeds) keyed by wheel name. steer_angles are radians at the
    steer joints (only the four corners appear); wheel_speeds are rad/s at the drive joints (all
    six wheels).
    """

    geometry: RoverGeometry = field(default_factory=RoverGeometry)

    def solve(self, linear_velocity: float, curvature: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Args:
            linear_velocity: forward speed of the rover centre, m/s. Negative drives backward.
            curvature: 1 / turn_radius, in 1/m. Positive curves LEFT, 0 is straight.
        """
        if abs(curvature) < STRAIGHT_CURVATURE_EPS:
            return self._solve_straight(linear_velocity)

        turn_radius = 1.0 / curvature
        yaw_rate = linear_velocity * curvature
        return self._solve_about_centre(turn_radius, yaw_rate)

    def solve_point_turn(self, yaw_rate: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Rotate in place. yaw_rate is rad/s, positive counter-clockwise (turning left).

        The turn centre sits at the rover centre, so the mid wheels lie exactly on the rotation
        axis line and end up counter-rotating with zero steer — no special case needed.
        """
        return self._solve_about_centre(turn_radius=0.0, yaw_rate=yaw_rate)

    def stop(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Zero speed with the wheels straightened."""
        return (
            {name: 0.0 for name in STEERED_WHEELS},
            {name: 0.0 for name in ALL_WHEELS},
        )

    # ── internals ────────────────────────────────────────────────────────────────
    def _solve_straight(self, linear_velocity: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        speed = _clamp(linear_velocity / self.geometry.wheel_radius, self.geometry.max_wheel_speed)
        return (
            {name: 0.0 for name in STEERED_WHEELS},
            {name: speed for name in ALL_WHEELS},
        )

    def _solve_about_centre(self, turn_radius: float, yaw_rate: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Common solver for arcs and point turns.

        The turn centre is at (turn_radius, 0) in the body frame — on the lateral line through the
        mid axle, positive to the left. A point turn is simply turn_radius = 0.
        """
        steer_angles: Dict[str, float] = {}
        wheel_speeds: Dict[str, float] = {}

        for name, (wheel_x, wheel_y) in self.geometry.wheel_positions().items():
            angle, speed = self._solve_wheel(
                wheel_x, wheel_y, turn_radius, yaw_rate, steerable=name in STEERED_WHEELS
            )
            if name in STEERED_WHEELS:
                steer_angles[name] = angle
            wheel_speeds[name] = speed

        return steer_angles, wheel_speeds

    def _solve_wheel(
        self, wheel_x: float, wheel_y: float, turn_radius: float, yaw_rate: float, steerable: bool
    ) -> Tuple[float, float]:
        """
        Solve one wheel against a turn centre at (turn_radius, 0).

        The wheel must roll tangentially about that centre. With the radius vector from centre to
        wheel being (dx, dy), the tangent for a positive (counter-clockwise) rotation is (dy, -dx),
        and the steer angle is that tangent measured from the forward axis.
        """
        dx = wheel_x - turn_radius
        dy = wheel_y

        angle = math.atan2(dy, -dx)
        radius = math.hypot(dx, dy)
        speed = yaw_rate * radius / self.geometry.wheel_radius

        # Steering is bidirectional: a wheel pointed backwards is the same as one pointed forwards
        # driving in reverse. Folding angles into +/-90 degrees keeps the steer joints near centre
        # and is what makes the unsteered mid wheels come out right — during a point turn their
        # angle is exactly +/-180 degrees, which folds to 0 with a negated speed, i.e. counter-rotation.
        if angle > math.pi / 2:
            angle -= math.pi
            speed = -speed
        elif angle < -math.pi / 2:
            angle += math.pi
            speed = -speed

        if not steerable:
            # Mid wheels have no steer joint. By construction their dy is 0, so the fold above
            # already produced 0; assert-by-clamp rather than silently steering something that cannot.
            angle = 0.0

        return _clamp(angle, self.geometry.max_steer_angle), _clamp(speed, self.geometry.max_wheel_speed)
