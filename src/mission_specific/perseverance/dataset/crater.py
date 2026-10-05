__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
A steep crater the rover drives into and cannot climb out of.

Not a fault: nothing on the rover breaks. The terrain does, so this lives beside the fault scheduler
rather than inside the injector, and it is applied to the height map between episodes rather than
mid-run. Stamping mid-run would work, but rebuilding the terrain collider stalls the simulation, rocks
already placed in the footprint would float, and nothing is gained - a crater is a passive hazard, and
the moment that matters is when the rover enters it, not when it appears.

WHY IT TRAPS. A wheel climbs only while grip beats the downhill pull: mu cos(theta) >= sin(theta), so
tan(theta) <= mu. At the 0.5 wheel friction that is 26.6 deg, however hard the motors push. Walls are
sampled steeper than that, and below ~45 deg so the rover slides in rather than tumbling. The wall is
also made longer along the slope than the rover, so it cannot bridge the lip with its front and rear
wheels on flat ground.

GEOMETRY. A flat floor, a straight wall at the sampled slope, a raised rim, a slightly elliptical
outline at a random rotation, and a little roughness. Heights are set relative to the terrain around
the rim, so the wall slope is the sampled slope rather than whatever the underlying undulation adds.
Everything a detector could key on - position, size, depth, slope, outline - is randomized per
episode.

DEM CONVENTION. TerrainManager writes np.flip(DEM, 0) row-major onto a grid whose vertex (x, y) sits
at world (x * res, y * res). So DEM[row, col] is world (col * res, (rows - 1 - row) * res).

No omni/pxr imports - plain numpy, testable with python3.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

OUTCOMES = ("trapped", "avoid", "skirt")

DEFAULTS = {
    # Terrain the crater has to fit in, [[xmin, xmax], [ymin, ymax]] in world metres, and its DEM
    # resolution. The launch script fills both in from the environment.
    "bounds": [[0.0, 20.0], [0.0, 20.0]],
    "resolution": 0.025,
    # Crater shape. The rover is ~0.8 m long at scale 0.35, so a floor radius of at least 1 m lets it
    # land flat, and a depth of 0.7 m or more makes the wall longer along the slope than the rover.
    "floor_radius_m": [1.0, 2.0],
    "wall_slope_deg": [30.0, 45.0],
    "depth_m": [0.7, 1.1],
    "rim_height_fraction": [0.0, 0.12],   # of depth
    "rim_width_m": 0.3,
    "axis_ratio": [0.9, 1.0],             # minor / major; the wall is steepest along the minor axis
    "roughness_m": [0.0, 0.02],
    "edge_margin_m": 1.0,                 # between the blended crater edge and the terrain edge
    # Episode. Spawn is measured from the rim, on flat ground, on a random bearing.
    "spawn_rim_distance_m": [2.5, 5.0],
    "outcome_weights": {"trapped": 0.7, "avoid": 0.15, "skirt": 0.15},
    "dash_window": [0.35, 0.60],          # fraction of the episode
    "keep_out_margin_m": 1.5,             # waypoints and paths stay this far outside the rim
    "skirt_margin_m": [0.4, 1.2],         # a skirt passes this far outside the rim
    "skirt_overshoot_m": [1.0, 3.0],      # how far past the tangent point the skirt waypoint sits
    "escape_timeout_s": [20.0, 45.0],     # a trapped rover's escape attempts are abandoned after this
    "fault_probability": 0.3,             # chance a crater episode also carries an injected fault
    # Labelling: the rover counts as in the crater once its base is this far inside the rim.
    "entry_margin_m": 0.3,
    "placement_attempts": 64,
}


def merged_config(config: Optional[Dict] = None) -> Dict:
    merged = {key: (dict(value) if isinstance(value, dict) else value) for key, value in DEFAULTS.items()}
    for key, value in dict(config or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        elif key in merged:
            merged[key] = value
    return merged


def _uniform(rng: random.Random, bounds: Sequence[float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    return rng.uniform(low, high) if high > low else low


@dataclass
class CraterSpec:
    """One crater's geometry, in world metres. reference_height is filled in by stamp()."""

    center_x: float
    center_y: float
    floor_radius: float
    wall_slope_deg: float
    depth: float
    rim_height: float
    rim_width: float
    axis_ratio: float
    rotation_rad: float
    roughness: float
    # (kx, ky, phase) per roughness component, so the same spec always stamps the same surface.
    roughness_waves: List[Tuple[float, float, float]] = field(default_factory=list)
    reference_height: Optional[float] = None

    @property
    def wall_width(self) -> float:
        return self.depth / math.tan(math.radians(self.wall_slope_deg))

    @property
    def outer_radius(self) -> float:
        """Radius of the rim crest along the major axis."""
        return self.floor_radius + self.wall_width

    @property
    def footprint_radius(self) -> float:
        """Everything the stamp touches, rim and blend included."""
        return self.outer_radius + 4.0 * self.rim_width

    def effective_radius(self, x, y):
        """
        Distance from the centre in crater units: equal to the major-axis radius on the ellipse.

        Along the minor axis a point at axis_ratio * r already reads r, so the outline is the ellipse
        and the wall there is steeper by 1 / axis_ratio.
        """
        dx, dy = np.asarray(x) - self.center_x, np.asarray(y) - self.center_y
        cos_r, sin_r = math.cos(self.rotation_rad), math.sin(self.rotation_rad)
        u = dx * cos_r + dy * sin_r
        v = -dx * sin_r + dy * cos_r
        return np.hypot(u, v / self.axis_ratio)

    def profile(self, r):
        """Height relative to the surrounding terrain at effective radius r."""
        r = np.asarray(r, dtype=float)
        tan_slope = math.tan(math.radians(self.wall_slope_deg))
        bowl = np.clip(-self.depth + (r - self.floor_radius) * tan_slope, -self.depth, 0.0)
        rim = self.rim_height * np.exp(-((r - self.outer_radius) / max(self.rim_width, 1e-6)) ** 2)
        return bowl + rim

    def roughness_at(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        total = np.zeros(np.broadcast(x, y).shape)
        for kx, ky, phase in self.roughness_waves:
            total += np.sin(kx * x + ky * y + phase)
        scale = self.roughness / max(1, len(self.roughness_waves)) ** 0.5
        return scale * total

    def as_dict(self) -> Dict:
        return {
            "center": [round(self.center_x, 4), round(self.center_y, 4)],
            "floor_radius_m": round(self.floor_radius, 4),
            "wall_slope_deg": round(self.wall_slope_deg, 3),
            "depth_m": round(self.depth, 4),
            "rim_height_m": round(self.rim_height, 4),
            "rim_width_m": round(self.rim_width, 4),
            "axis_ratio": round(self.axis_ratio, 4),
            "rotation_deg": round(math.degrees(self.rotation_rad), 3),
            "roughness_m": round(self.roughness, 4),
            "outer_radius_m": round(self.outer_radius, 4),
            "reference_height_m": None if self.reference_height is None else round(self.reference_height, 4),
        }


@dataclass
class CraterScenario:
    """What a crater episode does: the crater, where the rover starts, and how the episode ends."""

    spec: CraterSpec
    outcome: str                      # one of OUTCOMES
    spawn: Tuple[float, float]
    dash_s: Optional[float]           # None for "avoid": the rover never goes for the crater
    skirt_margin: float = 0.0
    skirt_overshoot: float = 0.0
    escape_timeout_s: float = 30.0

    def as_dict(self) -> Dict:
        return {
            "outcome": self.outcome,
            "spawn": [round(self.spawn[0], 4), round(self.spawn[1], 4)],
            "dash_s": None if self.dash_s is None else round(self.dash_s, 3),
            "skirt_margin_m": round(self.skirt_margin, 4),
            "skirt_overshoot_m": round(self.skirt_overshoot, 4),
            "escape_timeout_s": round(self.escape_timeout_s, 3),
            "crater": self.spec.as_dict(),
        }


# ── sampling ─────────────────────────────────────────────────────────────────────
def sample_spec(rng: random.Random, config: Dict, center: Optional[Tuple[float, float]] = None) -> CraterSpec:
    """A random crater. Placed anywhere that fits the bounds unless a centre is given."""
    depth = _uniform(rng, config["depth_m"])
    spec = CraterSpec(
        center_x=0.0,
        center_y=0.0,
        floor_radius=_uniform(rng, config["floor_radius_m"]),
        wall_slope_deg=_uniform(rng, config["wall_slope_deg"]),
        depth=depth,
        rim_height=_uniform(rng, config["rim_height_fraction"]) * depth,
        rim_width=float(config["rim_width_m"]),
        axis_ratio=_uniform(rng, config["axis_ratio"]),
        rotation_rad=rng.uniform(-math.pi, math.pi),
        roughness=_uniform(rng, config["roughness_m"]),
        roughness_waves=[
            (
                2.0 * math.pi / rng.uniform(0.5, 1.5) * math.cos(angle),
                2.0 * math.pi / rng.uniform(0.5, 1.5) * math.sin(angle),
                rng.uniform(0.0, 2.0 * math.pi),
            )
            for angle in (rng.uniform(-math.pi, math.pi) for _ in range(3))
        ],
    )

    if center is not None:
        spec.center_x, spec.center_y = float(center[0]), float(center[1])
        return spec

    (x_min, x_max), (y_min, y_max) = config["bounds"]
    reach = spec.footprint_radius + float(config["edge_margin_m"])
    if x_max - x_min < 2 * reach or y_max - y_min < 2 * reach:
        raise ValueError(f"terrain {config['bounds']} is too small for a crater reaching {reach:.2f} m")

    spec.center_x = rng.uniform(x_min + reach, x_max - reach)
    spec.center_y = rng.uniform(y_min + reach, y_max - reach)
    return spec


def sample_spawn(rng: random.Random, spec: CraterSpec, config: Dict) -> Tuple[float, float]:
    """A start on flat ground outside the rim, inside the bounds."""
    (x_min, x_max), (y_min, y_max) = config["bounds"]
    margin = float(config["edge_margin_m"])
    for _ in range(int(config["placement_attempts"])):
        bearing = rng.uniform(-math.pi, math.pi)
        # The outline is an ellipse no larger than the major-axis circle, so measuring from that
        # circle keeps the spawn at least this far from the rim in every direction.
        distance = spec.footprint_radius + _uniform(rng, config["spawn_rim_distance_m"])
        x = spec.center_x + distance * math.cos(bearing)
        y = spec.center_y + distance * math.sin(bearing)
        if x_min + margin <= x <= x_max - margin and y_min + margin <= y <= y_max - margin:
            return x, y
    raise ValueError("no spawn outside the crater fits inside the terrain; widen bounds or shrink the crater")


def sample_scenario(rng: random.Random, duration_s: float, config: Dict) -> CraterScenario:
    spec = sample_spec(rng, config)
    spawn = sample_spawn(rng, spec, config)

    weights = config["outcome_weights"]
    outcome = rng.choices(list(OUTCOMES), weights=[float(weights.get(name, 0.0)) for name in OUTCOMES], k=1)[0]
    dash_s = None if outcome == "avoid" else _uniform(rng, config["dash_window"]) * float(duration_s)

    return CraterScenario(
        spec=spec,
        outcome=outcome,
        spawn=spawn,
        dash_s=dash_s,
        skirt_margin=_uniform(rng, config["skirt_margin_m"]),
        skirt_overshoot=_uniform(rng, config["skirt_overshoot_m"]),
        escape_timeout_s=_uniform(rng, config["escape_timeout_s"]),
    )


def place_ahead_of(rng: random.Random, spawn: Tuple[float, float], config: Dict) -> CraterSpec:
    """For the interactive demo: a random crater whose nearest rim sits a sampled distance from spawn."""
    spec = sample_spec(rng, config, center=(0.0, 0.0))
    bearing = rng.uniform(-math.pi, math.pi)
    distance = spec.footprint_radius + _uniform(rng, config["spawn_rim_distance_m"])
    spec.center_x = float(spawn[0]) + distance * math.cos(bearing)
    spec.center_y = float(spawn[1]) + distance * math.sin(bearing)
    return spec


# ── the height map ───────────────────────────────────────────────────────────────
def world_grid(shape: Tuple[int, int], resolution: float):
    """World (x, y) of every DEM cell, following TerrainManager's flip - see the module docstring."""
    rows, cols = shape
    x = np.arange(cols, dtype=float) * resolution
    y = (rows - 1 - np.arange(rows, dtype=float)) * resolution
    return np.meshgrid(x, y)


def ground_height(dem: np.ndarray, resolution: float, x: float, y: float, radius: float) -> float:
    """
    Highest terrain point within radius of world (x, y), for placing the rover just above the ground.

    The highest point, not the one under the centre: a wheel resting on a bump must not start inside
    it. Coordinates outside the DEM are clamped to its edge.
    """
    rows, cols = dem.shape
    # Clamp the centre onto the DEM first: a negative slice bound would index from the far end.
    col = min(max(x / resolution, 0.0), cols - 1.0)
    row = min(max((rows - 1) - y / resolution, 0.0), rows - 1.0)
    reach = radius / resolution
    r0, r1 = int(max(0, math.floor(row - reach))), int(min(rows - 1, math.ceil(row + reach)))
    c0, c1 = int(max(0, math.floor(col - reach))), int(min(cols - 1, math.ceil(col + reach)))
    return float(np.max(dem[r0:r1 + 1, c0:c1 + 1]))


def stamp(dem: np.ndarray, mask: np.ndarray, spec: CraterSpec, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cut the crater into copies of the DEM and rock mask. Sets spec.reference_height.

    The crater is measured from the median height of a ring just outside its blend zone, and blended
    back into the original terrain over that zone, so the wall slope is exactly the sampled one and
    the join is smooth. The mask is cleared over the footprint so no rock lands in the crater or on
    the rim, where it could block the dash.
    """
    dem = np.array(dem, dtype=float, copy=True)
    mask = np.array(mask, copy=True)
    xs, ys = world_grid(dem.shape, resolution)
    r = spec.effective_radius(xs, ys)

    inner_blend = spec.outer_radius + 2.0 * spec.rim_width
    outer_blend = spec.footprint_radius
    ring = (r >= outer_blend) & (r <= outer_blend + 0.5)
    reference = float(np.median(dem[ring])) if np.any(ring) else float(np.median(dem))
    spec.reference_height = reference

    target = reference + spec.profile(r) + spec.roughness_at(xs, ys)
    # 1 inside the rim, easing to 0 across the blend zone, 0 beyond.
    t = np.clip((outer_blend - r) / max(outer_blend - inner_blend, 1e-6), 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    dem = weight * target + (1.0 - weight) * dem

    mask[r <= outer_blend] = 0
    return dem.astype(np.float32), mask


# ── the episode's geometry questions ─────────────────────────────────────────────
def keep_out_radius(spec: CraterSpec, config: Dict) -> float:
    return spec.footprint_radius + float(config["keep_out_margin_m"])


def path_clear(spec: CraterSpec, start: Tuple[float, float], end: Tuple[float, float], radius: float) -> bool:
    """True when the straight segment start -> end stays at least radius from the crater centre."""
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0.0 else max(0.0, min(1.0, ((spec.center_x - ax) * dx + (spec.center_y - ay) * dy) / length_sq))
    px, py = ax + t * dx, ay + t * dy
    return math.hypot(spec.center_x - px, spec.center_y - py) >= radius


def dash_target(spec: CraterSpec, position: Tuple[float, float]) -> Tuple[float, float]:
    """A goto that carries the rover over the rim and onto the floor, just past the centre."""
    dx, dy = spec.center_x - float(position[0]), spec.center_y - float(position[1])
    distance = math.hypot(dx, dy) or 1.0
    beyond = 0.5 * spec.floor_radius
    return spec.center_x + dx / distance * beyond, spec.center_y + dy / distance * beyond


def skirt_target(spec: CraterSpec, position: Tuple[float, float], margin: float, overshoot: float,
                 side: float = 1.0) -> Optional[Tuple[float, float]]:
    """
    A goto whose straight path passes the crater on one side, margin outside the rim at its closest.

    The path is the tangent from the rover to a circle of radius outer_radius + margin, carried past
    the tangent point by overshoot. None when the rover is already inside that circle.
    """
    px, py = float(position[0]), float(position[1])
    dx, dy = spec.center_x - px, spec.center_y - py
    distance = math.hypot(dx, dy)
    clearance = spec.outer_radius + float(margin)
    if distance <= clearance:
        return None

    alpha = math.asin(clearance / distance) * (1.0 if side >= 0 else -1.0)
    bearing = math.atan2(dy, dx) + alpha
    reach = math.sqrt(distance * distance - clearance * clearance) + float(overshoot)
    return px + reach * math.cos(bearing), py + reach * math.sin(bearing)


def oracle_row(spec: Optional[CraterSpec], position: Optional[Sequence[float]], config: Dict) -> Dict[str, object]:
    """
    The crater's oracle columns for one tick. Same keys with or without a crater, so every episode's
    truth.csv has the same header.

    in_crater is geometric ground truth: the base link is more than entry_margin_m inside the rim.
    rim_distance_m is signed (negative inside) so a consumer can pick a different threshold.
    """
    keys = ("in_crater", "rim_distance_m", "center_x", "center_y", "outer_radius_m", "depth_m",
            "wall_slope_deg")
    row: Dict[str, object] = {f"oracle.crater.{key}": "" for key in keys}
    row["oracle.crater.in_crater"] = 0
    if spec is None or position is None:
        return row

    x, y = float(position[0]), float(position[1])
    rim_distance = float(spec.effective_radius(x, y)) - spec.outer_radius
    row.update({
        "oracle.crater.in_crater": int(rim_distance < -float(config["entry_margin_m"])),
        "oracle.crater.rim_distance_m": round(rim_distance, 4),
        "oracle.crater.center_x": round(spec.center_x, 4),
        "oracle.crater.center_y": round(spec.center_y, 4),
        "oracle.crater.outer_radius_m": round(spec.outer_radius, 4),
        "oracle.crater.depth_m": round(spec.depth, 4),
        "oracle.crater.wall_slope_deg": round(spec.wall_slope_deg, 3),
    })
    return row
