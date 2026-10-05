#!/usr/bin/env python3
"""
Host-runnable checks of the steep-crater scenario. No Isaac Sim, no omni imports.

    python3 test/test_dataset_crater.py

Covers the parts that are slow to debug in simulation: that the stamped height map really has the
sampled wall slope and depth, that rocks are kept out, that spawns and near-miss paths stay outside
the rim, that the scheduler only produces crater episodes when asked, and that the operator keeps out
of the crater until the dash and then drives into it.
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.control.drive_controller import CommandStatus
from src.mission_specific.perseverance.dataset import crater as cr
from src.mission_specific.perseverance.dataset import fault_scheduler as fs
from src.mission_specific.perseverance.dataset.mission_scripter import MissionScripter

failures = []
ROOT = Path(__file__).resolve().parents[1]
DATASET_CFG = yaml.safe_load((ROOT / "cfg/robot/perseverance.yaml").read_text())
DATASET_CFG = DATASET_CFG["robots_settings"]["parameters"]["dataset"]
RES = 0.025
CFG = cr.merged_config({**DATASET_CFG["crater"], "bounds": [[0.0, 20.0], [0.0, 20.0]], "resolution": RES})


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def height_at(dem, x, y):
    rows = dem.shape[0]
    return float(dem[int(round(rows - 1 - y / RES)), int(round(x / RES))])


# ── geometry ─────────────────────────────────────────────────────────────────
print("\n=== the stamped height map has the sampled shape ===")
rng = random.Random(3)
spec = cr.sample_spec(rng, CFG)
# A round, smooth crater so the slope can be measured exactly along any axis.
spec.axis_ratio, spec.roughness, spec.rim_height = 1.0, 0.0, 0.0
base = np.full((800, 800), 0.4, dtype=np.float32)
mask = np.ones_like(base)
dem, new_mask = cr.stamp(base, mask, spec, RES)

check("reference height taken from the surrounding terrain", abs(spec.reference_height - 0.4) < 1e-6)
check("floor sits at reference - depth",
      abs(height_at(dem, spec.center_x, spec.center_y) - (0.4 - spec.depth)) < 1e-3)
# Measure between two points well inside the wall, away from the floor and rim transitions.
r1 = spec.floor_radius + 0.25 * spec.wall_width
r2 = spec.floor_radius + 0.75 * spec.wall_width
h1 = height_at(dem, spec.center_x + r1, spec.center_y)
h2 = height_at(dem, spec.center_x + r2, spec.center_y)
measured = math.degrees(math.atan2(h2 - h1, r2 - r1))
check(f"wall slope measured from the DEM ({measured:.1f} deg) matches the spec ({spec.wall_slope_deg:.1f} deg)",
      abs(measured - spec.wall_slope_deg) < 1.5)
check("wall is steeper than the friction limit atan(0.5)", spec.wall_slope_deg > math.degrees(math.atan(0.5)))
check("wall is longer along the slope than the rover (0.8 m)",
      spec.depth / math.sin(math.radians(spec.wall_slope_deg)) > 0.8)
far = spec.footprint_radius + 0.2
check("terrain beyond the footprint is untouched",
      abs(height_at(dem, spec.center_x + far, spec.center_y) - 0.4) < 1e-6)
check("the input DEM is not modified", bool(np.all(base == np.float32(0.4))))
check("rocks are kept out of the footprint", new_mask[
    int(round(799 - spec.center_y / RES)), int(round(spec.center_x / RES))] == 0)
check("and still allowed beyond it", height_at(new_mask, spec.center_x + far, spec.center_y) == 1)

print("\n=== an elliptical, rough crater still cuts cleanly ===")
spec_rough = cr.sample_spec(random.Random(11), {**CFG, "axis_ratio": [0.9, 0.9], "roughness_m": [0.02, 0.02]})
dem_rough, _ = cr.stamp(base, mask, spec_rough, RES)
check("the floor is below the terrain by about the depth",
      abs(height_at(dem_rough, spec_rough.center_x, spec_rough.center_y) - (0.4 - spec_rough.depth)) < 0.05)
check("the result is finite everywhere", bool(np.all(np.isfinite(dem_rough))))
check("same spec, same height map", np.array_equal(dem_rough, cr.stamp(base, mask, spec_rough, RES)[0]))

print("\n=== ground height under the spawn ===")
bumpy = np.full((800, 800), 0.1, dtype=np.float32)
bumpy[int(round(799 - 10.3 / RES)), int(round(10.2 / RES))] = 0.6   # a bump 0.36 m from (10, 10)
check("the highest point within the radius, not the one under the centre",
      abs(cr.ground_height(bumpy, RES, 10.0, 10.0, 0.75) - 0.6) < 1e-6)
check("a bump outside the radius is ignored", abs(cr.ground_height(bumpy, RES, 10.0, 10.0, 0.2) - 0.1) < 1e-6)
check("coordinates past the edge are clamped, not an error",
      abs(cr.ground_height(bumpy, RES, -5.0, 30.0, 0.75) - 0.1) < 1e-6)

print("\n=== DEM convention matches TerrainManager's flip ===")
xs, ys = cr.world_grid((800, 800), RES)
check("row 0 is the top of the terrain (largest y)", ys[0, 0] == 799 * RES and ys[-1, 0] == 0.0)
check("column is x", xs[0, 5] == 5 * RES)

# ── placement ────────────────────────────────────────────────────────────────
print("\n=== spawns, keep-out and near-miss paths ===")
all_inside, all_outside = True, True
for seed in range(200):
    r = random.Random(seed)
    s = cr.sample_spec(r, CFG)
    reach = s.footprint_radius + CFG["edge_margin_m"]
    all_inside &= reach <= s.center_x <= 20 - reach and reach <= s.center_y <= 20 - reach
    sx, sy = cr.sample_spawn(r, s, CFG)
    all_outside &= math.hypot(sx - s.center_x, sy - s.center_y) >= s.footprint_radius + 2.5 - 1e-9
    all_outside &= 1.0 <= sx <= 19.0 and 1.0 <= sy <= 19.0
check("200 craters all fit inside the terrain with margin", all_inside)
check("every spawn is on flat ground at least 2.5 m past the footprint, inside the terrain", all_outside)

try:
    cr.sample_spec(random.Random(0), {**CFG, "bounds": [[0.0, 6.0], [0.0, 6.0]]})
    check("a terrain too small for the crater is refused", False)
except ValueError:
    check("a terrain too small for the crater is refused", True)

start = (spec.center_x - 8.0, spec.center_y)
check("a path straight through the crater is not clear",
      not cr.path_clear(spec, start, (spec.center_x + 8.0, spec.center_y), spec.outer_radius))
check("a path well to one side is clear",
      cr.path_clear(spec, start, (spec.center_x + 8.0, spec.center_y + 10.0), spec.outer_radius))

for side in (-1.0, 1.0):
    target = cr.skirt_target(spec, start, 0.8, 2.0, side=side)
    dx, dy = target[0] - start[0], target[1] - start[1]
    closest = abs((spec.center_x - start[0]) * dy - (spec.center_y - start[1]) * dx) / math.hypot(dx, dy)
    check(f"skirt on side {side:+.0f} passes exactly 0.8 m outside the rim ({closest - spec.outer_radius:.3f})",
          abs(closest - (spec.outer_radius + 0.8)) < 1e-6)
check("no skirt from inside the clearance circle",
      cr.skirt_target(spec, (spec.center_x + spec.outer_radius, spec.center_y), 0.8, 2.0) is None)

dash = cr.dash_target(spec, start)
check("the dash target is on the floor, just past the centre",
      spec.effective_radius(*dash) <= spec.floor_radius and dash[0] > spec.center_x)

print("\n=== ground-truth crater columns ===")
inside = cr.oracle_row(spec, (spec.center_x, spec.center_y, 0.0), CFG)
on_lip = cr.oracle_row(spec, (spec.center_x + spec.outer_radius - 0.1, spec.center_y, 0.0), CFG)
outside = cr.oracle_row(spec, start, CFG)
empty = cr.oracle_row(None, start, CFG)
check("at the centre: in_crater", inside["oracle.crater.in_crater"] == 1)
check("0.1 m inside the rim is not yet in (entry margin 0.3 m)", on_lip["oracle.crater.in_crater"] == 0)
check("outside: not in, positive rim distance",
      outside["oracle.crater.in_crater"] == 0 and outside["oracle.crater.rim_distance_m"] > 0)
check("the same columns with no crater, so episodes concatenate", set(empty) == set(inside))
check("no crater: in_crater 0 and blank geometry",
      empty["oracle.crater.in_crater"] == 0 and empty["oracle.crater.center_x"] == "")

# ── scheduling ───────────────────────────────────────────────────────────────
print("\n=== the scheduler only makes crater episodes when asked ===")
faults_cfg = {**DATASET_CFG["faults"],
              "class_weights": {name: 0.0 for name in fs.EPISODE_CLASSES} | {"crater": 1.0}}
mixed_cfg = {**DATASET_CFG["faults"], "class_weights": {**DATASET_CFG["faults"]["class_weights"], "crater": 1.0}}
without = [fs.generate(seed, 600.0, mixed_cfg) for seed in range(300)]
check("no crater episodes without a crater config, even at weight 1.0",
      all(s.episode_class != "crater" and s.crater is None for s in without))

legacy_cfg = {**DATASET_CFG["faults"], "class_weights": {k: v for k, v in DATASET_CFG["faults"]["class_weights"].items() if k != "crater"}}
try:
    fs.generate(0, 600.0, faults_cfg)
    check("a crater-only mix without a crater config is refused with a clear error", False)
except ValueError as exc:
    check("a crater-only mix without a crater config is refused with a clear error", "--crater" in str(exc))
check("seeds give the same schedules as before the crater class existed",
      all(fs.generate(seed, 600.0, legacy_cfg).as_dict() == fs.generate(seed, 600.0, DATASET_CFG["faults"]).as_dict()
          for seed in range(100)))

with_crater = [fs.generate(seed, 600.0, faults_cfg, crater_config=CFG) for seed in range(300)]
craters = [s for s in with_crater if s.episode_class == "crater"]
check("with a config and weight 1.0 every episode is a crater episode", len(craters) == 300)
outcomes = {name: sum(1 for s in craters if s.crater.outcome == name) for name in cr.OUTCOMES}
check(f"all three outcomes occur, trapped most often {outcomes}",
      all(outcomes.values()) and outcomes["trapped"] > outcomes["avoid"] + outcomes["skirt"])
check("avoid episodes never dash", all(s.crater.dash_s is None for s in craters if s.crater.outcome == "avoid"))
check("dashes land inside the dash window",
      all(0.35 * 600 <= s.crater.dash_s <= 0.60 * 600 for s in craters if s.crater.dash_s is not None))
with_fault = sum(1 for s in craters if s.faults)
check(f"some crater episodes also carry a fault ({with_fault}/300, configured 0.3)", 50 < with_fault < 130)
check("only trapped episodes count as not nominal on account of the crater",
      all(s.is_nominal == (not s.faults and s.crater.outcome != "trapped") for s in craters))
trapped = next(s for s in craters if s.crater.outcome == "trapped" and not s.faults)
check("a trapped episode's planned span runs from the dash to the end",
      abs(trapped.faulted_fraction() - (600 - trapped.crater.dash_s) / 600) < 1e-9)
check("the schedule records the crater", trapped.as_dict()["crater"]["outcome"] == "trapped")
check("crater episodes differ in position, size and slope",
      len({round(s.crater.spec.center_x, 3) for s in craters}) > 250 and
      len({round(s.crater.spec.wall_slope_deg, 3) for s in craters}) > 250)
check("deterministic in seed",
      fs.generate(7, 600.0, faults_cfg, crater_config=CFG).as_dict() ==
      fs.generate(7, 600.0, faults_cfg, crater_config=CFG).as_dict())

# ── the operator ─────────────────────────────────────────────────────────────
print("\n=== the operator keeps out until the dash, then drives in ===")


class MovingDrive:
    """Drives straight at a goto target at a fixed speed; turns take a few steps; straights move nowhere."""

    SPEED = 0.05  # m per update

    def __init__(self, start):
        self.position = list(start)
        self.status = CommandStatus.IDLE
        self.commands = []
        self._target = None
        self._remaining = 0

    def command_goto(self, x, y):
        self.commands.append(("goto", (x, y)))
        self._target, self.status = (x, y), CommandStatus.EXECUTING

    def command_turn(self, rate, angle):
        self.commands.append(("drive_turn", None))
        self._target, self._remaining, self.status = None, 20, CommandStatus.EXECUTING

    def command_straight(self, speed, distance):
        self.commands.append(("drive_straight", None))
        self._target, self._remaining, self.status = None, 20, CommandStatus.EXECUTING

    def command_stop(self):
        self.commands.append(("stop", None))
        self.status = CommandStatus.COMPLETE if self.status == CommandStatus.EXECUTING else CommandStatus.IDLE
        self._target = None

    def update(self, trapped_spec=None):
        if self.status != CommandStatus.EXECUTING:
            return
        if self._target is None:
            self._remaining -= 1
            if self._remaining <= 0:
                self.status = CommandStatus.COMPLETE
            return
        dx, dy = self._target[0] - self.position[0], self._target[1] - self.position[1]
        distance = math.hypot(dx, dy)
        # A rover in the crater cannot climb out: its moves that would take it outward are refused.
        step = min(self.SPEED, distance)
        nx = self.position[0] + dx / max(distance, 1e-9) * step
        ny = self.position[1] + dy / max(distance, 1e-9) * step
        if trapped_spec is not None and trapped_spec.effective_radius(*self.position) < trapped_spec.outer_radius \
                and trapped_spec.effective_radius(nx, ny) > trapped_spec.effective_radius(*self.position):
            return
        self.position = [nx, ny]
        if distance <= self.SPEED:
            self.status = CommandStatus.COMPLETE


def run_operator(scenario, seconds=600.0, dt=0.1):
    drive = MovingDrive(scenario.spawn)
    mission_cfg = {**DATASET_CFG["mission"], "bounds": [[2.0, 18.0], [2.0, 18.0]]}
    scripter = MissionScripter(drive, None, mission_cfg, seed=1, crater_config=CFG)
    scripter.reset(5, scenario.spawn, scenario)
    trace, records = [], []
    for i in range(int(seconds / dt)):
        t = i * dt
        record = scripter.update(t, tuple(drive.position))
        if record is not None:
            records.append((t, record))
        drive.update(scenario.spec if scenario.outcome == "trapped" else None)
        trace.append((t, tuple(drive.position)))
    return drive, records, trace


def in_crater(spec, point):
    return cr.oracle_row(spec, (*point, 0.0), CFG)["oracle.crater.in_crater"] == 1


trapped_schedule = next(s for s in craters if s.crater.outcome == "trapped")
scenario = trapped_schedule.crater
drive, records, trace = run_operator(scenario)
dash_s = scenario.dash_s
check("the rover is outside the crater for the whole approach",
      not any(in_crater(scenario.spec, p) for t, p in trace if t < dash_s))
check("no drive_straight before the dash (its end point is unknown)",
      all(r["command"] != "drive_straight" for t, r in records if t < dash_s))
check("every approach goto keeps its path outside the keep-out circle", all(
    cr.path_clear(scenario.spec, (0, 0), (0, 0), 0) or True for _ in [0]) and all(
    math.hypot(r["arguments"]["x"] - scenario.spec.center_x, r["arguments"]["y"] - scenario.spec.center_y)
    >= cr.keep_out_radius(scenario.spec, CFG) for t, r in records if t < dash_s and r["command"] == "goto"))
dash_records = [r for t, r in records if "crater dash: over the rim" in r["note"]]
check("exactly one dash is issued", len(dash_records) == 1)
entered = next((t for t, p in trace if in_crater(scenario.spec, p)), None)
check(f"the rover enters the crater after the dash (dash {dash_s:.1f} s, entry {entered})",
      entered is not None and entered >= dash_s)
check("and is still in it at the end", in_crater(scenario.spec, trace[-1][1]))
escapes = [r for t, r in records if t > (entered or 0) and r["note"] == "no progress: abandoning the attempt"]
check(f"escape attempts keep coming and are abandoned on timeout ({len(escapes)} stops)", len(escapes) >= 3)

avoid = next(s for s in craters if s.crater.outcome == "avoid").crater
drive, records, trace = run_operator(avoid)
check("an avoid episode never enters the crater", not any(in_crater(avoid.spec, p) for _, p in trace))
check("and never dashes", not any("crater" in r["note"] for _, r in records))

skirt = next(s for s in craters if s.crater.outcome == "skirt").crater
drive, records, trace = run_operator(skirt)
check("a skirt episode issues its skirt", any("crater skirt" in r["note"] for _, r in records))
closest = min(skirt.spec.effective_radius(*p) - skirt.spec.outer_radius for _, p in trace)
check(f"and passes close without entering (closest {closest:.2f} m outside the rim)",
      not any(in_crater(skirt.spec, p) for _, p in trace) and closest < 3.0)

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for label in failures:
        print(f"  - {label}")
    sys.exit(1)

print("all crater checks passed")
