#!/usr/bin/env python3
"""
Host-runnable checks for the dataset fault scheduler. No Isaac Sim, no omni imports.

    python3 test/test_dataset_scheduler.py

The properties here are the ones that decide whether the dataset is trainable at all: majority
nominal, onsets in the middle so a transition exists to detect, overlap rare but real, and sequential
pairs that genuinely do not overlap. All of them are statistical, so they are checked over a large
sample rather than on one schedule.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.dataset import fault_scheduler as fs

failures = []
DURATION = 600.0
SAMPLE = 4000


def check(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def end_of(fault):
    return DURATION if fault.recovery_s is None else fault.recovery_s


def overlaps(schedule):
    faults = schedule.faults
    for i, first in enumerate(faults):
        for second in faults[i + 1:]:
            if first.onset_s < end_of(second) and second.onset_s < end_of(first):
                return True
    return False


schedules = [fs.generate(seed, DURATION) for seed in range(SAMPLE)]

# ── determinism ──────────────────────────────────────────────────────────────
print("\n=== determinism: the seed is the whole schedule ===")
check("same seed gives the same schedule", fs.generate(7, DURATION).as_dict() == fs.generate(7, DURATION).as_dict())
distinct = {repr(fs.generate(seed, DURATION).as_dict()) for seed in range(200)}
check("different seeds give different schedules", len(distinct) > 50)
check("duration is not baked into the seed",
      fs.generate(7, DURATION).as_dict() != fs.generate(7, DURATION * 2).as_dict())

# ── class balance ────────────────────────────────────────────────────────────
print("\n=== class balance: majority nominal, overlap rare ===")
classes = Counter(schedule.episode_class for schedule in schedules)
nominal = classes["nominal"] / SAMPLE
overlap = sum(overlaps(schedule) for schedule in schedules) / SAMPLE
faulted_samples = sum(schedule.faulted_fraction() for schedule in schedules) / SAMPLE

print(f"  nominal {nominal:.3f}  overlap {overlap:.3f}  sample-level faulted {faulted_samples:.3f}")
check("nominal episodes are the majority", nominal >= 0.55)
check("overlap is about 10% of episodes", 0.06 <= overlap <= 0.13)
check("every nominal episode really has no fault",
      all(not schedule.faults for schedule in schedules if schedule.episode_class == "nominal"))
check("every faulted episode really has a fault",
      all(schedule.faults for schedule in schedules if schedule.episode_class != "nominal"))
check("only concurrent episodes overlap",
      all(schedule.episode_class == "concurrent" for schedule in schedules if overlaps(schedule)))
check("every concurrent episode overlaps",
      all(overlaps(schedule) for schedule in schedules if schedule.episode_class == "concurrent"))

# The number that actually governs training, and the reason the manifest reports it: a clean
# lead-in inside every faulted episode pushes it far below the episode-level rate.
check("sample-level faulted fraction is well below the episode-level one",
      0.05 <= faulted_samples < (1.0 - nominal))

# ── onset placement ──────────────────────────────────────────────────────────
print("\n=== onsets sit in the middle, so a transition exists to detect ===")
first_onsets = [min(fault.onset_s for fault in s.faults) for s in schedules if s.faults]
last_onsets = [max(fault.onset_s for fault in s.faults) for s in schedules if s.faults]
check("no fault starts in the opening fifth", min(first_onsets) >= 0.20 * DURATION - 1e-9)
check("no fault starts in the closing fifth", max(last_onsets) <= 0.80 * DURATION + 1e-9)
check("single-fault onsets stay inside the configured central window",
      all(0.40 * DURATION - 1e-9 <= s.faults[0].onset_s <= 0.60 * DURATION + 1e-9
          for s in schedules if s.episode_class == "single"))

# ── recovery rules ───────────────────────────────────────────────────────────
print("\n=== recovery: only what can actually be lifted ===")
every_fault = [fault for schedule in schedules for fault in schedule.faults]
check("steer_stuck is never given a recovery",
      all(fault.recovery_s is None for fault in every_fault if fault.kind == "steer_stuck"))
check("a recovery is always after its onset",
      all(fault.recovery_s > fault.onset_s for fault in every_fault if fault.recovery_s is not None))
check("a recovery event drives every magnitude to zero",
      all(all(m == 0.0 for m in event.magnitudes)
          for fault in every_fault for event in fault.events() if event.action == "recovery"))

sequential = [s for s in schedules if s.episode_class == "sequential"]
check("sequential pairs never overlap", all(not overlaps(s) for s in sequential))
check("the recoverable half of a sequential pair is never steer_stuck",
      all(s.faults[0].kind != "steer_stuck" for s in sequential))
check("the first of a sequential pair always recovers",
      all(s.faults[0].recovery_s is not None for s in sequential))

# ── magnitudes and targets ───────────────────────────────────────────────────
print("\n=== magnitudes stay inside the shapes the injector accepts ===")
check("severities are in [0.3, 1.0]",
      all(0.30 - 1e-9 <= m <= 1.0 + 1e-9
          for fault in every_fault if fault.kind != "steer_stuck"
          for m in fault.magnitudes if m > 0.0))
check("a two-mode kind never has both magnitudes zero",
      all(any(m > 0.0 for m in fault.magnitudes)
          for fault in every_fault if fault.kind in fs.TWO_MODE_KINDS))
check("steer_stuck angles are away from zero",
      all(10.0 - 1e-9 <= abs(fault.magnitudes[0]) <= 45.0 + 1e-9
          for fault in every_fault if fault.kind == "steer_stuck"))
check("stuck angles go both ways",
      any(fault.magnitudes[0] < 0 for fault in every_fault if fault.kind == "steer_stuck") and
      any(fault.magnitudes[0] > 0 for fault in every_fault if fault.kind == "steer_stuck"))
check("targeted kinds always carry a target",
      all(fault.target is not None for fault in every_fault
          if fs.KIND_SHAPES[fault.kind][0] is not None))
check("untargeted kinds never carry one",
      all(fault.target is None for fault in every_fault
          if fs.KIND_SHAPES[fault.kind][0] is None))
check("magnitude count matches the injector's signature",
      all(len(fault.magnitudes) == fs.KIND_SHAPES[fault.kind][1] for fault in every_fault))

# ── playback ─────────────────────────────────────────────────────────────────
print("\n=== playback: events come out once, in order ===")
schedule = next(s for s in schedules if s.episode_class == "sequential")
seen = []
for tick in range(int(DURATION) + 1):
    seen.extend(schedule.pop_due(float(tick)))
check("every event is played exactly once",
      len(seen) == sum(len(fault.events()) for fault in schedule.faults))
check("events come out in time order", seen == sorted(seen, key=lambda e: e.time_s))
check("nothing is left after the episode", schedule.pop_due(DURATION * 2) == [])
schedule.reset_playback()
check("reset_playback replays the same events", len(schedule.pop_due(DURATION * 2)) == len(seen))

print("\n=== active_at agrees with the event stream ===")
schedule = next(s for s in schedules if s.episode_class == "concurrent")
mid = max(fault.onset_s for fault in schedule.faults) + 1.0
check("both concurrent faults are active together", len(schedule.active_at(mid)) == 2)
check("nothing is active before the first onset",
      schedule.active_at(min(fault.onset_s for fault in schedule.faults) - 1.0) == [])

nominal_schedule = next(s for s in schedules if s.is_nominal)
check("a nominal episode is never active", nominal_schedule.active_at(DURATION / 2) == [])
check("a nominal episode has zero faulted fraction", nominal_schedule.faulted_fraction() == 0.0)

# faulted_fraction unions overlapping spans rather than double counting them.
manual = fs.FaultSchedule("single", 100.0, [
    fs.ScheduledFault("battery", None, (0.5,), 20.0, 60.0),
    fs.ScheduledFault("imu", None, (0.5, 0.0), 40.0, 80.0),
])
check("overlapping spans are unioned, not summed", abs(manual.faulted_fraction() - 0.60) < 1e-9)

# ── config ───────────────────────────────────────────────────────────────────
print("\n=== config overrides reach the sampler ===")
all_nominal = fs.generate(3, DURATION, {"class_weights": {"nominal": 1.0, "single": 0.0,
                                                          "concurrent": 0.0, "sequential": 0.0}})
check("weights can force every episode nominal", all_nominal.is_nominal)
one_kind = [fs.generate(seed, DURATION, {
    "class_weights": {"nominal": 0.0, "single": 1.0, "concurrent": 0.0, "sequential": 0.0},
    "kind_weights": {kind: (1.0 if kind == "battery" else 0.0) for kind in fs.KIND_SHAPES},
}) for seed in range(50)]
check("kind weights can restrict the draw when the others are zeroed",
      all(fault.kind == "battery" for s in one_kind for fault in s.faults))
check("merged_config leaves DEFAULTS alone",
      fs.merged_config({"severity_range": [0.9, 1.0]})["severity_range"] == [0.9, 1.0]
      and fs.DEFAULTS["severity_range"] == [0.30, 1.00])

# ── fixed schedules (--fault with --dataset-out) ─────────────────────────────
print("\n=== fixed schedules: --fault records exactly what was asked ===")
fixed = fs.from_specs(["wheel_slip:front_left:0.8"], 7, DURATION)
only = fixed.faults[0] if len(fixed.faults) == 1 else None
check("one spec gives one single-class fault with its kind, target and magnitude",
      fixed.episode_class == "single" and only is not None
      and (only.kind, only.target, only.magnitudes, only.recovery_s) == ("wheel_slip", "front_left", (0.8,), None))
low, high = fs.DEFAULTS["onset_window"]
check("an omitted onset lands in the onset window", low * DURATION <= only.onset_s <= high * DURATION)
check("an omitted onset is deterministic in the seed",
      fs.from_specs(["battery:0.5"], 3, DURATION).as_dict() == fs.from_specs(["battery:0.5"], 3, DURATION).as_dict())
pair = fs.from_specs(["imu:0:0.5@120", "steer_stuck:ALL:-20@200"], 0, DURATION)
check("explicit onsets, two-mode mode and several specs",
      pair.episode_class == "concurrent" and [f.onset_s for f in pair.faults] == [120.0, 200.0]
      and pair.faults[0].mode == "noise" and pair.faults[1].target == fs.ALL)
bad_specs = ["foo:1", "wheel_slip:nope:1", "battery:x", "comms:0.5", "battery:0.5@700", "battery:0.5@-1"]
rejected = 0
for spec in bad_specs:
    try:
        fs.from_specs([spec], 0, DURATION)
    except ValueError:
        rejected += 1
check("malformed specs raise instead of recording a nominal episode", rejected == len(bad_specs))

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for label in failures:
        print(f"  - {label}")
    sys.exit(1)

print("all scheduler checks passed")
