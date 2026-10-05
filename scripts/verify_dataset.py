#!/usr/bin/env python3
"""
Check a generated fault-detection dataset the way a consumer would: with pandas, from the files
alone, with no knowledge of the code that wrote them.

    /isaac-sim/python.sh scripts/verify_dataset.py my_files/datasets/run01

Reports per episode: shape, commands, fault events, distance travelled, battery drain, how many
observable cells the comms fault emptied, and whether the label is clear before the first onset and
set after it. Then checks the one cross-episode property that decides whether the dataset can be
concatenated at all - that every episode has the same columns.

Run it inside the simulator container: the Isaac Sim python has a working pandas.
"""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
print(f"manifest: {manifest['episodes']} episodes  "
      f"classes={manifest['class_balance']['episodes']}  "
      f"sample-level faulted={manifest['class_balance']['sample_level_faulted_fraction']}")
print(f"stopped_early={manifest['stopped_early']!r}")

episode_dirs = sorted((root / "episodes").iterdir())
column_sets = []

for directory in episode_dirs:
    meta = json.loads((directory / "meta.json").read_text())
    obs = pd.read_csv(directory / "observable/telemetry.csv")
    oracle = pd.read_csv(directory / "oracle/truth.csv")
    events = [json.loads(l) for l in (directory / "oracle/faults.jsonl").read_text().splitlines() if l]
    commands = [json.loads(l) for l in (directory / "observable/commands.jsonl").read_text().splitlines() if l]
    schedule = json.loads((directory / "oracle/schedule.json").read_text())
    column_sets.append(tuple(obs.columns))

    print(f"\n--- {directory.name}  class={meta['episode_class']}  seed={meta['seed']} ---")
    print(f"  observable {obs.shape[0]}x{obs.shape[1]}   oracle {oracle.shape[0]}x{oracle.shape[1]}")
    print(f"  commands={len(commands)} (lost {sum(1 for c in commands if c.get('delivered') is False)})"
          f"  fault events={len(events)}  images saved={meta['images_saved']} lost={meta['images_lost']}")
    print(f"  dropped: {meta['dropped']}")

    # movement and battery are the two signs the episode really ran
    if "pose_ground_truth.position.x" in oracle:
        dx = oracle["pose_ground_truth.position.x"]
        dy = oracle["pose_ground_truth.position.y"]
        travel = ((dx.diff() ** 2 + dy.diff() ** 2) ** 0.5).sum()
        print(f"  travelled {travel:.2f} m   start=({dx.iloc[0]:.2f}, {dy.iloc[0]:.2f})")
    if "oracle.battery_charge_wh" in oracle:
        print(f"  battery {oracle['oracle.battery_charge_wh'].iloc[0]:.2f} -> "
              f"{oracle['oracle.battery_charge_wh'].iloc[-1]:.2f} Wh")
    if "battery_charge" in obs:
        print(f"  downlinked battery% {obs['battery_charge'].iloc[0]} -> {obs['battery_charge'].iloc[-1]}")

    # the label must be clean before the first onset and set after it
    if schedule["faults"]:
        onset = min(f["onset_s"] for f in schedule["faults"])
        before = oracle[oracle.time_s < onset]["oracle.fault_active"]
        after = oracle[oracle.time_s > onset + 2.0]["oracle.fault_active"]
        print(f"  first onset at {onset:.1f}s   label before={set(before)} after={set(after)}")
        print(f"  scheduled: {[(f['kind'], f['target'], f['magnitudes']) for f in schedule['faults']]}")
        print(f"  applied:   {[(e['kind'], e['targets'], e['magnitudes']) for e in events]}")
    else:
        active = set(oracle["oracle.fault_active"])
        print(f"  nominal: label values {active}  (must be {{0}})")

    nan_cells = int(obs.isna().sum().sum())
    print(f"  empty observable cells: {nan_cells}")
    if "oracle.imu_error.az" in oracle:
        print(f"  imu residual az: max |err| = {oracle['oracle.imu_error.az'].abs().max():.4f}")

print("\n=== cross-episode ===")
print("identical column sets across episodes:", len(set(column_sets)) == 1)
if len(set(column_sets)) == 1:
    print(f"columns: {len(column_sets[0])} observable")
