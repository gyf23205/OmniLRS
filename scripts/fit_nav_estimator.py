#!/usr/bin/env python3
"""
Fit the navigation EKF's config from nominal calibration dumps.

    python3 scripts/fit_nav_estimator.py DUMP_DIR [--out estimator_fit.yaml]

DUMP_DIR holds the .npz files written by run_perseverance.py --estimator-dump (one per episode;
run a few NOMINAL episodes - no faults - covering straights, arcs, point turns and slopes). Each row
has the raw 30 Hz sensor channels beside the ground-truth base-link pose.

Truth is used here, offline, to find what the sensors mean and how noisy they are. The running
filter never sees it. Steps:

  1. truth kinematics   heading, pitch, roll, forward/lateral speed and yaw rate from the pose
  2. imu mounting       the rover's forward / left axes in the imu frame, by matching the heading
                        derived from the imu orientation to the truth; derived signals checked
  3. noise sigmas       residual spread of each sensor against the truth
  4. torque model       least squares per wheel, one episode held out for validation
  5. replay             run the filter over every dump with the fitted config: NIS should be ~1, and
                        the spread of each residual's window mean sets the CUSUM window_sigma

Prints the fitted `estimator:` block and writes it to --out, for pasting into
cfg/robot/perseverance.yaml. Pure numpy + pyyaml, no Isaac Sim.
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission_specific.perseverance.control.ackermann_model import ALL_WHEELS, RoverGeometry
from src.mission_specific.perseverance.estimation import models as M
from src.mission_specific.perseverance.estimation import monitor as MON
from src.mission_specific.perseverance.estimation.nav_filter import RESIDUAL_KEYS, NavFilter, imu_orientation_matrix

ROBOT_CONFIG = Path(__file__).resolve().parents[1] / "cfg" / "robot" / "perseverance.yaml"


def smooth(x, n):
    """Centered moving average over n samples; edges shrink the window."""
    if n <= 1:
        return x
    kernel = np.ones(n) / n
    padded = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def truth_kinematics(d, forward_sign, smooth_n):
    """Heading, attitude and body-frame velocities from the recorded base-link pose (w, x, y, z)."""
    w, x, y, z = d["truth.q0"], d["truth.q1"], d["truth.q2"], d["truth.q3"]
    # Rotation matrix columns: body x (left) and body y (backward when forward_sign = -1) in the world.
    bx = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
    by = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)
    forward = by * forward_sign
    left = bx
    heading = np.arctan2(forward[:, 1], forward[:, 0])
    pitch = np.arcsin(np.clip(forward[:, 2], -1, 1))        # nose up positive
    roll = np.arcsin(np.clip(left[:, 2], -1, 1))            # left side up positive

    dt = d["dt"]
    p = np.stack([d["truth.p0"], d["truth.p1"], d["truth.p2"]], axis=1)
    vel = np.gradient(p, axis=0) / dt[:, None]
    vel = np.stack([smooth(vel[:, i], smooth_n) for i in range(3)], axis=1)
    v = np.sum(vel * forward, axis=1)
    v_lat = np.sum(vel * left, axis=1)
    omega = smooth(np.gradient(np.unwrap(heading)) / dt, smooth_n)
    return {"heading": heading, "pitch": pitch, "roll": roll, "v": v, "v_lat": v_lat, "omega": omega,
            "v_dot": smooth(np.gradient(v) / dt, smooth_n)}


def best_channel(d, candidates, target, smooth_n):
    """(channel, sign, |corr|) of the candidate most correlated with target."""
    best = None
    for name in candidates:
        signal = smooth(d[name], smooth_n)
        if np.std(signal) < 1e-9 or np.std(target) < 1e-9:
            continue
        corr = float(np.corrcoef(signal, target)[0, 1])
        if best is None or abs(corr) > best[2]:
            best = (name, 1.0 if corr >= 0 else -1.0, abs(corr))
    return best


class ReplayRobot:
    """Feeds a dump back through NavFilter exactly as Robot would have delivered it."""

    _steer_joint_names = [f"steer_joint_{n}" for n in ("front_left", "front_right", "rear_left", "rear_right")]

    def __init__(self, d):
        self.d = d
        self.k = 0

    def get_imu_readings(self):
        d, k = self.d, self.k
        accel = {a: float(d[f"imu.{a}"][k]) for a in ("ax", "ay", "az")}
        gyro = {g: float(d[f"imu.{g}"][k]) for g in ("gx", "gy", "gz")}
        orientation = {o: math.degrees(float(d[f"imu.{o}"][k])) for o in ("roll", "pitch", "yaw")}
        return accel, gyro, orientation

    def get_wheel_joint_velocities(self, names):
        return [float(self.d[f"rate.{n}"][self.k]) for n in names]

    def get_wheel_joint_efforts(self, names):
        return [float(self.d[f"effort.{n}"][self.k]) for n in names]

    def get_steer_angles(self):
        # The dump stores angles already divided by steer_sign, so replay with steer_sign = 1.
        return [float(self.d.get(f"steer.{n.replace('steer_joint_', '')}", np.zeros(self.k + 1))[self.k])
                for n in self._steer_joint_names]


class ReplayDrive:
    def __init__(self, d, wheels):
        self.d, self.wheels, self.k = d, wheels, 0

    @property
    def last_wheel_command(self):
        values = {n: float(self.d[f"cmd.{n}"][self.k]) for n in self.wheels if f"cmd.{n}" in self.d}
        return values if values and all(math.isfinite(v) for v in values.values()) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_dir")
    parser.add_argument("--out", default="estimator_fit.yaml")
    parser.add_argument("--gravity", type=float, default=1.62)
    parser.add_argument("--smooth-s", type=float, default=0.3, help="smoothing window for differentiated truth")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "*.npz")))
    if not files:
        sys.exit(f"no .npz dumps in {args.dump_dir}")
    dumps = [dict(np.load(f)) for f in files]
    print(f"{len(dumps)} dumps, {sum(len(d['t']) for d in dumps)} samples")

    params = yaml.safe_load(open(ROBOT_CONFIG))["robots_settings"]["parameters"]
    geometry = RoverGeometry.from_config(params.get("geometry", {}), float(params.get("scale", 1.0)))
    forward_sign = float(params.get("drive_control", {}).get("forward_axis_sign", 1.0))
    current = dict(params.get("estimator", {}))
    dt = float(np.median(dumps[0]["dt"]))
    n_smooth = max(int(round(args.smooth_s / dt)), 1)
    g = args.gravity

    # Keep samples where the truth is trustworthy (finite) and the rover is on the ground.
    truths = [truth_kinematics(d, forward_sign, n_smooth) for d in dumps]
    cat = lambda key, source: np.concatenate([s[key] for s in source])
    T = {k: cat(k, truths) for k in truths[0]}
    D = {k: np.concatenate([d[k] for d in dumps]) for k in dumps[0] if all(k in d for d in dumps)}

    # ── 2. imu mounting ─────────────────────────────────────────────────────────
    # Search the 24 signed axis pairs for the rover's forward and left axes in the imu frame, scored
    # by how well the heading derived from the imu orientation matches the true heading. Then check
    # the derived pitch, roll, specific forces and yaw rate against the truth.
    axes = [np.array(v, dtype=float) for v in np.vstack([np.eye(3), -np.eye(3)])]
    rotations = [imu_orientation_matrix(r, p, y) for r, p, y in zip(D["imu.roll"], D["imu.pitch"], D["imu.yaw"])]
    rotations = np.array(rotations)
    best = None
    for f in axes:
        for l in axes:
            if abs(f @ l) > 0.5:
                continue
            fw = rotations @ f
            err = np.abs(wrap(np.arctan2(fw[:, 1], fw[:, 0]) - T["heading"]))
            score = float(np.median(err))
            if best is None or score < best[0]:
                best = (score, f, l)
    _, forward_axis, left_axis = best
    up_axis = np.cross(forward_axis, left_axis)
    fw, lw = rotations @ forward_axis, rotations @ left_axis
    accel_vec = np.stack([D["imu.ax"], D["imu.ay"], D["imu.az"]], axis=1)
    gyro_vec = np.stack([D["imu.gx"], D["imu.gy"], D["imu.gz"]], axis=1)
    derived = {
        "heading": np.arctan2(fw[:, 1], fw[:, 0]),
        "pitch": np.arcsin(np.clip(fw[:, 2], -1, 1)),
        "roll": np.arcsin(np.clip(lw[:, 2], -1, 1)),
        "a_f": accel_vec @ forward_axis,
        "a_l": accel_vec @ left_axis,
        "yaw_rate": gyro_vec @ up_axis,
    }
    fwd_target = T["v_dot"] + g * np.sin(T["pitch"])
    lat_target = T["omega"] * T["v"] + g * np.sin(T["roll"]) * np.cos(T["pitch"])
    corr = lambda a, b: float(np.corrcoef(smooth(a, n_smooth), b)[0, 1])
    print(f"\nimu mounting: forward {forward_axis.tolist()}  left {left_axis.tolist()}  (up {up_axis.tolist()})")
    print(f"  heading error    median {math.degrees(best[0]):.3f} deg")
    for label, value in [
        ("pitch", corr(derived["pitch"], T["pitch"])), ("roll", corr(derived["roll"], T["roll"])),
        ("forward force", corr(derived["a_f"], fwd_target)), ("lateral force", corr(derived["a_l"], lat_target)),
        ("yaw rate", corr(derived["yaw_rate"], T["omega"])),
    ]:
        print(f"  {label:15s} corr with truth {value:+.3f}{'' if value > 0.8 else '   <- weak'}")
    imu_map = {"forward": forward_axis.tolist(), "left": left_axis.tolist()}

    # ── 2b. wheel yaw scale ─────────────────────────────────────────────────────
    # Least-squares twist (v, v_lat, omega) from the wheels alone, per sample, against the gyro yaw rate.
    positions = geometry.wheel_positions()
    wheel_omega = np.zeros(len(D["t"]))
    for k in range(len(D["t"])):
        rows, z = [], []
        for name in ALL_WHEELS:
            delta = D[f"steer.{name}"][k] if f"steer.{name}" in D else 0.0
            c, s_ = math.cos(delta), math.sin(delta)
            wx, wy = positions[name]
            rows += [[c, s_, wy * s_ - wx * c], [-s_, c, wy * c + wx * s_]]
            z += [D[f"rate.{name}"][k] * geometry.wheel_radius, 0.0]
        wheel_omega[k] = np.linalg.lstsq(np.array(rows), np.array(z), rcond=None)[0][2]
    turning = np.abs(yaw_rate_gyro := derived["yaw_rate"]) > 0.05
    wheel_yaw_scale = float(np.sum(wheel_omega[turning] * yaw_rate_gyro[turning]) / np.sum(yaw_rate_gyro[turning] ** 2))
    print(f"\nwheel yaw scale (wheel-implied / gyro yaw rate while turning): {wheel_yaw_scale:.4f}")

    # ── 3. noise sigmas ─────────────────────────────────────────────────────────
    roll_err, slip_err = [], []
    for name in ALL_WHEELS:
        delta = D.get(f"steer.{name}", np.zeros_like(T["v"]))
        c, s = np.cos(delta), np.sin(delta)
        wx, wy = positions[name]
        predicted = c * (T["v"] - wheel_yaw_scale * T["omega"] * wx) + s * (T["v_lat"] + wheel_yaw_scale * T["omega"] * wy)
        roll_err.append(smooth(D[f"rate.{name}"] * geometry.wheel_radius, n_smooth) - predicted)
        slip_err.append(-s * (T["v"] - wheel_yaw_scale * T["omega"] * wx) + c * (T["v_lat"] + wheel_yaw_scale * T["omega"] * wy))
    robust = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))     # MAD sigma, outlier-proof
    yaw_rate = derived["yaw_rate"]
    gyro_err = smooth(yaw_rate, n_smooth) - T["omega"]
    measurement = {
        "sigma_wheel": robust(np.concatenate(roll_err)),
        "sigma_sideslip": robust(np.concatenate(slip_err)),
        "sigma_gyro": float(np.std(yaw_rate - smooth(yaw_rate, n_smooth))) or robust(gyro_err),
        "sigma_heading": robust(wrap(derived["heading"] - T["heading"])),
    }
    a_f = derived["a_f"]
    sigma_accel = float(np.std(a_f - smooth(a_f, n_smooth)))
    print("\nmeasurement noise (robust sigma against truth):")
    for key, value in measurement.items():
        print(f"  {key:15s} {value:.5f}")
    print(f"  sigma_accel     {sigma_accel:.5f}")

    commanded = [(D[f"cmd.{n}"], D[f"rate.{n}"]) for n in ALL_WHEELS if f"cmd.{n}" in D]
    tracking = np.concatenate([c - r for c, r in commanded]) if commanded else np.array([0.2])
    tracking = tracking[np.isfinite(tracking)]
    # Standard deviation, not MAD: the error is near zero except in the lag after each command change,
    # so MAD (0.007 rad/s in the first run) makes those normal transients look like 20-sigma events.
    tracking_sigma = float(np.std(tracking)) if len(tracking) else 0.2
    print(f"  tracking_sigma  {tracking_sigma:.5f} rad/s")

    # ── 4. torque model ─────────────────────────────────────────────────────────
    coefficients, sigmas = {}, {}
    holdout = len(dumps) - 1 if len(dumps) > 1 else None
    print(f"\ntorque model (held-out episode: {os.path.basename(files[holdout]) if holdout is not None else 'none'}):")
    for name in ALL_WHEELS:
        rows, torques, test_rows, test_torques = [], [], [], []
        for i, d in enumerate(dumps):
            rate = d[f"rate.{name}"]
            accel_w = smooth(np.gradient(rate) / d["dt"], n_smooth)
            sf = np.stack([d["imu.ax"], d["imu.ay"], d["imu.az"]], axis=1) @ forward_axis - d.get("ekf.b_a", np.zeros_like(rate))
            omega = d.get("ekf.omega", np.zeros_like(rate))
            feats = np.array([MON.torque_features(sf[k], rate[k], accel_w[k], omega[k]) for k in range(len(rate))])
            (test_rows if i == holdout else rows).append(feats)
            (test_torques if i == holdout else torques).append(d[f"effort.{name}"])
        X, tau = np.concatenate(rows), np.concatenate(torques)
        coef, sigma = MON.fit_torque_model(X, tau)
        r2 = 1.0 - sigma ** 2 / max(float(np.var(tau)), 1e-18)
        if r2 < 0.5:
            print(f"  WARNING {name}: the model explains only {r2:.0%} of the measured effort variance "
                  f"(std {np.std(tau):.4f} N*m). If effort does not follow slope and acceleration, the "
                  f"effort residual cannot detect slip or sink - check what get_wheel_joint_efforts reports.")
        coefficients[name] = [round(float(c), 6) for c in coef]
        sigmas[name] = round(sigma, 6)
        if test_rows:
            r = (np.concatenate(test_torques) - np.concatenate(test_rows) @ coef) / sigma
            print(f"  {name:12s} sigma {sigma:.4f} N*m  R^2 {r2:.2f}  held-out residual mean {r.mean():+.2f}, std {r.std():.2f}")
        else:
            print(f"  {name:12s} sigma {sigma:.3f} N*m")

    fitted = dict(current)
    fitted["imu"] = imu_map
    fitted["wheel_yaw_scale"] = round(wheel_yaw_scale, 5)
    fitted["process_noise"] = {**current.get("process_noise", {}), "sigma_accel": round(sigma_accel, 6)}
    fitted["measurement_noise"] = {**current.get("measurement_noise", {}),
                                   **{k: round(v, 6) for k, v in measurement.items()}}
    fitted["tracking_sigma"] = round(tracking_sigma, 6)
    fitted["torque_model"] = {"coefficients": coefficients, "sigma": sigmas}

    # ── 5. replay ───────────────────────────────────────────────────────────────
    # R is a sensor property, Q a model property, and each is tuned from the NIS it controls:
    #   wheels   sigma_wheel was measured against differentiated truth, which is itself noisy:
    #            refine it, sigma <- sigma * sqrt(NIS)
    #   gyro     keep the measured sensor sigma; a NIS above 1 means the yaw rate changes faster than
    #            q_omega allows: q_omega <- q_omega * NIS
    #   heading  keep the measured sensor sigma; a NIS above 1 means heading propagation errs more than
    #            q_heading allows (terrain rocking, scrub): q_heading <- q_heading * NIS
    # Inflating the gyro and heading R instead (tried first) hides model error and blunts both sensors.
    def replay(config):
        window_means = {key: [] for key in RESIDUAL_KEYS}
        episode_means = {key: [] for key in RESIDUAL_KEYS}
        nis = {"wheels": [], "gyro": [], "heading": []}
        for d in dumps:
            start = {key: len(values) for key, values in window_means.items()}
            robot = ReplayRobot(d)
            drive = ReplayDrive(d, ALL_WHEELS)
            nav = NavFilter(robot, drive, geometry, {**config, "calibration_dump": None}, steer_sign=1.0, gravity=g)
            last = None
            for k in range(len(d["t"])):
                robot.k = drive.k = k
                nav.update(float(d["dt"][k]))
                snap = nav.telemetry()
                if snap and snap is not last:
                    last = snap
                    for key, value in snap["nis"].items():
                        nis[key].append(value)
                    for prefix in ("wheel_rolling", "wheel_sideslip", "wheel_effort", "wheel_tracking"):
                        if f"{prefix}_mean" in snap:
                            for idx, name in enumerate(ALL_WHEELS):
                                window_means[f"{prefix}.{name}"].append(snap[f"{prefix}_mean"][idx])
                    for key in ("gyro", "heading", "lateral_force"):
                        window_means[key].append(snap["imu_residual"][f"{key}_mean"])
            for key, values in window_means.items():
                episode_means[key].append(values[start[key]:])
        return {k: float(np.nanmean(v)) for k, v in nis.items()}, window_means, episode_means

    print("\nreplay, tuning sigma_wheel, q_omega and q_heading from the innovations (target mean NIS = 1):")
    for iteration in range(8):
        nis, window_means, episode_means = replay(fitted)
        print(f"  pass {iteration}: " + "  ".join(f"NIS {k} {v:.2f}" for k, v in nis.items()))
        if all(0.8 < v < 1.25 for v in nis.values()) or iteration == 7:
            break
        noise = dict(fitted["measurement_noise"])
        # Half steps in log space: the full correction overshoots once the reacquisition logic engages.
        noise["sigma_wheel"] = round(noise["sigma_wheel"] * float(np.clip(nis["wheels"] ** 0.25, 0.5, 2.0)), 6)
        fitted["measurement_noise"] = noise
        process = dict(fitted["process_noise"])
        process["q_omega"] = float(f'{process.get("q_omega", 0.005) * float(np.clip(nis["gyro"] ** 0.5, 0.3, 3.0)):.3g}')
        process["q_heading"] = float(f'{process.get("q_heading", 3e-5) * float(np.clip(nis["heading"] ** 0.5, 0.3, 3.0)):.3g}')
        fitted["process_noise"] = process
    print("  final measurement noise: " + ", ".join(f"{k} {v:.5g}" for k, v in fitted["measurement_noise"].items()))
    print("  final process noise:     " + ", ".join(f"{k} {v:.5g}" for k, v in fitted["process_noise"].items()))

    window_sigma = {}
    for key, values in window_means.items():
        values = np.array([v for v in values if math.isfinite(v)])
        if len(values) > 10:
            window_sigma[key] = round(float(np.std(values)), 6)
    worst = sorted(window_sigma, key=lambda key: -abs(np.nanmean(window_means[key])) / max(window_sigma[key], 1e-9))[:6]
    print("  largest nominal window-mean offsets, in units of their own spread (CUSUM drift is 0.5):")
    for key in worst:
        print(f"    {key:28s} {np.nanmean(window_means[key]) / max(window_sigma[key], 1e-9):+.2f}")
    # Per-residual CUSUM thresholds: run each nominal episode's window means through the CUSUM and
    # take 1.5x the largest level reached (never below the default). Nominal driving then never alarms.
    cusum_cfg = {**current.get("cusum", {}), "window_sigma": window_sigma}
    drift = float(cusum_cfg.get("drift", 0.5))
    floor = float(cusum_cfg.get("threshold", 10.0))
    thresholds = {}
    for key, per_episode in episode_means.items():
        if key not in window_sigma:
            continue
        peak = 0.0
        for values in per_episode:
            cusum = MON.Cusum(drift, float("inf"))
            for value in values:
                if math.isfinite(value):
                    cusum.update(value / window_sigma[key])
                    peak = max(peak, cusum.high, cusum.low)
        thresholds[key] = round(max(floor, 1.5 * peak), 2)
    cusum_cfg["thresholds"] = thresholds
    fitted["cusum"] = cusum_cfg
    raised = {k: v for k, v in thresholds.items() if v > floor}
    print(f"  cusum thresholds raised above {floor:g} for {len(raised)} residuals: "
          + ", ".join(f"{k} {v:g}" for k, v in sorted(raised.items(), key=lambda kv: -kv[1])[:8]))

    text = yaml.safe_dump({"estimator": fitted}, sort_keys=False, default_flow_style=None)
    Path(args.out).write_text(text)
    print(f"\nwrote {args.out}; paste under robots_settings.parameters in {ROBOT_CONFIG.name}")


if __name__ == "__main__":
    main()
