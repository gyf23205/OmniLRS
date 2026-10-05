__license__ = "BSD-3-Clause"
__status__ = "development"

"""
Process and measurement models for the perseverance navigation EKF.

Pure numpy: no omni / pxr / Isaac Sim imports, so every model and every Jacobian is testable with a
plain python3 interpreter. The filter mechanics live in ekf.py; this file only says what the rover
is expected to measure given a state.

State (STATE_DIM = 8), episode-local and planar:
    px, py   position in the frame fixed at episode start, m
    psi      heading of the forward axis, rad, counter-clockwise from world +x
    v        forward speed along the ground, m/s
    v_lat    sideways sliding speed, positive LEFT, m/s. Zero on a rover that has grip.
    omega    yaw rate about the body up axis, positive LEFT (counter-clockwise), rad/s
    b_g      gyro z bias, rad/s
    b_a      forward accelerometer bias, m/s^2

Body frame matches ackermann_model.py: x = lateral positive LEFT, y = longitudinal positive FORWARD,
wheel positions relative to the mid axle. A steer angle delta is measured from forward toward left.

Rigid-body velocity of the wheel at (x_i, y_i), in (forward, left) components:
    forward = v - omega * x_i
    left    = v_lat + omega * y_i
With v_lat = 0 this is exactly the motion AckermannModel._solve_wheel assumes (turn centre at
(R, 0), v = omega * R), which is why the wheel models below are that solver inverted.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

PX, PY, PSI, V, VLAT, OMEGA, BG, BA = range(8)
STATE_DIM = 8
STATE_NAMES = ("px", "py", "psi", "v", "v_lat", "omega", "b_g", "b_a")


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class ImuInput:
    """
    One IMU sample, already mapped into the body frame by the adapter.

    a_f, a_l are specific forces (what an accelerometer reads, gravity included): forward and left,
    m/s^2. roll and pitch are radians; pitch is positive nose-up, roll positive left side up, so a
    rover parked nose-up on a slope reads a_f = +g*sin(pitch).
    """

    a_f: float
    a_l: float
    roll: float
    pitch: float
    pitch_rate: float = 0.0      # gyro rate about the left axis, rad/s; positive nose DOWN (right-hand rule)


@dataclass
class ProcessNoise:
    """
    sigma_accel is the per-sample accelerometer noise (one reading at the filter rate), so its effect
    on the speeds over one step is (sigma_accel * dt)^2. The q_* entries are random-walk densities,
    discretised as q * dt.
    """

    sigma_accel: float = 0.05        # m/s^2, std of one accelerometer sample, drives v and v_lat
    q_omega: float = 5e-3            # (rad/s)^2 per s, yaw rate random walk: steering is not modelled
    q_gyro_bias: float = 1e-6        # (rad/s)^2 per s
    q_accel_bias: float = 1e-5       # (m/s^2)^2 per s
    q_position: float = 1e-6         # m^2 per s, keeps P well conditioned while parked
    q_heading: float = 3e-5          # rad^2 per s, unmodelled heading kinematics on rough terrain


# ── process model ──────────────────────────────────────────────────────────────
def propagate(x: np.ndarray, imu: ImuInput, dt: float, gravity: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    One prediction step. Returns (x_next, F) with F = d(x_next)/dx.

    Inertial navigation with measured acceleration: the accelerometer is the input, gravity is
    removed with the IMU attitude, and the omega * v terms are the rotating-frame (centripetal and
    Coriolis) terms of a body turning at omega.
    """
    psi, v, v_lat, omega, b_a = x[PSI], x[V], x[VLAT], x[OMEGA], x[BA]
    c_psi, s_psi = math.cos(psi), math.sin(psi)
    c_th, s_th = math.cos(imu.pitch), math.sin(imu.pitch)
    c_ph, s_ph = math.cos(imu.roll), math.sin(imu.roll)
    heading_gain = c_ph / c_th
    # Euler kinematics: psi_dot = (q sin(roll) + r cos(roll)) / cos(pitch). The q term matters on rough
    # ground, where the rover rocks in pitch while rolled; dropping it gave a heading NIS of ~240.
    heading_rate_input = imu.pitch_rate * s_ph / c_th

    x_next = np.array(x, dtype=float)
    x_next[PX] += (v * c_th * c_psi - v_lat * s_psi) * dt
    x_next[PY] += (v * c_th * s_psi + v_lat * c_psi) * dt
    x_next[PSI] = wrap_angle(psi + (omega * heading_gain + heading_rate_input) * dt)
    x_next[V] += (imu.a_f - b_a - gravity * s_th + omega * v_lat) * dt
    x_next[VLAT] += (imu.a_l - gravity * s_ph * c_th - omega * v) * dt
    # omega, b_g, b_a: random walks, unchanged in the mean

    F = np.eye(STATE_DIM)
    F[PX, PSI] = (-v * c_th * s_psi - v_lat * c_psi) * dt
    F[PX, V] = c_th * c_psi * dt
    F[PX, VLAT] = -s_psi * dt
    F[PY, PSI] = (v * c_th * c_psi - v_lat * s_psi) * dt
    F[PY, V] = c_th * s_psi * dt
    F[PY, VLAT] = c_psi * dt
    F[PSI, OMEGA] = heading_gain * dt
    F[V, VLAT] = omega * dt
    F[V, OMEGA] = v_lat * dt
    F[V, BA] = -dt
    F[VLAT, V] = -omega * dt
    F[VLAT, OMEGA] = -v * dt
    return x_next, F


def process_noise(noise: ProcessNoise, dt: float) -> np.ndarray:
    """Discrete Q for one step of length dt."""
    q = np.zeros(STATE_DIM)
    q[PX] = q[PY] = noise.q_position * dt
    q[PSI] = noise.q_heading * dt
    q[V] = q[VLAT] = (noise.sigma_accel * dt) ** 2
    q[OMEGA] = noise.q_omega * dt
    q[BG] = noise.q_gyro_bias * dt
    q[BA] = noise.q_accel_bias * dt
    return np.diag(q)


# ── measurement models ─────────────────────────────────────────────────────────
# Each returns (h, H): the predicted measurement and its 1 x STATE_DIM Jacobian row.

def wheel_rolling(x: np.ndarray, wheel_xy: Tuple[float, float], delta: float,
                  yaw_scale: float = 1.0) -> Tuple[float, np.ndarray]:
    """
    Ground speed along the rolling direction of a wheel steered by delta. The encoder measures it
    as wheel_radius * joint rate. AckermannModel.solve() inverted: see the module docstring.

    yaw_scale: the wheels turn as if the rover yawed yaw_scale * omega. A skid-steering
    rocker-bogie scrubs in turns, so its effective geometry is not the nominal one (the calibration
    run measured 1.05). Fitted on nominal data; the state omega stays the true, gyro-consistent rate.
    """
    wx, wy = wheel_xy
    c, s = math.cos(delta), math.sin(delta)
    H = np.zeros(STATE_DIM)
    H[V] = c
    H[VLAT] = s
    H[OMEGA] = (wy * s - wx * c) * yaw_scale
    return float(H @ x), H


def wheel_sideslip(x: np.ndarray, wheel_xy: Tuple[float, float], delta: float,
                   yaw_scale: float = 1.0) -> Tuple[float, np.ndarray]:
    """
    Ground speed along the wheel's axle, which is zero for a wheel that rolls without sliding. Used
    as a pseudo-measurement z = 0. For a mid wheel (delta = 0, y = 0) it reduces to v_lat.
    """
    wx, wy = wheel_xy
    c, s = math.cos(delta), math.sin(delta)
    H = np.zeros(STATE_DIM)
    H[V] = -s
    H[VLAT] = c
    H[OMEGA] = (wy * c + wx * s) * yaw_scale
    return float(H @ x), H


def gyro_z(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Body up-axis rate as the gyro reads it: true rate plus bias."""
    H = np.zeros(STATE_DIM)
    H[OMEGA] = 1.0
    H[BG] = 1.0
    return float(x[OMEGA] + x[BG]), H


def heading(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Heading from the IMU attitude, once the fixed mounting offset is removed. Innovation must be wrapped."""
    H = np.zeros(STATE_DIM)
    H[PSI] = 1.0
    return float(x[PSI]), H
