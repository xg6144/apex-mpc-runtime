"""CUDA rollout-cost backend for :mod:`race_track_mpc.mpc_core`.

Only independent candidate scoring runs on the device.  Sampling, constraint
application, and CEM updates remain on the host so backend selection cannot
change the candidate controls presented to the validated cost implementation.

Launch overhead dominates this workload on the Jetson (the kernel itself is
sub-millisecond), so the host side is organized around two ideas:

* Everything that is constant across the cost calls of one ``solve()`` -- the
  reference, obstacle, wall, and scalar parameters -- is packed into a few
  device buffers and uploaded once per solve, detected by value comparison so
  ``costs()`` keeps its stateless signature.
* Per call only the candidate controls go up (through a pinned staging
  buffer) and the per-sample costs come back.
"""

import math
import subprocess
import sys

import numpy as np

from .mpc_accel import prepare_cost_inputs

try:
    from numba import cuda, float32
    CUDA_IMPORTED = True
except Exception:  # pragma: no cover - CUDA is an optional accelerator
    cuda = None
    float32 = None
    CUDA_IMPORTED = False


# The production Jetson Orin NX has eight SMs.  At 512 CEM samples a
# 128-thread launch creates only four blocks and therefore leaves half of the
# SMs idle.  Sixty-four threads create eight blocks while still providing two
# warps per block; the rollout is register-heavy enough that larger blocks do
# not improve occupancy.
THREADS_PER_BLOCK = 64
_AVAILABLE = None

# Scalar parameter slots (float32 params array).
_P_X0 = 0
_P_Y0 = 1
_P_YAW0 = 2
_P_V0 = 3
_P_PREV_STEER = 4
_P_PREV_ACCEL = 5
_P_DT = 6
_P_WHEELBASE = 7
_P_UNDERSTEER = 8
_P_MIN_SPEED = 9
_P_SPEED_CEILING = 10
_P_CONTOUR_W = 11
_P_LAG_W = 12
_P_HEADING_W = 13
_P_SPEED_W = 14
_P_FF_W = 15
_P_ACCEL_W = 16
_P_STEER_RATE_W = 17
_P_ACCEL_RATE_W = 18
_P_TERMINAL_W = 19
_P_FRICTION_W = 20
_P_MAX_LAT = 21
_P_DISC_FRONT = 22
_P_OBSTACLE_W = 23
_P_OBSTACLE_COLL_W = 24
_P_WALL_RES = 25
_P_WALL_OX = 26
_P_WALL_OY = 27
_P_WALL_SAFE = 28
_P_WALL_HARD = 29
_P_WALL_W = 30
_P_WALL_COLL_W = 31
_P_DISC_REAR = 32
_P_STEERING_RETENTION = 33
_P_STEERING_RESPONSE = 34
_P_APPLIED_STEER = 35
_P_STEERING_AVERAGE_RETENTION = 36
_P_STEERING_AVERAGE_COMMAND = 37
_P_GRIP_HARD_W = 38
_N_PARAMS = 39

# Integer parameter slots (int32 iparams array).
_I_N_OBS = 0
_I_USE_WALL = 1
_I_WALL_H = 2
_I_WALL_W = 3
_N_IPARAMS = 4

# Rows of the packed (8, horizon) reference array.  The path tangent is
# precomputed once on the host: evaluating sin/cos(reference yaw) in every
# sample used to repeat the same two transcendental operations 10,240 times
# per 512-sample cost call.
_R_X = 0
_R_Y = 1
_R_YAW = 2
_R_SIN_YAW = 3
_R_COS_YAW = 4
_R_SPEED = 5
_R_FF_STEER = 6
_R_ALLOWED_WALL = 7

# Rows of the packed (3, capacity) obstacle array.
_O_SAFE = 0
_O_COLL = 1
_O_ALLOWED = 2

_PROBE_CODE = r'''
import numpy as np
from numba import cuda, float32

@cuda.jit
def probe(out):
    if cuda.grid(1) == 0:
        out[0] = float32(1.0)

if not cuda.is_available():
    raise SystemExit(2)
host = np.zeros(1, dtype=np.float32)
device = cuda.to_device(host)
probe[1, 1](device)
device.copy_to_host(host)
cuda.synchronize()
if host[0] != np.float32(1.0):
    raise SystemExit(3)
'''


def wrap_angle_formula(angle):
    """Target-independent angle wrap used by the CUDA device function."""
    values = np.asarray(angle)
    return values - 2.0 * np.pi * np.floor((values + np.pi) / (2.0 * np.pi))


if CUDA_IMPORTED:

    @cuda.jit(device=True, inline=True)
    def _wrap_angle_device(angle):
        pi = float32(3.141592653589793)
        two_pi = float32(6.283185307179586)
        turns = float32(math.floor((angle + pi) / two_pi))
        return angle - two_pi * turns

    @cuda.jit(device=True, inline=True)
    def _wall_clearance_device(dist, res, ox, oy, height, width, x, y):
        """Manual bilinear lookup because numba.cuda has no texture API."""
        half = float32(0.5)
        column = (x - ox) / res - half
        row = float32(height) - (y - oy) / res - half
        c0 = int(math.floor(column))
        r0 = int(math.floor(row))
        if c0 < 0 or c0 >= width - 1 or r0 < 0 or r0 >= height - 1:
            return float32(0.0)
        fc = column - float32(c0)
        fr = row - float32(r0)
        one = float32(1.0)
        return (dist[r0, c0] * (one - fr) * (one - fc)
                + dist[r0, c0 + 1] * (one - fr) * fc
                + dist[r0 + 1, c0] * fr * (one - fc)
                + dist[r0 + 1, c0 + 1] * fr * fc)

    @cuda.jit(fastmath=True)
    def _rollout_cost_kernel(controls, refs, params, iparams,
                             obs_centers, obs_pack, wall_dist, out, n_samples):
        s = cuda.grid(1)
        if s >= n_samples:
            return

        prev_steer = params[_P_PREV_STEER]
        prev_accel = params[_P_PREV_ACCEL]
        dt = params[_P_DT]
        wheelbase = params[_P_WHEELBASE]
        understeer = params[_P_UNDERSTEER]
        steering_retention = params[_P_STEERING_RETENTION]
        steering_response = params[_P_STEERING_RESPONSE]
        steering_average_retention = params[_P_STEERING_AVERAGE_RETENTION]
        steering_average_command = params[_P_STEERING_AVERAGE_COMMAND]
        min_speed = params[_P_MIN_SPEED]
        speed_ceiling = params[_P_SPEED_CEILING]
        contour_w = params[_P_CONTOUR_W]
        lag_w = params[_P_LAG_W]
        heading_w = params[_P_HEADING_W]
        speed_w = params[_P_SPEED_W]
        ff_w = params[_P_FF_W]
        accel_w = params[_P_ACCEL_W]
        steer_rate_w = params[_P_STEER_RATE_W]
        accel_rate_w = params[_P_ACCEL_RATE_W]
        terminal_w = params[_P_TERMINAL_W]
        friction_w = params[_P_FRICTION_W]
        grip_hard_w = params[_P_GRIP_HARD_W]
        max_lat = params[_P_MAX_LAT]
        disc_front = params[_P_DISC_FRONT]
        disc_rear = params[_P_DISC_REAR]
        obstacle_w = params[_P_OBSTACLE_W]
        obstacle_coll_w = params[_P_OBSTACLE_COLL_W]
        wall_res = params[_P_WALL_RES]
        wall_ox = params[_P_WALL_OX]
        wall_oy = params[_P_WALL_OY]
        wall_safe = params[_P_WALL_SAFE]
        wall_hard = params[_P_WALL_HARD]
        wall_w = params[_P_WALL_W]
        wall_coll_w = params[_P_WALL_COLL_W]
        n_obs = iparams[_I_N_OBS]
        use_wall = iparams[_I_USE_WALL] != 0
        wall_h = iparams[_I_WALL_H]
        wall_w_int = iparams[_I_WALL_W]

        zero = float32(0.0)
        half = float32(0.5)
        x = params[_P_X0]
        y = params[_P_Y0]
        yaw = params[_P_YAW0]
        v = params[_P_V0]
        p_steer = prev_steer
        p_accel = prev_accel
        applied_steer = params[_P_APPLIED_STEER]
        cost = zero
        horizon = controls.shape[0]

        for h in range(horizon):
            # Horizon-major SoA layout makes every warp read adjacent sample
            # controls.  The old [sample, horizon, control] layout had a
            # 160-byte lane stride and generated almost one memory transaction
            # per thread.
            steer = controls[h, 0, s]
            accel = controls[h, 1, s]

            d_steer = steer - p_steer
            d_accel = accel - p_accel
            cost += steer_rate_w * d_steer * d_steer
            cost += accel_rate_w * d_accel * d_accel

            v_next = v + dt * accel
            if v_next < min_speed:
                v_next = min_speed
            elif v_next > speed_ceiling:
                v_next = speed_ceiling
            avg = half * (v + v_next)
            eff_wb = wheelbase + understeer * avg * avg
            wheel_steer = (steering_average_retention * applied_steer
                           + steering_average_command * steer)
            applied_steer = (steering_retention * applied_steer
                             + steering_response * steer)
            tan_steer = math.tan(wheel_steer)

            lat = v * v * tan_steer / (
                wheelbase + understeer * v * v)
            excess = abs(lat) - max_lat
            if excess > zero:
                cost += friction_w * excess * excess
                cost += grip_hard_w * excess

            d_ff = steer - refs[_R_FF_STEER, h]
            cost += ff_w * d_ff * d_ff
            cost += accel_w * accel * accel

            nx = x + dt * avg * math.cos(yaw)
            ny = y + dt * avg * math.sin(yaw)
            nyaw = yaw + dt * avg * tan_steer / eff_wb

            ex = nx - refs[_R_X, h]
            ey = ny - refs[_R_Y, h]
            tyaw = refs[_R_YAW, h]
            s_t = refs[_R_SIN_YAW, h]
            c_t = refs[_R_COS_YAW, h]
            contour = -s_t * ex + c_t * ey
            lag = c_t * ex + s_t * ey
            herr = _wrap_angle_device(nyaw - tyaw)
            serr = v_next - refs[_R_SPEED, h]
            step = (contour_w * contour * contour + lag_w * lag * lag
                    + heading_w * herr * herr + speed_w * serr * serr)
            cost += step
            if h == horizon - 1:
                cost += terminal_w * step

            if n_obs > 0 or use_wall:
                dirx = math.cos(nyaw)
                diry = math.sin(nyaw)
                fx = nx + disc_front * dirx
                fy = ny + disc_front * diry
                rx = nx + disc_rear * dirx
                ry = ny + disc_rear * diry

                for m in range(n_obs):
                    cxm = obs_centers[h, m, 0]
                    cym = obs_centers[h, m, 1]

                    front_dx = fx - cxm
                    front_dy = fy - cym
                    front_distance_sq = (front_dx * front_dx
                                         + front_dy * front_dy)
                    safe = obs_pack[_O_SAFE, m]
                    collision = (obs_pack[_O_COLL, m]
                                 - obs_pack[_O_ALLOWED, m])
                    active_distance = max(safe, collision)
                    # Reject distant discs using squared distance so sqrt is
                    # paid only when either penalty can actually be nonzero.
                    if (active_distance > zero
                            and front_distance_sq
                            < active_distance * active_distance):
                        front_distance = math.sqrt(front_distance_sq)
                        front_intrusion = safe - front_distance
                        if front_intrusion > zero:
                            cost += (obstacle_w * front_intrusion
                                     * front_intrusion)
                        front_overlap = collision - front_distance
                        if front_overlap > zero:
                            cost += obstacle_coll_w * front_overlap

                    rear_dx = rx - cxm
                    rear_dy = ry - cym
                    rear_distance_sq = rear_dx * rear_dx + rear_dy * rear_dy
                    if (active_distance > zero
                            and rear_distance_sq
                            < active_distance * active_distance):
                        rear_distance = math.sqrt(rear_distance_sq)
                        rear_intrusion = safe - rear_distance
                        if rear_intrusion > zero:
                            cost += obstacle_w * rear_intrusion * rear_intrusion
                        rear_overlap = collision - rear_distance
                        if rear_overlap > zero:
                            cost += obstacle_coll_w * rear_overlap

                if use_wall:
                    front_clearance = _wall_clearance_device(
                        wall_dist, wall_res, wall_ox, wall_oy,
                        wall_h, wall_w_int, fx, fy)
                    front_deficit = wall_safe - front_clearance
                    if front_deficit > zero:
                        cost += wall_w * front_deficit * front_deficit
                    front_overlap = (wall_hard - front_clearance
                                     - refs[_R_ALLOWED_WALL, h])
                    if front_overlap > zero:
                        cost += wall_coll_w * front_overlap

                    rear_clearance = _wall_clearance_device(
                        wall_dist, wall_res, wall_ox, wall_oy,
                        wall_h, wall_w_int, rx, ry)
                    rear_deficit = wall_safe - rear_clearance
                    if rear_deficit > zero:
                        cost += wall_w * rear_deficit * rear_deficit
                    rear_overlap = (wall_hard - rear_clearance
                                    - refs[_R_ALLOWED_WALL, h])
                    if rear_overlap > zero:
                        cost += wall_coll_w * rear_overlap

            p_steer = steer
            p_accel = accel
            x = nx
            y = ny
            yaw = nyaw
            v = v_next

        out[s] = cost


def available() -> bool:
    """Return true only after a driver, context, and kernel execution succeed."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    if not CUDA_IMPORTED:
        _AVAILABLE = False
        return False
    try:
        # Some mismatched driver/runtime combinations terminate the process
        # inside the CUDA API instead of raising a Python exception.  Probe in
        # a fresh interpreter so even SIGSEGV becomes a safe false result.
        result = subprocess.run(
            [sys.executable, '-c', _PROBE_CODE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        _AVAILABLE = result.returncode == 0
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - host-specific
        _AVAILABLE = False
    return _AVAILABLE


class CudaCostBackend:
    """FP32 CUDA implementation of the validated rollout-cost interface."""

    name = 'gpu'
    # fp32 cost accumulation over the horizon reaches ~2e-3 relative error on
    # rollouts that pile up collision penalties spanning six decades; ordering
    # of candidates is unaffected at that scale, so accept it rather than
    # spuriously fall back on the first solve near an obstacle or wall.
    tolerance = 5e-3
    sample_dtype = np.float32

    def __init__(self, config):
        if not available():
            raise RuntimeError('CUDA runtime probe failed')
        self.config = config
        self._lateral_accel_limit = float(config.max_lateral_accel)
        self._stream = cuda.stream()
        self.max_samples = max(int(config.samples), 1)
        self._allocate_sample_buffers(self.max_samples)

        horizon = config.horizon
        self._host_refs = cuda.pinned_array((8, horizon), dtype=np.float32)
        self._d_refs = cuda.device_array((8, horizon), dtype=np.float32)
        self._host_params = cuda.pinned_array(_N_PARAMS, dtype=np.float32)
        self._d_params = cuda.device_array(_N_PARAMS, dtype=np.float32)
        self._host_iparams = cuda.pinned_array(_N_IPARAMS, dtype=np.int32)
        self._d_iparams = cuda.device_array(_N_IPARAMS, dtype=np.int32)

        self._obstacle_capacity = 1
        self._allocate_obstacle_buffers(self._obstacle_capacity)
        self._wall_dist = np.zeros((2, 2), dtype=np.float32)
        self._d_wall_dist = cuda.to_device(self._wall_dist, stream=self._stream)
        self._wall_res = 1.0
        self._wall_ox = 0.0
        self._wall_oy = 0.0
        self._wall_h = 2
        self._wall_w = 2
        self._use_wall = False
        self._solve_key = None
        # The final CEM mean is scored as a one-sample batch.  Keep a compact
        # SoA buffer for it instead of copying the full 512-sample capacity.
        self._host_single_control = cuda.pinned_array(
            (horizon, 2, 1), dtype=np.float32)
        self._d_single_control = cuda.device_array(
            (horizon, 2, 1), dtype=np.float32)
        self._warmup()

    def set_lateral_accel_limit(self, limit):
        value = float(limit)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError('lateral acceleration limit must be positive')
        if abs(value - self._lateral_accel_limit) <= 1e-9:
            return
        self._lateral_accel_limit = value
        self._solve_key = None

    def _allocate_sample_buffers(self, capacity):
        self._host_controls = cuda.pinned_array(
            (self.config.horizon, 2, capacity), dtype=np.float32)
        self._d_controls = cuda.device_array(
            (self.config.horizon, 2, capacity), dtype=np.float32)
        self._d_out = cuda.device_array(capacity, dtype=np.float32)
        self._host_out = cuda.pinned_array(capacity, dtype=np.float32)

    def _allocate_obstacle_buffers(self, capacity):
        horizon = self.config.horizon
        self._host_obs_centers = cuda.pinned_array(
            (horizon, capacity, 2), dtype=np.float32)
        self._d_obs_centers = cuda.device_array(
            (horizon, capacity, 2), dtype=np.float32)
        self._host_obs_pack = cuda.pinned_array((3, capacity), dtype=np.float32)
        self._d_obs_pack = cuda.device_array((3, capacity), dtype=np.float32)

    def _ensure_sample_capacity(self, samples):
        # Batch buffers are horizon-major and therefore only contiguous when
        # their last dimension exactly matches the active sample count.
        # Normal operation uses config.samples plus the dedicated one-sample
        # buffer, so this reallocates only for an explicit runtime size change.
        if samples == 1 or samples == self.max_samples:
            return
        self.max_samples = int(samples)
        self._allocate_sample_buffers(self.max_samples)

    def _ensure_obstacle_capacity(self, obstacles):
        if obstacles <= self._obstacle_capacity:
            return
        self._obstacle_capacity = int(obstacles)
        self._allocate_obstacle_buffers(self._obstacle_capacity)
        # The packed obstacle uploads live in the replaced buffers.
        self._solve_key = None

    def _warmup(self):
        horizon = self.config.horizon
        reference = np.zeros((horizon + 1, 6), dtype=np.float32)
        reference[:, 4] = np.float32(1.0)
        controls = np.zeros((1, horizon, 2), dtype=np.float32)
        self.costs(np.zeros(4, dtype=np.float32), controls, reference,
                   np.zeros(2, dtype=np.float32),
                   np.empty((0, 5), dtype=np.float32),
                   np.float32(self.config.max_speed))

    def set_wall_field(self, wall_field):
        self._solve_key = None
        if wall_field is None:
            self._use_wall = False
            return
        self._wall_dist = np.ascontiguousarray(
            wall_field._distance, dtype=np.float32)
        self._d_wall_dist = cuda.to_device(
            self._wall_dist, stream=self._stream)
        self._wall_res = float(wall_field.resolution)
        self._wall_ox = float(wall_field.origin_x)
        self._wall_oy = float(wall_field.origin_y)
        self._wall_h = int(wall_field.height)
        self._wall_w = int(wall_field.width)
        self._use_wall = True

    def _solve_ready(self, initial_state, reference, previous_control,
                     obstacles, speed_ceiling, initial_applied_steer):
        key = self._solve_key
        return (key is not None
                and key['speed_ceiling'] == float(speed_ceiling)
                and key['applied_steer'] == float(initial_applied_steer)
                and np.array_equal(key['state'], initial_state)
                and np.array_equal(key['previous'], previous_control)
                and np.array_equal(key['obstacles'], obstacles)
                and np.array_equal(key['reference'], reference))

    def _prepare_solve(self, initial_state, reference, previous_control,
                       obstacles, speed_ceiling, initial_applied_steer):
        """Upload everything that is shared by the cost calls of one solve."""
        cfg = self.config
        prepared = prepare_cost_inputs(
            cfg, initial_state, reference, obstacles,
            self._use_wall, self._wall_dist, self._wall_res,
            self._wall_ox, self._wall_oy, self._wall_h, self._wall_w,
            dtype=np.float32)
        self._ensure_obstacle_capacity(max(prepared.n_obs, 1))

        refs = self._host_refs
        refs[_R_X] = prepared.ref_x
        refs[_R_Y] = prepared.ref_y
        refs[_R_YAW] = prepared.ref_yaw
        refs[_R_SIN_YAW] = prepared.ref_sin_yaw
        refs[_R_COS_YAW] = prepared.ref_cos_yaw
        refs[_R_SPEED] = prepared.ref_speed
        refs[_R_FF_STEER] = prepared.ff_steer
        refs[_R_ALLOWED_WALL] = prepared.allowed_wall

        count = prepared.n_obs if prepared.n_obs > 0 else 1
        self._host_obs_centers.fill(0.0)
        self._host_obs_pack.fill(0.0)
        self._host_obs_centers[:, :count] = prepared.obs_centers
        self._host_obs_pack[_O_SAFE, :count] = prepared.obs_safe
        self._host_obs_pack[_O_COLL, :count] = prepared.obs_coll
        self._host_obs_pack[_O_ALLOWED, :count] = prepared.obs_allowed

        params = self._host_params
        params[_P_X0] = initial_state[0]
        params[_P_Y0] = initial_state[1]
        params[_P_YAW0] = initial_state[2]
        params[_P_V0] = initial_state[3]
        params[_P_PREV_STEER] = previous_control[0]
        params[_P_PREV_ACCEL] = previous_control[1]
        params[_P_DT] = cfg.dt
        params[_P_WHEELBASE] = cfg.wheelbase
        params[_P_UNDERSTEER] = cfg.understeer_gradient
        steering_coefficients = cfg.steering_step_coefficients(cfg.dt)
        params[_P_STEERING_RETENTION] = steering_coefficients[0]
        params[_P_STEERING_RESPONSE] = steering_coefficients[1]
        params[_P_APPLIED_STEER] = initial_applied_steer
        params[_P_STEERING_AVERAGE_RETENTION] = steering_coefficients[2]
        params[_P_STEERING_AVERAGE_COMMAND] = steering_coefficients[3]
        params[_P_MIN_SPEED] = cfg.min_speed
        params[_P_SPEED_CEILING] = speed_ceiling
        params[_P_CONTOUR_W] = cfg.contour_weight
        params[_P_LAG_W] = cfg.lag_weight
        params[_P_HEADING_W] = cfg.heading_weight
        params[_P_SPEED_W] = cfg.speed_weight
        params[_P_FF_W] = cfg.steer_feedforward_weight
        params[_P_ACCEL_W] = cfg.accel_weight
        params[_P_STEER_RATE_W] = cfg.steer_rate_weight
        params[_P_ACCEL_RATE_W] = cfg.accel_rate_weight
        params[_P_TERMINAL_W] = cfg.terminal_weight
        params[_P_FRICTION_W] = cfg.friction_weight
        params[_P_GRIP_HARD_W] = cfg.grip_hard_weight
        params[_P_MAX_LAT] = self._lateral_accel_limit
        params[_P_DISC_FRONT] = cfg.ego_center_offset + cfg.ego_disc_offset
        params[_P_DISC_REAR] = cfg.ego_center_offset - cfg.ego_disc_offset
        params[_P_OBSTACLE_W] = cfg.obstacle_weight
        params[_P_OBSTACLE_COLL_W] = cfg.obstacle_collision_weight
        params[_P_WALL_RES] = self._wall_res
        params[_P_WALL_OX] = self._wall_ox
        params[_P_WALL_OY] = self._wall_oy
        params[_P_WALL_SAFE] = prepared.wall_safe
        params[_P_WALL_HARD] = prepared.wall_hard
        params[_P_WALL_W] = cfg.wall_weight
        params[_P_WALL_COLL_W] = cfg.wall_collision_weight

        iparams = self._host_iparams
        iparams[_I_N_OBS] = prepared.n_obs
        iparams[_I_USE_WALL] = 1 if self._use_wall else 0
        iparams[_I_WALL_H] = self._wall_h
        iparams[_I_WALL_W] = self._wall_w

        stream = self._stream
        self._d_refs.copy_to_device(refs, stream=stream)
        self._d_params.copy_to_device(params, stream=stream)
        self._d_iparams.copy_to_device(iparams, stream=stream)
        self._d_obs_centers.copy_to_device(self._host_obs_centers,
                                           stream=stream)
        self._d_obs_pack.copy_to_device(self._host_obs_pack, stream=stream)

        self._solve_key = {
            'state': np.array(initial_state, dtype=np.float64, copy=True),
            'reference': np.array(reference, dtype=np.float64, copy=True),
            'previous': np.array(previous_control, dtype=np.float64,
                                 copy=True),
            'obstacles': np.array(obstacles, dtype=np.float64,
                                  copy=True).reshape(-1, 5),
            'speed_ceiling': float(speed_ceiling),
            'applied_steer': float(initial_applied_steer),
        }

    def costs(self, initial_state, controls, reference, previous_control,
              obstacles, speed_ceiling, initial_applied_steer=None):
        controls = np.asarray(controls)
        samples = int(controls.shape[0])
        if samples == 0:
            return np.empty(0, dtype=np.float64)
        self._ensure_sample_capacity(samples)

        if initial_applied_steer is None:
            initial_applied_steer = (
                self.config.steering_gain * float(previous_control[0]))

        obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 5)
        if not self._solve_ready(initial_state, reference, previous_control,
                                 obstacles, speed_ceiling,
                                 initial_applied_steer):
            self._prepare_solve(initial_state, reference, previous_control,
                                obstacles, speed_ceiling,
                                initial_applied_steer)

        stream = self._stream
        if samples == 1:
            staging = self._host_single_control
            device_controls = self._d_single_control
        else:
            staging = self._host_controls
            device_controls = self._d_controls
        staging[...] = controls.transpose(1, 2, 0)
        device_controls.copy_to_device(staging, stream=stream)

        blocks = (samples + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        _rollout_cost_kernel[blocks, THREADS_PER_BLOCK, stream](
            device_controls, self._d_refs, self._d_params,
            self._d_iparams, self._d_obs_centers, self._d_obs_pack,
            self._d_wall_dist, self._d_out, samples)
        self._d_out[:samples].copy_to_host(
            self._host_out[:samples], stream=stream)
        stream.synchronize()
        return self._host_out[:samples].astype(np.float64, copy=True)
