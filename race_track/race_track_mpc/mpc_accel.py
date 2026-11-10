"""Numba-accelerated rollout + cost for the sampling MPC.

The cross-entropy solver scores thousands of control rollouts per control
cycle; in pure NumPy each cost term is a separate pass over big (samples x
horizon) temporaries, single-threaded.  This module fuses the whole
rollout-and-cost into one native kernel, with each rollout doing its horizon
loop in registers and no large temporaries.  On the Jetson Orin NX the serial
kernel is deliberately preferred: at the competition population sizes the
Numba parallel scheduler's per-call fork/join and wake-up jitter cost more than
the saved arithmetic, and also contend with localization and the lidar driver.

The kernel is a bit-for-bit port of ``SamplingMPC._cost``/``_rollout``/
``_clearance_cost`` (and ``MapDistanceField.clearance``).  ``SamplingMPC``
validates it against the NumPy reference on the first real solve and silently
falls back to NumPy if anything does not match, so correctness never depends on
the accelerator being present or exact.
"""

import math
import os
from dataclasses import dataclass

import numpy as np

try:
    import numba
    from numba import njit
    AVAILABLE = True
except Exception:  # pragma: no cover - numba is an optional accelerator
    AVAILABLE = False


def default_threads() -> int:
    """Worker threads for the kernel, capped to leave cores for perception."""
    try:
        requested = int(os.environ.get('MPC_ACCEL_THREADS', '4'))
    except ValueError:
        requested = 4
    hardware = numba.config.NUMBA_NUM_THREADS if AVAILABLE else 1
    return max(1, min(requested, hardware))


def enabled() -> bool:
    """True unless numba is missing or explicitly disabled via the environment."""
    if not AVAILABLE:
        return False
    flag = os.environ.get('MPC_DISABLE_ACCEL', '').strip().lower()
    return flag not in ('1', 'true', 'yes', 'on')


if AVAILABLE:

    # These kernels are deliberately serial and non-fastmath: the work is a
    # cache-resident clamp pass where prange fork/join overhead exceeds the
    # arithmetic, and strict FP keeps them bit-identical to the NumPy path.

    @njit(cache=True)
    def _constraints_kernel(controls, min_steer, max_steer, min_accel,
                            max_accel, prev_steer, first_max_change,
                            max_change):
        for s in range(controls.shape[0]):
            previous = prev_steer
            for h in range(controls.shape[1]):
                steer = controls[s, h, 0]
                if steer < min_steer:
                    steer = min_steer
                elif steer > max_steer:
                    steer = max_steer
                allowed_change = first_max_change if h == 0 else max_change
                low = previous - allowed_change
                high = previous + allowed_change
                if steer < low:
                    steer = low
                elif steer > high:
                    steer = high
                controls[s, h, 0] = steer
                previous = steer
                accel = controls[s, h, 1]
                if accel < min_accel:
                    accel = min_accel
                elif accel > max_accel:
                    accel = max_accel
                controls[s, h, 1] = accel

    @njit(cache=True)
    def _sample_constrain_kernel(candidates, mean, std,
                                 min_steer, max_steer, min_accel, max_accel,
                                 prev_steer, first_max_change, max_change):
        for s in range(candidates.shape[0]):
            previous = prev_steer
            for h in range(candidates.shape[1]):
                if s == 0:
                    steer = mean[h, 0]
                    accel = mean[h, 1]
                else:
                    steer = mean[h, 0] + candidates[s, h, 0] * std[h, 0]
                    accel = mean[h, 1] + candidates[s, h, 1] * std[h, 1]
                if steer < min_steer:
                    steer = min_steer
                elif steer > max_steer:
                    steer = max_steer
                allowed_change = first_max_change if h == 0 else max_change
                low = previous - allowed_change
                high = previous + allowed_change
                if steer < low:
                    steer = low
                elif steer > high:
                    steer = high
                candidates[s, h, 0] = steer
                previous = steer
                if accel < min_accel:
                    accel = min_accel
                elif accel > max_accel:
                    accel = max_accel
                candidates[s, h, 1] = accel


def apply_control_constraints(controls, min_steer, max_steer, min_accel,
                              max_accel, prev_steer, first_max_change,
                              max_change):
    """In-place bound + steer-rate clamp; semantics identical to the NumPy
    loop in SamplingMPC._apply_constraints (clip to bounds, then clamp each
    step's steer to +-max_change around the previous clamped steer).

    Returns True when the numba kernel handled it, False if the caller must
    run the NumPy fallback.
    """
    if not AVAILABLE:
        return False
    _constraints_kernel(controls, float(min_steer), float(max_steer),
                        float(min_accel), float(max_accel),
                        float(prev_steer), float(first_max_change),
                        float(max_change))
    return True


def sample_constrained_candidates(noise, mean, std, min_steer, max_steer,
                                  min_accel, max_accel, prev_steer,
                                  first_max_change, max_change):
    """Turn a standard-normal draw into constrained CEM candidates in place.

    Equivalent to ``candidates = mean + noise * std; candidates[0] = mean``
    followed by the constraint clamp, without the temporaries.  ``noise`` is
    consumed and returned as the candidate array.  Returns None when numba is
    unavailable and the caller must build candidates the NumPy way.
    """
    if not AVAILABLE:
        return None
    _sample_constrain_kernel(noise, mean, std, float(min_steer),
                             float(max_steer), float(min_accel),
                             float(max_accel), float(prev_steer),
                             float(first_max_change), float(max_change))
    return noise


if AVAILABLE:

    @njit(cache=True, fastmath=True, inline='always')
    def _wall_clearance(dist, res, ox, oy, height, width, x, y):
        """Bilinear wall clearance at a world point; 0 outside the map.

        Matches obstacles.MapDistanceField.clearance exactly.
        """
        column = (x - ox) / res - 0.5
        row = height - (y - oy) / res - 0.5
        c0 = int(math.floor(column))
        r0 = int(math.floor(row))
        if c0 < 0 or c0 >= width - 1 or r0 < 0 or r0 >= height - 1:
            return 0.0
        fc = column - c0
        fr = row - r0
        return (dist[r0, c0] * (1.0 - fr) * (1.0 - fc)
                + dist[r0, c0 + 1] * (1.0 - fr) * fc
                + dist[r0 + 1, c0] * fr * (1.0 - fc)
                + dist[r0 + 1, c0 + 1] * fr * fc)

    @njit(cache=True, fastmath=True)
    def _rollout_cost_kernel(
            x0, y0, yaw0, v0, controls,
            ref_x, ref_y, ref_yaw, ref_sin_yaw, ref_cos_yaw,
            ref_speed, ff_steer,
            prev_steer, prev_accel,
            dt, wheelbase, understeer,
            steering_retention, steering_response,
            steering_average_retention, steering_average_command,
            initial_applied_steer,
            min_speed, speed_ceiling,
            contour_w, lag_w, heading_w, speed_w,
            ff_w, accel_w, steer_rate_w, accel_rate_w,
            terminal_w, friction_w, grip_hard_w, max_lat,
            disc_front, disc_rear,
            obs_centers, obs_safe, obs_coll, obs_allowed, n_obs,
            obstacle_w, obstacle_coll_w,
            use_wall, wall_dist, wall_res, wall_ox, wall_oy,
            wall_h, wall_w_int, wall_safe, wall_hard, allowed_wall,
            wall_w, wall_coll_w, out):
        samples = controls.shape[0]
        horizon = controls.shape[1]
        two_pi = 2.0 * math.pi
        for s in range(samples):
            x = x0
            y = y0
            yaw = yaw0
            v = v0
            p_steer = prev_steer
            p_accel = prev_accel
            applied_steer = initial_applied_steer
            cost = 0.0
            for h in range(horizon):
                steer = controls[s, h, 0]
                accel = controls[s, h, 1]

                # control-rate cost (delta from the previously applied control)
                d_steer = steer - p_steer
                d_accel = accel - p_accel
                cost += steer_rate_w * d_steer * d_steer
                cost += accel_rate_w * d_accel * d_accel

                # kinematic bicycle step with speed-dependent effective wheelbase
                v_next = v + dt * accel
                if v_next < min_speed:
                    v_next = min_speed
                elif v_next > speed_ceiling:
                    v_next = speed_ceiling
                avg = 0.5 * (v + v_next)
                eff_wb = wheelbase + understeer * avg * avg
                wheel_steer = (steering_average_retention * applied_steer
                               + steering_average_command * steer)
                applied_steer = (steering_retention * applied_steer
                                 + steering_response * steer)
                tan_steer = math.tan(wheel_steer)

                # lateral-accel (friction) penalty uses the pre-step speed
                lat = v * v * tan_steer / (wheelbase + understeer * v * v)
                excess = abs(lat) - max_lat
                if excess > 0.0:
                    cost += friction_w * excess * excess
                    cost += grip_hard_w * excess

                # feedforward-steer and acceleration effort
                d_ff = steer - ff_steer[h]
                cost += ff_w * d_ff * d_ff
                cost += accel_w * accel * accel

                nx = x + dt * avg * math.cos(yaw)
                ny = y + dt * avg * math.sin(yaw)
                nyaw = yaw + dt * avg * tan_steer / eff_wb

                # tracking cost of the resulting (predicted) state
                ex = nx - ref_x[h]
                ey = ny - ref_y[h]
                tyaw = ref_yaw[h]
                s_t = ref_sin_yaw[h]
                c_t = ref_cos_yaw[h]
                contour = -s_t * ex + c_t * ey
                lag = c_t * ex + s_t * ey
                herr = (nyaw - tyaw + math.pi) % two_pi - math.pi
                serr = v_next - ref_speed[h]
                step = (contour_w * contour * contour + lag_w * lag * lag
                        + heading_w * herr * herr + speed_w * serr * serr)
                cost += step
                if h == horizon - 1:
                    cost += terminal_w * step

                # obstacle / wall clearance on the two ego discs
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
                        safe = obs_safe[m]
                        collision = obs_coll[m] - obs_allowed[m]
                        active_distance = max(safe, collision)
                        fdx = fx - cxm
                        fdy = fy - cym
                        rdx = rx - cxm
                        rdy = ry - cym
                        df2 = fdx * fdx + fdy * fdy
                        dr2 = rdx * rdx + rdy * rdy
                        for dist2 in (df2, dr2):
                            if (active_distance > 0.0
                                    and dist2 < active_distance * active_distance):
                                dist = math.sqrt(dist2)
                                intr = safe - dist
                                if intr > 0.0:
                                    cost += obstacle_w * intr * intr
                                ovl = collision - dist
                                if ovl > 0.0:
                                    cost += obstacle_coll_w * ovl

                    if use_wall:
                        for dx, dy in ((fx, fy), (rx, ry)):
                            cl = _wall_clearance(wall_dist, wall_res, wall_ox,
                                                 wall_oy, wall_h, wall_w_int,
                                                 dx, dy)
                            deficit = wall_safe - cl
                            if deficit > 0.0:
                                cost += wall_w * deficit * deficit
                            ovl = wall_hard - cl - allowed_wall[h]
                            if ovl > 0.0:
                                cost += wall_coll_w * ovl

                p_steer = steer
                p_accel = accel
                x = nx
                y = ny
                yaw = nyaw
                v = v_next
            out[s] = cost


@dataclass(frozen=True)
class PreparedCostInputs:
    """Sample-independent arrays shared by CPU and CUDA cost kernels."""

    ref_x: np.ndarray
    ref_y: np.ndarray
    ref_yaw: np.ndarray
    ref_sin_yaw: np.ndarray
    ref_cos_yaw: np.ndarray
    ref_speed: np.ndarray
    ff_steer: np.ndarray
    obs_centers: np.ndarray
    obs_safe: np.ndarray
    obs_coll: np.ndarray
    obs_allowed: np.ndarray
    n_obs: int
    wall_safe: float
    wall_hard: float
    allowed_wall: np.ndarray


def prepare_cost_inputs(config, initial_state, reference, obstacles,
                        use_wall, wall_dist, wall_res, wall_ox, wall_oy,
                        wall_h, wall_w, dtype=np.float64):
    """Build invariant rollout inputs once, then cast for the target kernel.

    The geometry is intentionally computed in the same host-side order for
    both backends so adding an accelerator cannot create a second cost model.
    """
    cfg = config
    reference = np.asarray(reference, dtype=np.float64)
    target = reference[1:]
    ref_x = np.ascontiguousarray(target[:, 0], dtype=dtype)
    ref_y = np.ascontiguousarray(target[:, 1], dtype=dtype)
    ref_yaw = np.ascontiguousarray(target[:, 2], dtype=dtype)
    ref_sin_yaw = np.ascontiguousarray(np.sin(target[:, 2]), dtype=dtype)
    ref_cos_yaw = np.ascontiguousarray(np.cos(target[:, 2]), dtype=dtype)
    ref_speed = np.ascontiguousarray(target[:, 4], dtype=dtype)
    ff_steer = np.ascontiguousarray(np.clip(
        np.arctan((cfg.wheelbase + cfg.understeer_gradient * target[:, 4] ** 2)
                  * target[:, 3]) / cfg.steering_gain,
        cfg.min_steer, cfg.max_steer), dtype=dtype)

    horizon = cfg.horizon
    times = (np.arange(1, horizon + 1, dtype=np.float64) * cfg.dt
             + cfg.latency_compensation)

    direction = np.array([math.cos(initial_state[2]),
                          math.sin(initial_state[2])], dtype=np.float64)
    current_points = np.array([
        initial_state[0:2]
        + (cfg.ego_center_offset + cfg.ego_disc_offset) * direction,
        initial_state[0:2]
        + (cfg.ego_center_offset - cfg.ego_disc_offset) * direction,
    ], dtype=np.float64)

    obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 5)
    n_obs = obstacles.shape[0]
    if n_obs > 0:
        obs_centers = np.ascontiguousarray(
            obstacles[None, :, 0:2]
            + obstacles[None, :, 2:4] * times[:, None, None], dtype=dtype)
        obs_safe = np.ascontiguousarray(
            obstacles[:, 4] + cfg.ego_disc_radius + cfg.obstacle_margin,
            dtype=dtype)
        obs_coll = np.ascontiguousarray(
            obstacles[:, 4] + cfg.ego_disc_radius, dtype=dtype)
        current_distance = np.linalg.norm(
            current_points[:, None, :] - obstacles[None, :, 0:2],
            axis=-1).min(axis=0)
        obs_allowed = np.ascontiguousarray(
            np.maximum(obstacles[:, 4] + cfg.ego_disc_radius
                       - current_distance, 0.0), dtype=dtype)
    else:
        obs_centers = np.zeros((horizon, 1, 2), dtype=dtype)
        obs_safe = np.zeros(1, dtype=dtype)
        obs_coll = np.zeros(1, dtype=dtype)
        obs_allowed = np.zeros(1, dtype=dtype)

    wall_safe = cfg.ego_disc_radius + cfg.wall_margin
    wall_hard = cfg.ego_disc_radius + cfg.wall_hard_margin
    if use_wall:
        current_clearance = float(_np_wall_clearance(
            wall_dist, wall_res, wall_ox, wall_oy, wall_h, wall_w,
            current_points[:, 0], current_points[:, 1]).min())
        allowed_wall = np.maximum(
            max(wall_hard - current_clearance, 0.0) - 0.3 * times, 0.0)
    else:
        allowed_wall = np.zeros(horizon, dtype=np.float64)

    return PreparedCostInputs(
        ref_x=ref_x,
        ref_y=ref_y,
        ref_yaw=ref_yaw,
        ref_sin_yaw=ref_sin_yaw,
        ref_cos_yaw=ref_cos_yaw,
        ref_speed=ref_speed,
        ff_steer=ff_steer,
        obs_centers=obs_centers,
        obs_safe=obs_safe,
        obs_coll=obs_coll,
        obs_allowed=obs_allowed,
        n_obs=int(n_obs),
        wall_safe=float(wall_safe),
        wall_hard=float(wall_hard),
        allowed_wall=np.ascontiguousarray(allowed_wall, dtype=dtype),
    )


class CostBackend:
    """Prepares the sample-independent arrays and calls the fused kernel.

    Instances cache the wall distance field (as a contiguous float64 array) and
    reusable scratch so the per-solve Python overhead stays tiny.
    """

    name = 'cpu'
    tolerance = 1e-4

    def __init__(self, config):
        self.config = config
        self._lateral_accel_limit = float(config.max_lateral_accel)
        self._wall_dist = np.zeros((1, 1), dtype=np.float64)
        self._wall_res = 1.0
        self._wall_ox = 0.0
        self._wall_oy = 0.0
        self._wall_h = 1
        self._wall_w = 1
        self._use_wall = False
        if AVAILABLE:
            numba.set_num_threads(default_threads())

    def set_lateral_accel_limit(self, limit):
        value = float(limit)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError('lateral acceleration limit must be positive')
        self._lateral_accel_limit = value

    def set_wall_field(self, wall_field):
        if wall_field is None:
            self._use_wall = False
            return
        self._wall_dist = np.ascontiguousarray(wall_field._distance, dtype=np.float64)
        self._wall_res = float(wall_field.resolution)
        self._wall_ox = float(wall_field.origin_x)
        self._wall_oy = float(wall_field.origin_y)
        self._wall_h = int(wall_field.height)
        self._wall_w = int(wall_field.width)
        self._use_wall = True

    def prepare(self, initial_state, reference, obstacles):
        """Prepare inputs shared by every population in one MPC solve."""
        return prepare_cost_inputs(
            self.config, initial_state, reference, obstacles,
            self._use_wall, self._wall_dist, self._wall_res,
            self._wall_ox, self._wall_oy, self._wall_h, self._wall_w)

    def costs(self, initial_state, controls, reference, previous_control,
              obstacles, speed_ceiling, initial_applied_steer=None):
        """Return per-sample costs using the numba kernel.

        Mirrors SamplingMPC._cost exactly; sample-independent terms are built
        here in NumPy and the per-sample rollout/accumulation runs in the
        kernel.  ``obstacles`` are the (M, 5) rows the solver received.
        """
        prepared = self.prepare(initial_state, reference, obstacles)
        return self.costs_prepared(
            initial_state, controls, previous_control, speed_ceiling, prepared,
            initial_applied_steer)

    def costs_prepared(self, initial_state, controls, previous_control,
                       speed_ceiling, prepared, initial_applied_steer=None):
        """Score a CEM population using inputs prepared once for this solve."""
        cfg = self.config
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        if initial_applied_steer is None:
            initial_applied_steer = (
                cfg.steering_gain * float(previous_control[0]))
        steering_coefficients = cfg.steering_step_coefficients(cfg.dt)

        out = np.empty(controls.shape[0], dtype=np.float64)
        _rollout_cost_kernel(
            float(initial_state[0]), float(initial_state[1]),
            float(initial_state[2]), float(initial_state[3]), controls,
            prepared.ref_x, prepared.ref_y, prepared.ref_yaw,
            prepared.ref_sin_yaw, prepared.ref_cos_yaw,
            prepared.ref_speed, prepared.ff_steer,
            float(previous_control[0]), float(previous_control[1]),
            cfg.dt, cfg.wheelbase, cfg.understeer_gradient,
            steering_coefficients[0], steering_coefficients[1],
            steering_coefficients[2], steering_coefficients[3],
            float(initial_applied_steer), cfg.min_speed,
            float(speed_ceiling),
            cfg.contour_weight, cfg.lag_weight, cfg.heading_weight, cfg.speed_weight,
            cfg.steer_feedforward_weight, cfg.accel_weight, cfg.steer_rate_weight,
            cfg.accel_rate_weight, cfg.terminal_weight, cfg.friction_weight,
            cfg.grip_hard_weight, self._lateral_accel_limit,
            cfg.ego_center_offset + cfg.ego_disc_offset,
            cfg.ego_center_offset - cfg.ego_disc_offset,
            prepared.obs_centers, prepared.obs_safe, prepared.obs_coll,
            prepared.obs_allowed, prepared.n_obs,
            cfg.obstacle_weight, cfg.obstacle_collision_weight,
            bool(self._use_wall), self._wall_dist, self._wall_res,
            self._wall_ox, self._wall_oy, self._wall_h, self._wall_w,
            prepared.wall_safe, prepared.wall_hard, prepared.allowed_wall,
            cfg.wall_weight, cfg.wall_collision_weight, out)
        return out


def _np_wall_clearance(dist, res, ox, oy, height, width, x, y):
    """NumPy twin of the in-kernel bilinear clearance (for current_points)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    column = (x - ox) / res - 0.5
    row = height - (y - oy) / res - 0.5
    c0 = np.floor(column).astype(int)
    r0 = np.floor(row).astype(int)
    fc = column - c0
    fr = row - r0
    valid = (c0 >= 0) & (c0 < width - 1) & (r0 >= 0) & (r0 < height - 1)
    c0c = np.clip(c0, 0, width - 2)
    r0c = np.clip(r0, 0, height - 2)
    interpolated = (
        dist[r0c, c0c] * (1 - fr) * (1 - fc)
        + dist[r0c, c0c + 1] * (1 - fr) * fc
        + dist[r0c + 1, c0c] * fr * (1 - fc)
        + dist[r0c + 1, c0c + 1] * fr * fc
    )
    return np.where(valid, interpolated, 0.0)
