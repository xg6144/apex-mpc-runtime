"""ROS-independent IMU processing for the MPC controller.

The VESC IMU is noisy (motor/gear vibration) and its gyro carries a slowly
drifting bias, so every consumer here works on low-pass-filtered,
bias-corrected body-frame signals.  Position stays the localization stack's
job: the IMU only provides yaw rate, longitudinal acceleration for speed
support, and short-horizon pose propagation between sensor timestamps.

``ImuEstimator``  filtering, stationary bias estimation, complementary
                  speed fusion, and pose advancement over sub-100 ms windows.
``SafetyMonitor`` spin/oversteer and collision detection independent of the
                  optimizer.
``AdaptiveGripLimiter`` estimates the currently usable lateral acceleration
                  from commanded steering versus measured yaw response.
``GripObserver``  measured lateral-acceleration statistics while cornering,
                  to check max_lateral_accel against today's grip.
"""

import math
import threading
from collections import deque
from typing import Optional, Tuple

import numpy as np

_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def parse_axis(spec: str) -> Tuple[int, float]:
    """Map an axis spec like 'z' or '-y' to a (vector index, sign) pair.

    Used to express IMU mounting: on this car the VESC IMU is rotated ~90
    degrees about z, so vehicle yaw rate is '+z', longitudinal accel '-y',
    and lateral accel 'x' (identified from bag mpc_20260722 data).
    """
    text = str(spec).strip().lower()
    sign = 1.0
    if text.startswith('-'):
        sign = -1.0
        text = text[1:]
    elif text.startswith('+'):
        text = text[1:]
    if text not in _AXIS_INDEX:
        raise ValueError("IMU axis must be one of x, y, z with an optional "
                         "sign, got {!r}".format(spec))
    return _AXIS_INDEX[text], sign


class ImuEstimator:
    """Filter body-frame IMU samples and fuse them with odometry speed.

    Inputs are SI body-frame values: yaw rate (rad/s, counter-clockwise
    positive), longitudinal acceleration (m/s^2, forward positive), and
    lateral acceleration (m/s^2, leftward positive).  Axis mapping and unit
    conversion from the raw driver message happen in the ROS layer.
    """

    def __init__(self, lpf_cutoff: float = 8.0,
                 stationary_speed: float = 0.05,
                 stationary_hold: float = 0.4,
                 speed_fusion_time_constant: float = 0.4,
                 speed_hold_time: float = 0.15,
                 buffer_seconds: float = 1.0,
                 max_sample_gap: float = 0.2,
                 bias_window_samples: int = 128):
        if lpf_cutoff <= 0.0:
            raise ValueError('imu lpf_cutoff must be positive')
        if speed_fusion_time_constant <= 0.0:
            raise ValueError('speed_fusion_time_constant must be positive')
        self._lpf_cutoff = float(lpf_cutoff)
        self._stationary_speed = float(stationary_speed)
        self._stationary_hold = float(stationary_hold)
        self._speed_tau = float(speed_fusion_time_constant)
        self._speed_hold = abs(float(speed_hold_time))
        self._buffer_seconds = float(buffer_seconds)
        self._max_sample_gap = float(max_sample_gap)
        if int(bias_window_samples) < 1:
            raise ValueError('bias_window_samples must be at least one')
        self._bias_window_samples = int(bias_window_samples)

        # Samples arrive on the node's sensor callback thread while the
        # control timer reads/integrates on another; the lock protects the
        # compound buffer/filter updates (deque iteration during append
        # raises RuntimeError).
        self._lock = threading.Lock()
        self._last_time = None  # type: Optional[float]
        self._filtered = np.zeros(3, dtype=float)  # yaw_rate, ax, ay (biased)
        self._bias = np.zeros(3, dtype=float)
        self._bias_samples = 0
        # Raw samples seen while the car is held still; the bias is their
        # per-axis median.  ~2.5 s at the 50 Hz VESC IMU rate.
        self._standstill = deque(maxlen=self._bias_window_samples)
        self._buffer = deque()  # (t, yaw_rate, ax) bias-corrected, unfiltered
        self._stationary_since = None  # type: Optional[float]
        self._stationary = False
        self._fused_speed = None  # type: Optional[float]
        self._last_odom_speed = None  # type: Optional[float]
        self._last_odom_time = None  # type: Optional[float]

    @property
    def stationary(self) -> bool:
        return self._stationary

    @property
    def has_bias_estimate(self) -> bool:
        return self._bias_samples > 0

    def add_sample(self, timestamp: float, yaw_rate: float,
                   long_accel: float, lat_accel: float) -> None:
        """Ingest one SI body-frame IMU sample stamped by the driver."""
        raw = np.array([yaw_rate, long_accel, lat_accel], dtype=float)
        if not np.all(np.isfinite(raw)):
            return
        with self._lock:
            self._add_sample_locked(float(timestamp), raw)

    def _add_sample_locked(self, timestamp: float, raw: np.ndarray) -> None:
        dt = None
        if self._last_time is not None:
            dt = timestamp - self._last_time
            if dt <= 0.0:
                return  # out-of-order or duplicate sample
        self._last_time = timestamp

        if dt is not None and dt <= self._max_sample_gap:
            alpha = dt / (dt + 1.0 / (2.0 * math.pi * self._lpf_cutoff))
            self._filtered += alpha * (raw - self._filtered)
            # Acceleration integration is open-loop between odometry updates.
            # Keep it only while ERPM odometry is present to bound drift, but
            # never clamp it to ERPM: riding through a brief wheel slip/lock is
            # the reason this estimate exists.  This mirrors GHOST's measured
            # 0.15 s hold window (seven missed 50 Hz odometry samples).
            fresh_odom = (
                self._last_odom_time is not None
                and timestamp - self._last_odom_time <= self._speed_hold
            )
            if (self._fused_speed is not None and not self._stationary
                    and fresh_odom):
                self._fused_speed += (raw[1] - self._bias[1]) * dt
        else:
            self._filtered = raw.copy()

        if self._stationary:
            # The car is held still: the raw sample IS the bias (the accel
            # channels also absorb the gravity component from mounting tilt).
            # Take the median of a standstill window rather than tracking the
            # mean.  Chassis vibration makes this distribution strongly
            # skewed - the longitudinal channel of bag mpc_20260824_194608
            # read p50 -1.04 m/s^2 against a mean of -3.70 with std 4.79 -
            # and a mean-tracking filter converges to the outliers, not to
            # the offset.  The ~1.8 m/s^2 that was left over integrated
            # straight into the fused speed and held it 0.7 m/s above the
            # wheels for the whole of that run's launch.
            self._standstill.append(raw.copy())
            self._bias = np.median(np.stack(self._standstill), axis=0)
            self._bias_samples += 1

        self._buffer.append((timestamp, raw[0] - self._bias[0],
                             raw[1] - self._bias[1], raw[2] - self._bias[2]))
        cutoff = timestamp - self._buffer_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def update_speed(self, timestamp: float, odom_speed: float) -> None:
        """Ingest one odometry speed measurement (ERPM-based on this car)."""
        odom_speed = float(odom_speed)
        if not math.isfinite(odom_speed):
            return
        with self._lock:
            self._update_speed_locked(float(timestamp), odom_speed)

    def _update_speed_locked(self, timestamp: float,
                             odom_speed: float) -> None:
        self._last_odom_speed = odom_speed
        self._last_odom_time = timestamp

        if abs(odom_speed) < self._stationary_speed:
            if self._stationary_since is None:
                self._stationary_since = timestamp
            self._stationary = (
                timestamp - self._stationary_since >= self._stationary_hold)
        else:
            self._stationary_since = None
            self._stationary = False

        if self._fused_speed is None or self._stationary:
            self._fused_speed = odom_speed
            return
        # Complementary correction: the ~0.4 s time constant trusts the IMU
        # integral through brief wheel slip/lock while ERPM still anchors the
        # steady state faster than integration drift accumulates.
        gain = 1.0 - math.exp(-0.02 / self._speed_tau)
        if len(self._buffer) >= 2:
            spacing = (self._buffer[-1][0] - self._buffer[0][0]) / (
                len(self._buffer) - 1)
            if 0.0 < spacing <= self._max_sample_gap:
                gain = 1.0 - math.exp(-spacing / self._speed_tau)
        self._fused_speed += gain * (odom_speed - self._fused_speed)

    def yaw_rate(self) -> float:
        """Low-pass-filtered, bias-corrected yaw rate (rad/s)."""
        return float(self._filtered[0] - self._bias[0])

    def long_accel(self) -> float:
        """Low-pass-filtered, bias-corrected longitudinal accel (m/s^2)."""
        return float(self._filtered[1] - self._bias[1])

    def lat_accel(self) -> float:
        """Low-pass-filtered, bias-corrected lateral accel (m/s^2)."""
        return float(self._filtered[2] - self._bias[2])

    def raw_yaw_rate(self) -> float:
        """Latest bias-corrected yaw rate without filtering (rad/s)."""
        if not self._buffer:
            return 0.0
        return float(self._buffer[-1][1])

    def raw_long_accel(self) -> float:
        """Latest bias-corrected longitudinal accel without filtering."""
        if not self._buffer:
            return 0.0
        return float(self._buffer[-1][2])

    def raw_lat_accel(self) -> float:
        """Latest bias-corrected lateral accel without filtering."""
        if not self._buffer:
            return 0.0
        return float(self._buffer[-1][3])

    def fused_speed(self) -> Optional[float]:
        """Odometry speed with IMU support through slip; None before odom."""
        return self._fused_speed

    def last_sample_time(self) -> Optional[float]:
        return self._last_time

    def advance_pose(self, x: float, y: float, yaw: float, speed: float,
                     t_from: float, t_to: float):
        """Propagate a planar pose between two nearby timestamps.

        Integrates the buffered bias-corrected yaw rate over [t_from, t_to]
        (either order) and moves the position along the mean heading.  This
        aligns the localization pose with the scan timestamp; it is not a
        substitute for localization over long windows.
        """
        span = float(t_to) - float(t_from)
        if span == 0.0 or not self._buffer:
            return x, y, yaw
        lo, hi = (t_from, t_to) if span > 0.0 else (t_to, t_from)
        with self._lock:
            delta_yaw = self._integrate_yaw_rate(lo, hi)
        if span < 0.0:
            delta_yaw = -delta_yaw
        mid_yaw = yaw + 0.5 * delta_yaw
        return (x + speed * math.cos(mid_yaw) * span,
                y + speed * math.sin(mid_yaw) * span,
                yaw + delta_yaw)

    def _integrate_yaw_rate(self, lo: float, hi: float) -> float:
        """Trapezoidal integral of buffered yaw rate over [lo, hi]."""
        times = np.array([s[0] for s in self._buffer])
        rates = np.array([s[1] for s in self._buffer])
        # Clamp the window into the buffered range (nearest-sample hold at
        # the edges keeps sub-sample extrapolation bounded).
        grid = np.unique(np.clip(np.concatenate((
            [lo, hi], times[(times > lo) & (times < hi)])), lo, hi))
        if grid.size < 2:
            return float(np.interp(0.5 * (lo + hi), times, rates) * (hi - lo))
        values = np.interp(grid, times, rates)
        return float(np.trapz(values, grid))


class SafetyMonitor:
    """IMU-based stop decisions independent of the MPC optimizer.

    Spin/oversteer: the measured yaw rate disagreeing with the commanded
    kinematic yaw rate (v * tan(wheel steer) / effective wheelbase) for longer
    than a hold time.  The commanded steer runs through a first-order lag
    (servo + chassis response): without it, replaying bag mpc_20260723_171549
    shows corner-entry residuals of 3.5 rad/s in normal driving, while with a
    0.15 s lag the same bag never exceeds 2.3 rad/s -- the 2.5 default then
    separates real spins from aggressive cornering.  It caught the real
    obstacle-contact skid in bag mpc_20260723_175903 (residual -3.15 rad/s)
    1.4 s before the driver reacted.  Collision: an unfiltered acceleration
    spike beyond driving physics, latched for a hold time.  Clean fast laps
    peak at 13.5-18.0 m/s^2 while the measured obstacle contact spiked to
    19.9-20.8, so the 19.0 default sits between.
    """

    def __init__(self, wheelbase: float, understeer_gradient: float,
                 spin_yaw_residual: float = 2.5,
                 spin_hold_time: float = 0.2,
                 spin_min_speed: float = 1.5,
                 steer_lag_time_constant: float = 0.15,
                 collision_accel: float = 19.0,
                 collision_hold_time: float = 1.0,
                 steering_gain: float = 1.0):
        if spin_yaw_residual <= 0.0 or collision_accel <= 0.0:
            raise ValueError('safety thresholds must be positive')
        if steer_lag_time_constant < 0.0:
            raise ValueError('steer_lag_time_constant cannot be negative')
        if steering_gain <= 0.0:
            raise ValueError('steering_gain must be positive')
        self._wheelbase = float(wheelbase)
        self._understeer_gradient = float(understeer_gradient)
        self._steering_gain = float(steering_gain)
        self._spin_yaw_residual = float(spin_yaw_residual)
        self._spin_hold_time = float(spin_hold_time)
        self._spin_min_speed = float(spin_min_speed)
        self._steer_lag_tau = float(steer_lag_time_constant)
        self._collision_accel = float(collision_accel)
        self._collision_hold_time = float(collision_hold_time)
        self._spin_since = None  # type: Optional[float]
        self._spin_active = False
        self._collision_until = None  # type: Optional[float]
        self._last_update = None  # type: Optional[float]
        self._lagged_steer = 0.0

    def update(self, timestamp: float, measured_yaw_rate: float,
               commanded_steer: float, speed: float,
               raw_long_accel: float, raw_lat_accel: float) -> Optional[str]:
        """Return a stop reason for this instant, or None to keep driving."""
        timestamp = float(timestamp)
        target_steer = self._steering_gain * float(commanded_steer)
        dt = None
        if self._last_update is not None:
            dt = timestamp - self._last_update
            if dt > 0.5:
                # Monitoring paused (manual takeover, editor stop): stale
                # spin accumulation and the lagged steer state are invalid.
                self._spin_since = None
                self._spin_active = False
                self._lagged_steer = target_steer
                dt = None
        self._last_update = timestamp
        if dt is not None and dt > 0.0 and self._steer_lag_tau > 0.0:
            response = -math.expm1(-dt / self._steer_lag_tau)
            self._lagged_steer += response * (
                target_steer - self._lagged_steer)
        elif dt is None or self._steer_lag_tau <= 0.0:
            self._lagged_steer = target_steer

        if max(abs(raw_long_accel), abs(raw_lat_accel)) > self._collision_accel:
            self._collision_until = timestamp + self._collision_hold_time
        if self._collision_until is not None:
            if timestamp < self._collision_until:
                return 'collision: |accel| {:.1f} m/s^2 exceeded {:.1f}'.format(
                    max(abs(raw_long_accel), abs(raw_lat_accel)),
                    self._collision_accel)
            self._collision_until = None

        residual = self._yaw_residual(
            measured_yaw_rate, self._lagged_steer, speed)
        # Hysteresis: trigger above the threshold, release below 60% of it.
        threshold = self._spin_yaw_residual
        if self._spin_active:
            threshold *= 0.6
        if speed > self._spin_min_speed and abs(residual) > threshold:
            if self._spin_since is None:
                self._spin_since = timestamp
            if (self._spin_active
                    or timestamp - self._spin_since >= self._spin_hold_time):
                self._spin_active = True
                return ('spin: yaw rate off command by {:+.2f} rad/s'
                        .format(residual))
        else:
            self._spin_since = None
            self._spin_active = False
        return None

    def reset(self) -> None:
        self._spin_since = None
        self._spin_active = False
        self._collision_until = None
        self._last_update = None

    def _yaw_residual(self, measured_yaw_rate: float, steer: float,
                      speed: float) -> float:
        effective = (self._wheelbase
                     + self._understeer_gradient * speed * speed)
        expected = speed * math.tan(steer) / effective
        return float(measured_yaw_rate - expected)


class AdaptiveGripLimiter:
    """Estimate a conservative live lateral-acceleration limit.

    The commanded steering is passed through the same calibrated rack gain
    and first-order actuator lag as the MPC model.  While the car is turning,
    the measured yaw response is compared with that expected response.  A
    sustained shortfall (understeer) or excess/opposite response (incipient
    rotation) quickly lowers the limit.  Raising the limit is deliberately
    slow and only happens while the tires are actually exercising the current
    limit, so straight-line driving and IMU noise can never invent more grip.

    This class only returns a number.  It does not write the raceline and it
    does not command the VESC; the MPC node applies the number to its runtime
    reference and rollout costs.
    """

    def __init__(self, wheelbase: float, understeer_gradient: float,
                 steering_gain: float = 1.0,
                 steer_lag_time_constant: float = 0.15,
                 initial_limit: float = 5.0,
                 minimum_limit: float = 3.5,
                 maximum_limit: float = 6.0,
                 understeer_ratio: float = 0.82,
                 oversteer_ratio: float = 1.25,
                 slip_hold_time: float = 0.08,
                 attack_time_constant: float = 0.08,
                 release_time_constant: float = 8.0,
                 safety_factor: float = 0.90,
                 min_speed: float = 1.5,
                 min_expected_yaw_rate: float = 0.5):
        if wheelbase <= 0.0 or steering_gain <= 0.0:
            raise ValueError('grip vehicle parameters must be positive')
        if understeer_gradient < 0.0 or steer_lag_time_constant < 0.0:
            raise ValueError('grip model time/gradient cannot be negative')
        if not 0.0 < minimum_limit <= initial_limit <= maximum_limit:
            raise ValueError('grip limits must satisfy 0 < min <= initial <= max')
        if not 0.0 < understeer_ratio < 1.0 < oversteer_ratio:
            raise ValueError('grip response ratios must straddle one')
        if slip_hold_time < 0.0:
            raise ValueError('grip slip_hold_time cannot be negative')
        if attack_time_constant <= 0.0 or release_time_constant <= 0.0:
            raise ValueError('grip attack/release times must be positive')
        if not 0.0 < safety_factor <= 1.0:
            raise ValueError('grip safety_factor must be in (0, 1]')
        if min_speed < 0.0 or min_expected_yaw_rate <= 0.0:
            raise ValueError('grip observation thresholds are invalid')

        self._wheelbase = float(wheelbase)
        self._understeer_gradient = float(understeer_gradient)
        self._steering_gain = float(steering_gain)
        self._steer_lag_tau = float(steer_lag_time_constant)
        self._minimum_limit = float(minimum_limit)
        self._maximum_limit = float(maximum_limit)
        self._understeer_ratio = float(understeer_ratio)
        self._oversteer_ratio = float(oversteer_ratio)
        self._slip_hold_time = float(slip_hold_time)
        self._attack_tau = float(attack_time_constant)
        self._release_tau = float(release_time_constant)
        self._safety_factor = float(safety_factor)
        self._min_speed = float(min_speed)
        self._min_expected_yaw_rate = float(min_expected_yaw_rate)

        self._current_limit = float(initial_limit)
        self._last_update = None  # type: Optional[float]
        self._lagged_steer = 0.0
        self._slip_since = None  # type: Optional[float]
        self._slip_active = False
        self._response_ratio = float('nan')
        self._observed_lateral_accel = 0.0

    @property
    def current_limit(self) -> float:
        return self._current_limit

    @property
    def response_ratio(self) -> float:
        return self._response_ratio

    @property
    def observed_lateral_accel(self) -> float:
        return self._observed_lateral_accel

    @property
    def slip_active(self) -> bool:
        return self._slip_active

    def update(self, timestamp: float, measured_yaw_rate: float,
               commanded_steer: float, speed: float) -> float:
        """Update from one control instant and return the live grip limit."""
        if not all(math.isfinite(float(value)) for value in (
                timestamp, measured_yaw_rate, commanded_steer, speed)):
            return self._current_limit

        timestamp = float(timestamp)
        speed = abs(float(speed))
        target_steer = self._steering_gain * float(commanded_steer)
        dt = None
        if self._last_update is not None:
            dt = timestamp - self._last_update
            if dt <= 0.0:
                return self._current_limit
            if dt > 0.5:
                # Keep the learned tire limit across a manual/editor pause,
                # but discard transient actuator/slip state.
                dt = None
                self._slip_since = None
                self._slip_active = False
        self._last_update = timestamp

        if dt is None or self._steer_lag_tau <= 0.0:
            self._lagged_steer = target_steer
        else:
            response = -math.expm1(-dt / self._steer_lag_tau)
            self._lagged_steer += response * (
                target_steer - self._lagged_steer)

        effective = (self._wheelbase
                     + self._understeer_gradient * speed * speed)
        expected_yaw_rate = speed * math.tan(self._lagged_steer) / effective
        self._observed_lateral_accel = abs(
            speed * float(measured_yaw_rate))
        observable = (
            dt is not None
            and speed >= self._min_speed
            and abs(expected_yaw_rate) >= self._min_expected_yaw_rate)
        if not observable:
            self._response_ratio = float('nan')
            self._slip_since = None
            self._slip_active = False
            return self._current_limit

        measured = float(measured_yaw_rate)
        if measured * expected_yaw_rate <= 0.0:
            ratio = 0.0
        else:
            ratio = abs(measured / expected_yaw_rate)
        self._response_ratio = ratio
        slipping = (ratio < self._understeer_ratio
                    or ratio > self._oversteer_ratio)

        if slipping:
            if self._slip_since is None:
                self._slip_since = timestamp
            held = timestamp - self._slip_since >= self._slip_hold_time
            if held:
                self._slip_active = True
                if ratio < self._understeer_ratio:
                    # v*measured_yaw_rate is the lateral acceleration the
                    # tires actually produced at the failed steering demand.
                    target = (self._safety_factor
                              * self._observed_lateral_accel)
                else:
                    # Excess yaw is not proof of extra cornering grip; it is
                    # often the rear rotating.  Reduce relative to severity.
                    target = (self._safety_factor * self._current_limit
                              * self._oversteer_ratio / max(ratio, 1e-6))
                target = max(
                    self._minimum_limit,
                    min(float(target), self._current_limit))
                alpha = -math.expm1(-dt / self._attack_tau)
                self._current_limit += alpha * (
                    target - self._current_limit)
        else:
            self._slip_since = None
            self._slip_active = False
            # A matched response can raise the estimate only if this corner
            # is already using most of the available budget.  This makes the
            # recovery evidence-based and immune to straight-line samples.
            if (self._observed_lateral_accel
                    >= 0.8 * self._current_limit):
                proven = min(
                    self._maximum_limit,
                    self._observed_lateral_accel / self._safety_factor)
                target = max(self._current_limit, proven)
                alpha = -math.expm1(-dt / self._release_tau)
                self._current_limit += alpha * (
                    target - self._current_limit)

        self._current_limit = max(
            self._minimum_limit,
            min(self._current_limit, self._maximum_limit))
        return self._current_limit


class GripObserver:
    """Collect measured |lateral accel| while cornering for periodic reports.

    Report-only by design: max_lateral_accel stays a fixed parameter, and the
    log lets the operator verify it against today's grip every few seconds.
    """

    def __init__(self, min_speed: float = 1.5, min_yaw_rate: float = 0.8):
        self._min_speed = float(min_speed)
        self._min_yaw_rate = float(min_yaw_rate)
        self._samples = []

    def add(self, lat_accel: float, yaw_rate: float, speed: float) -> None:
        if speed >= self._min_speed and abs(yaw_rate) >= self._min_yaw_rate:
            self._samples.append(abs(float(lat_accel)))

    def report(self) -> Optional[str]:
        """Summarize and clear the window; None when barely cornering."""
        if len(self._samples) < 10:
            self._samples = []
            return None
        values = np.array(self._samples)
        self._samples = []
        return 'lat |a| p50 {:.1f} p95 {:.1f} max {:.1f} m/s^2'.format(
            float(np.percentile(values, 50)),
            float(np.percentile(values, 95)),
            float(values.max()))
