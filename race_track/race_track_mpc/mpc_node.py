"""ROS 2 node wrapping the F1TENTH sampling MPC with obstacle avoidance."""

import bisect
import dataclasses
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMessage
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float64MultiArray, String as StringMessage
from visualization_msgs.msg import Marker, MarkerArray

from .imu_fusion import (
    AdaptiveGripLimiter, GripObserver, ImuEstimator, SafetyMonitor, parse_axis)
from .mpc_core import (
    AvoidancePlanner, MPCConfig, Raceline, SamplingMPC, VehicleState)
from .mode_gate import ManualOverrideGate
from .obstacles import (
    MapDistanceField, ObstacleTracker, extrapolate_obstacles)

try:
    import pitwall
except ImportError:  # telemetry is optional; the car runs without the package
    pitwall = None


class OdometryHistory:
    """Recent localization poses, queryable at an arbitrary timestamp.

    The obstacle tracker needs the ego pose *at the scan's timestamp*.  Pairing
    a scan with whatever odometry arrived last rotates and translates every
    return by the pose error accumulated over that mismatch: the wall a beam
    actually hit is projected somewhere it is not, the real wall reads as free
    space, and the displaced returns cluster into an obstacle that was never
    there.  The error scales with speed and yaw rate, so it is worst exactly
    where the car is turning hardest and the corridor is narrowest.

    Keeping a second of history and interpolating removes the mismatch without
    needing an IMU, so it also holds when IMU fusion is off or its samples have
    gone stale - which is when the unaligned fallback used to take over.

    Timestamps are message header stamps throughout.  Never mix in an arrival
    or wall-clock time: subtracting one clock from another produces a
    plausible-looking offset that makes the alignment worse than none.
    """

    # How pose_at answered, most trustworthy first.  Reported in telemetry so a
    # regression in stamping shows up as a shift in the mix rather than as
    # phantom obstacles nobody can explain.
    INTERPOLATED = 'interpolated'
    EXTRAPOLATED = 'extrapolated'
    HELD = 'held'

    def __init__(self, window_seconds: float = 1.0):
        self._window = max(float(window_seconds), 0.05)
        self._stamps = deque()
        self._poses = deque()   # (x, y, yaw, speed, yaw_rate)
        self._lock = threading.Lock()

    def add(self, stamp: float, x: float, y: float, yaw: float,
            speed: float, yaw_rate: float) -> None:
        stamp = float(stamp)
        with self._lock:
            # A stamp that goes backwards means the source restarted (bag
            # replay, localization reset); the old poses describe a different
            # run, so drop them rather than interpolate across the seam.
            if self._stamps and stamp < self._stamps[-1]:
                self._stamps.clear()
                self._poses.clear()
            self._stamps.append(stamp)
            self._poses.append((float(x), float(y), float(yaw),
                                float(speed), float(yaw_rate)))
            horizon = stamp - self._window
            while len(self._stamps) > 2 and self._stamps[0] < horizon:
                self._stamps.popleft()
                self._poses.popleft()

    def newest_stamp(self):
        with self._lock:
            return self._stamps[-1] if self._stamps else None

    def slip_observation(self, window: float):
        """Wheel speed minus localization speed over one shared window.

        ERPM reads the wheel, so it cannot see the tyre spinning; the
        localization poses can.  Averaging both over the *same* window is
        what makes this usable while accelerating: each average lags the
        instantaneous speed by half a window, and taking the difference
        cancels that lag, leaving the slip itself rather than a stale speed.

        Path length is summed over the buffered samples rather than taken as
        the end-to-end chord, so a corner does not read as slip.  Returns
        None until the buffer spans at least half the window.
        """
        window = max(float(window), 1e-3)
        with self._lock:
            if len(self._stamps) < 3:
                return None
            stamps = list(self._stamps)
            poses = list(self._poses)
        newest = stamps[-1]
        start = 0
        for index in range(len(stamps) - 1, -1, -1):
            if newest - stamps[index] > window:
                start = index
                break
        span = newest - stamps[start]
        if span < 0.5 * window:
            return None
        segment = poses[start:]
        if len(segment) < 2:
            return None
        distance = 0.0
        for (x0, y0, *_), (x1, y1, *_) in zip(segment, segment[1:]):
            distance += math.hypot(x1 - x0, y1 - y0)
        wheel = sum(pose[3] for pose in segment) / len(segment)
        return wheel - distance / span

    def pose_at(self, stamp: float, max_extrapolation: float = 0.05):
        """Return (x, y, yaw, how) at `stamp`, or None when there is nothing.

        Within the buffered span the pose is interpolated between the two
        bracketing samples.  Past the newest sample it is dead-reckoned with a
        constant-turn-rate model for at most `max_extrapolation` seconds -
        beyond that the newest pose is held, because a longer dead-reckon is
        fiction and a held pose at least fails in a direction we can reason
        about.
        """
        stamp = float(stamp)
        with self._lock:
            if not self._stamps:
                return None
            stamps = list(self._stamps)
            poses = list(self._poses)

        if stamp <= stamps[0]:
            x, y, yaw = poses[0][:3]
            # Equality is an exact hit, not a hold - see the note below.
            return (x, y, yaw,
                    self.INTERPOLATED if stamp == stamps[0] else self.HELD)
        if stamp >= stamps[-1]:
            x, y, yaw, speed, yaw_rate = poses[-1]
            span = stamp - stamps[-1]
            if span <= 0.0:
                # Landed exactly on a sample.  That is an exact answer, not a
                # dead-reckon, and calling it extrapolated would misreport the
                # pairing mix for every publisher that stamps a scan and its
                # odometry from one clock read.
                return x, y, yaw, self.INTERPOLATED
            if span > max(float(max_extrapolation), 0.0):
                return x, y, yaw, self.HELD
            delta_yaw = yaw_rate * span
            mid_yaw = yaw + 0.5 * delta_yaw
            return (x + speed * math.cos(mid_yaw) * span,
                    y + speed * math.sin(mid_yaw) * span,
                    yaw + delta_yaw, self.EXTRAPOLATED)

        index = bisect.bisect_left(stamps, stamp)
        t0, t1 = stamps[index - 1], stamps[index]
        p0, p1 = poses[index - 1], poses[index]
        span = t1 - t0
        alpha = 0.0 if span <= 0.0 else (stamp - t0) / span
        # Yaw is interpolated on the shortest arc so a wrap at +-pi does not
        # spin the pose the long way round.
        delta_yaw = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
        return (p0[0] + alpha * (p1[0] - p0[0]),
                p0[1] + alpha * (p1[1] - p0[1]),
                p0[2] + alpha * delta_yaw,
                self.INTERPOLATED)


class MPCControllerNode(Node):
    """Track a closed raceline while avoiding scan-detected obstacles."""

    # Sizing/structural settings the backends allocate buffers from (or the
    # control thread reads exactly once): a live update cannot take effect,
    # so reject it instead of silently ignoring it.
    _RESTART_ONLY_PARAMETERS = frozenset({
        'horizon', 'dt', 'samples', 'iterations', 'elite_fraction',
        'random_seed', 'control_period', 'control_on_odometry',
        'wheelbase', 'steering_gain', 'steering_time_constant',
        'understeer_gradient',
        'enable_realtime_grip_limit', 'grip_limit_initial',
        'grip_limit_min', 'grip_limit_max', 'grip_understeer_ratio',
        'grip_oversteer_ratio', 'grip_slip_hold_time',
        'grip_attack_time_constant', 'grip_release_time_constant',
        'grip_safety_factor', 'grip_min_speed',
        'grip_min_expected_yaw_rate',
        'detection_range', 'wall_filter_margin', 'wall_center_gate',
        'obstacle_path_keep_distance', 'obstacle_path_keep_wall_floor',
        'obstacle_merge_gap', 'tracker_velocity_beta',
        'tracker_static_speed', 'enable_static_obstacle_bake', 'bake_age',
        'bake_position_std', 'bake_radius_margin', 'bake_absent_scans',
    })
    # Config fields baked into the AvoidancePlanner's precomputed lane
    # clearances: accept the update, then rebuild the planner off the control
    # thread through the pending-raceline swap.
    _PLANNER_REBUILD_PARAMETERS = frozenset({
        'avoid_max_offset', 'avoid_offset_step', 'ego_center_offset',
        'ego_disc_offset', 'ego_disc_radius', 'wall_hard_margin',
    })

    def __init__(self):
        super().__init__('mpc_controller')

        default_raceline = str(
            Path(get_package_share_directory('race_track_mpc'))
            / 'race_track'
            / 'track_file'
            / 'ifac_roboracer_raceline_apex.csv'
        )
        self.declare_parameter('raceline_path', default_raceline)
        default_map = str(
            Path(get_package_share_directory('race_track_mpc'))
            / 'race_track'
            / 'track_file'
            / 'ifac_roboracer.yaml'
        )
        self.declare_parameter('map_yaml_path', default_map)
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('enable_manual_override', True)
        self.declare_parameter('manual_override_topic', '/teleop')
        self.declare_parameter('manual_override_timeout', 0.25)
        self.declare_parameter('control_period', 0.04)
        # When the estimator already publishes at the desired control rate,
        # run immediately in that callback instead of adding a uniformly
        # distributed timer phase delay.  The real PF stream is 49.98 Hz;
        # simulation can retain the independent periodic loop.
        self.declare_parameter('control_on_odometry', False)
        self.declare_parameter('odom_timeout', 0.5)
        self.declare_parameter('maximum_path_distance', 1.5)
        # Absolute ceiling on the published speed command, measured against
        # the braking-feasible profile one preview window ahead.  The margin
        # is sized from real laps: over 2676 autonomous cycles of bag
        # mpc_20260824_194035 the command exceeded that profile by at most
        # 0.44 m/s, while the two wall crashes of mpc_20260824_194515/_194608
        # sat at +1.2 m/s median and +3.2 m/s peak.  0.5 m/s therefore never
        # binds a healthy lap and stops a corrupted speed estimate from
        # commanding a grip-limited bend at twice its profiled speed.
        self.declare_parameter('enable_speed_command_cap', True)
        self.declare_parameter('speed_command_cap_margin', 0.5)
        self.declare_parameter('enable_vesc_brake_command', False)
        # Wheel-slip correction: window shared by the wheel and localization
        # averages (0.20 s leaves the lag cancelled and the pose noise
        # averaged), a short smoothing constant, and a bound on how much
        # speed a localization fault may remove.  On bag mpc_20260824_194608
        # this recovers the ERPM over-read from +0.71 to +0.18 m/s.
        self.declare_parameter('enable_wheel_slip_correction', True)
        self.declare_parameter('wheel_slip_window', 0.20)
        self.declare_parameter('wheel_slip_time_constant', 0.05)
        self.declare_parameter('wheel_slip_max_correction', 1.0)
        self.declare_parameter('enable_emergency_stop', True)
        self.declare_parameter('emergency_stop_distance', 0.08)
        self.declare_parameter('emergency_stop_angle', 0.20)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('enable_obstacle_avoidance', True)
        self.declare_parameter('laser_offset', 0.275)
        self.declare_parameter('detection_range', 8.0)
        self.declare_parameter('wall_filter_margin', 0.12)
        self.declare_parameter('wall_center_gate', 0.0)
        self.declare_parameter('obstacle_path_keep_distance', 0.30)
        # Near-raceline returns may bypass the broad wall margin, but never
        # when they sit on the mapped wall shell itself.
        self.declare_parameter('obstacle_path_keep_wall_floor', 0.0)
        # Negative disables consolidation until a validated vehicle/sim config
        # explicitly opts in.
        self.declare_parameter('obstacle_merge_gap', -1.0)
        # Alpha-beta velocity tuning is separate from PF.  PF pose jitter moves
        # every lidar cluster in map coordinates; a low velocity gain plus a
        # realistic static snap threshold prevents that common-mode motion
        # from becoming a predicted moving obstacle inside the MPC.
        self.declare_parameter('tracker_velocity_beta', 0.25)
        self.declare_parameter('tracker_static_speed', 0.35)
        # Scan tracking is pipelined on the sensor executor.  Moving tracks
        # are extrapolated only over this short bounded age; an older
        # snapshot stops autonomous output instead of silently driving blind.
        self.declare_parameter('obstacle_prediction_max_age', 0.08)
        self.declare_parameter('obstacle_snapshot_timeout', 0.15)
        # Optional optimization that stamps stable obstacles into the solver
        # distance field.
        self.declare_parameter('enable_static_obstacle_bake', False)
        self.declare_parameter('bake_age', 20)
        self.declare_parameter('bake_position_std', 0.05)
        self.declare_parameter('bake_radius_margin', 0.05)
        self.declare_parameter('bake_absent_scans', 10)
        # RViz remains live, but marker construction is paced independently
        # so it cannot consume the MPC control deadline.
        self.declare_parameter('visualization_period', 0.20)
        self.declare_parameter('enable_raceline_editor', False)
        self.declare_parameter('raceline_editor_control_stride', 8)
        self.declare_parameter('raceline_editor_influence_points', 12)
        self.declare_parameter('raceline_editor_marker_scale', 0.16)
        # Editor handles carry the authored reference speed as height above
        # the floor; dragging one vertically is a speed edit.
        self.declare_parameter('raceline_editor_speed_height_scale', 0.10)
        self.declare_parameter('raceline_editor_speed_step', 0.5)
        self.declare_parameter('raceline_editor_save_path', '')
        # Dragging a handle in the map plane moves the LINE and re-derives
        # arc length, heading, curvature, speed and acceleration from it.
        # Set false for the historical speed-only editor.
        self.declare_parameter('raceline_editor_allow_geometry', True)
        # A geometry edit is refused if it leaves the swept body this close
        # to a wall. The car tracks to about 0.30 m p95, so a static margin
        # thinner than that is driven into the wall rather than near it -
        # but the corridor cannot always give 0.30, so the floor is the hard
        # one and the editor logs what it left.
        self.declare_parameter('raceline_editor_min_clearance', 0.10)
        self.declare_parameter('enable_imu', False)
        self.declare_parameter('imu_topic', '/sensors/imu/raw')
        # sensor_msgs/Imu uses SI units (m/s^2 and rad/s).  Keep the conversion
        # switches for compatibility with legacy drivers, but never apply them
        # to the current /sensors/imu/raw stream by default.
        self.declare_parameter('imu_accel_in_g', False)
        self.declare_parameter('imu_gyro_in_degrees', False)
        self.declare_parameter('imu_yaw_rate_axis', 'z')
        self.declare_parameter('imu_long_accel_axis', '-y')
        self.declare_parameter('imu_lat_accel_axis', 'x')
        self.declare_parameter('imu_lpf_cutoff', 8.0)
        self.declare_parameter('imu_timeout', 0.3)
        self.declare_parameter('imu_use_latency_compensation', True)
        self.declare_parameter('imu_use_scan_alignment', True)
        # Scan/pose time alignment (see OdometryHistory).  This path does not
        # need the IMU, so it also covers the case imu_use_scan_alignment
        # cannot: IMU disabled, or its samples stale.
        self.declare_parameter('odom_history_seconds', 1.0)
        self.declare_parameter('scan_pose_max_extrapolation', 0.05)
        # A scan's stamp is normally NEWER than the freshest pose (the
        # particle filter publishes after the scan it consumed), so there is
        # nothing to interpolate between until the next odometry arrives.
        # Holding the scan that long turns dead-reckoning into true
        # interpolation.  Measured on bag mpc_20260822_215337: frames with a
        # phantom cluster fall 13.4% -> 11.6% by extrapolating, and 13.4% ->
        # 9.8% by waiting, for 13 ms (p50) / 36 ms (p95) of extra obstacle
        # age.  Set false to keep the older, lower-latency behaviour.
        self.declare_parameter('scan_pose_wait_for_bracket', True)
        self.declare_parameter('scan_pose_max_wait', 0.03)
        self.declare_parameter('imu_use_speed_fusion', True)
        self.declare_parameter('imu_speed_fusion_time_constant', 0.4)
        self.declare_parameter('imu_speed_hold_time', 0.15)
        self.declare_parameter('enable_imu_safety', True)
        self.declare_parameter('imu_spin_yaw_residual', 2.5)
        self.declare_parameter('imu_spin_hold_time', 0.2)
        self.declare_parameter('imu_spin_min_speed', 1.5)
        self.declare_parameter('imu_steer_lag_time_constant', 0.15)
        self.declare_parameter('imu_collision_accel', 19.0)
        self.declare_parameter('imu_collision_hold_time', 1.0)
        # Runtime-only tire envelope.  It never changes the raceline CSV or
        # editor behavior; it tightens reference speeds and rollout costs from
        # observed steering-to-yaw response while the car is moving.
        self.declare_parameter('enable_realtime_grip_limit', False)
        self.declare_parameter('grip_limit_initial', 5.0)
        self.declare_parameter('grip_limit_min', 3.5)
        self.declare_parameter('grip_limit_max', 6.0)
        self.declare_parameter('grip_understeer_ratio', 0.82)
        self.declare_parameter('grip_oversteer_ratio', 1.25)
        self.declare_parameter('grip_slip_hold_time', 0.08)
        self.declare_parameter('grip_attack_time_constant', 0.08)
        self.declare_parameter('grip_release_time_constant', 8.0)
        self.declare_parameter('grip_safety_factor', 0.90)
        self.declare_parameter('grip_min_speed', 1.5)
        self.declare_parameter('grip_min_expected_yaw_rate', 0.5)
        defaults = MPCConfig()
        # Derived from the dataclass rather than from a parallel list.  The
        # config is then read back field by field, so any field missing from
        # a hand-maintained list is not a mis-tuned car but a node that will
        # not start at all - `avoid_profile_solver` bricked every sim launch
        # that way, and no unit test caught it because they all build
        # MPCConfig directly.  control_period is declared above with its own
        # default and documentation, hence the has_parameter guard.
        for name in defaults.__dataclass_fields__:
            if not self.has_parameter(name):
                self.declare_parameter(name, getattr(defaults, name))

        config_values = {
            name: self.get_parameter(name).value
            for name in defaults.__dataclass_fields__
        }
        self.config = MPCConfig(**config_values)
        self._wall_bake_enabled = bool(
            self.get_parameter('enable_static_obstacle_bake').value)
        self.config.validate()
        if not self.config.reference_speed_caps:
            self.get_logger().warn(
                'reference_speed_caps DISABLED: the reference follows the '
                'raceline/editor speeds as-is (no curvature or steering-reach '
                'braking, no minimum floor). Re-enable live with: ros2 param '
                'set /mpc_controller reference_speed_caps true')

        # Sector speed scaling removed 2026-08-24: the runtime layer (Sector<i>
        # params, global_limit, save_params, speed_scaling_path auto-load) was
        # only ever used for weaving-era workarounds and could silently load a
        # stale slow-down file.  Speed shaping now lives in the raceline CSV
        # (tools/reprofile_raceline.py) plus the runtime feasibility passes.

        raceline_path = self.get_parameter('raceline_path').value
        self.raceline = Raceline.from_csv(raceline_path)
        # A speed cap cannot make a geometric kink traceable.  The archived
        # weaving bag contained curvature spikes several times beyond rack
        # authority; fail before enabling drive output instead of carrying
        # their saturated opposite-steer tail into the following straight.
        self.raceline.validate_steering_reach(self.config)
        self.controller = SamplingMPC(self.config)
        # Startup can only report that a backend passed its warm-up probe.
        # The first real solve performs the NumPy equivalence check and may
        # promote GPU/CPU to active or fall back, so log that transition too.
        self._last_accel_status = self.controller.accel_status

        self._wall_field = None
        self._tracker = None
        map_yaml = str(self.get_parameter('map_yaml_path').value or '')
        avoidance_enabled = bool(self.get_parameter('enable_obstacle_avoidance').value)
        # The wall distance field serves the solver's wall costs and is kept
        # even with avoidance off; the flag only controls obstacle tracking.
        if map_yaml:
            self._wall_field = MapDistanceField.from_map_yaml(map_yaml)
            self.controller.set_wall_field(self._wall_field)
        if avoidance_enabled and self._wall_field is not None:
            self._tracker = ObstacleTracker(
                self._wall_field,
                detection_range=float(self.get_parameter('detection_range').value),
                wall_filter_margin=float(
                    self.get_parameter('wall_filter_margin').value),
                wall_center_gate=float(
                    self.get_parameter('wall_center_gate').value),
                path=self.raceline,
                path_keep_distance=float(self.get_parameter(
                    'obstacle_path_keep_distance').value),
                path_keep_wall_floor=float(self.get_parameter(
                    'obstacle_path_keep_wall_floor').value),
                velocity_beta=float(self.get_parameter(
                    'tracker_velocity_beta').value),
                static_speed=float(self.get_parameter(
                    'tracker_static_speed').value),
                merge_gap=float(self.get_parameter(
                    'obstacle_merge_gap').value),
                bake_enabled=self._wall_bake_enabled,
                bake_age=int(self.get_parameter('bake_age').value),
                bake_position_std=float(self.get_parameter(
                    'bake_position_std').value),
                bake_radius_margin=float(self.get_parameter(
                    'bake_radius_margin').value),
                bake_absent_scans=int(self.get_parameter(
                    'bake_absent_scans').value))
        elif avoidance_enabled:
            self.get_logger().warn(
                'Obstacle avoidance requested but map_yaml_path is empty; '
                'running raceline tracking only')
        self.planner = AvoidancePlanner(
            self.raceline, self.config, self._wall_field)

        self._state = None  # type: Optional[VehicleState]
        self._last_odom_time = None
        self._control_on_odometry = bool(
            self.get_parameter('control_on_odometry').value)
        self._odom_watchdog_stopped = True
        self._front_range = math.inf
        self._front_range_prev = math.inf
        self._scan = None
        self._scan_time = None
        self._last_warning_time = -math.inf
        self._last_path_s = None
        # Lap timing for the pitwall panel: armed at the first forward
        # start/finish crossing, so the initial partial lap is never counted.
        self._lap_start_time = None  # type: Optional[float]
        self._lap_clean = True
        self._lap_count = 0
        self._best_lap_time = None  # type: Optional[float]
        self._lap_crossing_pending = None  # type: Optional[float]
        self._raceline_editor = None
        self._odom_stamp = None  # header time (s), for scan-pose alignment
        self._scan_stamp = None
        # In periodic mode sensor callbacks run concurrently with control;
        # these tuples are swapped atomically.  In odometry-triggered mode
        # they also preserve one exact state/arrival pair for telemetry.
        self._odom_snapshot = None  # (VehicleState, header stamp, arrival perf)
        self._odom_history = OdometryHistory(
            float(self.get_parameter('odom_history_seconds').value))
        # How the last scans were paired with a pose, for the periodic log.
        self._pose_pairing = {OdometryHistory.INTERPOLATED: 0,
                              OdometryHistory.EXTRAPOLATED: 0,
                              OdometryHistory.HELD: 0}
        self._pose_pairing_warned = False
        # One scan held back waiting to be bracketed; the sensor callbacks
        # share a mutually-exclusive group, so a plain slot is enough.
        self._pending_scan = None
        self._scan_snapshot = None  # (LaserScan, arrival time, header stamp)
        # A tracker producer builds a fully immutable tuple then swaps this
        # one reference; control never observes a half-updated obstacle
        # array/count/bake generation.
        self._tracker_lock = threading.Lock()
        self._obstacle_snapshot = None
        self._last_obstacle_count = 0
        self._applied_bake_generation = 0
        self._viz_lock = threading.Lock()
        self._viz_snapshot = None
        self._imu = None
        self._imu_safety = None
        self._grip_observer = None
        self._grip_limiter = None
        self._imu_arrival = None
        self._imu_warned = False
        self._last_steer_command = 0.0
        self._last_speed_command = None  # type: Optional[float]
        # Low-passed wheel-over-ground speed, i.e. how much of what ERPM
        # reports is the tyre spinning rather than the car moving.
        self._wheel_slip = 0.0
        if bool(self.get_parameter('enable_imu').value):
            self._imu_yaw_axis = parse_axis(
                self.get_parameter('imu_yaw_rate_axis').value)
            self._imu_long_axis = parse_axis(
                self.get_parameter('imu_long_accel_axis').value)
            self._imu_lat_axis = parse_axis(
                self.get_parameter('imu_lat_accel_axis').value)
            self._imu_gyro_scale = (
                math.pi / 180.0
                if bool(self.get_parameter('imu_gyro_in_degrees').value)
                else 1.0)
            self._imu_accel_scale = (
                9.80665 if bool(self.get_parameter('imu_accel_in_g').value)
                else 1.0)
            self._imu = ImuEstimator(
                lpf_cutoff=float(self.get_parameter('imu_lpf_cutoff').value),
                speed_fusion_time_constant=float(self.get_parameter(
                    'imu_speed_fusion_time_constant').value),
                speed_hold_time=float(self.get_parameter(
                    'imu_speed_hold_time').value),
            )
            if bool(self.get_parameter('enable_imu_safety').value):
                self._imu_safety = SafetyMonitor(
                    wheelbase=self.config.wheelbase,
                    understeer_gradient=self.config.understeer_gradient,
                    steering_gain=self.config.steering_gain,
                    spin_yaw_residual=float(self.get_parameter(
                        'imu_spin_yaw_residual').value),
                    spin_hold_time=float(self.get_parameter(
                        'imu_spin_hold_time').value),
                    spin_min_speed=float(self.get_parameter(
                        'imu_spin_min_speed').value),
                    steer_lag_time_constant=float(self.get_parameter(
                        'imu_steer_lag_time_constant').value),
                    collision_accel=float(self.get_parameter(
                        'imu_collision_accel').value),
                    collision_hold_time=float(self.get_parameter(
                        'imu_collision_hold_time').value),
                )
            self._grip_observer = GripObserver()
            if bool(self.get_parameter(
                    'enable_realtime_grip_limit').value):
                self._grip_limiter = AdaptiveGripLimiter(
                    wheelbase=self.config.wheelbase,
                    understeer_gradient=self.config.understeer_gradient,
                    steering_gain=self.config.steering_gain,
                    steer_lag_time_constant=float(self.get_parameter(
                        'imu_steer_lag_time_constant').value),
                    initial_limit=float(self.get_parameter(
                        'grip_limit_initial').value),
                    minimum_limit=float(self.get_parameter(
                        'grip_limit_min').value),
                    maximum_limit=float(self.get_parameter(
                        'grip_limit_max').value),
                    understeer_ratio=float(self.get_parameter(
                        'grip_understeer_ratio').value),
                    oversteer_ratio=float(self.get_parameter(
                        'grip_oversteer_ratio').value),
                    slip_hold_time=float(self.get_parameter(
                        'grip_slip_hold_time').value),
                    attack_time_constant=float(self.get_parameter(
                        'grip_attack_time_constant').value),
                    release_time_constant=float(self.get_parameter(
                        'grip_release_time_constant').value),
                    safety_factor=float(self.get_parameter(
                        'grip_safety_factor').value),
                    min_speed=float(self.get_parameter(
                        'grip_min_speed').value),
                    min_expected_yaw_rate=float(self.get_parameter(
                        'grip_min_expected_yaw_rate').value),
                )
                self.controller.set_lateral_accel_limit(
                    self._grip_limiter.current_limit)
        if (bool(self.get_parameter('enable_realtime_grip_limit').value)
                and self._imu is None):
            self.get_logger().warn(
                'Realtime grip limit requested but IMU is disabled; using '
                'the fixed max_lateral_accel setting')
        self._manual_gate = ManualOverrideGate(
            enabled=bool(self.get_parameter('enable_manual_override').value),
            timeout=float(self.get_parameter('manual_override_timeout').value),
        )
        # The vehicle's optimized 50 Hz path is odometry-triggered on the
        # single executor: the PF stream is already 49.98 Hz, this removes
        # the old 0--20 ms timer phase wait, and bounded sub-10 ms solves no
        # longer starve sensors.  Periodic mode still runs in a plain thread,
        # NOT an executor timer.  History of why:
        # - One shared group starved sensors during 35-62 ms solves: 0.3 s
        #   odometry-stale stops mid-lap (5 in 15 s, bag mpc_20260723_175903)
        #   that also suspended the IMU safety layer during the very skid it
        #   should have caught.
        # - A MultiThreadedExecutor with the solve in a timer callback was
        #   worse: rclpy rebuilds its wait set in Python around every
        #   dispatch, so the idle workers' bookkeeping for 140 Hz of sensor
        #   traffic starved the solve thread of the GIL.  Measured on this
        #   Jetson: a 15 ms GPU solve became 28 ms, a 2.6 ms planner call
        #   31 ms, publishing an EMPTY marker array 9 ms - the whole callback
        #   ~86 ms regardless of sample count.
        # With the solve out of the executor every remaining callback is
        # bounded and light (offline real-scan tracker p95 ~1.3 ms), so a
        # single executor thread services them without starvation and the GIL
        # has exactly one light competitor.  The groups below only assert
        # that intent; they are serialized anyway.
        self._manual_callback_group = MutuallyExclusiveCallbackGroup()
        self._sensor_callback_group = MutuallyExclusiveCallbackGroup()
        self._shutdown = threading.Event()
        self._control_thread = None  # type: Optional[threading.Thread]
        self._visualization_thread = None  # type: Optional[threading.Thread]
        # Raceline replacements from the editor (executor thread) are handed
        # to the control thread as one reference swap, applied between cycles.
        self._pending_raceline = None
        self._autonomous_output_enabled = not self._manual_gate.enabled
        # Compute-rate reporting (printed every few seconds).
        self._rate_window_start = None  # type: Optional[float]
        self._rate_count = 0
        self._compute_sum = 0.0
        self._compute_max = 0.0
        self._stage_sum = 0.0
        self._stage_max = 0.0
        self._rate_report_period = 5.0

        self._drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter('drive_topic').value,
            10,
        )
        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        path_qos.reliability = ReliabilityPolicy.RELIABLE
        self._raceline_publisher = self.create_publisher(
            PathMessage, '/mpc/raceline', path_qos)
        self._prediction_publisher = self.create_publisher(
            PathMessage, '/mpc/prediction', 10)
        self._obstacle_publisher = self.create_publisher(
            MarkerArray, '/mpc/obstacles', 10)
        # One atomic sample per completed control cycle.  Keeping the timing
        # breakdown in one message lets benchmarks classify genuine avoidance
        # spline cycles without pairing independently delivered topics.
        self._compute_breakdown_publisher = self.create_publisher(
            Float64MultiArray, '/mpc/compute_breakdown', 100)
        # Control-loop state for the pitwall RViz panel: transient_local so a
        # panel opened mid-run immediately shows the current state.
        state_qos = QoSProfile(depth=1)
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._pit_state_publisher = self.create_publisher(
            StringMessage, '/mpc/state', state_qos)
        self._pit_state = ''
        if pitwall is not None:
            pitwall.init(self)

        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._odom_callback,
            10,
            callback_group=self._sensor_callback_group,
        )
        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self._scan_callback,
            10,
            callback_group=self._sensor_callback_group,
        )
        if self._imu is not None:
            # Best-effort matches the driver's reliable publisher and never
            # backpressures it; depth covers samples queued during a solve.
            imu_qos = QoSProfile(depth=20)
            imu_qos.reliability = ReliabilityPolicy.BEST_EFFORT
            self.create_subscription(
                Imu,
                self.get_parameter('imu_topic').value,
                self._imu_callback,
                imu_qos,
                callback_group=self._sensor_callback_group,
            )
        if self._manual_gate.enabled:
            self.create_subscription(
                AckermannDriveStamped,
                self.get_parameter('manual_override_topic').value,
                self._manual_override_callback,
                10,
                callback_group=self._manual_callback_group,
            )
        self._publish_raceline()
        if bool(self.get_parameter('enable_raceline_editor').value):
            # Import lazily so the real-car controller does not depend on RViz
            # editing packages unless the deliberately opt-in editor is used.
            from .raceline_markers import RacelineMarkerEditor

            configured_save_path = str(self.get_parameter(
                'raceline_editor_save_path').value).strip()
            save_path = (configured_save_path
                         if configured_save_path
                         else str(Path(raceline_path).expanduser()))
            allow_geometry = bool(self.get_parameter(
                'raceline_editor_allow_geometry').value)
            minimum_clearance = float(self.get_parameter(
                'raceline_editor_min_clearance').value)
            self._raceline_editor = RacelineMarkerEditor(
                self,
                self.raceline,
                self.config,
                self._replace_raceline,
                frame_id=self.get_parameter('frame_id').value,
                control_stride=int(self.get_parameter(
                    'raceline_editor_control_stride').value),
                influence_points=int(self.get_parameter(
                    'raceline_editor_influence_points').value),
                marker_scale=float(self.get_parameter(
                    'raceline_editor_marker_scale').value),
                save_path=save_path,
                speed_height_scale=float(self.get_parameter(
                    'raceline_editor_speed_height_scale').value),
                speed_step=float(self.get_parameter(
                    'raceline_editor_speed_step').value),
                speed_profile=(self._editor_speed_profile
                               if allow_geometry else None),
                clearance=(self._editor_body_clearance
                           if allow_geometry and self._wall_field is not None
                           else None),
                minimum_clearance=minimum_clearance,
            )
            self.get_logger().info(
                'Raceline editor ready: {} handles on '
                '/mpc/raceline_editor/update, save={}, speed 0..{:.2f} m/s, '
                'geometry {}'.format(
                    self._raceline_editor.handle_count,
                    self._raceline_editor.save_path,
                    self.config.max_speed,
                    ('ON, refused under {:.2f} m body clearance'.format(
                        minimum_clearance)
                     if allow_geometry and self._wall_field is not None
                     else ('ON, UNGUARDED (no map loaded)' if allow_geometry
                           else 'off (speed only)'))))
        self.get_logger().info(
            'MPC ready: {} points, {:.2f} m loop, N={}, samples={}, iters={}, '
            'speed scale={:.2f}, heading correction={:+.3f} rad, avoidance={}, '
            'accel={}'.format(
                len(self.raceline.s),
                self.raceline.total_length,
                self.config.horizon,
                self.config.samples,
                self.config.iterations,
                self.config.speed_scale,
                self.raceline.heading_offset,
                'on' if self._tracker is not None else 'off',
                self.controller.accel_status,
            )
        )
        if self._manual_gate.enabled:
            self.get_logger().info(
                'Manual override gate enabled: autonomous output remains muted while {} '
                'is active (timeout {:.2f} s)'.format(
                    self.get_parameter('manual_override_topic').value,
                    self._manual_gate.timeout,
                )
            )
        if self._imu is not None:
            self.get_logger().info(
                'IMU fusion enabled on {}: latency_comp={}, scan_alignment={}, '
                'speed_fusion={}, safety={}'.format(
                    self.get_parameter('imu_topic').value,
                    bool(self.get_parameter(
                        'imu_use_latency_compensation').value),
                    bool(self.get_parameter('imu_use_scan_alignment').value),
                    bool(self.get_parameter('imu_use_speed_fusion').value),
                    self._imu_safety is not None,
                )
            )
        # Deferred follow-up work outside the parameter callback (pending
        # raceline swaps from live parameter updates).
        self.create_timer(0.5, self._deferred_parameter_tasks,
                          callback_group=self._sensor_callback_group)
        self.create_timer(0.05, self._odom_watchdog_callback,
                          callback_group=self._sensor_callback_group)
        self.add_on_set_parameters_callback(self._on_set_parameters)

    def _odom_callback(self, message: Odometry) -> None:
        arrival_perf = time.perf_counter()
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        sin_yaw = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cos_yaw = 1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2)
        yaw = math.atan2(sin_yaw, cos_yaw)
        velocity = message.twist.twist.linear
        wheel_speed = math.hypot(velocity.x, velocity.y)
        stamp = (message.header.stamp.sec
                 + message.header.stamp.nanosec * 1e-9)
        # History first: the slip observation needs this sample in it.  It
        # stores the raw wheel speed on purpose - that is the measurement
        # being corrected, and re-deriving it from a corrected value would
        # feed the correction back into itself.
        self._odom_history.add(stamp, position.x, position.y, yaw,
                               wheel_speed,
                               float(message.twist.twist.angular.z))
        speed = self._ground_referenced_speed(wheel_speed)
        state = VehicleState(position.x, position.y, yaw, speed)
        self._state = state
        self._last_odom_time = self.get_clock().now()
        self._odom_stamp = stamp
        self._odom_snapshot = (state, stamp, arrival_perf)
        self._odom_watchdog_stopped = False
        if self._imu is not None:
            self._imu.update_speed(stamp, speed)
        if self._control_on_odometry:
            # SingleThreadedExecutor makes this non-reentrant.  Running at the
            # estimator's native 50 Hz removes timer phase latency and avoids
            # GIL contention with a separate control thread.
            self._run_control_cycle()

    def _odom_watchdog_callback(self) -> None:
        """Publish one explicit stop if odometry-triggered control goes stale."""
        if (not self._control_on_odometry or self._last_odom_time is None
                or self._odom_watchdog_stopped):
            return
        now = self.get_clock().now()
        age = (now - self._last_odom_time).nanoseconds * 1e-9
        if age <= float(self.get_parameter('odom_timeout').value):
            return
        self._odom_watchdog_stopped = True
        self._set_pit_state('NO_ODOM')
        self._publish_stop()
        self._reset_controllers()
        self._warn_throttled(
            'Odometry is stale ({:.2f} s); publishing stop command'.format(age))

    def _scan_callback(self, message: LaserScan) -> None:
        # A scan held back for a bracketing pose is released here rather than
        # from the odometry callback: with control_on_odometry the odometry
        # callback also runs the control cycle, and moving the tracker onto
        # that path costs latency where it is least affordable.  By now
        # odometry has advanced past the held scan's stamp, so it interpolates
        # rather than dead-reckons; the cost is that the obstacle map is one
        # scan period old instead of one odometry period.
        if self._pending_scan is not None:
            # force: a held scan is always tracked before the next one is
            # accepted, with whatever pose is available.  Dropping it instead
            # would let a stamping or odometry fault starve the tracker
            # frame by frame, which is how a bad alignment stops the car
            # rather than merely degrading it.  Worst case this falls back to
            # the dead-reckoned pose - the behaviour before any of this.
            self._release_pending_scan(self._odom_history.newest_stamp(),
                                       force=True)
        if not message.ranges or message.angle_increment == 0.0:
            self._front_range_prev = math.inf
            self._front_range = math.inf
            return
        self._front_range_prev = self._front_range
        self._scan = message
        self._scan_time = self.get_clock().now().nanoseconds * 1e-9
        self._scan_stamp = (message.header.stamp.sec
                            + message.header.stamp.nanosec * 1e-9)
        self._scan_snapshot = (message, self._scan_time, self._scan_stamp)
        half_angle = float(self.get_parameter('emergency_stop_angle').value)
        ranges = np.asarray(message.ranges, dtype=float)
        # Positive angle increments are standard LaserScan layout.  Slice the
        # emergency cone directly instead of allocating/traversing a full
        # angle vector at lidar rate; retain a generic fallback for unusual
        # reversed scans.
        if message.angle_increment > 0.0:
            low = max(0, int(math.ceil(
                (-half_angle - message.angle_min) / message.angle_increment)))
            high = min(ranges.size, int(math.floor(
                (half_angle - message.angle_min) / message.angle_increment)) + 1)
            front = ranges[low:high]
        else:
            angles = (message.angle_min + np.arange(ranges.size)
                      * message.angle_increment)
            front = ranges[np.abs(angles) <= half_angle]
        valid = (np.isfinite(front)
                 & (front >= message.range_min)
                 & (front <= message.range_max))
        self._front_range = (float(np.min(front[valid]))
                             if np.any(valid) else math.inf)
        # Front-beam protection above is deliberately published first.  The
        # heavier map filtering/tracking now runs once per arriving scan on
        # this sensor executor and is handed to the control thread atomically.
        self._process_obstacle_scan(
            message, self._scan_time, self._scan_stamp, ranges)

    def _manual_override_callback(self, message: AckermannDriveStamped) -> None:
        """Mute MPC output whenever the joystick's manual stream is alive."""
        now = self.get_clock().now().nanoseconds * 1e-9
        self._manual_gate.observe_manual_command(now)
        # The next autonomous solve must rate-limit from the rack angle the
        # driver actually commanded, not from the last pre-manual MPC angle.
        self._last_steer_command = float(message.drive.steering_angle)
        self._last_speed_command = float(message.drive.speed)

    def _imu_callback(self, message: Imu) -> None:
        """Map the VESC IMU sample onto vehicle axes in SI units."""
        stamp = (message.header.stamp.sec
                 + message.header.stamp.nanosec * 1e-9)
        gyro = (message.angular_velocity.x, message.angular_velocity.y,
                message.angular_velocity.z)
        accel = (message.linear_acceleration.x, message.linear_acceleration.y,
                 message.linear_acceleration.z)
        index, sign = self._imu_yaw_axis
        yaw_rate = sign * gyro[index] * self._imu_gyro_scale
        index, sign = self._imu_long_axis
        long_accel = sign * accel[index] * self._imu_accel_scale
        index, sign = self._imu_lat_axis
        lat_accel = sign * accel[index] * self._imu_accel_scale
        self._imu.add_sample(stamp, yaw_rate, long_accel, lat_accel)
        self._imu_arrival = self.get_clock().now().nanoseconds * 1e-9
        if self._grip_observer is not None:
            speed = self._imu.fused_speed()
            if speed is not None:
                self._grip_observer.add(
                    self._imu.lat_accel(), self._imu.yaw_rate(), speed)

    def _imu_ready(self, now_seconds: float) -> bool:
        """True when IMU fusion is enabled and data is fresh."""
        if self._imu is None:
            return False
        fresh = (
            self._imu_arrival is not None
            and now_seconds - self._imu_arrival
            <= float(self.get_parameter('imu_timeout').value)
        )
        if fresh:
            self._imu_warned = False
        elif not self._imu_warned:
            self._imu_warned = True
            self.get_logger().warn(
                'IMU enabled but data is missing or stale; IMU features '
                'suspended until samples resume')
        return fresh

    def _process_obstacle_scan(self, scan: LaserScan, scan_time: float,
                               scan_stamp: float,
                               ranges: np.ndarray) -> None:
        """Track this scan now, or hold it until its pose can be bracketed."""
        if (bool(self.get_parameter('scan_pose_wait_for_bracket').value)
                and self._tracker is not None):
            newest = self._odom_history.newest_stamp()
            max_wait = max(
                float(self.get_parameter('scan_pose_max_wait').value), 0.0)
            # Strictly newer: a pose stamped exactly at the scan's own time
            # is an exact hit, not something to wait for.  Some publishers
            # (the gym bridge among them) stamp a scan and its odometry from
            # one clock read, so >= here would park every single scan.
            if (newest is not None and newest < scan_stamp
                    and scan_stamp - newest < max_wait):
                # Nothing to interpolate between yet, but the bracketing
                # sample is due within the wait budget.  Park it; the next
                # odometry callback releases it.  Only the newest scan is
                # worth holding - an older one it displaced was going to be
                # superseded anyway.  When odometry is lagging by more than
                # the budget the test above fails and the scan goes straight
                # through on dead reckoning: a slightly misaligned obstacle
                # map beats none, and the odometry watchdog owns that fault.
                self._pending_scan = (scan, scan_time, scan_stamp, ranges)
                return
        self._pending_scan = None
        self._track_obstacle_scan(scan, scan_time, scan_stamp, ranges)

    def _release_pending_scan(self, newest_stamp, force: bool = False) -> None:
        """Run a held scan once it is bracketed, or once waiting stops paying."""
        pending = self._pending_scan
        if pending is None:
            return
        scan, scan_time, scan_stamp, ranges = pending
        waited = (newest_stamp - scan_stamp) if newest_stamp is not None else -1.0
        if waited < 0.0 and not force:
            # Still ahead of the newest pose.  Give up if odometry has fallen
            # so far behind that the scan would go stale waiting: a slightly
            # misaligned obstacle map beats no obstacle map.
            age = self.get_clock().now().nanoseconds * 1e-9 - scan_time
            if age < max(float(self.get_parameter('scan_pose_max_wait').value),
                         0.0):
                return
        self._pending_scan = None
        self._track_obstacle_scan(scan, scan_time, scan_stamp, ranges)

    def _track_obstacle_scan(self, scan: LaserScan, scan_time: float,
                             scan_stamp: float,
                             ranges: np.ndarray) -> None:
        """Build and atomically publish one immutable tracker snapshot."""
        odom_snapshot = self._odom_snapshot
        if self._tracker is None or odom_snapshot is None:
            return
        state, odom_stamp, _ = odom_snapshot
        x, y, yaw = state.x, state.y, state.yaw
        # The tracker's wall filter needs the ego pose AT the scan timestamp,
        # not at whichever odometry happened to arrive last.  Interpolating
        # the buffered poses to the scan's own header stamp removes the
        # mismatch exactly whenever the scan falls inside the buffered span,
        # which is the normal case and needs no IMU.
        offset = scan_stamp - odom_stamp
        aligned = None
        if abs(offset) < 0.5:
            aligned = self._odom_history.pose_at(
                scan_stamp,
                float(self.get_parameter(
                    'scan_pose_max_extrapolation').value))
        elif not self._pose_pairing_warned:
            # Half a second apart means the two streams are not sharing a
            # clock (an unstamped driver, or use_sim_time set on one side).
            # Interpolating across that would invent a pose, so hold and say
            # so once - the alignment is off until the stamping is fixed.
            self._pose_pairing_warned = True
            self._warn_throttled(
                'Scan and odometry stamps differ by {:.2f} s; pose alignment '
                'disabled (check header stamps / use_sim_time)'.format(offset))
        if aligned is not None:
            x, y, yaw, how = aligned
            self._pose_pairing[how] = self._pose_pairing.get(how, 0) + 1
            if (how == OdometryHistory.EXTRAPOLATED
                    and bool(self.get_parameter(
                        'imu_use_scan_alignment').value)
                    and self._imu_ready(
                        self.get_clock().now().nanoseconds * 1e-9)):
                # Ahead of the newest pose there is nothing to interpolate
                # between, so the extrapolation above held the odometry's yaw
                # rate.  A fresh IMU measures that rate directly over the same
                # window, which is strictly better than holding it.
                x, y, yaw = self._imu.advance_pose(
                    state.x, state.y, state.yaw, self._speed_estimate(),
                    odom_stamp, scan_stamp)
        started = time.perf_counter()
        with self._tracker_lock:
            obstacles = self._tracker.update(
                scan_time, ranges, scan.angle_min, scan.angle_increment,
                x, y, yaw,
                laser_offset=float(self.get_parameter('laser_offset').value),
            )
            raw_count = int(self._tracker.last_raw_count)
            bake_generation = int(self._tracker.bake_generation)
            baked = (self._tracker.baked_discs()
                     if self._wall_bake_enabled else np.empty((0, 3)))
            immutable_obstacles = np.array(
                obstacles, dtype=float, copy=True).reshape(-1, 5)
            immutable_obstacles.setflags(write=False)
            immutable_baked = np.array(
                baked, dtype=float, copy=True).reshape(-1, 3)
            immutable_baked.setflags(write=False)
            detection_seconds = time.perf_counter() - started
            self._obstacle_snapshot = (
                immutable_obstacles, float(scan_time), detection_seconds,
                raw_count, int(immutable_obstacles.shape[0]), bake_generation,
                immutable_baked)

    def _obstacle_frame(self, now_seconds: float):
        """Read one coherent scan result and compensate bounded mover age.

        Returns obstacles, tracker CPU seconds, scan age, raw/consolidated
        counts, bake generation/discs, and a stale flag.  The atomic tuple
        read is non-blocking so the MPC never waits behind the next lidar
        frame's filtering work.
        """
        snapshot = self._obstacle_snapshot
        if self._tracker is None:
            return None, 0.0, 0.0, 0, 0, 0, np.empty((0, 3)), False
        if snapshot is None:
            return None, 0.0, math.inf, 0, 0, 0, np.empty((0, 3)), True
        (obstacles, scan_time, detection_seconds, raw_count,
         consolidated_count, bake_generation, baked) = snapshot
        age = max(float(now_seconds) - scan_time, 0.0)
        timeout = max(float(self.get_parameter(
            'obstacle_snapshot_timeout').value), 0.01)
        stale = age > timeout
        maximum_age = max(float(self.get_parameter(
            'obstacle_prediction_max_age').value), 0.0)
        predicted = extrapolate_obstacles(obstacles, age, maximum_age)
        return (predicted, detection_seconds, age, raw_count,
                consolidated_count, bake_generation, baked, stale)

    def _ground_referenced_speed(self, wheel_speed: float) -> float:
        """The reported wheel speed with the observed tyre spin removed.

        ERPM is the only speed the drivetrain reports and it reads the wheel,
        not the ground.  Under the 5 m/s^2 launch of bag
        mpc_20260824_194608 it stood 0.7 m/s high (1.1 m/s in _194515) by the
        time the car reached the corner, and because every other estimate is
        anchored to it - the complementary IMU fusion included, which is why
        raising its time constant recovers only 0.09 m/s of that - the whole
        control loop inherited the error.  Localization is the one speed
        reference the tyres cannot corrupt.

        The correction only ever subtracts.  Wheels over-read under power and
        under-read under lock; the IMU integral already carries the car
        through lock, which is what it was added for, and letting this add
        speed would hand a localization glitch a way to invent it.  The
        bounded, low-passed result keeps a single bad pose from moving the
        estimate far in one cycle.
        """
        wheel_speed = float(wheel_speed)
        if not bool(self.get_parameter('enable_wheel_slip_correction').value):
            return wheel_speed
        window = float(self.get_parameter('wheel_slip_window').value)
        observed = self._odom_history.slip_observation(window)
        if observed is not None:
            limit = max(float(self.get_parameter(
                'wheel_slip_max_correction').value), 0.0)
            observed = min(max(observed, 0.0), limit)
            tau = max(float(self.get_parameter(
                'wheel_slip_time_constant').value), 1e-3)
            gain = 1.0 - math.exp(-self.config.control_period / tau)
            self._wheel_slip += gain * (observed - self._wheel_slip)
        return max(wheel_speed - self._wheel_slip, 0.0)

    def _speed_command(self, target_speed: float, measured_speed: float,
                       path_s: float, lateral_accel_limit) -> Tuple[float, float]:
        """Return the absolute speed and its preview acceleration for /drive.

        Cap the target against the reference profile before applying the
        per-cycle slew. Anchor the slew to the car's speed when decelerating
        so a stale high command cannot hold the car above its braking target.
        """
        period = self.config.control_period
        preview = max(self.config.speed_command_preview, period)
        measured_speed = max(float(measured_speed), 0.0)
        target = min(float(target_speed),
                     self._reference_speed_cap(path_s, measured_speed,
                                               lateral_accel_limit))
        previous = (measured_speed if self._last_speed_command is None
                    else float(self._last_speed_command))
        if target < measured_speed:
            anchor = min(previous, measured_speed)
        else:
            anchor = min(previous,
                         measured_speed + self.config.max_accel * preview)
        speed = float(np.clip(target,
                              anchor + self.config.min_accel * period,
                              anchor + self.config.max_accel * period))
        if bool(self.get_parameter('enable_vesc_brake_command').value):
            # Entry thresholds in vehicle_patches/vesc_motor_brake.patch.
            brake_accel = -1.0
            brake_speed_gap = max(0.15, -brake_accel * preview)
            brake_speed = measured_speed - brake_speed_gap
            speed_rounding_margin = 1e-6
            if (measured_speed >= 0.50 and self.config.min_accel <= brake_accel
                    and 0.0 <= brake_speed
                    and target <= brake_speed + speed_rounding_margin):
                # The pre-slew target already calls for braking. Give the
                # VESC enough setpoint gap in this command, with room for
                # AckermannDrive's float32 rounding at the entry threshold.
                speed = min(speed, max(0.0, brake_speed - speed_rounding_margin))

        # The preview speed carries the braking intent. The first CEM accel
        # cell can still be positive when the preview calls for deceleration.
        acceleration = float(np.clip((speed - measured_speed) / preview,
                                     self.config.min_accel, self.config.max_accel))
        return speed, acceleration

    def _reference_speed_cap(self, path_s: float, speed: float,
                             lateral_accel_limit) -> float:
        """Absolute ceiling for the published speed command at this station.

        Read one preview window down the road, because that is the point the
        speed command is steering the car's speed toward.  Approaching a
        corner the lookahead has already reached the slower profile, so the
        command starts falling before the geometry arrives; leaving one, the
        profile's own forward-acceleration pass supplies the headroom.

        A generous ``speed_command_cap_margin`` keeps this off the critical
        path in normal running - it exists to bound a speed estimate that has
        gone wrong, not to become a second speed controller.
        """
        if not bool(self.get_parameter('enable_speed_command_cap').value):
            return float(self.config.max_speed)
        raceline = self.raceline
        if raceline is None or not np.isfinite(path_s):
            return float(self.config.max_speed)
        margin = max(float(self.get_parameter(
            'speed_command_cap_margin').value), 0.0)
        preview = max(self.config.speed_command_preview,
                      self.config.control_period)
        lookahead = float(path_s) + max(float(speed), 0.0) * preview
        profile_speed = float(raceline.speed_at(
            lookahead, self.config, lateral_accel_limit))
        return min(float(self.config.max_speed), profile_speed + margin)

    def _speed_estimate(self) -> float:
        """Best current speed: IMU-supported fusion when enabled and fresh."""
        speed = self._state.speed if self._state is not None else 0.0
        if (self._imu is not None
                and bool(self.get_parameter('imu_use_speed_fusion').value)):
            fused = self._imu.fused_speed()
            if fused is not None:
                speed = max(0.0, float(fused))
        return speed

    def _front_hit_is_tracked(self, state: VehicleState,
                              obstacles: Optional[np.ndarray]) -> bool:
        """Whether the raw emergency beam belongs to a tracked obstacle."""
        if obstacles is None or obstacles.shape[0] == 0:
            return False
        laser_offset = float(self.get_parameter('laser_offset').value)
        laser_x = state.x + laser_offset * math.cos(state.yaw)
        laser_y = state.y + laser_offset * math.sin(state.yaw)
        dx = obstacles[:, 0] - laser_x
        dy = obstacles[:, 1] - laser_y
        distance = np.hypot(dx, dy)
        bearing = np.arctan2(
            np.sin(np.arctan2(dy, dx) - state.yaw),
            np.cos(np.arctan2(dy, dx) - state.yaw))
        radius = obstacles[:, 4]
        angular_radius = np.arcsin(np.clip(
            radius / np.maximum(distance, 1e-6), 0.0, 1.0))
        surface_range = np.maximum(distance - radius, 0.0)
        half_angle = float(self.get_parameter('emergency_stop_angle').value)
        threshold = float(self.get_parameter('emergency_stop_distance').value)
        tolerance = np.maximum(radius, 0.15)
        return bool(np.any(
            (dx * math.cos(state.yaw) + dy * math.sin(state.yaw) > 0.0)
            & (np.abs(bearing) <= half_angle + angular_radius)
            & (surface_range <= threshold + 0.25)
            & (np.abs(surface_range - self._front_range) <= tolerance)))

    def _escape_prediction_is_safe(self, state: VehicleState,
                                   predicted: np.ndarray,
                                   obstacles: np.ndarray) -> bool:
        """Allow only a slow trajectory that clears, rather than approaches.

        This is used solely to recover from a raw emergency stop after the hit
        has become a confirmed tracked obstacle.  The first stop still removes
        momentum; recovery is admitted only from near standstill and only when
        the MPC prediction maintains every hard clearance and improves the
        final obstacle clearance.
        """
        if obstacles.shape[0] == 0 or predicted.shape[0] < 2:
            return False
        cfg = self.config
        states = np.asarray(predicted, dtype=float)
        direction = np.stack(
            (np.cos(states[:, 2]), np.sin(states[:, 2])), axis=-1)
        centers = states[:, 0:2]
        discs = (
            centers + (cfg.ego_center_offset + cfg.ego_disc_offset) * direction,
            centers + (cfg.ego_center_offset - cfg.ego_disc_offset) * direction,
        )
        times = np.arange(states.shape[0], dtype=float) * cfg.dt
        obstacle_centers = (obstacles[None, :, 0:2]
                            + obstacles[None, :, 2:4] * times[:, None, None])
        clearance = np.full(states.shape[0], np.inf)
        for disc in discs:
            value = (np.linalg.norm(
                disc[:, None, :] - obstacle_centers, axis=-1)
                - obstacles[None, :, 4] - cfg.ego_disc_radius)
            clearance = np.minimum(clearance, value.min(axis=1))
        # A safe current state may not cross the hard boundary.  If localization
        # already reports a small overlap, tolerate only its existing depth.
        obstacle_floor = min(clearance[0] - 0.02, 0.0)
        if float(np.min(clearance[1:])) < obstacle_floor:
            return False
        if clearance[-1] < clearance[0] + 0.02:
            return False

        if self._wall_field is not None:
            wall_clearance = np.full(states.shape[0], np.inf)
            for disc in discs:
                wall_clearance = np.minimum(
                    wall_clearance,
                    self._wall_field.clearance(disc[:, 0], disc[:, 1]))
            hard = cfg.ego_disc_radius + cfg.wall_hard_margin
            wall_floor = min(wall_clearance[0] - 0.02, hard)
            if float(np.min(wall_clearance[1:])) < wall_floor:
                return False
        return float(np.hypot(states[-1, 0] - state.x,
                              states[-1, 1] - state.y)) > 0.05

    def _control_callback(self) -> None:
        now = self.get_clock().now()
        manual_active = self._manual_gate.manual_active(
            now.nanoseconds * 1e-9)
        if manual_active:
            self._set_pit_state('MANUAL')
            if self._autonomous_output_enabled:
                self._autonomous_output_enabled = False
                self._reset_controllers()
                if self._imu_safety is not None:
                    # The driver has the car; clear spin state and any
                    # collision latch for the next autonomous engagement.
                    self._imu_safety.reset()
                self.get_logger().info(
                    'MANUAL mode: MPC drive output suppressed')
            return

        if not self._autonomous_output_enabled:
            self._autonomous_output_enabled = True
            self._reset_controllers()
            self.get_logger().info(
                'AUTONOMOUS mode: MPC drive output enabled')

        if self._state is None or self._last_odom_time is None:
            self._set_pit_state('NO_ODOM')
            self._publish_stop()
            self._warn_throttled('Waiting for odometry; publishing stop command')
            return

        odom_age = (now - self._last_odom_time).nanoseconds * 1e-9
        if odom_age > float(self.get_parameter('odom_timeout').value):
            self._set_pit_state('NO_ODOM')
            self._publish_stop()
            self._reset_controllers()
            self._warn_throttled('Odometry is stale ({:.2f} s); publishing stop command'.format(odom_age))
            return

        compute_start = time.perf_counter()
        odom_snapshot = self._odom_snapshot
        odom_arrival_perf = (odom_snapshot[2]
                             if odom_snapshot is not None else None)
        # Scan filtering/tracking is pipelined on the sensor executor.  Read a
        # coherent immutable frame without waiting on the next scan, then
        # advance moving rows over the bounded producer-to-consumer age.
        (obstacles, detect_seconds, obstacle_scan_age, raw_count,
         consolidated_count, bake_generation, baked_discs,
         obstacle_stale) = self._obstacle_frame(now.nanoseconds * 1e-9)
        if obstacle_stale:
            self._set_pit_state('NO_SCAN')
            self._publish_stop()
            self._reset_mpc_warm_start()
            self.planner.reset()
            self._warn_throttled(
                'Obstacle scan is missing or stale ({:.0f} ms); '
                'publishing stop command'.format(1e3 * obstacle_scan_age))
            return
        if (self._tracker is not None
                and self._wall_bake_enabled
                and bake_generation != self._applied_bake_generation):
            # The frozen-disc set changed: swap in a composite wall field so
            # the solver's wall cost covers the baked obstacles.  Rebuilding
            # from the immutable base keeps un-bake exact, and set_wall_field
            # re-arms the accelerator validation chain on purpose - one
            # verified cycle per scene change is the intended safety cost.
            field = (self._wall_field if baked_discs.shape[0] == 0
                     else self._wall_field.with_static_discs(baked_discs))
            self.controller.set_wall_field(field)
            self._applied_bake_generation = bake_generation
            self.get_logger().info(
                'Static obstacle bake: {} disc(s) in the wall field'.format(
                    baked_discs.shape[0]))

        # Two consecutive scans must agree before stopping: a single-beam
        # transient at speed (seen at 5.5 m/s on the straight) otherwise slams
        # the brakes.  Costs one scan period (~66 ms) of reaction, which the
        # last-resort threshold does not meaningfully depend on.
        emergency_active = (
            bool(self.get_parameter('enable_emergency_stop').value)
            and max(self._front_range, self._front_range_prev)
                < float(self.get_parameter('emergency_stop_distance').value)
        )
        emergency_escape = False
        if emergency_active:
            tracked_front = self._front_hit_is_tracked(self._state, obstacles)
            emergency_escape = tracked_front and self._speed_estimate() < 0.25
        if emergency_active and not emergency_escape:
            self._set_pit_state('E_STOP')
            self._publish_stop()
            # Discard the optimizer warm start, but preserve obstacle history
            # and lane commitment so a confirmed obstacle can be planned around
            # from standstill on the next scan.
            self._reset_mpc_warm_start()
            self._warn_throttled(
                'Emergency stop: obstacle {:.2f} m ahead'.format(self._front_range))
            return

        imu_ready = self._imu_ready(now.nanoseconds * 1e-9)
        grip_limit = None
        if self._grip_limiter is not None:
            if imu_ready:
                self._grip_limiter.update(
                    now.nanoseconds * 1e-9,
                    self._imu.yaw_rate(),
                    self._last_steer_command,
                    self._speed_estimate(),
                )
            # Preserve the last conservative estimate through a short IMU
            # dropout/manual pause; loss of observations is never evidence
            # that grip improved.
            grip_limit = self._grip_limiter.current_limit
            self.controller.set_lateral_accel_limit(grip_limit)
        if self._imu_safety is not None and imu_ready:
            reason = self._imu_safety.update(
                now.nanoseconds * 1e-9,
                self._imu.yaw_rate(),
                self._last_steer_command,
                self._speed_estimate(),
                self._imu.raw_long_accel(),
                self._imu.raw_lat_accel(),
            )
            if reason is not None:
                self._set_pit_state('IMU_STOP')
                self._publish_stop()
                self._reset_controllers()
                self._warn_throttled('IMU safety stop: {}'.format(reason))
                return

        measured = (odom_snapshot[0]
                    if odom_snapshot is not None else self._state)
        if imu_ready and bool(
                self.get_parameter('imu_use_speed_fusion').value):
            # The odometry speed is ERPM-based on this car and collapses
            # under wheel slip/lock; the fused estimate keeps the initial
            # speed honest so the braking plan stays consistent.
            measured = VehicleState(
                measured.x, measured.y, measured.yaw, self._speed_estimate())
        if imu_ready and bool(
                self.get_parameter('imu_use_latency_compensation').value):
            state = self.controller.compensate_latency(
                measured,
                yaw_rate=self._imu.yaw_rate(),
                accel=self._imu.long_accel(),
            )
        else:
            state = self.controller.compensate_latency(measured)
        planner_start = time.perf_counter()
        (reference, path_s, path_distance, target_offset,
         relevant) = self.planner.reference(
             state, obstacles, hint_s=self._last_path_s,
             lateral_accel_limit=grip_limit)
        planner_seconds = time.perf_counter() - planner_start
        if path_distance > float(self.get_parameter('maximum_path_distance').value):
            self._set_pit_state('OFF_LINE')
            self._publish_stop()
            self._reset_controllers()
            # Drop the projection hint: with it, the next cycle projects onto
            # the raceline region where the vehicle went off, so a pose reset
            # to a different track section could never leave OFF_LINE.  The
            # unhinted projection is global, which is exactly what recovery
            # from an arbitrary reset pose needs.
            self._last_path_s = None
            self._warn_throttled(
                'Vehicle is {:.2f} m from the raceline; reset near the start pose'.format(path_distance))
            return

        if self._last_path_s is not None:
            jump = abs(path_s - self._last_path_s)
            if jump > 0.45 * self.raceline.total_length:
                # Normal start/finish wrapping should retain the warm start.
                wrapped_jump = self.raceline.total_length - jump
                if wrapped_jump > 2.0:
                    self._reset_controllers()
                    # Teleport (e.g. sim reset): the running lap is meaningless.
                    self._lap_start_time = None
                elif path_s < self._last_path_s:
                    # Forward crossing of the start/finish line.
                    # Finalize only after this cycle has actually published a
                    # finite command and assigned AUTO/FOLLOW.  Otherwise the
                    # boundary frame could hide an obstacle or safety event
                    # from the completed lap's learning verdict.
                    self._lap_crossing_pending = now.nanoseconds * 1e-9
        self._last_path_s = path_s

        # Wall-baked rows are already represented in the composite field;
        # keeping them in the solver list would double-count them.  The rows
        # are emitted verbatim from the frozen registry, so exact matching on
        # (x, y, radius) identifies them.  The planner, follow governor, and
        # escape check above all still use the full list.
        solver_obstacles = relevant
        if (self._tracker is not None and self._wall_bake_enabled
                and relevant.shape[0]):
            discs = self._tracker.baked_discs()
            if discs.shape[0]:
                baked_row = np.all(np.isclose(
                    relevant[:, None, [0, 1, 4]], discs[None, :, :],
                    atol=1e-9), axis=2).any(axis=1)
                solver_obstacles = relevant[~baked_row]
        solve_start = time.perf_counter()
        try:
            result = self.controller.solve(
                state, reference, solver_obstacles,
                speed_ceiling=self.planner.last_follow_cap,
                # ``state`` was propagated to the expected command-application
                # time above.  Advance the rollout's steering state to that
                # same instant; otherwise pose and rack angle are 50 ms out of
                # phase on the real car and steering reversals are exaggerated.
                actuation_delay=self.config.latency_compensation,
                collision_obstacles=obstacles)
        except (ValueError, FloatingPointError) as error:
            self._set_pit_state('SOLVE_FAIL')
            self._publish_stop()
            self._reset_controllers()
            self._warn_throttled('MPC solve failed: {}'.format(error))
            return
        solve_seconds = time.perf_counter() - solve_start
        accel_status = self.controller.accel_status
        if accel_status != self._last_accel_status:
            self.get_logger().info(
                'MPC accelerator status: {}'.format(accel_status))
            self._last_accel_status = accel_status

        # The manual heartbeat may arrive while solve() is running in the
        # other executor thread.  Re-check at the final publication boundary
        # before publishing either a drive command or a stop.
        if self._manual_gate.manual_active(
                self.get_clock().now().nanoseconds * 1e-9):
            self._set_pit_state('MANUAL')
            if self._autonomous_output_enabled:
                self._autonomous_output_enabled = False
                self._reset_controllers()
                if self._imu_safety is not None:
                    # The driver has the car; clear spin state and any
                    # collision latch for the next autonomous engagement.
                    self._imu_safety.reset()
                self.get_logger().info(
                    'MANUAL mode: MPC drive output suppressed')
            return

        if not result.has_safe_trajectory:
            self._publish_stop()
            self._set_pit_state('E_STOP')
            self._reset_mpc_warm_start()
            self._warn_throttled(
                'MPC found no collision-free trajectory; publishing stop command')
            return

        if not np.all(np.isfinite([result.steering, result.target_speed, result.cost])):
            self._set_pit_state('SOLVE_FAIL')
            self._publish_stop()
            self._reset_controllers()
            self._warn_throttled('MPC produced a non-finite command')
            return

        if emergency_escape and not self._escape_prediction_is_safe(
                state, result.predicted_states, relevant):
            self._set_pit_state('ESCAPE')
            self._publish_stop()
            self._reset_mpc_warm_start()
            self._warn_throttled(
                'Emergency hold: tracked obstacle has no improving escape path')
            return

        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = 'ego_racecar/base_link'
        command.drive.steering_angle = result.steering
        target_speed = (min(result.target_speed, 0.30)
                        if emergency_escape else result.target_speed)
        measured_speed = max(self._speed_estimate(), 0.0)
        command.drive.speed, planned_accel = self._speed_command(
            target_speed, measured_speed, path_s, grip_limit)
        command.drive.acceleration = (min(planned_accel, 1.0)
                                      if emergency_escape else planned_accel)
        self._drive_publisher.publish(command)
        # This is the user's control deadline: state/scene read through the
        # completed /drive publish call.  Visualization, telemetry assembly,
        # and logging happen after the actuator-bound command and must never
        # be charged as if they delayed it (or, conversely, hidden inside the
        # solver-only number).
        command_publish_perf = time.perf_counter()
        command_seconds = command_publish_perf - compute_start
        odom_to_command_seconds = (
            command_publish_perf - odom_arrival_perf
            if odom_arrival_perf is not None else math.nan)
        self._last_steer_command = float(result.steering)
        self._last_speed_command = float(command.drive.speed)
        viz_start = time.perf_counter()
        self._queue_visualization(result.predicted_states, obstacles)
        viz_snapshot_seconds = time.perf_counter() - viz_start
        follow_cap = self.planner.last_follow_cap
        self._set_pit_state(
            'ESCAPE' if emergency_escape
            else ('FOLLOW' if follow_cap is not None else 'AUTO'))
        if self._lap_crossing_pending is not None:
            crossing_stamp = self._lap_crossing_pending
            self._lap_crossing_pending = None
            self._record_lap_crossing(crossing_stamp)
        full_cycle_seconds = time.perf_counter() - compute_start
        # Critical-path latency benefits from scan/control pipelining, while
        # serial-equivalent stage sum proves that work was not merely hidden.
        # Scan age is measured at control start; adding this cycle's compute
        # yields sensor-arrival-to-command latency for the selected frame.
        stage_sum_seconds = command_seconds + detect_seconds
        sensor_to_command_seconds = obstacle_scan_age + command_seconds
        breakdown = Float64MultiArray()
        # Columns: control-start through /drive publish, async detection,
        # avoidance planner, MPC solve,
        # selected offset, relevant obstacles, raw tracks, consolidated
        # tracks, full-replan flag, visualization snapshot, scan age,
        # serial-equivalent stage sum, and scan-arrival-to-command latency
        # (times in ms), followed by clear-track recovery, latest-odom-arrival
        # through /drive publish, full callback bookkeeping time, live lateral
        # grip limit, and measured/expected yaw-response ratio.
        breakdown.data = [
            1e3 * command_seconds,
            1e3 * detect_seconds,
            1e3 * planner_seconds,
            1e3 * solve_seconds,
            float(target_offset),
            float(relevant.shape[0]),
            float(raw_count),
            float(consolidated_count),
            1.0 if self.planner.last_replanned else 0.0,
            1e3 * viz_snapshot_seconds,
            1e3 * obstacle_scan_age,
            1e3 * stage_sum_seconds,
            1e3 * sensor_to_command_seconds,
            1.0 if result.clear_track_recovery else 0.0,
            1e3 * odom_to_command_seconds,
            1e3 * full_cycle_seconds,
            (-1.0 if grip_limit is None else float(grip_limit)),
            (-1.0 if self._grip_limiter is None
             or not np.isfinite(self._grip_limiter.response_ratio)
             else float(self._grip_limiter.response_ratio)),
        ]
        self._compute_breakdown_publisher.publish(breakdown)
        if pitwall is not None:
            pitwall.log('speed', state.speed)
            pitwall.log('target_speed', result.target_speed)
            pitwall.log('steer', result.steering)
            pitwall.log('compute_ms', 1e3 * command_seconds)
            # -1 marks "no obstacle governs the speed" for the panel.
            pitwall.log('follow_cap',
                        -1.0 if follow_cap is None else follow_cap)
            pitwall.log('obstacles',
                        0 if obstacles is None else int(obstacles.shape[0]))
            if self._grip_limiter is not None:
                pitwall.log('grip_limit', self._grip_limiter.current_limit)
                ratio = self._grip_limiter.response_ratio
                pitwall.log('grip_response_ratio',
                            -1.0 if not np.isfinite(ratio) else ratio)
        self._report_rate(command_seconds, stage_sum_seconds)

    def _report_rate(self, compute_seconds: float,
                     stage_sum_seconds: float) -> None:
        """Accumulate one control-cycle compute time; print the rate periodically."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._rate_window_start is None:
            self._rate_window_start = now
        self._rate_count += 1
        self._compute_sum += compute_seconds
        self._compute_max = max(self._compute_max, compute_seconds)
        self._stage_sum += stage_sum_seconds
        self._stage_max = max(self._stage_max, stage_sum_seconds)
        window = now - self._rate_window_start
        if window >= self._rate_report_period and self._rate_count > 0:
            target = 1.0 / max(float(self.get_parameter('control_period').value), 1e-6)
            text = ('MPC {:.1f} Hz (target {:.0f}) | critical avg {:.1f} ms, '
                    'max {:.1f} ms | stage-sum avg {:.1f} ms, max {:.1f} ms').format(
                self._rate_count / window,
                target,
                1e3 * self._compute_sum / self._rate_count,
                1e3 * self._compute_max,
                1e3 * self._stage_sum / self._rate_count,
                1e3 * self._stage_max,
            )
            if self._grip_observer is not None:
                grip = self._grip_observer.report()
                if grip is not None:
                    text += ' | {} (cap {:.1f})'.format(
                        grip, (self._grip_limiter.current_limit
                               if self._grip_limiter is not None
                               else self.config.max_lateral_accel))
            if self._grip_limiter is not None:
                ratio = self._grip_limiter.response_ratio
                ratio_text = ('n/a' if not np.isfinite(ratio)
                              else '{:.2f}'.format(ratio))
                text += ' | live grip {:.2f} m/s^2, yaw response {}{}'.format(
                    self._grip_limiter.current_limit,
                    ratio_text,
                    ' SLIP' if self._grip_limiter.slip_active else '')
            # Scan/pose pairing mix.  Reported because this is the failure
            # that hides: a stamping regression shows up here as the balance
            # tipping to held/extrapolated, long before anyone connects it to
            # phantom obstacles in the corners.
            paired = sum(self._pose_pairing.values())
            if paired:
                text += ' | scan pose {:.0f}% interp, {:.0f}% extrap, ' \
                        '{:.0f}% held'.format(
                            *(100.0 * self._pose_pairing[key] / paired
                              for key in (OdometryHistory.INTERPOLATED,
                                          OdometryHistory.EXTRAPOLATED,
                                          OdometryHistory.HELD)))
                for key in self._pose_pairing:
                    self._pose_pairing[key] = 0
            self.get_logger().info(text)
            self._rate_window_start = now
            self._rate_count = 0
            self._compute_sum = 0.0
            self._compute_max = 0.0
            self._stage_sum = 0.0
            self._stage_max = 0.0

    def start_control_loop(self) -> None:
        """Start visualization and, when needed, periodic control threads."""
        self._visualization_thread = threading.Thread(
            target=self._visualization_loop, name='mpc_visualization',
            daemon=True)
        self._visualization_thread.start()
        if self._control_on_odometry:
            return
        self._control_thread = threading.Thread(
            target=self._control_loop, name='mpc_control', daemon=True)
        self._control_thread.start()

    def stop_control_loop(self) -> None:
        self._shutdown.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None
        if self._visualization_thread is not None:
            self._visualization_thread.join(timeout=2.0)
            self._visualization_thread = None

    def _control_loop(self) -> None:
        period = float(self.get_parameter('control_period').value)
        deadline = time.monotonic()
        while not self._shutdown.is_set() and rclpy.ok():
            self._run_control_cycle()
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                self._shutdown.wait(remaining)
            else:
                # Overran the period: restart pacing instead of bursting to
                # catch up, so command spacing stays even.
                deadline = time.monotonic()

    def _run_control_cycle(self) -> None:
        """Apply pending editor state and run one fail-safe control cycle."""
        self._apply_pending_raceline()
        try:
            self._control_callback()
        except Exception as error:  # pragma: no cover - defensive
            self.get_logger().error(
                'Control cycle failed: {}'.format(error))
            try:
                self._set_pit_state('CONTROL_FAIL')
            except Exception:
                pass
            try:
                self._publish_stop()
            except Exception:
                pass

    def _apply_pending_raceline(self) -> None:
        pending = self._pending_raceline
        if pending is None:
            return
        self._pending_raceline = None
        self.raceline, self.planner = pending
        if self._tracker is not None:
            with self._tracker_lock:
                self._tracker.set_path(self.raceline)
                # The previous scan used the previous corridor filter.  Wait
                # for one scan through the new mask instead of mixing epochs.
                self._obstacle_snapshot = None
        self._last_path_s = None
        # The s coordinate just changed meaning; drop the running lap.
        self._lap_start_time = None
        self._reset_mpc_warm_start()

    def _reset_mpc_warm_start(self) -> None:
        """Reset CEM history from the steering angle held at the actuator."""
        self.controller.reset(np.array(
            [self._last_steer_command, 0.0], dtype=float))

    def _reset_controllers(self) -> None:
        self._reset_mpc_warm_start()
        self.planner.reset()
        if self._tracker is not None:
            with self._tracker_lock:
                self._tracker.reset()
                self._obstacle_snapshot = None

    # Body rectangle used to judge a hand geometry edit.  This is the physical
    # car, not the solver's two-disc collision proxy: the discs are 0.44 m
    # wide against a real 0.31 m body, which would refuse edits on the line
    # the car already drives.
    _EDITOR_BODY_LENGTH = 0.58
    _EDITOR_BODY_WIDTH = 0.31
    _EDITOR_BODY_SAMPLE = 0.02

    def _editor_speed_profile(self, data: np.ndarray) -> np.ndarray:
        """Reference speed for hand-moved geometry, from the node's own limits.

        Routed through Raceline.speed_at - the same cached profile the
        reference sampler uses - so an edited line cannot promise a speed the
        controller would refuse a moment later.  speed_at resamples the
        internal profile grid back onto the stations; the profile itself is
        not station-indexed.
        """
        line = Raceline(data)
        return np.asarray(line.speed_at(data[:, 0], self.config), dtype=float)

    def _editor_body_clearance(self, x, y, heading) -> np.ndarray:
        """Swept-body clearance, sampled off the wall distance field.

        Distance from a set to the walls is the minimum over its points, and
        the field is 1-Lipschitz, so a grid at `_EDITOR_BODY_SAMPLE` spacing
        overestimates by at most half that; half a sample is subtracted back
        off so the guard errs towards refusing.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        heading = np.atleast_1d(np.asarray(heading, dtype=float))
        half_length = 0.5 * self._EDITOR_BODY_LENGTH
        half_width = 0.5 * self._EDITOR_BODY_WIDTH
        step = self._EDITOR_BODY_SAMPLE
        along = np.linspace(-half_length, half_length,
                            max(2, int(round(2 * half_length / step)) + 1))
        across = np.linspace(-half_width, half_width,
                             max(2, int(round(2 * half_width / step)) + 1))
        body_x, body_y = (grid.ravel() for grid in np.meshgrid(along, across))
        # base_link is the rear axle on the car, so the footprint centre sits
        # ego_center_offset ahead of the station.
        centre = self.config.ego_center_offset + body_x[None, :]
        cos_h = np.cos(heading)[:, None]
        sin_h = np.sin(heading)[:, None]
        sample_x = x[:, None] + centre * cos_h - body_y[None, :] * sin_h
        sample_y = y[:, None] + centre * sin_h + body_y[None, :] * cos_h
        field = self._wall_field.clearance(sample_x.ravel(), sample_y.ravel())
        return np.maximum(
            field.reshape(sample_x.shape).min(axis=1) - 0.5 * step, 0.0)

    def _replace_raceline(self, raceline: Raceline) -> None:
        """Hand a speed-edited raceline to the control thread atomically.

        Runs on the executor thread (editor callback).  The planner is built
        here so the control thread only performs reference assignments.
        """
        planner = AvoidancePlanner(raceline, self.config, self._wall_field)
        self._pending_raceline = (raceline, planner)
        self._publish_raceline(raceline)

    def _on_set_parameters(self, parameters) -> SetParametersResult:
        """Apply ros2 param / rqt_reconfigure updates to the running MPC.

        Cost weights, speed caps, and avoidance gains land in the shared
        MPCConfig, which every solve re-reads; lane-geometry fields also
        rebuild the avoidance planner off the control thread.  Sizing
        fields (samples, horizon, ...) still require a restart.
        """
        config_fields = MPCConfig.__dataclass_fields__
        candidate = None
        rebuild_planner = False
        for parameter in parameters:
            name = parameter.name
            if name in self._RESTART_ONLY_PARAMETERS:
                if parameter.value != self.get_parameter(name).value:
                    return SetParametersResult(
                        successful=False,
                        reason='{} requires a restart'.format(name))
                continue
            if name in config_fields:
                if candidate is None:
                    candidate = dataclasses.replace(self.config)
                setattr(candidate, name,
                        type(getattr(self.config, name))(parameter.value))
                if name in self._PLANNER_REBUILD_PARAMETERS:
                    rebuild_planner = True
                continue
            # Everything else (topics, editor, IMU flags, ...) keeps its
            # declare-time behavior; several are already re-read per cycle.
        if candidate is not None:
            try:
                candidate.validate()
            except ValueError as error:
                return SetParametersResult(
                    successful=False, reason=str(error))
            if (self.config.reference_speed_caps
                    and not candidate.reference_speed_caps):
                self.get_logger().warn(
                    'reference_speed_caps disabled live: reference now '
                    'follows the raceline/editor speeds as-is')
            # Mutate the shared config in place: the solver, planner, and
            # editor all hold this object and re-read it every cycle.
            for name in config_fields:
                setattr(self.config, name, getattr(candidate, name))
        if rebuild_planner:
            pending = self._pending_raceline
            raceline = pending[0] if pending is not None else self.raceline
            self._pending_raceline = (
                raceline,
                AvoidancePlanner(raceline, self.config, self._wall_field))
        return SetParametersResult(successful=True)

    def _deferred_parameter_tasks(self) -> None:
        """Executor-side follow-ups a set-parameters callback cannot do."""
        # (Currently none: the sector-scaling save_params machinery this
        # handled was removed 2026-08-24.  The timer stays as the hook point.)

    def _publish_stop(self) -> None:
        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = 'ego_racecar/base_link'
        # Hold the last autonomous rack angle while stopping.  Zeroing this
        # field made the servo snap to centre and also defeated the VESC
        # converter's steering-aware regen derate exactly when an emergency
        # stop occurred in a corner.
        command.drive.steering_angle = self._last_steer_command
        command.drive.speed = 0.0
        command.drive.acceleration = self.config.min_accel
        self._drive_publisher.publish(command)
        self._last_speed_command = 0.0

    def _publish_raceline(self, raceline: Optional[Raceline] = None) -> None:
        raceline = self.raceline if raceline is None else raceline
        message = PathMessage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.get_parameter('frame_id').value
        for x, y, yaw in zip(raceline.x, raceline.y, raceline.heading):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.z = math.sin(0.5 * float(yaw))
            pose.pose.orientation.w = math.cos(0.5 * float(yaw))
            message.poses.append(pose)
        self._raceline_publisher.publish(message)

    def _publish_prediction(self, states: np.ndarray) -> None:
        message = PathMessage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.get_parameter('frame_id').value
        for state in states:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(state[0])
            pose.pose.position.y = float(state[1])
            pose.pose.orientation.z = math.sin(0.5 * float(state[2]))
            pose.pose.orientation.w = math.cos(0.5 * float(state[2]))
            message.poses.append(pose)
        self._prediction_publisher.publish(message)

    def _queue_visualization(self, states: np.ndarray,
                             obstacles: Optional[np.ndarray]) -> None:
        """Hand the latest numeric result to the independent RViz loop."""
        prediction_wanted = self._prediction_publisher.get_subscription_count() > 0
        obstacles_wanted = self._obstacle_publisher.get_subscription_count() > 0
        if not prediction_wanted and not obstacles_wanted:
            return
        prediction = np.asarray(states, dtype=float).copy() \
            if prediction_wanted else None
        obstacle_snapshot = None
        if obstacles_wanted and obstacles is not None:
            obstacle_snapshot = np.asarray(obstacles, dtype=float).copy()
        with self._viz_lock:
            # A slow renderer never queues old frames: only the newest MPC
            # result matters to the driver watching RViz2.
            self._viz_snapshot = (prediction, obstacle_snapshot)

    def _visualization_loop(self) -> None:
        period = max(
            float(self.get_parameter('visualization_period').value), 0.02)
        deadline = time.monotonic()
        while not self._shutdown.is_set() and rclpy.ok():
            with self._viz_lock:
                snapshot = self._viz_snapshot
                self._viz_snapshot = None
            if snapshot is not None:
                prediction, obstacles = snapshot
                try:
                    if (prediction is not None
                            and self._prediction_publisher.get_subscription_count() > 0):
                        self._publish_prediction(prediction)
                    if self._obstacle_publisher.get_subscription_count() > 0:
                        self._publish_obstacles(obstacles)
                except RuntimeError:
                    if rclpy.ok():
                        raise
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                self._shutdown.wait(remaining)
            else:
                deadline = time.monotonic()

    def _publish_obstacles(self, obstacles: Optional[np.ndarray]) -> None:
        count = 0 if obstacles is None else obstacles.shape[0]
        # Markers expire on their own (0.3 s lifetime); publishing an empty
        # array every idle cycle is pure overhead.
        if count == 0 and self._last_obstacle_count == 0:
            return
        self._last_obstacle_count = count
        message = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        frame = self.get_parameter('frame_id').value
        for index in range(count):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame
            marker.ns = 'obstacles'
            marker.id = index
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(obstacles[index, 0])
            marker.pose.position.y = float(obstacles[index, 1])
            marker.pose.position.z = 0.15
            diameter = 2.0 * float(obstacles[index, 4])
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = 0.3
            moving = math.hypot(obstacles[index, 2], obstacles[index, 3]) > 0.1
            marker.color.r = 1.0 if moving else 1.0
            marker.color.g = 0.5 if moving else 0.1
            marker.color.b = 0.0 if moving else 0.1
            marker.color.a = 0.8
            marker.lifetime.nanosec = 300000000
            message.markers.append(marker)
        self._obstacle_publisher.publish(message)

    def _record_lap_crossing(self, stamp: float) -> None:
        """Time laps at forward start/finish crossings for the pitwall panel."""
        if self._lap_start_time is None:
            self._lap_start_time = stamp
            self._lap_clean = True
            if pitwall is not None:
                pitwall.event('lap timing armed at start/finish')
            return
        lap_time = stamp - self._lap_start_time
        if lap_time < 5.0:
            # Localization jitter around the line (e.g. holding position near
            # s=0) re-crosses within seconds; that is not a lap.  Keep the
            # original lap start.
            return
        self._lap_count += 1
        clean = self._lap_clean
        if clean and (self._best_lap_time is None
                      or lap_time < self._best_lap_time):
            self._best_lap_time = lap_time
        suffix = '' if clean else ' (interrupted)'
        if self._best_lap_time is not None:
            suffix += ' | best {:.2f} s'.format(self._best_lap_time)
        self.get_logger().info(
            'Lap {}: {:.2f} s{}'.format(self._lap_count, lap_time, suffix))
        if pitwall is not None:
            pitwall.log('lap_time', lap_time)
            if self._best_lap_time is not None:
                pitwall.log('best_lap', self._best_lap_time)
            pitwall.event('lap {}: {:.2f} s{}'.format(
                self._lap_count, lap_time, suffix))
        self._lap_start_time = stamp
        self._lap_clean = True

    def _set_pit_state(self, state: str) -> None:
        """Publish the control-loop state for the pitwall RViz panel."""
        if state == self._pit_state:
            return
        self._pit_state = state
        if state != 'AUTO':
            # Any interruption (manual, stops, escapes, edits) taints the
            # running lap; it still gets timed but never sets the best.
            self._lap_clean = False
        if state != 'AUTO' and self._lap_crossing_pending is not None:
            crossing_stamp = self._lap_crossing_pending
            self._lap_crossing_pending = None
            try:
                self._record_lap_crossing(crossing_stamp)
            finally:
                # The interruption belongs to both sides of the boundary:
                # the just-finished lap and the newly started one.
                self._lap_clean = False
        message = StringMessage()
        message.data = state
        self._pit_state_publisher.publish(message)
        if pitwall is not None:
            pitwall.event('state -> {}'.format(state))

    def _warn_throttled(self, text: str) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_warning_time >= 2.0:
            self.get_logger().warn(text)
            self._last_warning_time = now

    def stop(self) -> None:
        """Send an explicit stop before node shutdown."""
        self.stop_control_loop()
        if self._raceline_editor is not None:
            self._raceline_editor.shutdown()
        # Do not inject even a zero autonomous command while manual control is
        # selected; the joystick/mux owns the actuator in that mode.
        now = self.get_clock().now().nanoseconds * 1e-9
        if not self._manual_gate.manual_active(now):
            try:
                self._publish_stop()
            except Exception:
                # rclpy context may already be torn down on SIGTERM shutdown.
                pass


def main(args=None):
    rclpy.init(args=args)
    node = MPCControllerNode()
    # Vehicle control executes directly on its native-rate odometry callback;
    # simulation's periodic mode uses the node's own thread.  Both use a
    # single executor for deterministic sensor ordering.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    node.start_control_loop()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # launch sends SIGINT to every process at once. Humble can invalidate
        # the rclpy context while a subscription callback is being taken;
        # that is a normal shutdown race, not a controller failure.
        if rclpy.ok():
            raise
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
