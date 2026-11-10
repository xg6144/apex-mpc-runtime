"""ROS-independent raceline and nonlinear MPC implementation.

The controller uses the cross-entropy method (CEM) to optimize a bounded
steering/acceleration sequence over a kinematic bicycle model.  Keeping this
module independent of ROS makes both tuning and regression testing easy.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import mpc_accel


def wrap_angle(angle):
    """Wrap a scalar or array of angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class VehicleState:
    """Planar kinematic bicycle state."""

    x: float
    y: float
    yaw: float
    speed: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.speed], dtype=float)


@dataclass
class MPCConfig:
    """Numerical, vehicle, constraint, and cost settings for the MPC."""

    horizon: int = 24
    dt: float = 0.05
    samples: int = 256
    iterations: int = 3
    elite_fraction: float = 0.10
    wheelbase: float = 0.3302
    # ROS/VESC steering commands are not necessarily wheel angles.  Keep the
    # measured wheelbase physical and model the calibrated rack authority
    # explicitly: wheel_steer = steering_gain * commanded_steer.
    steering_gain: float = 1.0
    # First-order command-to-wheel response used by every rollout.  Zero keeps
    # the ideal instantaneous actuator used by the simulator/default tests.
    steering_time_constant: float = 0.0
    min_steer: float = -0.48
    max_steer: float = 0.51
    max_steer_rate: float = 3.20
    min_accel: float = -5.0
    max_accel: float = 3.5
    min_speed: float = 0.0
    max_speed: float = 5.0
    speed_scale: float = 0.40
    minimum_reference_speed: float = 0.8
    max_lateral_accel: float = 7.0
    # When False the reference honors the (sector-scaled) raceline profile
    # as-is: no curvature/steering-reach caps and no minimum_reference_speed
    # floor, only the solver's [0, max_speed] bound.  Free-tuning mode - the
    # profile itself must then be physically sensible.
    reference_speed_caps: bool = True
    # Global longitudinal feasibility of the speed reference (applied only
    # while reference_speed_caps is true).  The pointwise curvature/steering
    # caps alone leave discontinuous speed drops at corner entries; at
    # speed_scale 1.3+ the implied braking exceeds the measured +-3.8 m/s^2
    # capability and the excess is carried INTO the corner (bags
    # mpc_20260824_0206xx/0217xx: in-corner speeds 36% over the lateral cap,
    # repeated wall contacts while running wide).  A circular backward pass
    # limits reference deceleration to reference_brake_accel and a forward
    # pass limits its acceleration demand to reference_accel_limit, so
    # braking starts automatically early enough at ANY speed scale and
    # corner-exit references stop demanding more thrust than the drivetrain
    # has.  0 disables a pass.
    reference_brake_accel: float = 3.5
    reference_accel_limit: float = 3.5
    contour_weight: float = 18.0
    lag_weight: float = 2.0
    heading_weight: float = 4.0
    speed_weight: float = 2.0
    terminal_weight: float = 3.0
    steer_feedforward_weight: float = 0.7
    accel_weight: float = 0.06
    steer_rate_weight: float = 2.0
    accel_rate_weight: float = 0.03
    initial_steer_std: float = 0.17
    initial_accel_std: float = 1.5
    minimum_steer_std: float = 0.012
    minimum_accel_std: float = 0.12
    random_seed: int = 7
    # Research ablations.  Both default to the deployed snapshot behaviour;
    # experiment harnesses may disable them without maintaining a second
    # controller implementation.  ``warm_start_mode='configured'`` preserves
    # the historical fractional_plan_transport switch.  The explicit modes
    # support reviewer-facing controls without changing the production
    # default: whole_cell, fractional, phase_accumulated, and cold_start.
    fractional_plan_transport: bool = True
    warm_start_mode: str = 'configured'
    deterministic_progress_anchor: bool = True
    # The historical anchor switch bundled three distinct interventions.
    # Keep it as a master switch for backward compatibility, while exposing
    # each component for a factorial research ablation.
    progress_mean_refresh: bool = True
    # Fraction of the deterministic feed-forward sequence mixed into the
    # transported CEM mean when progress_mean_refresh is enabled.  Exposing
    # the historical 0.20 value makes the intervention tunable and permits a
    # reviewer-facing sensitivity analysis without forking the controller.
    progress_mean_refresh_fraction: float = 0.20
    # Research-only source-provenance control.  The first frozen study applied
    # actuator constraints only after blending the transported mean with the
    # feasible feed-forward sequence; the later source constrained both before
    # and after the blend.  Keep the current behaviour as the default while
    # allowing an explicit common-seed crossover of the two operators.
    progress_refresh_constraint_order: str = 'pre_and_post'
    progress_candidate_insertion: bool = True
    clear_track_fallback: bool = True
    # Budget-matched iCEM-style temporal sample reuse comparator.  A positive
    # fraction selects that fraction of the cost-sorted final elite set, shifts
    # each by one whole prediction cell, samples a new tail action, and replaces
    # the same number of stochastic candidates in the next first population.
    # Full iCEM appends these samples; replacement here isolates elite memory
    # without increasing the trajectory-evaluation budget.
    shifted_elite_reuse_fraction: float = 0.0
    # Explicit integer dose for reviewer-facing K sweeps.  A positive count is
    # mutually exclusive with the legacy fraction above and is capped by the
    # stored final elite count at run time.
    shifted_elite_reuse_count: int = 0
    # Full iCEM comparator settings following Pinneri et al.  The production
    # controller remains ``standard``; experiments opt into colored noise,
    # population decay, within-iteration elite retention, temporal elite
    # shifting, and momentum updates together.
    cem_variant: str = 'standard'
    icem_population_decay: float = 1.25
    icem_elite_count: int = 10
    icem_reuse_fraction: float = 0.30
    icem_noise_beta: float = 3.0
    icem_momentum: float = 0.10
    # The configured iCEM package historically executed the lowest-cost
    # evaluated candidate while transporting the fitted mean.  This switch
    # isolates that execution rule without adding an objective evaluation.
    icem_execution_rule: str = 'best'
    # The simulator realizes speed commands through a proportional controller,
    # so command the speed the plan reaches this many seconds ahead.
    speed_command_preview: float = 0.20
    # Expected odometry-to-actuation delay compensated before each solve.
    latency_compensation: float = 0.02
    # A time-indexed MPC can settle into the locally cheap "park and avoid
    # contour error" solution after it falls behind a tight corner.  On a
    # clear track, admit a deterministic Pure-Pursuit/feed-forward candidate
    # and keep its speed above this floor.  Obstacles/follow caps bypass this
    # path completely.
    clear_track_speed_floor: float = 0.0
    clear_track_lookahead: float = 0.70
    # Ego footprint as two discs centred this far ahead/behind the footprint
    # centre.  The centre itself sits ego_center_offset ahead of the pose:
    # 0.0 when the pose is the body centre (the simulator's CoG pose),
    # wheelbase/2 (~0.16) on the real car where base_link is the REAR AXLE -
    # without that shift the nose corner sits ~8 cm outside the collision
    # model and grazes obstacles during avoidance.
    ego_center_offset: float = 0.0
    ego_disc_offset: float = 0.145
    ego_disc_radius: float = 0.22
    # Obstacle avoidance cost shaping.  The collision weights are effectively
    # hard constraints: any rollout touching the hard boundary must always
    # lose against every rollout that stays clear.
    obstacle_weight: float = 200.0
    obstacle_margin: float = 0.20
    obstacle_collision_weight: float = 50000.0
    # Wall clearance cost shaping (uses the map distance field when set).
    wall_weight: float = 120.0
    wall_margin: float = 0.15
    # Extra distance outside the conservative ego discs treated as a hard
    # boundary.  The measured real body already sits ~4.8 cm inside these
    # discs, so a zero extra margin represents a physical ~5 cm wall gap.
    wall_hard_margin: float = 0.05
    wall_collision_weight: float = 50000.0
    # Avoidance reference planner.
    avoid_max_offset: float = 0.70
    avoid_offset_step: float = 0.05
    # Search a sparse lateral grid first, then expand only the best regions
    # at avoid_offset_step resolution.  This retains the fine collision grid
    # without rebuilding every fine candidate on every replan.
    avoid_coarse_offset_step: float = 0.15
    avoid_refine_top_k: int = 2
    # Solve the evasion offset as a C2 profile (the vendored advanced_pp QP:
    # free-space tube, obstacle keyhole, exact steering-lock box) instead of
    # picking one constant offset out of the fan above.  The fan stays as the
    # fallback for movers and for any cycle the solver declines.
    avoid_profile_solver: bool = False
    # A scene keeps its selected lane briefly.  A materially changed obstacle
    # geometry invalidates it immediately and a periodic full replan bounds
    # how long vehicle progress can reuse the local choice.
    avoid_replan_period: float = 0.0
    avoid_lateral_slew: float = 1.5
    # Minimum longitudinal distance over which an offset change is blended.
    # The old time-step ramp could demand more lateral than longitudinal motion
    # in a slow corner, producing a nearly sideways, impossible reference.
    avoid_transition_distance: float = 2.5
    avoid_comfort_clearance: float = 0.40
    avoid_min_speed_factor: float = 0.45
    # Post-avoidance recovery governor.  For recovery_hold_time seconds after
    # the last cycle with relevant obstacles, a vehicle still displaced more
    # than recovery_error_deadband laterally from its reference keeps a
    # surge-limited speed reference (current speed + recovery_surge_accel*t)
    # instead of jumping straight back to the full profile.  Otherwise the
    # profile speeds consume the whole lateral-accel budget on a straight and
    # the grip-barrier cost leaves no curvature authority to close the offset:
    # bag mpc_20260824_014406 held a -0.7 m offset for 5 m after passing a box
    # and drove into the wall at 5.4 m/s while accelerating.  Zero on a clean
    # lap: the timer only arms while obstacles are actually relevant.
    recovery_hold_time: float = 3.0
    recovery_error_deadband: float = 0.30
    recovery_surge_accel: float = 2.5
    # Lap-persistent avoidance-lane memory.  Once a static obstacle has been
    # passed on a hard-safe lane, the chosen lateral offset is remembered in
    # track coordinates (s, lateral, radius).  On the next approach, if a
    # static obstacle matches within the tolerances below, the remembered
    # lane is re-validated against the CURRENT scene and, when it still
    # passes every hard gate, adopted directly - the car repeats the exact
    # avoidance path lap after lap instead of re-deriving a possibly
    # different one from noisy tracker geometry.  A failed re-validation
    # falls back to the normal lane search, whose new outcome overwrites the
    # memory.  Moving obstacles are never memorized or matched.
    lane_memory_enabled: bool = True
    lane_memory_s_tolerance: float = 0.60
    lane_memory_lateral_tolerance: float = 0.30
    lane_memory_radius_tolerance: float = 0.15
    lane_memory_size: int = 12
    # Minimum combined free clearance left around the conservative footprint
    # (obstacle-side gap + wall-side gap) for a collision-free lane to run
    # without the FOLLOW governor or proximity speed factor.  The requested
    # 0.35 m therefore means a raw aperture of at least footprint width 0.44 m
    # + 0.35 m, rather than a redundant raw-aperture test below car width.
    avoid_full_speed_gap: float = 0.35
    # Executable-path validation already checks both footprint discs, the
    # wall field and the steering rack.  Enter on the first fully safe plan so
    # obstacle detection itself cannot inject a one-cycle brake command;
    # unsafe geometry still revokes the pass immediately.
    avoid_full_speed_confirm_cycles: int = 1
    # Follow governor: while no passing lane clears an obstacle ahead, cap the
    # reference speed to its track speed plus this gain times the gap beyond
    # the minimum following distance.
    follow_distance: float = 1.2
    follow_gain: float = 2.5
    # Extra lateral clearance (beyond just-touching) over which the follow
    # speed cap blends from held to full speed.  On tight tracks the achievable
    # side offset is small, so a large value pins the car to a crawl even once
    # it is geometrically past the obstacle; lower it to pass tight gaps at
    # speed (the solver's hard clearance cost still prevents contact).
    follow_clear_margin: float = 0.40
    # Following a moving obstacle was reworked on 2026-08-26 and ROLLED BACK
    # the same day at the driver's call.  The qualifying bag's four collision
    # stops were all the other car, and three changes were measured against
    # that geometry (tools/obstacle_sim.py --opponent, 16 scenarios of the ego
    # catching a 0.30-0.40x car on the main straight): contacts 5/16 -> 2/16.
    #   * a braking-feasible cap, sqrt(along^2 + 2a(gap-hold)) in place of the
    #     linear law below, which only binds under 2*brake/gain = 3.04 m/s;
    #   * refusing a full-speed pass over a mover no lane could clear;
    #   * a guard forbidding any speed increase inside 1.5 m of a tracked
    #     object (the ego ran 0.7-1.0 m behind for 2.5 s and ACCELERATED
    #     2.59 -> 3.85 m/s into it).
    # A fourth idea, padding the lane-blocked requirement for movers, was
    # measured and rejected before the rollback: it changed nothing (2/16 at
    # pad 0, 0.10 and 0.25 alike) and cost 1.6 s a lap queued behind a slower
    # car.  None of it was ever driven on the car.
    # Penalty on predicted lateral acceleration beyond max_lateral_accel: the
    # kinematic rollout has no friction limit, so infeasible plans must cost.
    friction_weight: float = 30.0
    # Linear near-hard barrier paired with the quadratic friction shaping.
    # Kept zero by default for backwards-compatible/simulator behavior; the
    # real-car profile enables it with the adaptive runtime grip limit.
    grip_hard_weight: float = 0.0
    # Single-track tire slip reduces yaw response at speed: the effective
    # curvature is tan(wheel_steer) /
    # (wheelbase + understeer_gradient * v^2).
    # Identified from the simulator at the jeju1 hairpin (saturated steering
    # achieved kappa 1.12 at 2.5 m/s instead of the kinematic 1.35).
    understeer_gradient: float = 0.011

    # Wall-clock spacing of control cycles.  The model step stays dt; this
    # only scales per-cycle planner dynamics (lateral slew, stall timing) so
    # their physical rates do not change when the control rate does.
    control_period: float = 0.04

    @property
    def reference_steer_limit(self) -> float:
        """Guaranteed steering magnitude when no turn direction is known."""
        return min(abs(self.min_steer), abs(self.max_steer))

    def reference_steer_limit_for(self, curvature):
        """Direction-aware steering magnitude for signed curvature."""
        curvature = np.asarray(curvature, dtype=float)
        limit = np.where(curvature >= 0.0,
                         self.max_steer, abs(self.min_steer))
        if limit.ndim == 0:
            return float(limit)
        return limit

    def steering_response_fraction(self, duration: float) -> float:
        """Exact first-order actuator response over ``duration`` seconds."""
        if self.steering_time_constant <= 0.0:
            return 1.0
        return float(-np.expm1(-max(float(duration), 0.0)
                               / self.steering_time_constant))

    def steering_step_coefficients(
            self, duration: float) -> Tuple[float, float, float, float]:
        """Branchless end/average wheel-angle recurrence coefficients.

        Returns coefficients ``r, c, ar, ac`` such that one held command is
        integrated exactly as ``end = r*start + c*command`` and
        ``average = ar*start + ac*command``.  Precomputing these four scalars
        keeps the real-car lag model out of the cost kernel's inner branch.
        """
        duration = max(float(duration), 0.0)
        if self.steering_time_constant <= 0.0 or duration <= 0.0:
            return 0.0, self.steering_gain, 0.0, self.steering_gain
        fraction = self.steering_response_fraction(duration)
        retention = 1.0 - fraction
        average_retention = (
            self.steering_time_constant * fraction / duration)
        return (
            retention,
            fraction * self.steering_gain,
            average_retention,
            (1.0 - average_retention) * self.steering_gain,
        )

    def validate(self) -> None:
        """Raise ValueError for settings that cannot produce an MPC problem."""
        if self.horizon < 2:
            raise ValueError('horizon must be at least 2')
        if self.dt <= 0.0:
            raise ValueError('dt must be positive')
        if self.control_period <= 0.0:
            raise ValueError('control_period must be positive')
        if self.warm_start_mode not in {
                'configured', 'whole_cell', 'fractional',
                'phase_accumulated', 'cold_start'}:
            raise ValueError('warm_start_mode is invalid')
        if not 0.0 <= self.progress_mean_refresh_fraction <= 1.0:
            raise ValueError(
                'progress_mean_refresh_fraction must be in [0, 1]')
        if self.progress_refresh_constraint_order not in {
                'pre_and_post', 'post_blend_only'}:
            raise ValueError(
                'progress_refresh_constraint_order is invalid')
        if not 0.0 <= self.shifted_elite_reuse_fraction < 1.0:
            raise ValueError(
                'shifted_elite_reuse_fraction must be in [0, 1)')
        if (not isinstance(self.shifted_elite_reuse_count, int)
                or isinstance(self.shifted_elite_reuse_count, bool)
                or self.shifted_elite_reuse_count < 0):
            raise ValueError(
                'shifted_elite_reuse_count must be a non-negative integer')
        if (self.shifted_elite_reuse_fraction > 0.0
                and self.shifted_elite_reuse_count > 0):
            raise ValueError(
                'shifted elite reuse fraction and count are mutually exclusive')
        if ((self.shifted_elite_reuse_fraction > 0.0
                or self.shifted_elite_reuse_count > 0)
                and self.warm_start_mode != 'whole_cell'):
            raise ValueError(
                'shifted elite reuse requires whole_cell warm_start_mode')
        if self.cem_variant not in {'standard', 'icem'}:
            raise ValueError('cem_variant must be standard or icem')
        if self.cem_variant == 'icem' and self.warm_start_mode != 'whole_cell':
            raise ValueError('iCEM comparator requires whole_cell warm_start_mode')
        if self.icem_population_decay < 1.0:
            raise ValueError('icem_population_decay must be at least one')
        if self.icem_elite_count < 2:
            raise ValueError('icem_elite_count must be at least two')
        if not 0.0 <= self.icem_reuse_fraction <= 1.0:
            raise ValueError('icem_reuse_fraction must be in [0, 1]')
        if self.icem_noise_beta < 0.0:
            raise ValueError('icem_noise_beta must be non-negative')
        if not 0.0 <= self.icem_momentum <= 1.0:
            raise ValueError('icem_momentum must be in [0, 1]')
        if self.icem_execution_rule not in {'best', 'fitted_mean'}:
            raise ValueError('icem_execution_rule must be best or fitted_mean')
        if (self.cem_variant != 'icem'
                and self.icem_execution_rule != 'best'):
            raise ValueError(
                'fitted_mean execution is defined only for the iCEM comparator')
        if self.samples < 8:
            raise ValueError('samples must be at least 8')
        if self.iterations < 1:
            raise ValueError('iterations must be positive')
        if not 0.0 < self.elite_fraction < 1.0:
            raise ValueError('elite_fraction must be between zero and one')
        if self.wheelbase <= 0.0:
            raise ValueError('wheelbase must be positive')
        if self.steering_gain <= 0.0:
            raise ValueError('steering_gain must be positive')
        if self.steering_time_constant < 0.0:
            raise ValueError('steering_time_constant must be non-negative')
        if self.min_steer >= self.max_steer:
            raise ValueError('min_steer must be smaller than max_steer')
        if self.min_steer > 0.0 or self.max_steer < 0.0:
            raise ValueError('the steering range must contain zero')
        if (max(abs(self.min_steer), abs(self.max_steer))
                * self.steering_gain >= np.pi / 2.0):
            raise ValueError('physical steering range must stay below pi/2')
        if self.min_accel >= self.max_accel:
            raise ValueError('min_accel must be smaller than max_accel')
        if self.min_speed < 0.0 or self.min_speed >= self.max_speed:
            raise ValueError('speed bounds are invalid')
        if self.speed_command_preview < 0.0:
            raise ValueError('speed_command_preview must be non-negative')
        if self.latency_compensation < 0.0:
            raise ValueError('latency_compensation must be non-negative')
        if not 0.0 <= self.clear_track_speed_floor <= self.max_speed:
            raise ValueError('clear_track_speed_floor is outside speed bounds')
        if self.clear_track_lookahead <= 0.0:
            raise ValueError('clear_track_lookahead must be positive')
        if self.understeer_gradient < 0.0:
            raise ValueError('understeer_gradient must be non-negative')
        if self.max_lateral_accel <= 0.0:
            raise ValueError('max_lateral_accel must be positive')
        if self.friction_weight < 0.0 or self.grip_hard_weight < 0.0:
            raise ValueError('grip cost weights must be non-negative')
        if self.ego_disc_radius <= 0.0:
            raise ValueError('ego_disc_radius must be positive')
        if self.wall_margin < 0.0 or self.wall_hard_margin < 0.0:
            raise ValueError('wall margins must be non-negative')
        if self.avoid_max_offset < 0.0 or self.avoid_offset_step <= 0.0:
            raise ValueError('avoidance offset settings are invalid')
        if self.avoid_coarse_offset_step < self.avoid_offset_step:
            raise ValueError(
                'avoid_coarse_offset_step must be at least avoid_offset_step')
        if self.avoid_refine_top_k < 1:
            raise ValueError('avoid_refine_top_k must be positive')
        if self.avoid_replan_period < 0.0:
            raise ValueError('avoid_replan_period must be non-negative')
        if self.avoid_transition_distance <= 0.0:
            raise ValueError('avoid_transition_distance must be positive')
        if not 0.0 < self.avoid_min_speed_factor <= 1.0:
            raise ValueError('avoid_min_speed_factor must be in (0, 1]')
        if self.recovery_hold_time < 0.0:
            raise ValueError('recovery_hold_time must be non-negative')
        if self.recovery_error_deadband < 0.0:
            raise ValueError('recovery_error_deadband must be non-negative')
        if self.recovery_surge_accel <= 0.0:
            raise ValueError('recovery_surge_accel must be positive')
        if self.lane_memory_s_tolerance < 0.0 \
                or self.lane_memory_lateral_tolerance < 0.0 \
                or self.lane_memory_radius_tolerance < 0.0:
            raise ValueError('lane memory tolerances must be non-negative')
        if self.lane_memory_size < 1:
            raise ValueError('lane_memory_size must be positive')
        if self.reference_brake_accel < 0.0 \
                or self.reference_accel_limit < 0.0:
            raise ValueError(
                'reference brake/accel limits must be non-negative')
        if self.avoid_full_speed_gap <= 0.0:
            raise ValueError('avoid_full_speed_gap must be positive')
        if self.avoid_full_speed_confirm_cycles < 1:
            raise ValueError(
                'avoid_full_speed_confirm_cycles must be positive')


@dataclass(frozen=True)
class MPCResult:
    """First command and full optimal prediction returned by the MPC."""

    steering: float
    acceleration: float
    target_speed: float
    controls: np.ndarray
    predicted_states: np.ndarray
    cost: float
    clear_track_recovery: bool = False
    clear_track_recovery_attempted: bool = False
    warm_start_used: bool = False
    mean_feedforward_distance_pre_refresh: float = float('nan')
    mean_feedforward_distance_post_refresh: float = float('nan')
    mean_feedforward_distance_final: float = float('nan')
    mean_feedforward_distance_by_iteration: tuple = ()
    elite_spread_by_iteration: tuple = ()
    shifted_elites_inserted: int = 0
    population_evaluations_by_iteration: tuple = ()
    population_candidates_by_iteration: tuple = ()
    shifted_elite_selected: bool = False
    retained_elite_selected: bool = False
    cost_evaluates_controls: bool = True
    has_safe_trajectory: bool = True


class Raceline:
    """Closed-loop raceline with periodic interpolation and projection."""

    def __init__(self, data: np.ndarray):
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[1] < 7:
            raise ValueError('raceline must contain s, x, y, psi, kappa, v, a')
        if values.shape[0] < 4 or not np.all(np.isfinite(values[:, :7])):
            raise ValueError('raceline is too short or contains non-finite values')
        if np.any(np.diff(values[:, 0]) <= 0.0):
            raise ValueError('raceline distance values must be strictly increasing')

        closing_duplicate = np.linalg.norm(values[-1, 1:3] - values[0, 1:3]) < 1e-6
        if closing_duplicate:
            self.total_length = float(values[-1, 0])
            values = values[:-1]
        else:
            closing_gap = float(np.linalg.norm(values[-1, 1:3] - values[0, 1:3]))
            self.total_length = float(values[-1, 0] + closing_gap)

        if self.total_length <= values[-1, 0]:
            raise ValueError('raceline loop length is invalid')

        self.s = values[:, 0].copy()
        self.x = values[:, 1].copy()
        self.y = values[:, 2].copy()
        self.raw_heading = values[:, 3].copy()
        self.curvature = values[:, 4].copy()
        self.speed = values[:, 5].copy()
        self.acceleration = values[:, 6].copy()

        # Some trajectory exporters use a heading axis rotated from the map's
        # x-axis.  Estimate a single convention offset from the actual path
        # tangent.  This IFAC trajectory requires a +pi/2 correction.
        forward_heading = np.arctan2(
            np.roll(self.y, -1) - self.y,
            np.roll(self.x, -1) - self.x,
        )
        heading_error = wrap_angle(forward_heading - self.raw_heading)
        circular_mean = np.arctan2(
            np.mean(np.sin(heading_error)),
            np.mean(np.cos(heading_error)),
        )
        concentration = float(np.hypot(
            np.mean(np.sin(heading_error)),
            np.mean(np.cos(heading_error)),
        ))
        centered_error = circular_mean + wrap_angle(heading_error - circular_mean)
        detected_offset = float(np.median(centered_error))
        if concentration > 0.95 and abs(detected_offset) > 0.05:
            self.heading_offset = detected_offset
        else:
            self.heading_offset = 0.0
        self.heading = wrap_angle(self.raw_heading + self.heading_offset)

        # Sector speed scaling (Unicorn-style runtime tuning layer).  ``speed``
        # stays the raceline's own profile so the editor and CSV round-trips
        # keep base values; only the interpolant the reference reads is scaled.
        self.sector_scale = np.ones(self.speed.shape[0], dtype=float)
        # Longitudinally feasible reference profile cache (see
        # feasible_speed_profile).  The version bumps on sector re-scaling.
        self._profile_version = 0
        self._feasible_key = None
        self._feasible_interp = None

        self._interp_s = np.append(self.s, self.total_length)
        self._interp_x = np.append(self.x, self.x[0])
        self._interp_y = np.append(self.y, self.y[0])
        self._interp_sin_heading = np.append(np.sin(self.heading), np.sin(self.heading[0]))
        self._interp_cos_heading = np.append(np.cos(self.heading), np.cos(self.heading[0]))
        self._interp_curvature = np.append(self.curvature, self.curvature[0])
        self._interp_speed = np.append(self.speed, self.speed[0])
        self._interp_acceleration = np.append(self.acceleration, self.acceleration[0])

        next_x = np.roll(self.x, -1)
        next_y = np.roll(self.y, -1)
        self._segment_x = next_x - self.x
        self._segment_y = next_y - self.y
        self._segment_length_sq = self._segment_x ** 2 + self._segment_y ** 2
        if np.any(self._segment_length_sq < 1e-12):
            raise ValueError('raceline contains a zero-length segment')
        self._segment_ds = np.append(np.diff(self.s), self.total_length - self.s[-1])

    @classmethod
    def from_csv(cls, path) -> 'Raceline':
        """Load the semicolon-delimited trajectory format used by global_racetrajectory_optimization."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError('raceline file does not exist: {}'.format(source))
        return cls(np.loadtxt(str(source), delimiter=';', comments='#'))

    def set_sector_scale(self, scale) -> None:
        """Apply a per-waypoint multiplicative speed scale (None clears it).

        The scaled interpolant is swapped in a single assignment, so a control
        thread interpolating concurrently sees either the old or the new
        profile, never a mixture.
        """
        if scale is None:
            scale = np.ones(self.speed.shape[0], dtype=float)
        scale = np.asarray(scale, dtype=float)
        if scale.shape != self.speed.shape:
            raise ValueError('sector scale length must match the raceline')
        if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
            raise ValueError('sector scale must be finite and non-negative')
        self.sector_scale = scale.copy()
        scaled = self.speed * self.sector_scale
        self._interp_speed = np.append(scaled, scaled[0])
        self._profile_version += 1

    def feasible_speed_profile(self, config: MPCConfig,
                               lateral_accel_limit: Optional[float] = None
                               ) -> np.ndarray:
        """Per-waypoint reference speeds with global braking feasibility.

        Starts from the sector-scaled profile through the pointwise
        curvature/steering caps, optionally tightened by a runtime lateral
        limit, then runs a circular backward pass (deceleration no steeper
        than ``reference_brake_accel``) and a forward pass (acceleration
        demand no steeper than ``reference_accel_limit``).  Returned padded
        for ``_interp_s`` interpolation and cached until the inputs change.
        """
        cfg = config
        runtime = (None if lateral_accel_limit is None
                   else round(float(lateral_accel_limit), 2))
        key = (self._profile_version, cfg.speed_scale,
               cfg.reference_speed_caps, cfg.max_lateral_accel,
               cfg.minimum_reference_speed, cfg.max_speed,
               cfg.reference_brake_accel, cfg.reference_accel_limit,
               cfg.understeer_gradient, cfg.steering_gain,
               cfg.min_steer, cfg.max_steer, runtime)
        if key == self._feasible_key and self._feasible_interp is not None:
            return self._feasible_interp
        base = self._interp_speed[:-1]
        profile = self._safe_speed_array(base, self.curvature, cfg)
        if runtime is not None:
            profile = np.minimum(profile, np.sqrt(
                float(runtime) / np.maximum(np.abs(self.curvature), 1e-3)))
        if cfg.reference_speed_caps:
            count = profile.shape[0]
            ds = self._segment_ds

            # Measured drivetrain envelope (bags mpc_20260824_011849/_012148):
            # +-3.8 m/s^2 plateau up to ~6.3 m/s, decaying roughly as 6.3/v
            # above it (motor power / regen limits).  Budgets above the knee
            # scale the same way so high-speed braking distances stay honest.
            def _budget(base: float, speed: float) -> float:
                if speed <= 6.3:
                    return base
                return base * max(6.3 / speed, 0.5)

            brake = float(cfg.reference_brake_accel)
            if brake > 1e-6:
                # Two circular sweeps let the constraint propagate across the
                # start/finish wrap.
                for _ in range(2):
                    for index in range(count - 1, -1, -1):
                        downstream = profile[(index + 1) % count]
                        allowed = math.sqrt(
                            downstream * downstream
                            + 2.0 * _budget(brake, downstream) * ds[index])
                        if profile[index] > allowed:
                            profile[index] = allowed
            accel = float(cfg.reference_accel_limit)
            if accel > 1e-6:
                for _ in range(2):
                    for index in range(count):
                        upstream = profile[index]
                        target = (index + 1) % count
                        allowed = math.sqrt(
                            upstream * upstream
                            + 2.0 * _budget(accel, upstream) * ds[index])
                        if profile[target] > allowed:
                            profile[target] = allowed
        self._feasible_interp = np.append(profile, profile[0])
        self._feasible_key = key
        return self._feasible_interp

    def speed_at(self, distance, config: MPCConfig,
                 lateral_accel_limit: Optional[float] = None):
        """Feasible reference speed at one or more path stations.

        Same cached profile the reference sampler uses, exposed on its own so
        a caller can bound something in absolute terms - the published speed
        command above all - without rebuilding a reference.  Because the
        profile carries the braking feasibility pass, the value at a station
        is the fastest the car may be *there* and still slow for everything
        ahead of it, which is a bound no speed estimate participates in.
        """
        profile = self.feasible_speed_profile(config, lateral_accel_limit)
        query = np.mod(np.asarray(distance, dtype=float), self.total_length)
        return np.interp(query, self._interp_s, profile)

    def validate_steering_reach(self, config: MPCConfig,
                                tolerance: float = 1.15) -> None:
        """Reject authored kinks that the steering rack cannot trace.

        Speed limiting cannot make a geometric corner reachable when its
        curvature exceeds ``tan(steer_limit) / wheelbase`` even at zero
        speed.  A small tolerance absorbs discrete-waypoint curvature noise,
        while still rejecting multi-radian spikes in a malformed input path.
        """
        if tolerance < 1.0:
            raise ValueError('steering reach tolerance must be at least one')
        positive_limit = (
            tolerance * np.tan(config.steering_gain * config.max_steer)
            / config.wheelbase)
        negative_limit = (
            tolerance * np.tan(config.steering_gain * abs(config.min_steer))
            / config.wheelbase)
        segment_length = np.sqrt(self._segment_length_sq)
        segment_heading = np.arctan2(self._segment_y, self._segment_x)
        geometric_curvature = wrap_angle(
            segment_heading - np.roll(segment_heading, 1)) / (
                0.5 * (segment_length + np.roll(segment_length, 1)))
        for source, curvature in (
                ('stored', self.curvature),
                ('geometric', geometric_curvature)):
            allowed = np.where(curvature >= 0.0,
                               positive_limit, negative_limit)
            ratio = np.abs(curvature) / allowed
            worst = int(np.argmax(ratio))
            if ratio[worst] > 1.0:
                raise ValueError(
                    'raceline {} curvature {:.3f} 1/m at waypoint {} '
                    'exceeds steering-reachable {:.3f} 1/m'.format(
                        source, curvature[worst], worst, allowed[worst]))

    def sample(self, distances) -> np.ndarray:
        """Interpolate [x, y, heading, curvature, speed, acceleration] at loop distances."""
        query = np.mod(np.asarray(distances, dtype=float), self.total_length)
        sin_heading = np.interp(query, self._interp_s, self._interp_sin_heading)
        cos_heading = np.interp(query, self._interp_s, self._interp_cos_heading)
        return np.column_stack((
            np.interp(query, self._interp_s, self._interp_x),
            np.interp(query, self._interp_s, self._interp_y),
            np.arctan2(sin_heading, cos_heading),
            np.interp(query, self._interp_s, self._interp_curvature),
            np.interp(query, self._interp_s, self._interp_speed),
            np.interp(query, self._interp_s, self._interp_acceleration),
        ))

    def project(self, x: float, y: float, hint_s: Optional[float] = None,
                search_distance: float = 3.0) -> Tuple[float, float]:
        """Return closest loop distance and Euclidean distance to the raceline.

        A progress hint prevents projection onto the wrong side of tight
        hairpins.  The initial projection remains global, so pose resets work.
        """
        rel_x = float(x) - self.x
        rel_y = float(y) - self.y
        fraction = (rel_x * self._segment_x + rel_y * self._segment_y) / self._segment_length_sq
        fraction = np.clip(fraction, 0.0, 1.0)
        projection_x = self.x + fraction * self._segment_x
        projection_y = self.y + fraction * self._segment_y
        distance_sq = (float(x) - projection_x) ** 2 + (float(y) - projection_y) ** 2
        if hint_s is None:
            index = int(np.argmin(distance_sq))
        else:
            circular_delta = np.abs(
                (self.s - float(hint_s) + 0.5 * self.total_length)
                % self.total_length - 0.5 * self.total_length
            )
            candidates = np.flatnonzero(circular_delta <= float(search_distance))
            if candidates.size == 0:
                index = int(np.argmin(distance_sq))
            else:
                index = int(candidates[np.argmin(distance_sq[candidates])])
        distance_s = (self.s[index] + fraction[index] * self._segment_ds[index]) % self.total_length
        return float(distance_s), float(np.sqrt(distance_sq[index]))

    def project_many(self, x, y, hint_s: Optional[float] = None,
                     search_distance: float = 3.0) -> Tuple[np.ndarray,
                                                            np.ndarray]:
        """Vectorized equivalent of :meth:`project` for one local scene.

        Avoidance projects every tracked obstacle against the same raceline
        and progress window.  Doing that as one small (obstacles x segments)
        array removes a Python call plus repeated hint-mask construction per
        obstacle while retaining exactly the scalar segment projection.
        """
        query_x = np.asarray(x, dtype=float).reshape(-1)
        query_y = np.asarray(y, dtype=float).reshape(-1)
        if query_x.shape != query_y.shape:
            raise ValueError('x and y must have matching shapes')
        if query_x.size == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        rel_x = query_x[:, None] - self.x[None, :]
        rel_y = query_y[:, None] - self.y[None, :]
        fraction = (
            rel_x * self._segment_x[None, :]
            + rel_y * self._segment_y[None, :]
        ) / self._segment_length_sq[None, :]
        np.clip(fraction, 0.0, 1.0, out=fraction)
        projection_x = self.x[None, :] + fraction * self._segment_x[None, :]
        projection_y = self.y[None, :] + fraction * self._segment_y[None, :]
        distance_sq = ((query_x[:, None] - projection_x) ** 2
                       + (query_y[:, None] - projection_y) ** 2)
        if hint_s is None:
            indices = np.argmin(distance_sq, axis=1)
        else:
            circular_delta = np.abs(
                (self.s - float(hint_s) + 0.5 * self.total_length)
                % self.total_length - 0.5 * self.total_length)
            candidates = np.flatnonzero(
                circular_delta <= float(search_distance))
            if candidates.size == 0:
                indices = np.argmin(distance_sq, axis=1)
            else:
                indices = candidates[
                    np.argmin(distance_sq[:, candidates], axis=1)]
        rows = np.arange(query_x.size)
        distance_s = (
            self.s[indices]
            + fraction[rows, indices] * self._segment_ds[indices]
        ) % self.total_length
        return distance_s, np.sqrt(distance_sq[rows, indices])

    def reference(self, state: VehicleState, config: MPCConfig,
                  hint_s: Optional[float] = None,
                  speed_limit: Optional[float] = None,
                  lateral_accel_limit: Optional[float] = None,
                  projection: Optional[Tuple[float, float]] = None
                  ) -> Tuple[np.ndarray, float, float]:
        """Build an (N+1)x6 locally continuous reference trajectory.

        ``speed_limit`` caps both the sampled speeds and the forward march of
        the reference points; without it a blocked vehicle would still be
        dragged forward by the lag cost toward whatever blocks it.
        ``projection`` reuses an already computed (nearest_s, distance) pair
        so a caller rebuilding the reference within one control cycle does
        not project the same pose twice.
        """
        if lateral_accel_limit is not None:
            lateral_accel_limit = float(lateral_accel_limit)
            if (not np.isfinite(lateral_accel_limit)
                    or lateral_accel_limit <= 0.0):
                raise ValueError('lateral_accel_limit must be positive and finite')
        if projection is None:
            nearest_s, path_distance = self.project(
                state.x, state.y, hint_s=hint_s)
        else:
            nearest_s, path_distance = projection
        distances = np.empty(config.horizon + 1, dtype=float)
        distances[0] = nearest_s

        # All caps (scale, curvature/steering reach, runtime grip limit) plus
        # the global braking/acceleration feasibility passes live in one
        # cached per-waypoint profile; the sequential march only interpolates
        # it.  This is also what makes corner braking start early enough at
        # any speed scale - a pointwise cap alone told the car to keep
        # profile speed until the corner itself.
        profile = self.feasible_speed_profile(config, lateral_accel_limit)
        for index in range(config.horizon):
            query = np.mod(float(distances[index]), self.total_length)
            target_speed = float(np.interp(query, self._interp_s, profile))
            preview_speed = max(0.5 * (max(state.speed, 0.0) + target_speed), 0.5)
            if speed_limit is not None:
                # Keep a minimal march even when held: a point-collapsed
                # reference blinds the lane evaluation and the lag cost.
                preview_speed = min(preview_speed, max(speed_limit, 0.3))
            distances[index + 1] = distances[index] + preview_speed * config.dt

        reference = self.sample(distances)
        reference[:, 4] = np.interp(
            np.mod(distances, self.total_length), self._interp_s, profile)
        if lateral_accel_limit is not None:
            # The profile already carries this cap at waypoint resolution;
            # re-apply it against the sampled curvature so the pointwise
            # guarantee survives interpolation across curvature transitions.
            runtime_speed = np.sqrt(
                lateral_accel_limit
                / np.maximum(np.abs(reference[:, 3]), 1e-3))
            reference[:, 4] = np.minimum(reference[:, 4], runtime_speed)
        if speed_limit is not None:
            reference[:, 4] = np.minimum(reference[:, 4], max(speed_limit, 0.0))

        continuous_heading = np.unwrap(reference[:, 2])
        continuous_heading += round((state.yaw - continuous_heading[0]) / (2.0 * np.pi)) * 2.0 * np.pi
        reference[:, 2] = continuous_heading
        return reference, nearest_s, path_distance

    @staticmethod
    def _safe_speed(raceline_speed: float, curvature: float, config: MPCConfig) -> float:
        if not config.reference_speed_caps:
            return float(np.clip(float(raceline_speed) * config.speed_scale,
                                 0.0, config.max_speed))
        scaled_speed = max(float(raceline_speed) * config.speed_scale,
                           config.minimum_reference_speed)
        signed_curvature = float(curvature)
        kappa = max(abs(signed_curvature), 1e-3)
        curvature_speed = np.sqrt(config.max_lateral_accel / kappa)
        # With tire slip the needed steering grows with speed; cap the speed
        # so this curvature stays reachable within 95% of the steering range.
        if config.understeer_gradient > 1e-9:
            steer_limit = config.reference_steer_limit_for(signed_curvature)
            reachable = (0.95 * np.tan(config.steering_gain * steer_limit) / kappa
                         - config.wheelbase) / config.understeer_gradient
            if reachable <= 0.0:
                steer_speed = config.minimum_reference_speed
            else:
                steer_speed = float(np.sqrt(reachable))
            curvature_speed = min(curvature_speed, steer_speed)
        return float(np.clip(min(scaled_speed, curvature_speed),
                             config.minimum_reference_speed, config.max_speed))

    @staticmethod
    def _safe_speed_array(raceline_speed: np.ndarray, curvature: np.ndarray,
                          config: MPCConfig) -> np.ndarray:
        """Vectorized twin of _safe_speed (identical element-wise arithmetic)."""
        if not config.reference_speed_caps:
            return np.clip(raceline_speed * config.speed_scale,
                           0.0, config.max_speed)
        scaled_speed = np.maximum(raceline_speed * config.speed_scale,
                                  config.minimum_reference_speed)
        curvature = np.asarray(curvature, dtype=float)
        kappa = np.maximum(np.abs(curvature), 1e-3)
        curvature_speed = np.sqrt(config.max_lateral_accel / kappa)
        if config.understeer_gradient > 1e-9:
            steer_limit = config.reference_steer_limit_for(curvature)
            reachable = (0.95 * np.tan(config.steering_gain * steer_limit) / kappa
                         - config.wheelbase) / config.understeer_gradient
            steer_speed = np.where(
                reachable <= 0.0,
                config.minimum_reference_speed,
                np.sqrt(np.maximum(reachable, 0.0)))
            curvature_speed = np.minimum(curvature_speed, steer_speed)
        return np.clip(np.minimum(scaled_speed, curvature_speed),
                       config.minimum_reference_speed, config.max_speed)


class SamplingMPC:
    """Warm-started, bounded nonlinear MPC solved using CEM sampling."""

    def __init__(self, config: Optional[MPCConfig] = None,
                 backend_factories=None):
        self.config = config if config is not None else MPCConfig()
        self.config.validate()
        self._rng = np.random.default_rng(self.config.random_seed)
        # Temporal-elite terminal actions must not consume the population RNG:
        # otherwise changing K also changes every stochastic CEM candidate and
        # defeats the common-random-number dose comparison.
        self._shifted_tail_rng = np.random.default_rng(np.random.SeedSequence(
            [self.config.random_seed, 0x53484946]))
        self._mean = None  # type: Optional[np.ndarray]
        self._previous_elites = None  # type: Optional[np.ndarray]
        self._warm_start_phase = 0.0
        self._previous_control = np.zeros(2, dtype=float)
        # Estimated physical wheel angle at the previous solve instant.  It is
        # propagated one control period at the next solve, then used as the
        # persistent actuator state for the next cycle.
        self._applied_steer = 0.0
        # Initial wheel angle for the current prediction.  Production passes
        # a latency-compensated pose into solve(), so the actuator state must
        # be propagated over that same delay without advancing the persistent
        # state (the next real solve is still only one control period later).
        self._prediction_steer = 0.0
        self._wall_field = None
        self._obstacles = np.empty((0, 5), dtype=float)
        self._speed_ceiling = self.config.max_speed
        self._lateral_accel_limit = self.config.max_lateral_accel
        # Accelerators stay behind a validation chain because availability is
        # not evidence that a backend is safe to use for vehicle control.
        self._backend_candidates = []
        self._backend_failures = []
        self._backend = None
        self._accel_validated = False
        self._accel_status = 'disabled'
        self._configure_backends(backend_factories)
        # Backends may prefer float32 candidates (the CUDA kernel computes in
        # fp32 anyway); generating them in that dtype halves the host-side
        # sampling/constraint cost.  Both backends and the NumPy reference
        # always see the identical candidate array, so validation still holds.
        primary = self._backend_candidates[0] if self._backend_candidates else None
        self._sample_dtype = np.dtype(
            getattr(primary, 'sample_dtype', np.float64)
            if primary is not None else np.float64)
        # Reuse the CEM noise storage and draw antithetic pairs.  Balanced
        # +/- perturbations remove sample-mean bias (important when reducing
        # the population for a hard deadline) and halve RNG work/allocations.
        self._noise = np.empty(
            (self.config.samples, self.config.horizon, 2),
            dtype=self._sample_dtype)
        self._warmup_constraints()

    @staticmethod
    def _default_backend_factories():
        def gpu_backend(config):
            from . import mpc_cuda
            if not mpc_cuda.available():
                raise RuntimeError('CUDA runtime probe failed')
            return mpc_cuda.CudaCostBackend(config)

        def cpu_backend(config):
            if not mpc_accel.AVAILABLE:
                raise RuntimeError('numba is unavailable')
            return mpc_accel.CostBackend(config)

        return {'gpu': gpu_backend, 'cpu': cpu_backend}

    def _configure_backends(self, backend_factories) -> None:
        """Build the requested validation chain without assuming concrete types."""
        disabled = os.environ.get('MPC_DISABLE_ACCEL', '').strip().lower()
        if disabled in ('1', 'true', 'yes', 'on'):
            self._accel_status = 'disabled'
            return

        requested = os.environ.get('MPC_ACCEL_BACKEND', 'auto').strip().lower()
        if requested not in ('auto', 'gpu', 'cpu', 'numpy'):
            self._accel_status = (
                "invalid MPC_ACCEL_BACKEND={!r} -> numpy active".format(requested))
            return
        if requested == 'numpy':
            self._accel_status = 'numpy selected'
            return

        factories = (self._default_backend_factories()
                     if backend_factories is None else backend_factories)
        names = ('gpu', 'cpu') if requested == 'auto' else (requested,)
        for name in names:
            factory = factories.get(name)
            if factory is None:
                self._backend_failures.append('{} unavailable (no factory)'.format(name))
                continue
            try:
                backend = factory(self.config)
                if not all(hasattr(backend, attr)
                           for attr in ('costs', 'tolerance', 'name')):
                    raise TypeError('backend interface is incomplete')
                self._warmup_backend(backend)
                self._backend_candidates.append(backend)
            except Exception as error:  # pragma: no cover - defensive
                self._backend_failures.append(
                    '{} unavailable ({})'.format(name, error))

        if self._backend_candidates:
            self._backend = self._backend_candidates[0]
            ready = self._backend_ready_status(self._backend)
            self._accel_status = self._join_backend_status(ready)
        else:
            self._accel_status = self._join_backend_status('numpy active')

    @property
    def accel_status(self) -> str:
        """Human-readable accelerator chain state for controller logging."""
        return self._accel_status

    def _warmup_backend(self, backend) -> None:
        """Trigger JIT compilation now so the first real solve is not stalled."""
        horizon = self.config.horizon
        reference = np.zeros((horizon + 1, 6), dtype=float)
        reference[:, 4] = 1.0
        controls = np.zeros((4, horizon, 2), dtype=float)
        backend.costs(np.zeros(4), controls, reference,
                      np.zeros(2), np.empty((0, 5)), self.config.max_speed,
                      0.0)

    def _warmup_constraints(self) -> None:
        """Compile the sampling/constraint kernels for every dtype up front."""
        if not mpc_accel.enabled():
            return
        for dtype in {self._sample_dtype, np.dtype(np.float64)}:
            warm = np.zeros((2, self.config.horizon, 2), dtype=dtype)
            first_max_change = (
                self.config.max_steer_rate * self.config.control_period)
            max_change = self.config.max_steer_rate * self.config.dt
            mpc_accel.apply_control_constraints(
                warm, self.config.min_steer, self.config.max_steer,
                self.config.min_accel, self.config.max_accel, 0.0,
                first_max_change, max_change)
            mpc_accel.sample_constrained_candidates(
                warm, warm[0], warm[0], self.config.min_steer,
                self.config.max_steer, self.config.min_accel,
                self.config.max_accel, 0.0,
                first_max_change, max_change)

    def _join_backend_status(self, result: str) -> str:
        return ' -> '.join(self._backend_failures + [result])

    @staticmethod
    def _backend_ready_status(backend) -> str:
        if backend.name == 'cpu':
            return 'cpu ready ({} threads)'.format(mpc_accel.default_threads())
        if backend.name == 'gpu' and hasattr(backend, 'max_samples'):
            return 'gpu ready ({} max)'.format(backend.max_samples)
        return '{} ready'.format(backend.name)

    @staticmethod
    def _backend_active_status(backend, worst: float) -> str:
        if backend.name == 'cpu':
            return 'cpu active ({} threads, match {:.1e})'.format(
                mpc_accel.default_threads(), worst)
        if backend.name == 'gpu' and hasattr(backend, 'max_samples'):
            return 'gpu active ({} max, match {:.1e})'.format(
                backend.max_samples, worst)
        return '{} active (match {:.1e})'.format(backend.name, worst)

    def set_wall_field(self, wall_field) -> None:
        """Attach a MapDistanceField used for wall clearance costs."""
        was_validated = self._accel_validated
        self._wall_field = wall_field
        survivors = []
        for backend in self._backend_candidates:
            try:
                setter = getattr(backend, 'set_wall_field', None)
                if setter is not None:
                    setter(wall_field)
                survivors.append(backend)
            except Exception as error:  # pragma: no cover - defensive
                self._backend_failures.append(
                    '{} unavailable ({})'.format(backend.name, error))
        self._backend_candidates = survivors
        self._backend = survivors[0] if survivors else None
        if was_validated and survivors:
            # A backend proven without wall lookups must prove the newly
            # enabled cost terms before it can influence another command.
            self._accel_validated = False
        if not self._accel_validated:
            status = (self._backend_ready_status(self._backend)
                      if self._backend is not None else 'numpy active')
            self._accel_status = self._join_backend_status(status)

    @property
    def lateral_accel_limit(self) -> float:
        """Lateral-acceleration limit currently enforced by every backend."""
        return self._lateral_accel_limit

    def set_lateral_accel_limit(self, limit: Optional[float] = None) -> None:
        """Apply a runtime grip limit without mutating the MPC configuration.

        ``None`` restores the configured ceiling.  The adaptive estimator may
        only tighten that ceiling, never use it to loosen the operator's base
        setting.
        """
        value = (self.config.max_lateral_accel if limit is None
                 else float(limit))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError('lateral acceleration limit must be positive')
        value = min(value, self.config.max_lateral_accel)
        if abs(value - self._lateral_accel_limit) <= 1e-9:
            return
        self._lateral_accel_limit = value

        survivors = []
        for backend in self._backend_candidates:
            setter = getattr(backend, 'set_lateral_accel_limit', None)
            if setter is None:
                if abs(value - self.config.max_lateral_accel) <= 1e-12:
                    survivors.append(backend)
                else:
                    self._backend_failures.append(
                        '{} unavailable (no dynamic grip interface)'.format(
                            backend.name))
                continue
            try:
                setter(value)
                survivors.append(backend)
            except Exception as error:  # pragma: no cover - defensive
                self._backend_failures.append(
                    '{} unavailable ({})'.format(backend.name, error))
        if len(survivors) != len(self._backend_candidates):
            self._backend_candidates = survivors
            self._backend = survivors[0] if survivors else None
            self._accel_validated = False
            status = (self._backend_ready_status(self._backend)
                      if self._backend is not None else 'numpy active')
            self._accel_status = self._join_backend_status(status)

    def reset(self, previous_control: Optional[np.ndarray] = None) -> None:
        """Discard the warm start while preserving the actuator boundary.

        ``previous_control`` is the command actually held by the vehicle.  A
        stop in a corner keeps the rack at its last angle, so restarting from
        an artificial zero would bypass the first-command slew constraint and
        snap the servo back toward centre.
        """
        self._mean = None
        self._previous_elites = None
        self._warm_start_phase = 0.0
        if previous_control is None:
            self._previous_control[:] = 0.0
            self._applied_steer = 0.0
            self._prediction_steer = 0.0
        else:
            previous = np.asarray(previous_control, dtype=float).reshape(2)
            self._previous_control[0] = np.clip(
                previous[0], self.config.min_steer, self.config.max_steer)
            self._previous_control[1] = np.clip(
                previous[1], self.config.min_accel, self.config.max_accel)
            # A reset has no trustworthy actuator history.  Initializing to
            # the held command avoids inventing a centre-to-lock transient.
            self._applied_steer = (
                self.config.steering_gain * self._previous_control[0])
            self._prediction_steer = self._applied_steer

    def commit_applied_steering(self, steering: float) -> None:
        """Commit a downstream-modified steering command for the next solve.

        Research comparators may place a causal output filter after solve().
        The plant and the next first-command slew/actuator prediction must use
        the same command; otherwise the comparator introduces an avoidable
        controller--plant boundary mismatch.
        """
        if not np.isfinite(steering):
            raise ValueError('applied steering must be finite')
        self._previous_control[0] = np.clip(
            steering, self.config.min_steer, self.config.max_steer)

    def _steering_after(self, initial: float, commanded: float,
                        duration: float) -> float:
        target = self.config.steering_gain * float(commanded)
        fraction = self.config.steering_response_fraction(duration)
        return float(initial + fraction * (target - initial))

    def _shift_warm_start(self) -> np.ndarray:
        """Advance the prior plan by one real control interval.

        The optimizer replans every ``control_period`` while its piecewise
        control cells are ``dt`` seconds wide.  Dropping one whole cell per
        solve is only correct when those periods match.  On the car they are
        20 ms and 50 ms, so the legacy shift advanced a curve-exit steering
        reversal 2.5 times too quickly and fed it into the next first action.
        Linear interpolation gives the prior plan at ``t + control_period``;
        the final command is held beyond the finite horizon.
        """
        if self._mean is None:
            raise RuntimeError('warm start is not initialized')
        cfg = self.config
        horizon = cfg.horizon
        mode = cfg.warm_start_mode
        if mode == 'configured':
            mode = ('fractional' if cfg.fractional_plan_transport
                    else 'whole_cell')
        if mode == 'cold_start':
            raise RuntimeError('cold_start does not transport a prior plan')
        if mode == 'whole_cell':
            # Conventional receding-horizon hot start: discard one complete
            # prediction cell.  This is the exact legacy baseline even when
            # the physical controller advances by less than one model cell.
            shifted = np.concatenate((self._mean[1:], self._mean[-1:]), axis=0)
            return shifted.astype(self._sample_dtype, copy=False)
        if mode == 'phase_accumulated':
            # Retain the sub-cell phase and advance only after the elapsed
            # real control time crosses a prediction-cell boundary.  Unlike
            # fractional interpolation, this control changes no plan values.
            self._warm_start_phase += cfg.control_period
            cells = int(np.floor(
                (self._warm_start_phase + 1e-12) / cfg.dt))
            self._warm_start_phase -= cells * cfg.dt
            if cells <= 0:
                return self._mean.copy()
            cells = min(cells, horizon - 1)
            shifted = np.concatenate(
                (self._mean[cells:], np.repeat(
                    self._mean[-1:], cells, axis=0)), axis=0)
            return shifted.astype(self._sample_dtype, copy=False)
        if mode != 'fractional':  # validate() should make this unreachable.
            raise RuntimeError('unsupported warm-start mode')
        positions = (np.arange(horizon, dtype=float)
                     + cfg.control_period / cfg.dt)
        lower = np.minimum(
            np.floor(positions).astype(np.int64), horizon - 1)
        upper = np.minimum(lower + 1, horizon - 1)
        fraction = (positions - lower)[:, None]
        shifted = ((1.0 - fraction) * self._mean[lower]
                   + fraction * self._mean[upper])
        return shifted.astype(self._sample_dtype, copy=False)

    def _normalized_control_distance(
            self, left: np.ndarray, right: np.ndarray) -> float:
        """Range-normalized RMS distance between two control sequences."""
        cfg = self.config
        scale = np.array([
            cfg.max_steer - cfg.min_steer,
            cfg.max_accel - cfg.min_accel,
        ], dtype=float)
        difference = (
            np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
        return float(np.sqrt(np.mean((difference / scale) ** 2)))

    def _shift_previous_elites(
            self, *, random_tail: bool = False,
            mean: Optional[np.ndarray] = None,
            standard_deviation: Optional[np.ndarray] = None,
            count: Optional[int] = None) -> Optional[np.ndarray]:
        """Whole-cell transport of the preceding solve's final elite set.

        The budget-matched comparator holds the last action, matching the
        controller's mean transport.  Full iCEM instead samples a new terminal
        action, as in Pinneri et al.'s ``elites_2_action_sequences``.
        """
        if self._previous_elites is None:
            return None
        elites = self._previous_elites
        if count is not None:
            elites = elites[:max(int(count), 0)]
        if elites.shape[0] == 0:
            return None
        if random_tail:
            if mean is None or standard_deviation is None:
                raise ValueError('iCEM random tail requires mean and deviation')
            tail = (
                np.asarray(mean[-1], dtype=self._sample_dtype)[None, :]
                + self._shifted_tail_rng.standard_normal(
                    (elites.shape[0], 2), dtype=self._sample_dtype)
                * np.asarray(
                    standard_deviation[-1], dtype=self._sample_dtype)[None, :])
            tail[:, 0] = np.clip(
                tail[:, 0], self.config.min_steer, self.config.max_steer)
            tail[:, 1] = np.clip(
                tail[:, 1], self.config.min_accel, self.config.max_accel)
            tail = tail[:, None, :]
        else:
            tail = elites[:, -1:, :]
        shifted = np.concatenate((elites[:, 1:, :], tail), axis=1)
        return self._apply_constraints(shifted).astype(
            self._sample_dtype, copy=False)

    def _colored_standard_normal(self, count: int, beta: float) -> np.ndarray:
        """Generate unit-variance power-law noise along the horizon axis."""
        shape = (int(count), self.config.horizon, 2)
        white = self._rng.standard_normal(shape).astype(
            self._sample_dtype, copy=False)
        if beta <= 0.0:
            return white
        spectrum = np.fft.rfft(white.astype(float), axis=1)
        frequencies = np.fft.rfftfreq(self.config.horizon)
        scale = np.ones_like(frequencies)
        positive = frequencies > 0.0
        scale[positive] = frequencies[positive] ** (-0.5 * beta)
        scale[~positive] = 0.0
        colored = np.fft.irfft(
            spectrum * scale[None, :, None],
            n=self.config.horizon, axis=1)
        colored -= colored.mean(axis=1, keepdims=True)
        deviation = colored.std(axis=1, keepdims=True)
        colored /= np.maximum(deviation, 1e-9)
        return colored.astype(self._sample_dtype, copy=False)

    def compensate_latency(self, state: VehicleState,
                           yaw_rate: Optional[float] = None,
                           accel: Optional[float] = None) -> VehicleState:
        """Propagate the measured state over the odometry-to-actuation delay.

        By default uses the previously commanded steering and acceleration,
        which are the controls acting on the vehicle while this solve is in
        flight.  Measured ``yaw_rate`` and ``accel`` (IMU) override the
        command-based propagation so servo lag, speed lag, and slip are
        reflected instead of assumed away.
        """
        cfg = self.config
        delay = cfg.latency_compensation
        if delay <= 0.0:
            return state
        command = float(self._previous_control[0])
        if accel is None:
            accel = float(self._previous_control[1])
        speed = float(np.clip(state.speed + 0.5 * accel * delay,
                              cfg.min_speed, cfg.max_speed))
        if yaw_rate is None:
            # Odometry already contains the previous cycle's motion.  Estimate
            # the wheel angle at this cycle, then integrate its continued
            # response over the odometry-to-actuation delay.
            current_steer = self._steering_after(
                self._applied_steer, command, cfg.control_period)
            _, _, average_retention, average_command = (
                cfg.steering_step_coefficients(delay))
            average_steer = (
                average_retention * current_steer
                + average_command * command)
            effective = cfg.wheelbase + cfg.understeer_gradient * speed ** 2
            yaw_rate = speed * np.tan(average_steer) / effective
        mid_yaw = state.yaw + 0.5 * yaw_rate * delay
        return VehicleState(
            x=state.x + speed * np.cos(mid_yaw) * delay,
            y=state.y + speed * np.sin(mid_yaw) * delay,
            yaw=state.yaw + yaw_rate * delay,
            speed=float(np.clip(state.speed + accel * delay,
                                cfg.min_speed, cfg.max_speed)),
        )

    def solve(self, state: VehicleState, reference: np.ndarray,
              obstacles: Optional[np.ndarray] = None,
              speed_ceiling: Optional[float] = None,
              actuation_delay: float = 0.0, *,
              collision_obstacles: Optional[np.ndarray] = None) -> MPCResult:
        """Optimize and return one receding-horizon drive command.

        ``obstacles`` rows are (x, y, vx, vy, radius) in map coordinates,
        measured at the current control instant.  ``speed_ceiling`` clamps
        the rollout speeds (e.g. while following a blocked section); braking
        down from a higher current speed remains possible.  When ``state``
        has already been propagated to its expected actuation time, pass the
        same non-negative delay as ``actuation_delay``.  This aligns the
        rollout's first-order steering state with the compensated pose while
        leaving the persistent real-time actuator estimate unchanged.

        Pass the full obstacle scene as ``collision_obstacles`` when the cost
        uses a filtered subset. With no collision-free candidate the result
        requests a stop and sets ``has_safe_trajectory`` to false.
        """
        cfg = self.config
        actuation_delay = float(actuation_delay)
        if not np.isfinite(actuation_delay) or actuation_delay < 0.0:
            raise ValueError('actuation_delay must be finite and non-negative')
        if speed_ceiling is None:
            self._speed_ceiling = cfg.max_speed
        else:
            self._speed_ceiling = float(np.clip(
                max(speed_ceiling, state.speed - cfg.dt * abs(cfg.min_accel)),
                cfg.min_speed, cfg.max_speed))
        initial_state = state.as_array()
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (cfg.horizon + 1, 6):
            raise ValueError('reference must have shape ({}, 6)'.format(cfg.horizon + 1))
        if not np.all(np.isfinite(initial_state)) or not np.all(np.isfinite(reference)):
            raise ValueError('state and reference must be finite')
        if obstacles is None:
            self._obstacles = np.empty((0, 5), dtype=float)
        else:
            self._obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 5)
            if not np.all(np.isfinite(self._obstacles)):
                raise ValueError('obstacles must be finite')
        if collision_obstacles is None:
            collision_obstacles = self._obstacles
        else:
            collision_obstacles = np.asarray(
                collision_obstacles, dtype=float).reshape(-1, 5)
            if not np.all(np.isfinite(collision_obstacles)):
                raise ValueError('collision obstacles must be finite')
        times = actuation_delay + cfg.dt * np.arange(cfg.horizon + 1)
        obstacle_centers = (collision_obstacles[None, :, :2]
                            + times[:, None, None]
                            * collision_obstacles[None, :, 2:4])
        obstacle_radius_sq = (collision_obstacles[:, 4] + cfg.ego_disc_radius) ** 2

        # The last command has acted for one real control period since the
        # preceding solve.  Keep this persistent estimate at the real solve
        # instant; the prediction-only copy below may be farther in the future.
        self._applied_steer = self._steering_after(
            self._applied_steer, self._previous_control[0],
            cfg.control_period)
        self._prediction_steer = self._steering_after(
            self._applied_steer, self._previous_control[0],
            actuation_delay)

        # Reference trigonometry, predicted obstacle centres and wall limits
        # are invariant across all CEM populations in this solve.  The CPU
        # backend can prepare them once; other/validation backends retain the
        # original costs() interface and behavior.
        prepared_cost_inputs = None
        if self._accel_validated and self._backend is not None:
            prepare = getattr(self._backend, 'prepare', None)
            costs_prepared = getattr(self._backend, 'costs_prepared', None)
            if prepare is not None and costs_prepared is not None:
                prepared_cost_inputs = prepare(
                    initial_state, reference, self._obstacles)

        dtype = self._sample_dtype
        feedforward = self._feedforward_controls(state, reference).astype(dtype)
        # Keep a feasible deterministic progress candidate in every CEM
        # iteration.  Previously it was used only on the first solve; once a
        # warm start converged to zero speed, two stochastic iterations could
        # fail to rediscover the coupled accelerate+turn basin for seconds.
        feedforward = self._apply_constraints(feedforward[None, :, :])[0]
        cold_start = cfg.warm_start_mode == 'cold_start'
        warm_start_used = self._mean is not None and not cold_start
        if not warm_start_used:
            mean = feedforward
            distance_pre_refresh = float('nan')
            distance_post_refresh = float('nan')
        else:
            mean = self._shift_warm_start()
            if cfg.progress_refresh_constraint_order == 'pre_and_post':
                mean = self._apply_constraints(mean[None, :, :])[0]
            distance_pre_refresh = self._normalized_control_distance(
                mean, feedforward)
            if (cfg.deterministic_progress_anchor
                    and cfg.progress_mean_refresh):
                # Refresh the shifted warm start as the reference changes.
                # The exposed fraction supports mechanism/sensitivity tests;
                # zero keeps only the transported CEM plan.
                fraction = cfg.progress_mean_refresh_fraction
                mean = ((1.0 - fraction) * mean
                        + fraction * feedforward).astype(dtype)
            mean = self._apply_constraints(mean[None, :, :])[0]
            distance_post_refresh = self._normalized_control_distance(
                mean, feedforward)
        standard_deviation = np.tile(
            [cfg.initial_steer_std, cfg.initial_accel_std],
            (cfg.horizon, 1),
        ).astype(dtype)
        variance_floor = np.array(
            [cfg.minimum_steer_std, cfg.minimum_accel_std], dtype=dtype) ** 2
        icem = cfg.cem_variant == 'icem'
        elite_count = (
            min(cfg.icem_elite_count, cfg.samples // 2)
            if icem else
            max(2, int(round(cfg.samples * cfg.elite_fraction))))
        shifted_elites = None
        if warm_start_used:
            if icem and self._previous_elites is not None:
                reuse_count = int(
                    self._previous_elites.shape[0] * cfg.icem_reuse_fraction)
                shifted_elites = self._shift_previous_elites(
                    random_tail=True, mean=mean,
                    standard_deviation=standard_deviation,
                    count=reuse_count)
            elif (cfg.shifted_elite_reuse_fraction > 0.0
                  or cfg.shifted_elite_reuse_count > 0):
                reuse_count = (
                    cfg.shifted_elite_reuse_count
                    if cfg.shifted_elite_reuse_count > 0 else
                    int(self._previous_elites.shape[0]
                        * cfg.shifted_elite_reuse_fraction))
                shifted_elites = self._shift_previous_elites(
                    random_tail=True, mean=mean,
                    standard_deviation=standard_deviation,
                    count=reuse_count)

        first_max_change = cfg.max_steer_rate * cfg.control_period
        max_change = cfg.max_steer_rate * cfg.dt
        best_controls = None
        best_cost = np.inf
        mean_feedforward_distance_by_iteration = []
        elite_spread_by_iteration = []
        shifted_elites_inserted = 0
        final_elites = None
        final_elite_costs = None
        population_evaluations_by_iteration = []
        population_candidates_by_iteration = []
        shifted_elite_selected = False
        retained_elite_selected = False
        for iteration_index in range(cfg.iterations):
            sample_count = (
                max(2 * elite_count, int(
                    cfg.samples / cfg.icem_population_decay ** iteration_index))
                if icem else cfg.samples)
            if icem:
                noise = self._colored_standard_normal(
                    sample_count, cfg.icem_noise_beta)
            else:
                positive_count = (sample_count + 1) // 2
                positive = self._noise[:positive_count]
                self._rng.standard_normal(
                    positive.shape, dtype=dtype, out=positive)
                negative_count = sample_count - positive_count
                if negative_count:
                    self._noise[positive_count:sample_count] = (
                        -positive[:negative_count])
                noise = self._noise[:sample_count]
            if mpc_accel.enabled() and not icem:
                candidates = mpc_accel.sample_constrained_candidates(
                    noise, mean, standard_deviation,
                    cfg.min_steer, cfg.max_steer, cfg.min_accel,
                    cfg.max_accel, float(self._previous_control[0]),
                    first_max_change, max_change)
            else:
                candidates = None
            if candidates is None:
                candidates = mean[None, :, :] + noise * standard_deviation[None, :, :]
                if not icem:
                    candidates[0] = mean
                candidates = self._apply_constraints(candidates)
            if icem and iteration_index == cfg.iterations - 1:
                candidates[0] = mean
            candidate_origin = np.zeros(candidates.shape[0], dtype=np.int8)
            if (cfg.samples > 1
                    and cfg.deterministic_progress_anchor
                    and cfg.progress_candidate_insertion):
                candidates[1] = feedforward
            if iteration_index == 0 and shifted_elites is not None:
                # Preserve candidates 0 (current mean) and 1 when it is the
                # deterministic feed-forward candidate.  Temporal elites go
                # at the population tail so neither reserved slot is hidden.
                reserved = 2 if (
                    cfg.deterministic_progress_anchor
                    and cfg.progress_candidate_insertion) else 1
                if icem:
                    shifted_elites_inserted = shifted_elites.shape[0]
                    candidates = np.concatenate(
                        (candidates, shifted_elites), axis=0)
                    candidate_origin = np.concatenate((
                        candidate_origin,
                        np.ones(shifted_elites_inserted, dtype=np.int8)))
                else:
                    shifted_elites_inserted = min(
                        shifted_elites.shape[0],
                        candidates.shape[0] - reserved)
                    if shifted_elites_inserted > 0:
                        candidates[-shifted_elites_inserted:] = (
                            shifted_elites[:shifted_elites_inserted])
                        candidate_origin[-shifted_elites_inserted:] = 1
            retained_elites = None
            retained_costs = None
            if icem and iteration_index > 0 and final_elites is not None:
                keep_count = int(
                    final_elites.shape[0] * cfg.icem_reuse_fraction)
                if keep_count > 0:
                    retained_elites = final_elites[:keep_count]
                    retained_costs = final_elite_costs[:keep_count]
            evaluated_count = candidates.shape[0]
            if prepared_cost_inputs is None:
                costs = self._eval_costs(initial_state, candidates, reference)
            else:
                costs = costs_prepared(
                    initial_state, candidates, self._previous_control,
                    self._speed_ceiling, prepared_cost_inputs,
                    self._prediction_steer)
            if retained_elites is not None:
                # These trajectories were evaluated under the identical state,
                # reference, and constraints in the preceding inner iteration;
                # iCEM retains their rollouts and costs without re-evaluation.
                candidates = np.concatenate(
                    (candidates, retained_elites), axis=0)
                costs = np.concatenate((costs, retained_costs), axis=0)
                candidate_origin = np.concatenate((
                    candidate_origin,
                    np.full(retained_elites.shape[0], 2, dtype=np.int8)))
            population_evaluations_by_iteration.append(evaluated_count)
            population_candidates_by_iteration.append(candidates.shape[0])

            # Keep the penalties for fitting the search distribution, even
            # when there are too few feasible candidates to fit it by itself.
            best_index = self._best_safe_index(
                initial_state, candidates, costs, obstacle_centers,
                obstacle_radius_sq, np.inf if icem else best_cost)
            if best_index is not None:
                best_cost = float(costs[best_index])
                best_controls = candidates[best_index].copy()
                shifted_elite_selected = candidate_origin[best_index] == 1
                retained_elite_selected = candidate_origin[best_index] == 2

            if icem:
                elite_indices = np.argsort(costs)[:elite_count]
            else:
                elite_indices = np.argpartition(
                    costs, elite_count - 1)[:elite_count]
            elites = candidates[elite_indices]
            elite_cost = costs[elite_indices]
            # Standard CEM must retain its historical argpartition fitting
            # order.  Sort only the copy stored for next-step temporal reuse;
            # otherwise enabling shifted elites also changes floating-point
            # accumulation order in the current weighted fit.
            if icem:
                final_elites = elites.copy()
                final_elite_costs = elite_cost.copy()
            elif (cfg.shifted_elite_reuse_fraction > 0.0
                  or cfg.shifted_elite_reuse_count > 0):
                storage_order = np.argsort(elite_cost)
                final_elites = elites[storage_order].copy()
                final_elite_costs = elite_cost[storage_order].copy()
            else:
                final_elites = elites.copy()
                final_elite_costs = elite_cost.copy()
            if icem:
                new_mean = np.mean(elites, axis=0)
                new_deviation = np.std(elites, axis=0)
                momentum = cfg.icem_momentum
                mean = ((1.0 - momentum) * new_mean
                        + momentum * mean).astype(dtype)
                standard_deviation = (
                    (1.0 - momentum) * new_deviation
                    + momentum * standard_deviation).astype(dtype)
                standard_deviation = np.maximum(
                    standard_deviation, np.sqrt(variance_floor))
                variance = new_deviation ** 2
            else:
                normalized = ((elite_cost - elite_cost.min())
                              / (elite_cost.std() + 1e-9))
                weights = np.exp(-normalized)
                weights = (weights / weights.sum()).astype(dtype)
                mean = np.sum(elites * weights[:, None, None], axis=0)
                centered = elites - mean[None, :, :]
                variance = np.sum(
                    weights[:, None, None] * centered ** 2, axis=0)
                standard_deviation = np.sqrt(
                    np.maximum(variance, variance_floor))
            mean = self._apply_constraints(mean[None, :, :])[0]
            mean_feedforward_distance_by_iteration.append(
                self._normalized_control_distance(mean, feedforward))
            scale = np.array([
                cfg.max_steer - cfg.min_steer,
                cfg.max_accel - cfg.min_accel,
            ], dtype=float)
            elite_spread_by_iteration.append(float(np.sqrt(np.mean(
                variance.astype(float) / scale[None, :] ** 2))))

        if not icem:
            if prepared_cost_inputs is None:
                mean_costs = self._eval_costs(
                    initial_state, mean[None, :, :], reference)
            else:
                mean_costs = costs_prepared(
                    initial_state, mean[None, :, :], self._previous_control,
                    self._speed_ceiling, prepared_cost_inputs,
                    self._prediction_steer)
            mean_cost = float(mean_costs[0])
            if (np.isfinite(mean_cost) and mean_cost <= best_cost
                    and self._collision_free_mask(
                        self._rollout(initial_state, mean[None, :, :]),
                        obstacle_centers, obstacle_radius_sq)[0]):
                best_controls = mean
                best_cost = mean_cost
                # The post-fit mean, rather than a population candidate, is
                # the executed sequence.  Do not attribute its selection to a
                # temporal or retained elite that happened to be population
                # best before this final comparison.
                shifted_elite_selected = False
                retained_elite_selected = False

        if best_controls is None:
            # Keep the rack at its last commanded angle while braking. This
            # is a stop request, even if braking alone cannot avoid contact.
            stop_control = np.array([self._previous_control[0], cfg.min_accel])
            stop_controls = np.repeat(stop_control[None, :], cfg.horizon, axis=0)
            self.reset(stop_control)
            return MPCResult(
                steering=float(stop_control[0]), acceleration=cfg.min_accel,
                target_speed=0.0, controls=stop_controls,
                predicted_states=self._rollout(
                    initial_state, stop_controls[None])[0],
                cost=np.inf, cost_evaluates_controls=False,
                has_safe_trajectory=False)

        execute_fitted_mean = (
            icem and cfg.icem_execution_rule == 'fitted_mean'
            and self._collision_free_mask(
                self._rollout(initial_state, mean[None, :, :]),
                obstacle_centers, obstacle_radius_sq)[0])
        if execute_fitted_mean:
            # Deliberately do not objective-evaluate the post-final-fit mean:
            # the ablation changes only the executed sequence and retains the
            # configured 40/32/25 evaluation schedule.
            best_controls = mean.copy()
            shifted_elite_selected = False
            retained_elite_selected = False

        predicted_states = self._rollout(initial_state, best_controls[None, :, :])[0]
        preview_index = int(np.clip(
            round(cfg.speed_command_preview / cfg.dt), 1, cfg.horizon))
        target_speed = float(np.clip(
            predicted_states[preview_index, 3],
            cfg.min_speed,
            cfg.max_speed,
        ))
        clear_track_recovery = False
        clear_track_recovery_attempted = False
        reference_floor = min(
            cfg.clear_track_speed_floor,
            float(np.max(reference[1:preview_index + 1, 4])))
        if (cfg.deterministic_progress_anchor
                and cfg.clear_track_fallback
                and self._obstacles.shape[0] == 0
                and self._speed_ceiling >= reference_floor
                and reference_floor > cfg.min_speed + 1e-6
                and target_speed < reference_floor):
            clear_track_recovery_attempted = True
            recovered = self._clear_track_recovery(
                initial_state, reference, feedforward, reference_floor,
                preview_index)
            if recovered is not None and self._collision_free_mask(
                    recovered[1][None, :, :], obstacle_centers,
                    obstacle_radius_sq)[0]:
                best_controls, predicted_states = recovered
                target_speed = max(
                    reference_floor,
                    float(predicted_states[preview_index, 3]))
                clear_track_recovery = True

        first_control = best_controls[0]
        # Standard deployed CEM historically transported the selected best
        # sequence.  iCEM instead shift-initializes the fitted distribution
        # mean while executing the best evaluated elite.
        self._mean = mean.copy() if icem else best_controls.copy()
        if (icem or cfg.shifted_elite_reuse_fraction > 0.0
                or cfg.shifted_elite_reuse_count > 0):
            self._previous_elites = (
                None if final_elites is None else final_elites.copy())
        else:
            self._previous_elites = None
        self._previous_control = first_control.copy()
        return MPCResult(
            steering=float(first_control[0]),
            acceleration=float(first_control[1]),
            target_speed=target_speed,
            controls=best_controls,
            predicted_states=predicted_states,
            cost=best_cost,
            clear_track_recovery=clear_track_recovery,
            clear_track_recovery_attempted=clear_track_recovery_attempted,
            warm_start_used=warm_start_used,
            mean_feedforward_distance_pre_refresh=distance_pre_refresh,
            mean_feedforward_distance_post_refresh=distance_post_refresh,
            mean_feedforward_distance_final=(
                self._normalized_control_distance(mean, feedforward)
                if warm_start_used else float('nan')),
            mean_feedforward_distance_by_iteration=tuple(
                mean_feedforward_distance_by_iteration),
            elite_spread_by_iteration=tuple(elite_spread_by_iteration),
            shifted_elites_inserted=shifted_elites_inserted,
            population_evaluations_by_iteration=tuple(
                population_evaluations_by_iteration),
            population_candidates_by_iteration=tuple(
                population_candidates_by_iteration),
            shifted_elite_selected=shifted_elite_selected,
            retained_elite_selected=retained_elite_selected,
            cost_evaluates_controls=not execute_fitted_mean,
        )

    def _best_safe_index(self, state, controls, costs, obstacle_centers,
                         obstacle_radius_sq, cost_limit) -> Optional[int]:
        eligible = np.flatnonzero(np.isfinite(costs) & (costs < cost_limit))
        if not eligible.size:
            return None
        if self._wall_field is None and obstacle_radius_sq.size == 0:
            return int(eligible[np.argmin(costs[eligible])])

        # Most cycles need only the cheapest rollout. Batch the rest to
        # amortize rollout overhead while bounding temporary geometry arrays.
        order = eligible[np.argsort(costs[eligible], kind='stable')]
        for batch in np.split(order, range(1, order.size, 128)):
            predicted = self._rollout(state, controls[batch])
            safe = self._collision_free_mask(
                predicted, obstacle_centers, obstacle_radius_sq)
            if np.any(safe):
                return int(batch[np.argmax(safe)])
        return None

    def _collision_free_mask(self, predicted, obstacle_centers,
                             obstacle_radius_sq) -> np.ndarray:
        """Test the two body discs over the full predicted horizon."""
        safe = np.all(np.isfinite(predicted), axis=(1, 2))
        if self._wall_field is None and not obstacle_radius_sq.size:
            return safe
        radius = self.config.ego_disc_radius
        discs = self._ego_discs(predicted)
        for centers in discs:
            if not np.any(safe):
                break
            if self._wall_field is not None:
                disc = centers[safe]
                clearance = self._wall_field.clearance(disc[:, :, 0], disc[:, :, 1])
                safe[safe] = np.all(clearance > radius, axis=1)
            if obstacle_radius_sq.size and np.any(safe):
                # Closest approach of the moving discs within each model step.
                relative = centers[safe, :, None, :] - obstacle_centers
                start = relative[:, :-1]
                step = np.diff(relative, axis=1)
                dx, dy = step[..., 0], step[..., 1]
                x, y = start[..., 0], start[..., 1]
                fraction = np.clip(
                    -(x * dx + y * dy)
                    / np.maximum(dx * dx + dy * dy, np.finfo(float).tiny), 0.0, 1.0)
                closest_sq = (x + fraction * dx) ** 2 + (y + fraction * dy) ** 2
                safe[safe] = np.all(closest_sq > obstacle_radius_sq, axis=(1, 2))
        if self._wall_field is not None and np.any(safe):
            # Check intermediate disc positions after the cheaper tests.
            indices = np.flatnonzero(safe)
            for centers in discs:
                disc = centers[indices]
                step = np.diff(disc, axis=1)
                distance = np.max(np.hypot(step[:, :, 0], step[:, :, 1]))
                subdivisions = max(1, math.ceil(distance / (0.5 * radius)))
                fraction = np.arange(1, subdivisions) / subdivisions
                between = (disc[:, :-1, None, :]
                           + step[:, :, None, :] * fraction[None, None, :, None])
                clearance = self._wall_field.clearance(
                    between[..., 0], between[..., 1])
                safe[indices] &= np.all(clearance > radius, axis=(1, 2))
        return safe

    def _clear_track_recovery(
            self, state: np.ndarray, reference: np.ndarray,
            feedforward: np.ndarray, speed_floor: float,
            preview_index: int):
        """Return a safe geometric progress plan when clear-track MPC stalls.

        This is deliberately not an obstacle escape path.  It runs only when
        the planner passed zero relevant obstacles and uses the same map-wall
        hard boundary as the MPC.  Pure Pursuit supplies a stable steering
        direction at zero speed, where steering has almost no rollout cost
        and CEM otherwise dithers between locks.
        """
        cfg = self.config
        recovery = np.asarray(feedforward, dtype=self._sample_dtype).copy()
        delta = reference[1:, 0:2] - state[None, 0:2]
        distance = np.hypot(delta[:, 0], delta[:, 1])
        candidates = np.flatnonzero(distance >= cfg.clear_track_lookahead)
        target_index = int(candidates[0]) if candidates.size else len(distance) - 1
        target_distance = max(float(distance[target_index]), 1e-3)
        bearing = np.arctan2(delta[target_index, 1], delta[target_index, 0])
        alpha = float(wrap_angle(bearing - state[2]))
        effective_wheelbase = (
            cfg.wheelbase
            + cfg.understeer_gradient * max(state[3], speed_floor) ** 2)
        pursuit_steer = np.arctan(
            2.0 * effective_wheelbase * np.sin(alpha) / target_distance
        ) / cfg.steering_gain

        # Reach the non-zero floor by the command preview point.  The VESC
        # consumes target_speed directly, while this consistent acceleration
        # prevents latency compensation and the next warm start from still
        # believing that the car intends to park.
        recovery_accel = max(
            (speed_floor - state[3])
            / max(preview_index * cfg.dt, cfg.dt), 0.0)
        recovery[:preview_index, 1] = np.maximum(
            recovery[:preview_index, 1], recovery_accel)

        # Pure Pursuit is normally the best deterministic escape from the
        # CEM's parked local minimum. At a malformed path kink, however,
        # its one steering direction can point into the nearby wall.  The old
        # code then rejected recovery outright and kept returning zero speed
        # forever.  Fan a small set of rate-constrained steering directions
        # and use the closest wall-safe progress candidate.  This does not
        # modify or reject the authored path and runs only on a clear track;
        # obstacle/follow cases still retain full authority to stop.
        steer_targets = np.unique(np.concatenate((
            # The three dynamic directions below cover the local basin; nine
            # rack-wide anchors retain kink/wall escape coverage without
            # paying for twenty extra rollouts on every recovery cycle.
            np.linspace(cfg.min_steer, cfg.max_steer, 9),
            np.array([
                pursuit_steer,
                feedforward[0, 0],
                self._previous_control[0],
            ]),
        )))
        recovery_candidates = np.repeat(
            recovery[None, :, :], steer_targets.size, axis=0)
        recovery_candidates[:, :, 0] = steer_targets[:, None]
        recovery_candidates = self._apply_constraints(recovery_candidates)
        predicted_candidates = self._rollout(state, recovery_candidates)

        safe = self._clear_track_recovery_safe_mask(
            state, predicted_candidates, preview_index)
        if not np.any(safe):
            return None

        # Compare only the command-preview portion that this cycle can
        # commit.  Longer-tail ranking would again favor parking before an
        # impossible single-point kink even though a safe immediate turn is
        # available and the controller replans every 20 ms in the studied setup.
        predicted = predicted_candidates[:, 1:preview_index + 1]
        target = reference[None, 1:preview_index + 1]
        error_x = predicted[:, :, 0] - target[:, :, 0]
        error_y = predicted[:, :, 1] - target[:, :, 1]
        target_yaw = target[:, :, 2]
        contour = -np.sin(target_yaw) * error_x + np.cos(target_yaw) * error_y
        lag = np.cos(target_yaw) * error_x + np.sin(target_yaw) * error_y
        heading = wrap_angle(predicted[:, :, 2] - target_yaw)
        score = np.sum(
            cfg.contour_weight * contour ** 2
            + cfg.lag_weight * lag ** 2
            + cfg.heading_weight * heading ** 2,
            axis=1)
        first_delta = (recovery_candidates[:, 0, 0]
                       - self._previous_control[0])
        score += cfg.steer_rate_weight * first_delta ** 2
        score[~safe] = np.inf
        best = int(np.argmin(score))
        return recovery_candidates[best], predicted_candidates[best]

    def _clear_track_recovery_is_safe(
            self, state: np.ndarray, predicted: np.ndarray,
            check_steps: Optional[int] = None) -> bool:
        predicted = np.asarray(predicted, dtype=float)
        if predicted.ndim != 2:
            raise ValueError('predicted states must have shape (N, 4)')
        return bool(self._clear_track_recovery_safe_mask(
            state, predicted[None, :, :], check_steps)[0])

    def _clear_track_recovery_safe_mask(
            self, state: np.ndarray, predicted: np.ndarray,
            check_steps: Optional[int] = None) -> np.ndarray:
        """Vectorized wall gate for clear-track recovery candidates."""
        predicted = np.asarray(predicted, dtype=float)
        if predicted.ndim != 3 or predicted.shape[2] != 4:
            raise ValueError('predicted states must have shape (S, N, 4)')
        sample_count, horizon, _ = predicted.shape
        if horizon < 2:
            return np.zeros(sample_count, dtype=bool)
        progress = np.hypot(
            predicted[:, -1, 0] - state[0],
            predicted[:, -1, 1] - state[1]) >= 0.05
        if self._wall_field is None:
            return progress
        # Only the command-preview segment is committed: this controller
        # replans every 40 ms.  Rejecting a useful first turn because the
        # uncommitted feed-forward tail brushes a wall kept the old controller
        # parked indefinitely.  Include the actual current pose explicitly;
        # ``_rollout`` starts one model step in the future.
        if check_steps is None:
            check_steps = horizon
        count = int(np.clip(check_steps, 1, horizon))
        current = np.broadcast_to(
            np.asarray(state, dtype=float), (sample_count, 1, 4))
        checked = np.concatenate((current, predicted[:, :count, :]), axis=1)
        cfg = self.config
        direction = np.stack(
            (np.cos(checked[:, :, 2]), np.sin(checked[:, :, 2])), axis=-1)
        centers = checked[:, :, 0:2]
        clearance = np.full(checked.shape[:2], np.inf)
        for offset in (cfg.ego_center_offset + cfg.ego_disc_offset,
                       cfg.ego_center_offset - cfg.ego_disc_offset):
            discs = centers + offset * direction
            disc_clearance = self._wall_field.clearance(
                discs[:, :, 0].ravel(), discs[:, :, 1].ravel()
            ).reshape(sample_count, count + 1)
            np.minimum(clearance, disc_clearance, out=clearance)
        hard = cfg.ego_disc_radius + cfg.wall_hard_margin
        # A localization estimate already inside the hard band may move only
        # if it does not worsen that existing intrusion.
        floor = np.minimum(clearance[:, 0] - 0.02, hard)
        return progress & (np.min(clearance[:, 1:], axis=1) >= floor)

    def _eval_costs(self, initial_state: np.ndarray, controls: np.ndarray,
                    reference: np.ndarray) -> np.ndarray:
        """Per-sample rollout cost via the numba backend, else NumPy.

        The backend is checked against the NumPy reference exactly once, on the
        first real solve, and dropped on any mismatch so a bad accelerator can
        never steer the car.
        """
        if not self._backend_candidates:
            return self._cost(initial_state, controls, reference)
        if not self._accel_validated:
            return self._validate_backend(initial_state, controls, reference)
        return self._backend.costs(
            initial_state, controls, reference, self._previous_control,
            self._obstacles, self._speed_ceiling, self._prediction_steer)

    def _validate_backend(self, initial_state: np.ndarray, controls: np.ndarray,
                          reference: np.ndarray) -> np.ndarray:
        self._accel_validated = True
        reference_costs = self._cost(initial_state, controls, reference)
        for backend in self._backend_candidates:
            try:
                accel_costs = backend.costs(
                    initial_state, controls, reference, self._previous_control,
                    self._obstacles, self._speed_ceiling,
                    self._prediction_steer)
                # Costs are sums of non-negative penalty terms, so a reference
                # cost near zero means every term is near zero; the unit floor
                # makes the comparison absolute there instead of amplifying
                # float32 noise on a perfect rollout into a spurious mismatch.
                relative = (np.abs(reference_costs - accel_costs)
                            / np.maximum(np.abs(reference_costs), 1.0))
                worst = float(np.max(relative)) if relative.size else 0.0
                if not np.all(np.isfinite(accel_costs)) or worst > backend.tolerance:
                    self._backend_failures.append(
                        '{} fell back (mismatch {:.1e})'.format(backend.name, worst))
                    continue
                self._backend = backend
                self._backend_candidates = [backend]
                self._accel_status = self._join_backend_status(
                    self._backend_active_status(backend, worst))
                return accel_costs
            except Exception as error:  # pragma: no cover - defensive
                self._backend_failures.append(
                    '{} fell back ({})'.format(backend.name, error))

        self._backend = None
        self._backend_candidates = []
        self._accel_status = self._join_backend_status('numpy active')
        return reference_costs

    def _feedforward_controls(self, state: VehicleState, reference: np.ndarray) -> np.ndarray:
        cfg = self.config
        controls = np.zeros((cfg.horizon, 2), dtype=float)
        effective_wheelbase = (cfg.wheelbase
                               + cfg.understeer_gradient * reference[1:, 4] ** 2)
        controls[:, 0] = (
            np.arctan(effective_wheelbase * reference[1:, 3])
            / cfg.steering_gain)
        speed_delta = np.diff(reference[:, 4]) / cfg.dt
        controls[:, 1] = speed_delta
        controls[0, 1] = (reference[1, 4] - state.speed) / cfg.dt
        controls[:, 0] = np.clip(controls[:, 0], cfg.min_steer, cfg.max_steer)
        controls[:, 1] = np.clip(controls[:, 1], cfg.min_accel, cfg.max_accel)
        return controls

    def _apply_constraints(self, controls: np.ndarray) -> np.ndarray:
        cfg = self.config
        source = np.asarray(controls)
        dtype = source.dtype if source.dtype == np.float32 else np.float64
        constrained = np.array(source, dtype=dtype)
        first_max_change = cfg.max_steer_rate * cfg.control_period
        max_change = cfg.max_steer_rate * cfg.dt
        if mpc_accel.enabled() and mpc_accel.apply_control_constraints(
                constrained, cfg.min_steer, cfg.max_steer,
                cfg.min_accel, cfg.max_accel,
                float(self._previous_control[0]), first_max_change,
                max_change):
            return constrained

        constrained[:, :, 0] = np.clip(constrained[:, :, 0], cfg.min_steer, cfg.max_steer)
        constrained[:, :, 1] = np.clip(constrained[:, :, 1], cfg.min_accel, cfg.max_accel)

        previous_steer = np.full(constrained.shape[0], self._previous_control[0])
        for index in range(cfg.horizon):
            allowed_change = first_max_change if index == 0 else max_change
            constrained[:, index, 0] = np.clip(
                constrained[:, index, 0],
                previous_steer - allowed_change,
                previous_steer + allowed_change,
            )
            previous_steer = constrained[:, index, 0]
        return constrained

    def _rollout_with_steering(
            self, state: np.ndarray,
            controls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        sample_count = controls.shape[0]
        states = np.empty((sample_count, cfg.horizon + 1, 4), dtype=float)
        wheel_steers = np.empty((sample_count, cfg.horizon), dtype=float)
        states[:, 0, :] = state
        applied_steer = np.full(
            sample_count, self._prediction_steer, dtype=float)
        (steering_retention, steering_response,
         steering_average_retention,
         steering_average_command) = cfg.steering_step_coefficients(cfg.dt)
        for index in range(cfg.horizon):
            x = states[:, index, 0]
            y = states[:, index, 1]
            yaw = states[:, index, 2]
            speed = states[:, index, 3]
            commanded_steer = controls[:, index, 0]
            accel = controls[:, index, 1]
            step_steer = (
                steering_average_retention * applied_steer
                + steering_average_command * commanded_steer)
            applied_steer = (
                steering_retention * applied_steer
                + steering_response * commanded_steer)
            wheel_steers[:, index] = step_steer
            next_speed = np.clip(speed + cfg.dt * accel, cfg.min_speed,
                                 self._speed_ceiling)
            average_speed = 0.5 * (speed + next_speed)
            effective_wheelbase = (cfg.wheelbase
                                   + cfg.understeer_gradient * average_speed ** 2)
            states[:, index + 1, 0] = x + cfg.dt * average_speed * np.cos(yaw)
            states[:, index + 1, 1] = y + cfg.dt * average_speed * np.sin(yaw)
            states[:, index + 1, 2] = (
                yaw + cfg.dt * average_speed * np.tan(step_steer)
                / effective_wheelbase)
            states[:, index + 1, 3] = next_speed
        return states, wheel_steers

    def _rollout(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        return self._rollout_with_steering(state, controls)[0]

    def _cost(self, state: np.ndarray, controls: np.ndarray, reference: np.ndarray) -> np.ndarray:
        cfg = self.config
        states, wheel_steers = self._rollout_with_steering(state, controls)
        predicted = states[:, 1:, :]
        target = reference[None, 1:, :]

        error_x = predicted[:, :, 0] - target[:, :, 0]
        error_y = predicted[:, :, 1] - target[:, :, 1]
        target_yaw = target[:, :, 2]
        contour_error = -np.sin(target_yaw) * error_x + np.cos(target_yaw) * error_y
        lag_error = np.cos(target_yaw) * error_x + np.sin(target_yaw) * error_y
        heading_error = wrap_angle(predicted[:, :, 2] - target_yaw)
        speed_error = predicted[:, :, 3] - target[:, :, 4]

        costs = np.sum(
            cfg.contour_weight * contour_error ** 2
            + cfg.lag_weight * lag_error ** 2
            + cfg.heading_weight * heading_error ** 2
            + cfg.speed_weight * speed_error ** 2,
            axis=1,
        )

        feedforward_steer = np.clip(
            np.arctan((cfg.wheelbase
                       + cfg.understeer_gradient * reference[1:, 4] ** 2)
                      * reference[1:, 3]) / cfg.steering_gain,
            cfg.min_steer,
            cfg.max_steer,
        )
        costs += cfg.steer_feedforward_weight * np.sum(
            (controls[:, :, 0] - feedforward_steer[None, :]) ** 2,
            axis=1,
        )
        costs += cfg.accel_weight * np.sum(controls[:, :, 1] ** 2, axis=1)

        previous = np.broadcast_to(self._previous_control, (controls.shape[0], 1, 2))
        control_delta = np.diff(np.concatenate((previous, controls), axis=1), axis=1)
        costs += cfg.steer_rate_weight * np.sum(control_delta[:, :, 0] ** 2, axis=1)
        costs += cfg.accel_rate_weight * np.sum(control_delta[:, :, 1] ** 2, axis=1)

        terminal_contour = contour_error[:, -1]
        terminal_lag = lag_error[:, -1]
        terminal_heading = heading_error[:, -1]
        terminal_speed = speed_error[:, -1]
        costs += cfg.terminal_weight * (
            cfg.contour_weight * terminal_contour ** 2
            + cfg.lag_weight * terminal_lag ** 2
            + cfg.heading_weight * terminal_heading ** 2
            + cfg.speed_weight * terminal_speed ** 2
        )
        speeds = states[:, :-1, 3]
        lateral_accel = (speeds ** 2 * np.tan(wheel_steers)
                         / (cfg.wheelbase + cfg.understeer_gradient * speeds ** 2))
        excess = np.maximum(
            np.abs(lateral_accel) - self._lateral_accel_limit, 0.0)
        costs += cfg.friction_weight * np.sum(excess ** 2, axis=1)
        costs += cfg.grip_hard_weight * np.sum(excess, axis=1)

        costs += self._clearance_cost(state, predicted)
        return costs

    def _ego_discs(self, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Front and rear disc centres for every predicted state, each (S, N, 2)."""
        cfg = self.config
        direction = np.stack(
            (np.cos(predicted[:, :, 2]), np.sin(predicted[:, :, 2])), axis=-1)
        center = predicted[:, :, 0:2]
        front = cfg.ego_center_offset + cfg.ego_disc_offset
        rear = cfg.ego_center_offset - cfg.ego_disc_offset
        return (center + front * direction, center + rear * direction)

    def _clearance_cost(self, state: np.ndarray,
                        predicted: np.ndarray) -> np.ndarray:
        """Penalties keeping every rollout clear of obstacles and walls.

        The near-hard collision terms are measured relative to the current
        state's own violation: when the vehicle already sits inside a hard
        band, staying put must not beat driving out of it.
        """
        cfg = self.config
        costs = np.zeros(predicted.shape[0], dtype=float)
        discs = self._ego_discs(predicted)
        direction = np.array([np.cos(state[2]), np.sin(state[2])])
        current_points = np.array([
            state[0:2] + (cfg.ego_center_offset + cfg.ego_disc_offset) * direction,
            state[0:2] + (cfg.ego_center_offset - cfg.ego_disc_offset) * direction,
        ])

        if self._obstacles.shape[0] > 0:
            steps = np.arange(1, cfg.horizon + 1, dtype=float)
            times = steps * cfg.dt + cfg.latency_compensation
            centers = (self._obstacles[None, :, 0:2]
                       + self._obstacles[None, :, 2:4] * times[:, None, None])
            safe = (self._obstacles[:, 4] + cfg.ego_disc_radius
                    + cfg.obstacle_margin)
            collision = self._obstacles[:, 4] + cfg.ego_disc_radius
            current_distance = np.linalg.norm(
                current_points[:, None, :] - self._obstacles[None, :, 0:2],
                axis=-1).min(axis=0)
            # Tolerate the existing intrusion without decay: squeezing past a
            # tight obstacle holds a constant clearance for a while, and that
            # motion must not cost more than parking next to it.  Worsening
            # beyond the current intrusion stays forbidden.
            allowed = np.broadcast_to(
                np.maximum(collision - current_distance, 0.0)[None, :],
                (times.shape[0], collision.shape[0]))
            for disc in discs:
                distance = np.linalg.norm(
                    disc[:, :, None, :] - centers[None, :, :, :], axis=-1)
                intrusion = np.maximum(safe[None, None, :] - distance, 0.0)
                costs += cfg.obstacle_weight * np.sum(intrusion ** 2, axis=(1, 2))
                overlap = np.maximum(
                    collision[None, None, :] - distance - allowed[None, :, :],
                    0.0)
                costs += cfg.obstacle_collision_weight * np.sum(overlap, axis=(1, 2))

        if self._wall_field is not None:
            safe = cfg.ego_disc_radius + cfg.wall_margin
            # The plant tracks the prediction imperfectly; the hard boundary
            # keeps a tracking-error buffer beyond the disc radius.
            hard = cfg.ego_disc_radius + cfg.wall_hard_margin
            current_clearance = float(self._wall_field.clearance(
                current_points[:, 0], current_points[:, 1]).min())
            wall_times = (np.arange(1, cfg.horizon + 1, dtype=float) * cfg.dt
                          + cfg.latency_compensation)
            allowed_wall = np.maximum(
                max(hard - current_clearance, 0.0) - 0.3 * wall_times, 0.0)
            for disc in discs:
                clearance = self._wall_field.clearance(
                    disc[:, :, 0].ravel(), disc[:, :, 1].ravel()
                ).reshape(disc.shape[0], disc.shape[1])
                deficit = np.maximum(safe - clearance, 0.0)
                costs += cfg.wall_weight * np.sum(deficit ** 2, axis=1)
                overlap = np.maximum(hard - clearance - allowed_wall[None, :], 0.0)
                costs += cfg.wall_collision_weight * np.sum(overlap, axis=1)
        return costs


class _ProfileObstacle:
    """The handful of fields the vendored profile solver reads."""

    __slots__ = ('id', 'x', 'y', 'radius', 'confirmed')

    def __init__(self, identifier: int, x: float, y: float, radius: float):
        self.id = identifier
        self.x = x
        self.y = y
        self.radius = radius
        self.confirmed = True

    @property
    def position(self):
        return (self.x, self.y)


def _corridor_widths(raceline: 'Raceline', wall_field,
                     reach: float = 3.0, step: float = 0.02) -> np.ndarray:
    """Left/right wall distances beside every raceline point.

    The profile solver's tube is built from these; apex carries a distance
    field rather than the ray-marched corridor columns advanced_pp's CSVs
    hold, so they are marched here once at construction.
    """
    points = np.column_stack([raceline.x, raceline.y])
    normal = np.column_stack([-np.sin(raceline.heading),
                              np.cos(raceline.heading)])
    offsets = np.arange(step, reach + 0.5 * step, step)
    widths = np.empty((points.shape[0], 2))
    for column, sign in ((0, 1.0), (1, -1.0)):
        probe = (points[None, :, :]
                 + (sign * offsets)[:, None, None] * normal[None, :, :])
        clearance = np.asarray(wall_field.clearance(
            probe[:, :, 0].ravel(), probe[:, :, 1].ravel())
        ).reshape(offsets.size, -1)
        blocked = clearance < 0.5 * step
        # First blocked step per station; never blocked means the full reach.
        first = np.where(blocked.any(axis=0), blocked.argmax(axis=0),
                         offsets.size)
        widths[:, column] = offsets[np.clip(first - 1, 0, offsets.size - 1)]
        widths[first == 0, column] = 0.0
    return widths


class AvoidancePlanner:
    """Shift the raceline reference laterally to pass blocked track sections.

    Evaluates a fan of lateral offset candidates against tracked obstacles
    (propagated with their velocities) and the map wall field, then returns a
    smoothly ramped reference the sampling MPC refines.  Without obstacles the
    reference relaxes back onto the raceline.
    """

    LANE_GRID_STEP = 0.25
    # 9.0 was tried on 2026-08-26 and REVERTED.  The reasoning for it was
    # sound - the tracker reaches `detection_range` 8.0 m and
    # tools/obstacle_sim.py measures first detection of a rule box on the ifac
    # straights at 7.0-8.4 m, so 6.0 does discard sight the car already has -
    # but the A/B says it does not help.  Twelve random 3-obstacle layouts,
    # same seeds, 36 encounters: contacts 5 -> 5, grazes 3 -> 1, and layouts
    # completing laps 9/12 -> 8/12.  Three reasons it loses what it gains:
    #   * 9 m sits at the edge of detection_range, so the fan commits from a
    #     track whose radius (0.6*old + 0.4*new) and depth push have not
    #     settled, and _commit() then locks that side;
    #   * `maneuver_length` stretches with it, so the car leaves the line
    #     earlier and holds the offset longer - costly in a ~2 m corridor;
    #   * it re-opens the folded-track problem the relevance gate below was
    #     added to close (wall fragments from the far side of the hairpin
    #     around the ifac wrap).
    # And it does not touch the binding constraint: at the 8.5 m/s reference
    # stopping needs hold + v^2/(2*brake) = 10.35 m, which 9 m does not reach
    # either, while the lane change itself needs only 4.2-4.8 m and already
    # fitted inside 6.  Revisit only together with a braking-feasible follow
    # cap, which is what actually fires mid-swerve and steals the grip.
    LANE_LOOKAHEAD = 6.0
    # After a scene has been continuously unchanged this long, stretch the
    # periodic lane refresh by this factor (e.g. 0.1 s -> 0.3 s).  The
    # threshold must fit inside one obstacle encounter: the 6 m corridor at
    # racing speed is traversed in roughly 1.5-2.5 s and the stability timer
    # restarts whenever the relevant set changes, so a longer threshold never
    # engages in practice.
    STABLE_SCENE_TIME = 0.75
    STABLE_REPLAN_FACTOR = 3.0

    def __init__(self, raceline: Raceline, config: MPCConfig, wall_field=None):
        self.raceline = raceline
        self.config = config
        self.wall_field = wall_field
        self._target_offset = 0.0
        self._cached_target = 0.0
        self._cached_blocked = float('inf')
        self._cached_scene = None
        self._replan_elapsed = float('inf')
        self._scene_stable_time = 0.0
        self.last_replanned = False
        self.last_follow_cap = None  # type: Optional[float]
        # Geometry of the lane selected by the most recent fan search.  These
        # are exposed for telemetry and for deciding whether FOLLOW is still
        # necessary once a complete collision-free pass path exists.
        self.last_lane_clearance = float('inf')
        self.last_lane_safe = False
        self.last_open_pass = False
        self.last_executed_obstacle_clearance = float('inf')
        self.last_executed_obstacle_start = float('inf')
        self.last_executed_obstacle_end = float('inf')
        self.last_executed_wall_clearance = float('inf')
        self.last_executed_gap = float('inf')
        self._open_pass_confirmations = 0
        self._lane_committed = False
        self._stall_time = 0.0
        self._stall_relax = 0.0
        self._recovery_time = 0.0
        self._lane_memory = []
        self.last_memory_hit = False
        self._active_lateral_accel_limit = config.max_lateral_accel
        self._lane_offsets = np.arange(
            -config.avoid_max_offset,
            config.avoid_max_offset + 0.5 * config.avoid_offset_step,
            config.avoid_offset_step)
        self._profile_planner = self._build_profile_planner(raceline, config,
                                                            wall_field)
        self._lane_clearance = None
        if wall_field is not None:
            # Precomputed wall clearance of every candidate lane along the
            # whole loop: lane viability must look a fixed distance ahead,
            # not just over the speed-scaled horizon (nearly zero when slow).
            s_grid = np.arange(0.0, raceline.total_length, self.LANE_GRID_STEP)
            points = raceline.sample(s_grid)
            normal_x = -np.sin(points[:, 2])
            normal_y = np.cos(points[:, 2])
            self._lane_grid_s = s_grid
            direction_x = np.cos(points[:, 2])
            direction_y = np.sin(points[:, 2])
            front = config.ego_center_offset + config.ego_disc_offset
            rear = config.ego_center_offset - config.ego_disc_offset
            lanes = []
            for offset in self._lane_offsets:
                lane_x = points[:, 0] + offset * normal_x
                lane_y = points[:, 1] + offset * normal_y
                front_clearance = wall_field.clearance(
                    lane_x + front * direction_x,
                    lane_y + front * direction_y)
                rear_clearance = wall_field.clearance(
                    lane_x + rear * direction_x,
                    lane_y + rear * direction_y)
                lanes.append(np.minimum(front_clearance, rear_clearance))
            self._lane_clearance = np.stack(lanes)

    def reset(self) -> None:
        self._target_offset = 0.0
        self._cached_target = 0.0
        self._cached_blocked = float('inf')
        self._cached_scene = None
        self._replan_elapsed = float('inf')
        self._scene_stable_time = 0.0
        self.last_replanned = False
        self.last_follow_cap = None
        self.last_lane_clearance = float('inf')
        self.last_lane_safe = False
        self.last_open_pass = False
        self.last_executed_obstacle_clearance = float('inf')
        self.last_executed_obstacle_start = float('inf')
        self.last_executed_obstacle_end = float('inf')
        self.last_executed_wall_clearance = float('inf')
        self.last_executed_gap = float('inf')
        self._open_pass_confirmations = 0
        self._lane_committed = False
        self._stall_time = 0.0
        self._stall_relax = 0.0
        self._recovery_time = 0.0
        # ``_lane_memory`` deliberately survives reset(): a manual takeover
        # or safety stop invalidates transient planner state, not the track
        # knowledge of which side worked for a given static obstacle.  A
        # raceline swap rebuilds the planner and clears it naturally.
        self.last_memory_hit = False

    def _lane_memory_match(self, s_obs: float, lateral_obs: float,
                           radius_obs: float):
        """Return the stored lane entry matching this obstacle, or None."""
        cfg = self.config
        total = self.raceline.total_length
        for entry in self._lane_memory:
            delta_s = abs(((float(s_obs) - entry['s'] + 0.5 * total) % total)
                          - 0.5 * total)
            if (delta_s <= cfg.lane_memory_s_tolerance
                    and abs(float(lateral_obs) - entry['lateral'])
                    <= cfg.lane_memory_lateral_tolerance
                    and abs(float(radius_obs) - entry['radius'])
                    <= cfg.lane_memory_radius_tolerance):
                return entry
        return None

    def _lane_memory_record(self, s_obs: float, lateral_obs: float,
                            radius_obs: float, offset: float) -> None:
        """Store/refresh the lane used while actually passing an obstacle."""
        entry = self._lane_memory_match(s_obs, lateral_obs, radius_obs)
        if entry is None:
            entry = {}
            self._lane_memory.append(entry)
            if len(self._lane_memory) > int(self.config.lane_memory_size):
                self._lane_memory.pop(0)
        entry['s'] = float(s_obs)
        entry['lateral'] = float(lateral_obs)
        entry['radius'] = float(radius_obs)
        entry['offset'] = float(offset)

    def _recovery_governor(self, reference, state: VehicleState,
                           lateral_error: float, times) -> None:
        """Surge-limit the reference speed while recovering a lateral offset.

        Runs only while the post-avoidance hold timer is armed (the timer
        decays here, one control period per cycle) and the vehicle is still
        more than the deadband off its reference.  It never brakes below the
        current speed; it only stops the reference from demanding full
        profile speed while the grip budget is needed to close the offset.
        """
        cfg = self.config
        self._recovery_time = max(
            0.0, self._recovery_time - cfg.control_period)
        if self._recovery_time <= 0.0:
            return
        if abs(lateral_error) <= cfg.recovery_error_deadband:
            return
        surge = (max(state.speed, cfg.minimum_reference_speed)
                 + cfg.recovery_surge_accel * times)
        reference[:, 4] = np.minimum(reference[:, 4], surge)

    def reference(self, state: VehicleState, obstacles: Optional[np.ndarray] = None,
                  hint_s: Optional[float] = None,
                  lateral_accel_limit: Optional[float] = None):
        """Return (reference, nearest_s, raceline_distance, target_offset,
        relevant_obstacles).

        ``relevant_obstacles`` is the subset actually near the reference
        corridor; feed exactly this to the solver so returns from behind a
        dividing wall cannot distort the optimization.
        """
        cfg = self.config
        if lateral_accel_limit is None:
            self._active_lateral_accel_limit = cfg.max_lateral_accel
            runtime_limit = None
        else:
            value = float(lateral_accel_limit)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError('lateral_accel_limit must be positive')
            self._active_lateral_accel_limit = min(
                value, cfg.max_lateral_accel)
            runtime_limit = self._active_lateral_accel_limit
        base, nearest_s, path_distance = self.raceline.reference(
            state, cfg, hint_s=hint_s,
            lateral_accel_limit=runtime_limit)
        obstacles = (np.empty((0, 5)) if obstacles is None
                     else np.asarray(obstacles, dtype=float).reshape(-1, 5))

        normal_x = -np.sin(base[:, 2])
        normal_y = np.cos(base[:, 2])
        ego_lateral = float(np.clip(
            normal_x[0] * (state.x - base[0, 0])
            + normal_y[0] * (state.y - base[0, 1]),
            -cfg.avoid_max_offset, cfg.avoid_max_offset))

        steps = np.arange(base.shape[0], dtype=float)
        times = steps * cfg.dt + cfg.latency_compensation
        if obstacles.shape[0] > 0:
            centers = (obstacles[None, :, 0:2]
                       + obstacles[None, :, 2:4] * times[:, None, None])
            # Obstacle relevance must use a fixed spatial lookahead.  Using the
            # speed-scaled MPC reference delayed detection in slow corners even
            # though the lidar had already seen the blocker several metres out.
            corridor_s = nearest_s + np.arange(
                0.0, self.LANE_LOOKAHEAD + self.LANE_GRID_STEP,
                self.LANE_GRID_STEP)
            corridor = self.raceline.sample(corridor_s)
            relevant = self._relevant(
                corridor, obstacles[None, :, 0:2], obstacles[:, 4])
        else:
            relevant = np.zeros(0, dtype=bool)

        if not np.any(relevant):
            self.last_replanned = False
            self._cached_target = 0.0
            self._cached_blocked = float('inf')
            self._cached_scene = None
            self._replan_elapsed = float('inf')
            self.last_follow_cap = None
            self.last_lane_clearance = float('inf')
            self.last_lane_safe = False
            self._reset_open_pass()
            self._target_offset = self._slew(self._target_offset, 0.0)
            if abs(self._target_offset) < 0.5 * cfg.avoid_offset_step:
                self._lane_committed = False
            profile = self._ramp(ego_lateral, self._target_offset, base)
            reference = self._shift(base, profile, None, None)
            self._recovery_governor(
                reference, state, ego_lateral - self._target_offset, times)
            return reference, nearest_s, \
                path_distance, self._target_offset, np.empty((0, 5))

        # Geometry of each relevant obstacle relative to the raceline: lane
        # viability must not depend on the speed-scaled reference, which
        # collapses to a point at standstill and then sees no obstacles.
        # Anything clearly behind us is the follower's problem (racing
        # convention) - keeping it in the cost can pin the car in place
        # between a blocker ahead and a queued vehicle behind.
        total = self.raceline.total_length
        rows = obstacles[relevant]
        row_indices = np.flatnonzero(relevant)
        projected, _ = self.raceline.project_many(
            rows[:, 0], rows[:, 1], hint_s=nearest_s,
            search_distance=self.LANE_LOOKAHEAD + 3.0)
        signed = ((projected - nearest_s + 0.5 * total) % total) - 0.5 * total
        projected_points = self.raceline.sample(projected)
        projected_laterals = (
            -np.sin(projected_points[:, 2])
            * (rows[:, 0] - projected_points[:, 0])
            + np.cos(projected_points[:, 2])
            * (rows[:, 1] - projected_points[:, 1]))
        lateral_reach = (
            rows[:, 4] + cfg.ego_disc_radius + cfg.obstacle_margin
            + cfg.avoid_max_offset + 0.3)
        # Use one spatial horizon consistently.  Previously _relevant() used
        # the 6 m corridor but the follow governor retained projected rows as
        # far as 13 m ahead.  Around the ifac wrap that admitted wall fragments
        # on the far side of the hairpin and made FOLLOW start several seconds
        # before the actual corner-exit obstacle entered the lane planner.  A
        # lateral gate is also necessary on folded sections: a return can be
        # spatially close to one branch but project several metres outside the
        # branch the ego is actually following (bag mpc_20260821_215058 had 27
        # displayed fragments and up to 17 planner rows around s=28..32).
        outside_corridor = (
            (signed < -0.6)
            | (signed > self.LANE_LOOKAHEAD)
            | (np.abs(projected_laterals) > lateral_reach))
        relevant[row_indices[outside_corridor]] = False
        keep = row_indices[~outside_corridor]
        geometry = []
        if keep.size:
            ahead_rows = rows[~outside_corridor]
            points = projected_points[~outside_corridor]
            laterals = projected_laterals[~outside_corridor]
            gaps = signed[~outside_corridor]
            proj_s = projected[~outside_corridor]
            # The anti-stall relax may shave planning margin but never the
            # hard collision distance itself.  Padding this for movers was
            # measured and rejected; see MPCConfig above.
            required = np.maximum(
                ahead_rows[:, 4] + cfg.ego_disc_radius + 0.05
                - self._stall_relax,
                ahead_rows[:, 4] + cfg.ego_disc_radius + 0.02)
            geometry = list(zip(gaps.tolist(), laterals.tolist(),
                                required.tolist()))
        if not keep.size:
            self.last_replanned = False
            self._cached_target = 0.0
            self._cached_blocked = float('inf')
            self._cached_scene = None
            self._replan_elapsed = float('inf')
            self.last_follow_cap = None
            self.last_lane_clearance = float('inf')
            self.last_lane_safe = False
            self._reset_open_pass()
            self._target_offset = self._slew(self._target_offset, 0.0)
            if abs(self._target_offset) < 0.5 * cfg.avoid_offset_step:
                self._lane_committed = False
            profile = self._ramp(ego_lateral, self._target_offset, base)
            reference = self._shift(base, profile, None, None)
            self._recovery_governor(
                reference, state, ego_lateral - self._target_offset, times)
            return reference, nearest_s, \
                path_distance, self._target_offset, np.empty((0, 5))
        centers = (obstacles[None, relevant, 0:2]
                   + obstacles[None, relevant, 2:4] * times[:, None, None])
        radii = obstacles[relevant, 4]
        ahead = [g for g in geometry if 0.05 < g[0] < self.LANE_LOOKAHEAD]
        maneuver_length = float(np.clip(
            max(g[0] for g in ahead) + 1.5, 2.0, self.LANE_LOOKAHEAD)) \
            if ahead else 2.0

        moving = np.hypot(obstacles[relevant, 2],
                          obstacles[relevant, 3]) > 0.1
        # Lap memory lookup: the nearest STATIC obstacle ahead that matches a
        # remembered pass proposes its stored lane.  _select_target still
        # re-validates it against the current scene before adopting it.
        memory_offset = None
        if cfg.lane_memory_enabled:
            static_rows = np.hypot(ahead_rows[:, 2], ahead_rows[:, 3]) <= 0.1
            ahead_static = [
                (float(g), float(s_o), float(l_o), float(r_o))
                for g, s_o, l_o, r_o, is_static in zip(
                    gaps, proj_s, laterals, ahead_rows[:, 4], static_rows)
                if is_static and 0.05 < g < self.LANE_LOOKAHEAD]
            if ahead_static:
                _, s_o, l_o, r_o = min(ahead_static)
                entry = self._lane_memory_match(s_o, l_o, r_o)
                if entry is not None:
                    memory_offset = float(entry['offset'])
        target, best_blocked = self._select_target(
            base, normal_x, normal_y, ego_lateral, centers, radii, ahead,
            nearest_s, maneuver_length, memory_offset=memory_offset)

        if best_blocked > 0.02 and np.any(moving):
            # No lane passes a moving obstacle cleanly: do not start a
            # side-by-side attempt into a shrinking corridor - hold the
            # centre and let the follow governor wait for a real opening.
            target = 0.0
            self._cached_target = target
            self.last_lane_safe = False

        # Keep the chosen side while the obstacle is beside the car.  The fan
        # scores only its forward corridor; without this commitment, noisy
        # cluster geometry can make every candidate tie as the obstacle leaves
        # that corridor and command a lane change back across the obstacle.
        target = self._commit(target, gaps, laterals, required)
        if abs(target) >= 0.5 * cfg.avoid_offset_step:
            self._lane_committed = True

        # ``target`` is the terminal lane, not a lateral actuator command.
        # Slewing it delayed an otherwise smooth quintic path by 0.3-0.5 s
        # and made the car brake before the lane change had developed.  Move
        # the terminal choice immediately; _ramp() still starts at the actual
        # ego lateral position with zero first/second derivative, so the
        # published path remains smooth and physically steerable.
        self._target_offset = float(np.clip(
            target, -cfg.avoid_max_offset, cfg.avoid_max_offset))
        relevant_obstacles = obstacles[relevant]

        # Validate the path that will actually be published after commitment.
        # A safe terminal fan target alone is insufficient:
        # the car may still be midway through the quintic lane change.
        profile = self._ramp(ego_lateral, self._target_offset, base)
        preview = self._shift(base, profile, None, None)
        clearance_profile = self._footprint_obstacle_clearance_profile(
            preview, centers, radii)
        obstacle_clearance = float(np.min(clearance_profile))
        obstacle_start = float(clearance_profile[0])
        obstacle_end = float(clearance_profile[-1])
        wall_clearance_profile = self._footprint_wall_clearance_profile(
            preview)
        wall_clearance = float(np.min(wall_clearance_profile))
        steerable = self._path_is_steerable(preview[:, 3])
        # Measure the obstacle-to-wall opening at one physical cross-section:
        # the preview sample closest to the blocker.  Adding independent
        # minima from different corner/horizon samples understated the actual
        # passage in bag 20260823_215739 (measured residual ~0.365 m).
        closest = int(np.argmin(clearance_profile))
        free_gap = (max(float(clearance_profile[closest]), 0.0)
                    + max(float(wall_clearance_profile[closest]), 0.0))
        # A full-speed pass is never inferred from a terminal lane alone and
        # never admitted while the conservative footprint overlaps anything.
        # Tracker-inflated overlap remains under the normal solver/follow
        # safety policy; only positive executable clearance may disable
        # obstacle-induced speed shaping.
        executed_safe = (
            (obstacle_clearance >= 0.02)
            and wall_clearance >= 0.01
            and steerable
            and free_gap >= cfg.avoid_full_speed_gap)
        if executed_safe:
            self._open_pass_confirmations += 1
        else:
            # Unsafe geometry revokes the fast pass immediately; only entry
            # is debounced against PF/tracker jitter.
            self._open_pass_confirmations = 0
        self.last_open_pass = (
            self._open_pass_confirmations
            >= cfg.avoid_full_speed_confirm_cycles)
        self.last_executed_obstacle_clearance = obstacle_clearance
        self.last_executed_obstacle_start = obstacle_start
        self.last_executed_obstacle_end = obstacle_end
        self.last_executed_wall_clearance = wall_clearance
        self.last_executed_gap = free_gap

        # Record the lane actually used while passing each static obstacle
        # (inside the lane-change distance, moving, on a currently
        # collision-free executable path).  The final update as the obstacle
        # falls behind is what the next lap's memory lookup reuses.
        if (cfg.lane_memory_enabled and state.speed > 0.8 and steerable
                and obstacle_clearance >= 0.02 and wall_clearance >= 0.01):
            memorable = (
                (np.hypot(ahead_rows[:, 2], ahead_rows[:, 3]) <= 0.1)
                & (gaps < cfg.avoid_transition_distance))
            for s_o, l_o, r_o in zip(proj_s[memorable],
                                     laterals[memorable],
                                     ahead_rows[memorable, 4]):
                self._lane_memory_record(s_o, l_o, r_o, self._target_offset)

        follow_cap = (None if self.last_open_pass else self._follow_cap(
            state, relevant_obstacles, gaps, laterals, points, required,
            self._target_offset))
        self.last_follow_cap = follow_cap

        # Anti-stall: a prolonged standstill near static blockers gradually
        # relaxes the wait decision (never the collision constraints in the
        # solver) so the widest physically drivable gap gets used.  Any
        # standstill with obstacles around counts - freezes also happen with
        # a partially released cap when the cost equilibrium pins the car.
        moving_blockers = np.any(np.hypot(relevant_obstacles[:, 2],
                                          relevant_obstacles[:, 3]) > 0.1)
        if self.last_open_pass:
            # A previously accumulated anti-stall state must not turn a newly
            # certified full-speed pass back into a permanent 0.5 m/s creep.
            self._stall_time = 0.0
            self._stall_relax = 0.0
        elif state.speed < 0.2 and not moving_blockers:
            self._stall_time += cfg.control_period
        elif state.speed > 0.5:
            self._stall_time = 0.0
        # A standstill with flickering detections keeps its stall time: the
        # escape pressure must keep building until the vehicle truly moves.
        # Wall-clock based (3 s grace, 0.06/s ramp - the values previously
        # tuned as 60 cycles / 0.003 per cycle at the 20 Hz control rate) so
        # the anti-stall behavior is invariant to the control rate.
        self._stall_relax = min(0.20, 0.06 * max(self._stall_time - 3.0, 0.0))
        if self._stall_relax > 0.0 and follow_cap is not None:
            creep = min(0.5, 3.0 * self._stall_relax)
            follow_cap = max(follow_cap, creep)
            self.last_follow_cap = follow_cap

        if follow_cap is not None:
            # Rebuild the base so the reference points themselves stop
            # advancing: capping only the speed column would leave the lag
            # cost dragging the vehicle into whatever blocks it.  No
            # minimum-speed floor - waiting beats squeezing past.  The pose
            # projection and the propagated obstacle centres are unchanged by
            # the cap; only the reference march is redone.
            base, nearest_s, path_distance = self.raceline.reference(
                state, cfg, hint_s=hint_s, speed_limit=max(follow_cap, 0.0),
                lateral_accel_limit=runtime_limit,
                projection=(nearest_s, path_distance))
            normal_x = -np.sin(base[:, 2])
            normal_y = np.cos(base[:, 2])

        profile = self._solved_profile(state, obstacles, relevant, base,
                                       nearest_s)
        if profile is None:
            profile = self._ramp(ego_lateral, self._target_offset, base)
        reference = self._shift(base, profile, None, None)
        if not self.last_open_pass:
            # Unlike the retired centre-point estimate, use the same front and
            # rear footprint discs as the solver for proximity speed shaping.
            clearance = self._footprint_obstacle_clearance_profile(
                reference, centers, radii)
            self._apply_obstacle_speed_factor(reference, clearance)
        # No speed surges while maneuvering around obstacles: the reference
        # may accelerate only moderately from the current speed.
        surge_limit = max(state.speed, cfg.minimum_reference_speed) \
            + 2.5 * times
        reference[:, 4] = np.minimum(reference[:, 4], surge_limit)
        if self._stall_relax > 0.0:
            # Positive escape incentive: without a floor the cost equilibrium
            # can keep the car parked even after the cap is released.
            creep = min(0.5, 3.0 * self._stall_relax)
            reference[:, 4] = np.maximum(reference[:, 4], creep)
        # Obstacles are actively relevant: arm the post-avoidance recovery
        # governor so the return to the line stays speed-disciplined after
        # they drop out of the corridor.
        self._recovery_time = cfg.recovery_hold_time
        return reference, nearest_s, path_distance, self._target_offset, \
            relevant_obstacles

    @staticmethod
    def _build_profile_planner(raceline: 'Raceline', config: MPCConfig,
                               wall_field):
        """The vendored advanced_pp profile solver, or None when unused."""
        if not config.avoid_profile_solver or wall_field is None:
            return None
        from . import pp_avoidance
        widths = _corridor_widths(raceline, wall_field)
        data = np.column_stack([
            raceline.s, raceline.x, raceline.y, raceline.heading,
            raceline.curvature, raceline.speed, raceline.acceleration,
            widths[:, 0], widths[:, 1]])
        # heading_offset 0: apex stores the driving heading already, so the
        # TUM quarter-turn detection must not fire a second time.
        line = pp_avoidance.Raceline(data, name='apex', heading_offset=0.0)
        return pp_avoidance.AvoidancePlanner(
            line, pp_avoidance.VehicleParams(),
            pp_avoidance.AvoidanceConfig(), wall_field)

    def _solved_profile(self, state: VehicleState, obstacles, relevant,
                        base: np.ndarray, nearest_s: float):
        """Lateral offset per reference station from the C2 profile solver.

        Returns None whenever the solver is disabled or declines, and the
        caller falls back to the constant-offset ramp.

        The fan search picks ONE lateral offset and ramps onto it, so a lane
        is only viable if it fits every pinch of the window; the profile
        solver shapes ``d(s)`` against the tube instead.  On the qualifying
        scene of bag mpc_20260825_141120 that is 0.50 m of body clearance
        past the obstacle against the fan's 0.09 m, and the pass is committed
        on the cycle after detection rather than 0.6 s later.
        """
        planner = self._profile_planner
        if planner is None or obstacles is None or not np.any(relevant):
            return None
        rows = obstacles[relevant]
        # The solver plans around things that stay put; movers keep the fan
        # search, whose obstacle propagation this does not reproduce.
        static = np.hypot(rows[:, 2], rows[:, 3]) <= 0.1
        if not np.any(static):
            return None
        tracked = [_ProfileObstacle(index, float(row[0]), float(row[1]),
                                    float(row[4]))
                   for index, row in enumerate(rows[static])]
        try:
            planner.update(tracked, (state.x, state.y), float(state.speed))
        except Exception:                      # never let it stop the car
            self._profile_planner = None
            return None
        plan = planner.plan
        if plan is None or plan.profile is None:
            return None
        grid = planner.raceline.s
        stations = np.mod(nearest_s + self._progress(base),
                          self.raceline.total_length)
        profile = np.interp(stations, grid, plan.profile[0],
                            period=self.raceline.total_length)
        # No re-anchoring on the car's own cross-track error: the profile is
        # where the LINE should be, and closing the error to it is the
        # tracker's job.  Biasing the whole manoeuvre by the current lateral
        # dragged the reference to the blocked side for 1.08 s.
        # Bounded by the solver's OWN tube, not by avoid_max_offset: the
        # fixed ceiling is what the fan search needs because a constant
        # offset has no other bound, and applying it here would throw away
        # exactly the displacement the tube says is available (+1.00 m on the
        # qualifying scene against a 0.70 m ceiling).  A correct solve never
        # touches this; a broken one cannot leave the corridor.
        hi = np.interp(stations, grid, planner._tube_hi,
                       period=self.raceline.total_length)
        lo = np.interp(stations, grid, planner._tube_lo,
                       period=self.raceline.total_length)
        profile = np.clip(profile, lo, hi)
        if not np.all(np.isfinite(profile)):
            return None
        # Downstream - lane memory, the follow cap, the recovery governor and
        # the telemetry - all speak the scalar lane.  Report the deepest point
        # of the profile so they keep describing the same manoeuvre.
        deepest = int(np.argmax(np.abs(profile)))
        self._target_offset = float(profile[deepest])
        return profile

    def _reset_open_pass(self) -> None:
        self._open_pass_confirmations = 0
        self.last_open_pass = False
        self.last_executed_obstacle_clearance = float('inf')
        self.last_executed_obstacle_start = float('inf')
        self.last_executed_obstacle_end = float('inf')
        self.last_executed_wall_clearance = float('inf')
        self.last_executed_gap = float('inf')

    def _select_target(self, base: np.ndarray, normal_x: np.ndarray,
                       normal_y: np.ndarray, ego_lateral: float,
                       centers: np.ndarray, radii: np.ndarray, ahead,
                       nearest_s: float,
                       maneuver_length: float,
                       memory_offset: Optional[float] = None
                       ) -> Tuple[float, float]:
        """Choose a lane with cached validation and coarse-to-fine search.

        Repeated tracker frames reuse the last winning fine-grid lane for a
        short time.  Scene changes invalidate immediately and the short
        periodic refresh bounds reuse as the ego and real movers advance.
        ``memory_offset`` is a lane remembered from a previous pass of the
        matched obstacle; it wins outright if it is still hard-safe against
        the current scene, otherwise the normal search runs.
        """
        cfg = self.config
        self.last_replanned = False
        self._replan_elapsed += cfg.control_period
        # Tracker output is commonly repeated across several control
        # cycles.  Re-solving the same static lane fan cannot improve safety.
        # A new row, a 15 cm centre displacement, or an 8 cm radius growth
        # invalidates immediately; anything smaller stays inside the 20 cm
        # obstacle margin until the mandatory periodic refresh.  The gates
        # must exceed the real car's cycle-to-cycle PF pose jitter, which
        # moves every obstacle's WORLD coordinates: an earlier 8 cm gate
        # degenerated into a full replan at the control rate exactly in the
        # corners where the PF error peaks (bag mpc_20260820_185730: 27-55 ms
        # avg cycles while avoiding).
        scene = np.column_stack((centers[0, :, :], radii))
        scene = scene[np.lexsort((scene[:, 1], scene[:, 0]))]
        scene_unchanged = (
            self._cached_scene is not None
            and scene.shape == self._cached_scene.shape
            and (scene.shape[0] == 0 or (
                np.max(np.linalg.norm(
                    scene[:, :2] - self._cached_scene[:, :2], axis=1)) < 0.15
                and np.max(np.abs(
                    scene[:, 2] - self._cached_scene[:, 2])) < 0.08)))
        # A scene that has already sat still for a while earns a longer
        # refresh: unchanged geometry cannot change lane viability, only ego
        # progress can, and the tiebreak's current-lane preference plus the
        # immediate scene-change invalidation still bound staleness.  At the
        # stretched period the ego advances at most ~1.5 m between refreshes
        # against the planner's 6 m corridor.
        if not scene_unchanged:
            self._scene_stable_time = 0.0
        else:
            self._scene_stable_time += cfg.control_period
        effective_period = cfg.avoid_replan_period
        if self._scene_stable_time >= self.STABLE_SCENE_TIME:
            effective_period *= self.STABLE_REPLAN_FACTOR
        cache_due = (
            cfg.avoid_replan_period <= 0.0
            or self._replan_elapsed + 1e-12 >= effective_period)
        if not cache_due and scene_unchanged:
            return self._cached_target, self._cached_blocked

        # Remembered lane first: if the lane used to pass this obstacle on a
        # previous lap is still hard-safe in the current scene, repeat it
        # verbatim instead of re-deriving a new one from noisy geometry.
        if memory_offset is not None:
            memory_index = int(np.argmin(
                np.abs(self._lane_offsets - memory_offset)))
            m_blocked, m_clearance, m_safe, _ = self._candidate_metrics(
                base, normal_x, normal_y, ego_lateral,
                np.array([memory_index], dtype=int),
                centers, radii, ahead, nearest_s, maneuver_length)
            if bool(m_safe[0]):
                target = float(self._lane_offsets[memory_index])
                self.last_lane_clearance = float(m_clearance[0])
                self.last_lane_safe = True
                self.last_memory_hit = True
                self._cached_target = target
                self._cached_blocked = float(m_blocked[0])
                self._cached_scene = scene.copy()
                self._replan_elapsed = 0.0
                self.last_replanned = True
                return target, float(m_blocked[0])
        self.last_memory_hit = False

        lane_count = self._lane_offsets.shape[0]
        stride = max(1, int(round(
            cfg.avoid_coarse_offset_step / cfg.avoid_offset_step)))
        coarse_indices = np.arange(0, lane_count, stride, dtype=int)
        if coarse_indices[-1] != lane_count - 1:
            coarse_indices = np.append(coarse_indices, lane_count - 1)
        coarse_blocked, coarse_clearance, coarse_safe, coarse_drivable = \
            self._candidate_metrics(
                base, normal_x, normal_y, ego_lateral, coarse_indices,
                centers, radii, ahead, nearest_s, maneuver_length)
        coarse_candidates = self._lane_offsets[coarse_indices]
        coarse_tiebreak = self._candidate_tiebreak(
            coarse_candidates, coarse_clearance)
        # Refine physically drivable basins first.  A low obstacle-overlap
        # score is useless if the shifted path itself runs through a wall or
        # asks for curvature beyond the steering rack.
        coarse_score = coarse_blocked + 1e3 * (~coarse_drivable)
        coarse_order = np.lexsort((coarse_tiebreak, coarse_score))

        # Refine the best few coarse basins.  Including the currently applied
        # offset preserves the original lane hysteresis even when it does not
        # happen to lie on the coarse grid.
        refine = set()
        # Each coarse anchor owns only the half-cell up to its neighbours.
        # Expanding a full coarse stride evaluated most neighbouring basins a
        # second time; the selected top basins already represent those cells.
        refine_radius = max(1, (stride + 1) // 2)
        for position in coarse_order[:min(cfg.avoid_refine_top_k,
                                          coarse_order.size)]:
            centre = int(coarse_indices[position])
            begin = max(0, centre - refine_radius)
            end = min(lane_count, centre + refine_radius + 1)
            refine.update(range(begin, end))
        current_index = int(np.argmin(
            np.abs(self._lane_offsets - self._target_offset)))
        refine.add(current_index)
        coarse_set = set(coarse_indices.tolist())
        fine_indices = np.array(sorted(refine - coarse_set), dtype=int)

        indices = coarse_indices
        blocked_depth = coarse_blocked
        lane_clearance = coarse_clearance
        hard_safe = coarse_safe
        drivable = coarse_drivable
        if fine_indices.size:
            fine_blocked, fine_clearance, fine_safe, fine_drivable = \
                self._candidate_metrics(
                    base, normal_x, normal_y, ego_lateral, fine_indices,
                    centers, radii, ahead, nearest_s, maneuver_length)
            indices = np.concatenate((indices, fine_indices))
            blocked_depth = np.concatenate((blocked_depth, fine_blocked))
            lane_clearance = np.concatenate((lane_clearance, fine_clearance))
            hard_safe = np.concatenate((hard_safe, fine_safe))
            drivable = np.concatenate((drivable, fine_drivable))

        candidates = self._lane_offsets[indices]
        tiebreak = self._candidate_tiebreak(candidates, lane_clearance)
        safe_positions = np.flatnonzero(hard_safe)
        drivable_positions = np.flatnonzero(drivable)
        if safe_positions.size:
            pool = safe_positions
        elif drivable_positions.size:
            # A narrow corner can have no collision-free lane at the current
            # transition distance.  Keep creeping toward the least-blocked
            # path, but only among wall- and steering-feasible candidates.
            pool = drivable_positions
        else:
            # Do not command the old +/-0.85 m escape into the wall when the
            # whole local fan is infeasible.  Relax toward the base line while
            # FOLLOW regulates speed and the next spatial refresh progresses.
            pool = np.array([int(np.argmin(np.abs(candidates)))])
        order = np.lexsort((tiebreak[pool], blocked_depth[pool]))
        best_index = int(pool[int(order[0])])
        best_blocked = float(blocked_depth[best_index])
        target = float(candidates[best_index])

        # Lane hysteresis: while the current target lane stays clean, keep
        # it.  The explicit current_index insertion above guarantees that the
        # safety score is present even in a sparse coarse search.
        current_position = np.flatnonzero(indices == current_index)
        if current_position.size:
            position = int(current_position[-1])
            if drivable[position] and blocked_depth[position] < 0.01:
                best_blocked = float(blocked_depth[position])
                target = float(candidates[position])
                best_index = position

        self.last_lane_clearance = float(lane_clearance[best_index])
        self.last_lane_safe = bool(hard_safe[best_index])

        self._cached_target = target
        self._cached_blocked = best_blocked
        self._cached_scene = scene.copy()
        self._replan_elapsed = 0.0
        self.last_replanned = True
        return target, best_blocked

    def _commit(self, target: float, gaps: np.ndarray,
                laterals: np.ndarray, required: np.ndarray) -> float:
        """Keep one side of every obstacle until the rear of the car is past.

        The bounds are accumulated before applying them so two obstacle rows
        cannot produce a row-order-dependent answer.  If obstacles on both
        sides make the bounds conflict, holding the current lane is the only
        non-flipping choice; the executable-path check keeps the speed
        governor active in that case.
        """
        # Commit as soon as a blocker enters the complete lane-change
        # distance.  Waiting until half that distance allowed tracker jitter
        # to select the opposite side while the car was already beginning
        # its first manoeuvre (bag 20260823_215739 at t=139.10..139.30), which
        # consumed the remaining approach and forced FOLLOW to stop.
        alongside = np.asarray(gaps) < self.config.avoid_transition_distance
        if not np.any(alongside):
            return float(target)

        current = float(self._target_offset)
        lower = -self.config.avoid_max_offset
        upper = self.config.avoid_max_offset
        for lateral, needed in zip(
                np.asarray(laterals)[alongside],
                np.asarray(required)[alongside]):
            # Before a terminal lane is actually clear of this row, ``0.0``
            # is not a committed side.  Using it to break a centred tie
            # inverted the fan's only safe + lane into the - lane in bag
            # 20260823_215739.  Let the validated fan target choose the first
            # side; once the previous terminal lane has the required lateral
            # separation, hold that side until the obstacle is behind.
            anchor = current if self._lane_committed else float(target)
            if anchor <= lateral:
                upper = min(upper, current, float(lateral - needed))
            else:
                lower = max(lower, current, float(lateral + needed))
        if lower > upper:
            return float(np.clip(current, -self.config.avoid_max_offset,
                                 self.config.avoid_max_offset))
        return float(np.clip(target, lower, upper))

    def _footprint_obstacle_clearance_profile(
            self, reference: np.ndarray, centers: np.ndarray,
            radii: np.ndarray) -> np.ndarray:
        """Per-step obstacle gap for the same two discs used by the solver."""
        cfg = self.config
        direction = np.stack(
            (np.cos(reference[:, 2]), np.sin(reference[:, 2])), axis=-1)
        origin = reference[:, 0:2]
        clearance = np.full(reference.shape[0], np.inf)
        for offset in (cfg.ego_center_offset + cfg.ego_disc_offset,
                       cfg.ego_center_offset - cfg.ego_disc_offset):
            disc = origin + offset * direction
            gap = (np.linalg.norm(
                disc[:, None, :] - centers, axis=-1)
                - radii[None, :] - cfg.ego_disc_radius)
            clearance = np.minimum(clearance, gap.min(axis=1))
        return clearance

    def _footprint_obstacle_clearance(
            self, reference: np.ndarray, centers: np.ndarray,
            radii: np.ndarray) -> float:
        return float(np.min(self._footprint_obstacle_clearance_profile(
            reference, centers, radii)))

    def _footprint_wall_clearance_profile(
            self, reference: np.ndarray) -> np.ndarray:
        if self.wall_field is None:
            return np.full(reference.shape[0], np.inf)
        cfg = self.config
        direction = np.stack(
            (np.cos(reference[:, 2]), np.sin(reference[:, 2])), axis=-1)
        origin = reference[:, 0:2]
        clearance = np.full(reference.shape[0], np.inf)
        for offset in (cfg.ego_center_offset + cfg.ego_disc_offset,
                       cfg.ego_center_offset - cfg.ego_disc_offset):
            disc = origin + offset * direction
            value = self.wall_field.clearance(disc[:, 0], disc[:, 1]) \
                - cfg.ego_disc_radius
            clearance = np.minimum(clearance, value)
        return clearance

    def _footprint_wall_clearance(self, reference: np.ndarray) -> float:
        return float(np.min(self._footprint_wall_clearance_profile(reference)))

    def _path_is_steerable(self, curvature: np.ndarray) -> bool:
        cfg = self.config
        positive = np.tan(cfg.steering_gain * cfg.max_steer) / cfg.wheelbase
        negative = abs(
            np.tan(cfg.steering_gain * cfg.min_steer) / cfg.wheelbase)
        curvature = np.asarray(curvature, dtype=float)
        return bool(np.all(np.where(curvature >= 0.0,
                                    curvature <= positive + 1e-9,
                                    -curvature <= negative + 1e-9)))

    def _apply_obstacle_speed_factor(
            self, reference: np.ndarray, clearance: np.ndarray) -> None:
        cfg = self.config
        extra = np.asarray(clearance, dtype=float) - cfg.obstacle_margin
        factor = np.clip(
            cfg.avoid_min_speed_factor
            + (1.0 - cfg.avoid_min_speed_factor)
            * extra / max(cfg.avoid_comfort_clearance, 1e-3),
            cfg.avoid_min_speed_factor, 1.0)
        reference[:, 4] *= factor

    def _candidate_tiebreak(self, candidates: np.ndarray,
                            lane_clearance: np.ndarray) -> np.ndarray:
        return (0.6 * np.abs(candidates - self._target_offset)
                + 0.4 * np.abs(candidates)
                - 0.8 * np.clip(lane_clearance, 0.0, 0.5))

    def _candidate_metrics(self, base: np.ndarray, normal_x: np.ndarray,
                           normal_y: np.ndarray, ego_lateral: float,
                           lane_indices: np.ndarray, centers: np.ndarray,
                           radii: np.ndarray, ahead, nearest_s: float,
                           maneuver_length: float) -> Tuple[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray]:
        """Return obstacle, clearance, safety and drivability lane metrics."""
        cfg = self.config
        lane_indices = np.asarray(lane_indices, dtype=int).reshape(-1)
        candidates = self._lane_offsets[lane_indices]
        profiles = self._profiles(ego_lateral, candidates, base)
        shifted = np.stack((
            base[None, :, 0] + normal_x[None, :] * profiles,
            base[None, :, 1] + normal_y[None, :] * profiles), axis=-1)
        candidate_heading, candidate_curvature = self._geometry_batch(
            shifted, base[:, 2])
        direction = np.stack(
            (np.cos(candidate_heading), np.sin(candidate_heading)), axis=-1)
        front = cfg.ego_center_offset + cfg.ego_disc_offset
        rear = cfg.ego_center_offset - cfg.ego_disc_offset
        candidate_discs = (
            shifted + front * direction,
            shifted + rear * direction,
        )
        lane_clearance = np.full(candidates.shape[0], np.inf)
        for disc in candidate_discs:
            clearance = (np.linalg.norm(
                disc[:, :, None, :] - centers[None, :, :, :], axis=-1)
                - radii[None, None, :] - cfg.ego_disc_radius)
            lane_clearance = np.minimum(
                lane_clearance, clearance.min(axis=(1, 2)))

        time_deficit = np.maximum(
            0.5 * cfg.obstacle_margin - lane_clearance, 0.0)
        if ahead:
            ahead_gaps = np.array([g[0] for g in ahead])
            ahead_lateral = np.array([g[1] for g in ahead])
            ahead_required = np.array([g[2] for g in ahead])
            candidate_delta = candidates - ego_lateral
            transition = np.maximum(
                cfg.avoid_transition_distance,
                4.0 * np.abs(candidate_delta))
            u = np.clip(
                ahead_gaps[None, :] / transition[:, None], 0.0, 1.0)
            smooth = u ** 3 * (10.0 + u * (-15.0 + 6.0 * u))
            lateral_at_obstacle = (
                ego_lateral + candidate_delta[:, None] * smooth)
            geometry_deficit = np.maximum(
                ahead_required[None, :]
                - np.abs(lateral_at_obstacle - ahead_lateral[None, :]),
                0.0).max(axis=1)
        else:
            geometry_deficit = np.zeros(candidates.shape[0])
        obstacle_deficit = np.maximum(time_deficit, geometry_deficit)

        wall_deficit = np.zeros(candidates.shape[0])
        wall_clearance = np.full(candidates.shape[0], np.inf)
        if self._lane_clearance is not None:
            threshold = cfg.ego_disc_radius + 0.5 * cfg.wall_margin
            start = int(nearest_s / self.LANE_GRID_STEP)
            count = max(
                int(np.ceil(maneuver_length / self.LANE_GRID_STEP)), 1)
            grid = (start + np.arange(count)) % self._lane_grid_s.shape[0]
            # Judge each station against the offset the RAMP will have
            # reached there, not against the target held constant from the
            # car to the end of the manoeuvre.  Holding it constant makes a
            # lane's viability depend on the narrowest point of the whole
            # window even when the car will still be near the raceline
            # there, and that is what left every lane beyond +-0.20 m
            # undrivable at s=16.6 in bag mpc_20260825_141120: the corridor
            # 0.4 m ahead is 0.57 m wide, the corridor at the obstacle 4.7 m
            # ahead is 1.32 m, and a constant +0.70 m fails on the former
            # while the ramp is still at +0.05 m there.  Past the transition
            # the ramp has arrived and this is the same test as before.
            distances = np.arange(count) * self.LANE_GRID_STEP
            ramp_delta = candidates - ego_lateral
            ramp_transition = np.maximum(
                cfg.avoid_transition_distance, 4.0 * np.abs(ramp_delta))
            ramp_u = np.clip(
                distances[None, :] / ramp_transition[:, None], 0.0, 1.0)
            ramp_smooth = ramp_u ** 3 * (
                10.0 + ramp_u * (-15.0 + 6.0 * ramp_u))
            ramp = ego_lateral + ramp_delta[:, None] * ramp_smooth
            ramp_index = np.clip(np.rint(
                (ramp - self._lane_offsets[0]) / cfg.avoid_offset_step
            ).astype(int), 0, self._lane_offsets.shape[0] - 1)
            grid_clearance = self._lane_clearance[
                ramp_index, grid[None, :]].min(axis=1)
            wall_clearance = np.minimum(wall_clearance, grid_clearance)
            wall_deficit = np.maximum(threshold - grid_clearance, 0.0)
            transition_clearance = np.full(candidates.shape[0], np.inf)
            for disc in candidate_discs:
                clearance = self.wall_field.clearance(
                    disc[:, :, 0].ravel(), disc[:, :, 1].ravel()
                ).reshape(disc.shape[0], disc.shape[1])
                transition_clearance = np.minimum(
                    transition_clearance, clearance.min(axis=1))
            wall_deficit = np.maximum(
                wall_deficit,
                np.maximum(threshold - transition_clearance, 0.0))
            wall_clearance = np.minimum(
                wall_clearance, transition_clearance)

        positive_limit = (
            np.tan(cfg.steering_gain * cfg.max_steer) / cfg.wheelbase)
        negative_limit = (
            abs(np.tan(cfg.steering_gain * cfg.min_steer)) / cfg.wheelbase)
        curvature_excess = np.maximum(
            np.where(candidate_curvature >= 0.0,
                     candidate_curvature - positive_limit,
                     -candidate_curvature - negative_limit),
            0.0).max(axis=1)
        blocked_depth = (1.5 * obstacle_deficit + wall_deficit
                         + 0.5 * curvature_excess)
        drivable = (
            (wall_clearance >= cfg.ego_disc_radius + cfg.wall_hard_margin)
            & (curvature_excess <= 1e-9))
        hard_safe = (
            drivable
            & (lane_clearance >= 0.02)
            & (geometry_deficit <= 1e-9))
        return blocked_depth, lane_clearance, hard_safe, drivable

    def _follow_cap(self, state: VehicleState, obstacles: np.ndarray,
                    gaps: np.ndarray, laterals: np.ndarray,
                    points: np.ndarray, required: np.ndarray,
                    offset: float) -> Optional[float]:
        """Speed cap while an obstacle ahead is not cleared laterally.

        Tracking the raceline speed into the back of a slower vehicle is
        otherwise cheaper for the optimizer than braking behind it.  The
        per-row raceline geometry (arc gap, lateral offset, sampled point,
        required clearance) is reused from the lane-selection pass instead of
        re-projecting every obstacle onto the raceline.
        """
        cfg = self.config
        cap = None
        lookahead = self.LANE_LOOKAHEAD
        for row, gap, lateral, point, row_required in zip(
                obstacles, gaps, laterals, points, required):
            gap = float(gap)
            if gap < 0.05 or gap > lookahead:
                continue
            # In corners the straight-line separation is much smaller than
            # the arc gap; follow at whichever distance is tighter.
            gap = min(gap, float(np.hypot(row[0] - state.x, row[1] - state.y)))
            lateral_gap = abs(offset - float(lateral))
            clear_margin = max(cfg.follow_clear_margin, 1e-3)
            if lateral_gap >= row_required + clear_margin:
                continue
            # Blend the cap out continuously as the lane clears laterally;
            # a hard blocked/clear switch oscillates on noisy clusters.
            blend = max((lateral_gap - row_required) / clear_margin, 0.0)
            along = (row[2] * np.cos(point[2]) + row[3] * np.sin(point[2]))
            moving = float(np.hypot(row[2], row[3])) > 0.1
            if moving:
                # Hang further back through corners: passing or tailing a
                # vehicle mid-corner is where side contact happens, so close
                # in (and overtake) on the straights instead.
                curvature = abs(float(point[3]))
                hold_distance = cfg.follow_distance * (
                    1.0 + 1.5 * min(curvature, 1.5))
            else:
                hold_distance = 0.7 * cfg.follow_distance
            # Negative gap error must slow us below the obstacle speed so a
            # shrunken following distance opens back up.
            blocked = max(max(along, 0.0)
                          + cfg.follow_gain * (gap - hold_distance), 0.0)
            limit = blocked + blend * (cfg.max_speed - blocked)
            if not moving:
                # Never park in front of a static obstacle: a full stop is
                # where lane commitment degenerates and freezes develop.  The
                # solver's clearance constraints handle the final protection
                # while crawling past.
                limit = max(limit, 0.6)
            cap = limit if cap is None else min(cap, limit)
        return cap

    def _relevant(self, base: np.ndarray, centers: np.ndarray,
                  radii: np.ndarray) -> np.ndarray:
        cfg = self.config
        distance = np.linalg.norm(
            base[:, None, 0:2] - centers, axis=-1).min(axis=0)
        reach = (radii + cfg.ego_disc_radius + cfg.obstacle_margin
                 + cfg.avoid_max_offset + 0.3)
        return distance < reach

    def _slew(self, current: float, target: float) -> float:
        # Per control cycle, not per model step: the physical slew rate must
        # not scale with the control rate.
        limit = self.config.avoid_lateral_slew * self.config.control_period
        return current + float(np.clip(target - current, -limit, limit))

    @staticmethod
    def _progress(base: np.ndarray) -> np.ndarray:
        segment = np.hypot(np.diff(base[:, 0]), np.diff(base[:, 1]))
        # Raceline.reference always marches, but retain a numerical floor so a
        # zero speed cap or duplicate input point cannot poison the geometry.
        return np.concatenate(([0.0], np.cumsum(np.maximum(segment, 1e-6))))

    def _profiles(self, start: float, targets: np.ndarray,
                  base: np.ndarray) -> np.ndarray:
        """Distance-parameterized quintic lane changes for all targets."""
        targets = np.asarray(targets, dtype=float).reshape(-1)
        delta = targets - float(start)
        transition = np.maximum(
            self.config.avoid_transition_distance, 4.0 * np.abs(delta))
        u = np.clip(
            self._progress(base)[None, :] / transition[:, None], 0.0, 1.0)
        # Quintic smoothstep: zero first and second derivative at both ends.
        smooth = u ** 3 * (10.0 + u * (-15.0 + 6.0 * u))
        return float(start) + delta[:, None] * smooth

    def _ramp(self, start: float, target: float, base: np.ndarray) -> np.ndarray:
        return self._profiles(start, np.array([target]), base)[0]

    @staticmethod
    def _gradient_batch(values: np.ndarray,
                        coordinates: np.ndarray) -> np.ndarray:
        """Vectorized first derivative for row-wise nonuniform coordinates."""
        values = np.asarray(values, dtype=float)
        coordinates = np.asarray(coordinates, dtype=float)
        result = np.empty_like(values)
        left = np.maximum(coordinates[:, 1:-1] - coordinates[:, :-2], 1e-6)
        right = np.maximum(coordinates[:, 2:] - coordinates[:, 1:-1], 1e-6)
        span = left + right
        result[:, 1:-1] = (
            -right / (left * span) * values[:, :-2]
            + (right - left) / (left * right) * values[:, 1:-1]
            + left / (right * span) * values[:, 2:])
        result[:, 0] = ((values[:, 1] - values[:, 0])
                        / np.maximum(coordinates[:, 1] - coordinates[:, 0], 1e-6))
        result[:, -1] = ((values[:, -1] - values[:, -2])
                         / np.maximum(coordinates[:, -1] - coordinates[:, -2], 1e-6))
        return result

    @classmethod
    def _geometry_batch(cls, paths: np.ndarray,
                        base_heading: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Headings and geometric curvatures of sampled candidate paths."""
        paths = np.asarray(paths, dtype=float)
        segment = np.hypot(np.diff(paths[:, :, 0], axis=1),
                           np.diff(paths[:, :, 1], axis=1))
        progress = np.concatenate((
            np.zeros((paths.shape[0], 1)),
            np.cumsum(np.maximum(segment, 1e-6), axis=1)), axis=1)
        tangent_x = cls._gradient_batch(paths[:, :, 0], progress)
        tangent_y = cls._gradient_batch(paths[:, :, 1], progress)
        heading = np.unwrap(np.arctan2(tangent_y, tangent_x), axis=1)
        heading += (np.round(
            (base_heading[0] - heading[:, 0]) / (2.0 * np.pi))
            * 2.0 * np.pi)[:, None]
        curvature = cls._gradient_batch(heading, progress)
        return heading, curvature

    def _shift(self, base: np.ndarray, profile: np.ndarray,
               clearance: Optional[np.ndarray], _unused) -> np.ndarray:
        cfg = self.config
        reference = base.copy()
        reference[:, 0] += -np.sin(base[:, 2]) * profile
        reference[:, 1] += np.cos(base[:, 2]) * profile

        if np.max(np.abs(profile)) > 1e-6:
            paths = reference[None, :, 0:2]
            heading, shifted_curvature = self._geometry_batch(
                paths, base[:, 2])
            reference[:, 2] = heading[0]
            shifted_curvature = shifted_curvature[0]
            # Use the curvature of the path we actually generated.  The
            # constant-offset formula k/(1-k*d) omits the ramp derivatives and
            # badly understated corner-entry steering demand.
            curve_limit = np.sqrt(
                self._active_lateral_accel_limit
                / np.maximum(np.abs(shifted_curvature), 1e-3))
            reference[:, 3] = shifted_curvature
            reference[:, 4] = np.minimum(reference[:, 4], curve_limit)

        if clearance is not None:
            extra = clearance - cfg.obstacle_margin
            factor = np.clip(
                cfg.avoid_min_speed_factor
                + (1.0 - cfg.avoid_min_speed_factor)
                * extra / max(cfg.avoid_comfort_clearance, 1e-3),
                cfg.avoid_min_speed_factor, 1.0)
            reference[:, 4] *= factor
        return reference
