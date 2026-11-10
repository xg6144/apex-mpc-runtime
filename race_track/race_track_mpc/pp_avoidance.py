"""Vendored from pp_ws/src/advanced_pp - the C2 evasion-profile planner.

Ported verbatim on 2026-08-25 so apex_ws solves the avoidance offset the way
advanced_pp does: the profile ``d(s)`` is found by a QP inside the track's
free-space tube, an obstacle keyhole and an exact steering-lock box, rather
than by picking one constant lateral offset out of a fan.

Measured on the qualifying scene of bag mpc_20260825_141120 (s=20.6, obstacle
0.22 m left of the line, 1.32 m of room on the left and 0.15 m on the right):
the fan planner produced a 0.091 m body-to-obstacle pass after first
committing 0.6 s to the blocked side, while this solver produced a +1.005 m
profile with 0.497 m of body clearance on the cycle after detection.

Only the imports are rewired; the algorithm and its comments are unchanged so
the two stacks can still be diffed against each other.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def wrap_angle(angle):
    """Wrap a scalar or array of angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi



class VehicleParams:
    """Chassis, tyre and actuator parameters of this car.

    Provenance: `f1tenth_stack/config/vesc.yaml` for the servo map (the driver
    CLIPS commands to servo [0.15, 0.85], i.e. steering [-0.2705, +0.2278] rad
    after the B210 swap), and `tools/fit_lookup_from_bag.py` on the
    mpc_/map_ bags of 2026-08-21 for mu, the command gain and the delay.
    """

    mass: float = 3.8
    wheelbase: float = 0.325
    l_f: float = 0.163
    l_r: float = 0.162
    inertia_z: float = 0.055

    # Tyres: simplified Pacejka, Fy = mu*Fz * sin(C * atan(B * alpha)).
    mu: float = 0.47
    tyre_b_front: float = 8.0
    tyre_c_front: float = 1.55
    tyre_b_rear: float = 14.0
    tyre_c_rear: float = 1.55

    # Actuation.  Asymmetric because the servo map is: the steering reaches
    # servo 0.000 at -0.48 rad and servo 1.000 at +0.51 rad after the
    # 2026-08-22 recalibration (`f1tenth_stack/config/vesc.yaml`, gain 1.0101
    # / offset 0.4848).  Before it, servo_min/max 0.15/0.85 clipped the same
    # car to [-0.2705, +0.2278] - little more than half the travel.
    min_steer: float = -0.48
    max_steer: float = 0.51
    # rad/s of COMMAND, which is what the bringup's servo smoother limits
    # (`max_servo_speed`, converted to servo units with the map's own gain).
    # It moves with the map: the servo hardware does 4.50 servo units/s, which
    # was 3.2 rad/s of command at the old gain and is 4.45 at the new one.
    # Leaving it at 3.2 makes the WHEELS 28 % slower and shows up as tracking
    # error on the approach to the hairpin, where the wheel has to travel.
    max_steer_rate: float = 4.45
    # Front-wheel angle delivered per unit of command.  PREDICTED from the
    # recalibration, not yet re-identified: three bags put the linkage at
    # 0.9039 rad of wheel angle per unit of servo (at the old clip it reached
    # +0.289 rad, measured +0.286..+0.293), and the new map is 1.0101 servo
    # units per rad of command, so 0.9039 * 1.0101 = 0.913.  Re-fit it from
    # the first bag on the new calibration before trusting a fast lap to it.
    steer_command_gain: float = 0.913
    # `pp_20260824_000209` measures 100 ms command -> yaw response.  Older
    # pre-calibration bags measured about 140 ms.
    control_delay: float = 0.10

    # Envelope.  max_speed is the VESC ERPM clamp (70000 ERPM = 10.4 m/s);
    # the raceline speed profile is what actually decides pace.
    max_speed: float = 10.4
    max_accel: float = 5.0
    max_decel: float = 4.0
    # Sustained lateral acceleration the car delivers (0.47 g measured).
    max_lateral_accel: float = 4.6

    @property
    def wheelbase_dynamic(self) -> float:
        return self.l_f + self.l_r

    @property
    def command_wheelbase(self) -> float:
        """Wheelbase in COMMAND space.

        Pure pursuit inverts `delta = atan(2 L sin(eta) / L1)`, and `delta` is
        what is published - not what the front wheels reach.  The servo link
        delivers `steer_command_gain` times the command, so the geometry has
        to be inverted with `L / gain` (0.256 m here) or every command comes
        out 27 % too large and the car saws.
        """
        return self.wheelbase / max(self.steer_command_gain, 1e-3)

    @property
    def max_curvature(self) -> float:
        """Curvature the car can hold at full lock, whichever way is worse."""
        return min(self.curvature_towards(1.0), self.curvature_towards(-1.0))

    def curvature_towards(self, side: float) -> float:
        """Curvature available turning `side` (+1 left, -1 right).

        The servo does not reach the same angle both ways - this car goes to
        +0.51 rad left and -0.48 right - so a manoeuvre that turns left has
        7 % more curvature to spend than the symmetric figure admits.  Judging
        a left-hand evasion against the right-hand limit refuses paths the car
        can drive.
        """
        reach = abs(self.max_steer) if side >= 0.0 else abs(self.min_steer)
        return float(np.tan(reach * self.steer_command_gain) / self.wheelbase_dynamic)

    def update(self, values: Dict[str, Any]):
        return _apply(self, values)



class AvoidanceConfig:
    """Static obstacle avoidance.

    The manoeuvre is the one described in
    https://hmcl-unist.github.io/unicorn-racing/posts/static-obstacle-avoidance-en/:
    an evasion path in the Frenet frame of the global line,

        p_evade(s) = p_race(s) + d(s) * n(s),

    where `d` rises to an apex beside the obstacle and returns to zero.  Here
    `d(s)` is a quintic smoothstep, which is C2 - the curvature of the evasion
    line is continuous, so the speed profile computed from it is meaningful
    and the car is never asked for a step in steering.

    Detection is the emergency stop's own test read the other way round: a
    return that is NOT on a mapped wall is an obstacle (see obstacles.py).
    """

    enable_avoidance: bool = True

    # --- detection ---------------------------------------------------------
    # How far ahead a cone has to be found for there to be room to go round
    # it smoothly.  It has to beat the emergency stop, which at 7.5 m/s wants
    # 9.4 m of clear corridor: a manoeuvre that cannot be planned before then
    # is one the car brakes for instead of steering round.  Long range used
    # to mean false positives; `track_offset` and the centre check are what
    # make it affordable now.
    detect_range: float = 10.0
    detect_fov: float = 1.75
    # Map clearance below which a return is "a wall the map already knows",
    # applied to the returns AND to the estimated centre.  Wider than the
    # emergency stop's own margin: a false obstacle here makes the car swerve,
    # which is worse than a false brake.  Swept on ifac_fin2 bag
    # pp_20260823_220233: after correcting its fixed omitted-wall patch, 0.40
    # leaves no confirmed ghosts or avoidance episodes.  Synthetic legal
    # obstacles on the line remain detectable and are passed on the open side.
    obstacle_wall_margin: float = 0.40
    # Only track what is near the racing line.  This planner leaves the line
    # to get round something standing ON it; a detection further off than this
    # will never be planned around, and tracking it only fills the display,
    # the events and the obstacle count with things nobody will act on.  On
    # the bag above it removed 95 % of the ghosts and none of the cone.
    # Anything further away cannot overlap this 0.28 m car plus the largest
    # competition obstacle.  On the same bag 0.70 removes the repeated
    # |d|=0.9 m edge ghosts while retaining the fixed |d|=0.42 m blocker.
    track_offset: float = 0.70
    cluster_gap: float = 0.10
    cluster_beam_gap: int = 4
    # A 0.3 m box at the 10 m detection range spans ~7 beams, so four is a bar
    # noise does not clear and the smallest competition obstacle does.
    cluster_min_points: int = 4
    # The competition runs SQUARE obstacles between these two sizes and
    # nothing else.  Every number below is derived from them, so the rule is
    # written down once and the derivation is checked by a test.
    obstacle_min_size: float = 0.30
    obstacle_max_size: float = 0.50
    # What has to be planned around is the half-DIAGONAL, because the lidar
    # sees one face and the box may be at any angle to it: 0.30*sqrt(2)/2 and
    # 0.50*sqrt(2)/2.  Never believe an obstacle is smaller than the smallest
    # one that can be on this track, and never plan around more than the
    # largest - anything bigger is a wall the map is wrong about, and swerving
    # on the strength of the map being wrong is not a good bet.  The emergency
    # stop still sees it either way.
    obstacle_min_radius: float = 0.212
    obstacle_max_radius: float = 0.354
    # The lidar reads a silhouette SHORT: the first and last returns land
    # inside the true edges, so a 0.30 m box measures 0.129-0.146 m of
    # half-chord rather than the geometric 0.15.  Measured against synthetic
    # scans from 2 m to 10 m (tools: see test_obstacles.py):
    #
    #     0.30 box face-on   0.129 - 0.146   <- smallest legal silhouette
    #     0.50 box corner-on 0.330 - 0.347   <- largest legal silhouette
    #     0.60 box corner-on 0.386 - 0.405   <- must be rejected
    #     r = 0.10 speck     0.086 - 0.094   <- must be rejected
    #
    # These two put the accept band at 0.110 - 0.369, which clears the
    # smallest legal obstacle by 19 mm and the largest illegal one by 17 mm.
    silhouette_shortfall: float = 0.040
    silhouette_overshoot: float = 0.015
    # The lidar sees the near face only, so the centroid of the returns sits
    # in front of the centre and has to be pushed back through it.  Geometry
    # says 2R/pi = 0.64 R for a fully resolved half-circle; measured against
    # synthetic scans of competition-sized obstacles it is 0.8 R, because the
    # beams do not resolve the tangents.  At 0.6 the centre of a 0.5 m box
    # came out 0.08 m too close, at 0.8 it is within 0.02 m from 3 m to 7 m.
    centre_push: float = 0.8

    # --- tracking ----------------------------------------------------------
    match_distance: float = 0.40
    smoothing: float = 0.35
    min_hits: int = 4
    forget_time: float = 1.0
    # "It is not moving" as a RATE: how far the estimate has moved over the
    # last `static_window` seconds.  A total excursion does not work - the
    # estimate of a stationary cone walked 0.8 m down the track over 1.5 s as
    # the car closed from 6 m to 3 m (bag pp_20260822_194215), which a 0.35 m
    # total rejects and a 0.5 m-per-half-second rate does not, while a car
    # pulling away at 2 m/s covers 1.0 m in the same window.
    static_window: float = 0.5
    static_tolerance: float = 0.5
    # ...and how long it has to have been tracked before that rate is trusted
    # at all.  Separate from the window because every metre spent confirming
    # is a metre taken off the turn-out: at 7.5 m/s half a second of
    # confirmation is 3.8 m of the run-up gone.
    confirm_time: float = 0.25
    # Once a CONFIRMED obstacle is this close, its estimate is frozen: the
    # centre-push calibration holds from 3 m to 7 m, and inside ~2 m the
    # near-face geometry drags the estimated centre towards the car - on a
    # 0.50 m box the estimate walked 0.2 m up the track DURING the pass,
    # the replanned profile cleared the walking estimate by 3 cm and hit
    # the real box by 13.  The far estimate was good; keep it.
    track_freeze_range: float = 1.8

    # --- the manoeuvre -----------------------------------------------------
    # Hard admission gate on the PHYSICAL opening between the obstacle edge
    # and the mapped wall.  Even if the body/margin model can squeeze a path
    # through a smaller opening, do not attempt it.  Requested for ifac_fin2:
    # obstacle breakthroughs need at least 35 cm of real free width.
    min_passage_width: float = 0.35
    # This car is 0.28 m wide.  The admission tests below are GEOMETRIC (can
    # the body fit at all), and the robustness against tracking error comes
    # from slowing down through what is tight - see `pass_error_quad`.
    ego_half_width: float = 0.14
    # Half the car's LENGTH, which matters whenever it is not parallel to the
    # thing it is passing - and in the middle of an evasion it never is.  A
    # car yawed by theta presents `half_width*cos(theta) +
    # ego_half_length*sin(theta)` towards the obstacle, and booking only the
    # half-width is how a manoeuvre that reads as clear by 0.06 m puts a
    # corner 0.19 m inside a 0.30 m cone.  Measured in the gym at s=28: the
    # plan cleared by the disc model and the body clipped it anyway.
    ego_half_length: float = 0.29
    # How many discs cover the body.  Three (the old cover) have radius
    # hypot(L/6, W/2) = 0.170, which books the 0.28 m car as 0.34 m wide and
    # makes narrow gaps unpassable BY MODEL.  Nine put the cover 6 mm over
    # the true half-width (0.146 vs 0.140), while `min_passage_width` remains
    # the independent hard lower bound on the physical opening.
    body_disc_count: int = 9
    # Clearance the profile AIMS for between the car's flank and everything
    # it passes.  Deliberately larger than the tracking error of a solved
    # evasion (0.19 m p95): when there is room, the pass is planned deep
    # enough that the error never reaches the obstacle.
    lateral_margin: float = 0.30
    # ...and the least the corridor may squeeze it to before the placement
    # is refused.  GEOMETRIC, not a comfort figure - the honesty of a margin
    # this thin comes from `pass_error_quad` below: the car slows until its
    # predicted tracking error fits the margin it actually got.  0.03 is
    # where that model bottoms out at the crawl floor (error floor 0.02 plus
    # the crawl's own few millimetres); below it the promise cannot be kept
    # at any speed.
    lateral_margin_min: float = 0.03
    # How much clear air the CAR BODY keeps from a wall while going round an
    # obstacle.  Not a clearance for the line - the planner derives that by
    # adding the body's own disc cover, and checks the finished manoeuvre at
    # each disc rather than at the point the body is centred on, so this is
    # the number it sounds like: metres between the car and the wall.
    #
    # 0.05 m is a deliberate choice to attempt manoeuvres that a larger margin
    # refuses.  It is thinner than the car tracks: cross-track error at race
    # pace is p95 0.16 m, so a manoeuvre planned to this margin relies on
    # being tracked far better than the racing line is.  It applies ONLY to
    # evasions; the racing line keeps its own 0.22 m of body clearance.
    wall_body_margin: float = 0.05
    # The finished evasion must still fit geometrically, but a nearby mapped
    # side wall must never lower its speed.  Obstacle slack and front E-stop
    # remain active independently.
    enable_wall_speed_cap: bool = False
    # An obstacle blocks the line when its edge reaches this close to the
    # car's flank on the line.
    block_margin: float = 0.05
    # Closer than this there is no manoeuvre left to make - it is the
    # emergency stop's problem.
    stand_off: float = 0.5
    plan_horizon: float = 9.0
    # The window the profile is solved in: how much run-up before the
    # obstacle (more where the raceline is quick, because the same sideways
    # shift needs more metres at speed to stay gentle) and how much return
    # room after it.  The profile is free to leave the line late and return
    # early WITHIN the window - these bound the search, they do not shape
    # the answer, which is why there is no length ladder any more.
    window_back: float = 5.0
    window_back_fast: float = 8.0
    window_fast_speed: float = 4.5
    window_out: float = 6.0
    # The profile solver: second derivatives at knots every `qp_knot_spacing`
    # metres, constraints sampled every `qp_sample_spacing`, at most
    # `qp_iterations` ADMM sweeps per planning cycle (the hard bound on the
    # planner's compute), re-linearising the curvature box
    # `qp_relinearizations` times.  The solver is advisory - a bad solve is
    # caught by the same gates as ever and can only cause a refusal.
    qp_knot_spacing: float = 0.5
    qp_sample_spacing: float = 0.25
    qp_iterations: int = 400
    qp_relinearizations: int = 2
    # Fraction of the steering lock the PLANNED profile may use; what is
    # held back is authority for the tracking error the controller still
    # has to take out.  The gates still allow full lock, and the base line's
    # own demand is always allowed.
    lock_fraction: float = 0.96
    # How far inside the true free-space tube the solver stays: an ADMM
    # solution converges to tolerance, not exactly, and a profile ON the
    # boundary would fail the wall gate by micrometres.
    tube_margin: float = 0.03
    # How the tracking error of a solved evasion grows with speed and path
    # curvature, measured closed-loop against the identified plant:
    #   e_p95 ~ quad * |kappa| * v^2 + lin * v
    # (0.19 m at 2.4 m/s through the 1.4 1/m hairpin S, 0.09 m at ~5 m/s on
    # a straight bump of ~0.2, 0.021 m at a 0.74 m/s crawl - one constant
    # fits all three, and it is ~1.5x the 140 ms delay squared, which is the
    # physics it is standing in for).  After the gates pass, the evasion's
    # speed is capped SAMPLE BY SAMPLE so that `factor` times the error this
    # model predicts fits inside the slack the profile actually has there
    # (to the obstacle and to the walls), less `keep`.  This is what makes
    # "any 0.30 m gap is taken" safe: a thin gap is not refused, it is taken
    # at the speed its thinness affords - down to `min_pass_speed`, where
    # the predicted error is millimetres and the e-stop still stands behind
    # everything.
    # The quadratic limb is small where the servo has authority and grows
    # without bound as the PLANNED curvature approaches the lock (the
    # inflation term in `_cap_from_slack`): 0.006 with the clip at 7.5
    # reproduces all three anchors - 0.09 m at 5 m/s on a straight bump,
    # 0.19 m at 2.4 m/s through the near-lock hairpin S, and does not slow
    # a straight-line pass below the raceline at all.  A single flat
    # coefficient could only fit one end and over-braked the other 2x.
    pass_error_quad: float = 0.006
    pass_error_lin: float = 0.010
    pass_error_floor: float = 0.02
    pass_error_factor: float = 0.7
    # Tracker-estimate uncertainty booked against the obstacle-side slack
    # in the speed caps (walls are map-truth and carry none): a frozen
    # close-range estimate is ~0.1 m of centre bias on the biggest box.
    pass_obstacle_reserve: float = 0.08
    pass_error_keep: float = 0.015
    min_pass_speed: float = 0.35
    # Lateral acceleration the evasion line's own speed profile is allowed to
    # plan for, as a fraction of what the car delivers - the same convention
    # as the raceline generator's --grip-margin.  This is what decides how
    # much the car slows for the manoeuvre, and it slows for the geometry it
    # is actually about to drive rather than for a rule of thumb.
    grip_margin: float = 0.85
    # An EXTRA flat speed cap while alongside the obstacle, on top of that.
    # Off (0) by default: going round a cone should not mean slowing for it,
    # and a turn-out long enough to be comfortable is already gentle enough
    # that its own profile asks for no reduction at all.  Set it if the
    # surface beside the racing line is worse than the line itself.
    pass_speed: float = 0.0
    pass_window: float = 1.0
    # Re-plan only when the estimate has really moved.  A path rebuilt every
    # cycle around a jittering centroid jitters with it.
    replan_tolerance: float = 0.12

    def update(self, values: Dict[str, Any]):
        return _apply(self, values)



S, X, Y, PSI, KAPPA, VX, AX, W_LEFT, W_RIGHT = range(9)

# `line_type, id, s, d, x, y, d_right, d_left, psi, kappa, vx, ax` remapped
# onto this module's `s, x, y, psi, kappa, vx, ax, w_left, w_right`.
_COMBINED_COLUMNS = (2, 4, 5, 8, 9, 10, 11, 7, 6)


class Raceline:
    """A closed raceline sampled at (roughly) constant arc length."""

    def __init__(self, data: np.ndarray, name: str = '', heading_offset: float = 0.0):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] < 7 or data.shape[0] < 8:
            raise ValueError('raceline needs >= 8 rows of >= 7 columns, got '
                             f'{data.shape}')
        if data.shape[1] < 9:
            # No corridor in the file: report it as unknown, not as zero.
            pad = np.full((data.shape[0], 9 - data.shape[1]), np.inf)
            data = np.hstack([data, pad])
        self.name = name
        self.data = data
        self.heading_offset = float(heading_offset)

        self.s = data[:, S].copy()
        self.points = data[:, [X, Y]].copy()
        self.psi = wrap_angle(data[:, PSI] + heading_offset)
        self.kappa = data[:, KAPPA].copy()
        self.vx = data[:, VX].copy()
        self.ax = data[:, AX].copy()
        self.w_left = data[:, W_LEFT].copy()
        self.w_right = data[:, W_RIGHT].copy()

        closing = float(np.hypot(*(self.points[0] - self.points[-1])))
        self.length = float(self.s[-1] + closing)
        self.spacing = float(np.mean(np.diff(self.s))) if len(self.s) > 1 else 0.1
        # Wrapped copies so np.interp works across the start/finish line.
        self._s_wrapped = np.concatenate([self.s, [self.length]])
        self._vx_wrapped = np.concatenate([self.vx, [self.vx[0]]])
        self._ax_wrapped = np.concatenate([self.ax, [self.ax[0]]])
        self._kappa_wrapped = np.concatenate([self.kappa, [self.kappa[0]]])
        self._x_wrapped = np.concatenate([self.points[:, 0], [self.points[0, 0]]])
        self._y_wrapped = np.concatenate([self.points[:, 1], [self.points[0, 1]]])
        # Heading has to be interpolated on a CONTINUOUS copy: either side of
        # the +-pi branch cut a wrapped column interpolates across the whole
        # circle and comes back pointing at the far wall.
        continuous = np.unwrap(self.psi)
        self._psi_wrapped = np.concatenate(
            [continuous, [continuous[-1] + wrap_angle(self.psi[0] - self.psi[-1])]])

    # ------------------------------------------------------------------ load
    @classmethod
    def from_csv(cls, path: str, heading_offset: Optional[float] = None) -> 'Raceline':
        if not os.path.isfile(path):
            raise FileNotFoundError(f'raceline not found: {path}')
        data = None
        for delimiter in (';', ',', None):
            try:
                data = np.loadtxt(path, delimiter=delimiter, comments='#')
            except ValueError:
                continue
            if data.ndim == 2 and data.shape[1] >= 7:
                break
            data = None
        if data is None:
            # combined_waypoints.csv has a text first column and a text header.
            data = cls._load_combined(path)
        if data is None:
            raise ValueError(f'could not parse a >=7 column raceline from {path}')
        if heading_offset is None:
            heading_offset = cls.detect_heading_offset(data)
        return cls(data, name=os.path.basename(path), heading_offset=heading_offset)

    @staticmethod
    def _load_combined(path: str) -> Optional[np.ndarray]:
        rows = []
        with open(path) as handle:
            for line in handle:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) < 12 or not parts[0].startswith('raceline'):
                    continue
                try:
                    rows.append([float(parts[index]) for index in _COMBINED_COLUMNS])
                except ValueError:
                    continue
        return np.asarray(rows) if len(rows) >= 8 else None

    @staticmethod
    def detect_heading_offset(data: np.ndarray) -> float:
        """Offset that maps the psi column onto ``atan2(dy, dx)``.

        Only the three conventions seen in the wild are considered (0, +pi/2,
        -pi/2); anything else is reported as 0 so a genuinely odd file shows up
        as tracking error rather than being silently rotated.
        """
        tangent = np.arctan2(np.gradient(data[:, Y]), np.gradient(data[:, X]))
        residual = np.angle(np.mean(np.exp(1j * (tangent - data[:, PSI]))))
        for candidate in (0.0, np.pi / 2.0, -np.pi / 2.0):
            if abs(wrap_angle(residual - candidate)) < 0.35:
                return candidate
        return 0.0

    # --------------------------------------------------------------- queries
    def nearest_index(self, position, hint: Optional[int] = None,
                      window: int = 40) -> int:
        """Index of the closest raceline sample to `position`.

        With a `hint` only +-`window` samples around it are searched; if the
        winner sits on the window edge the search falls back to the whole
        raceline, so a relocalisation jump cannot latch the projection.
        """
        position = np.asarray(position, dtype=float)
        count = len(self.s)
        if hint is None or window <= 0 or 2 * window + 1 >= count:
            deltas = self.points - position
            return int(np.argmin(np.einsum('ij,ij->i', deltas, deltas)))
        indices = (np.arange(hint - window, hint + window + 1) % count)
        deltas = self.points[indices] - position
        local = int(np.argmin(np.einsum('ij,ij->i', deltas, deltas)))
        if local in (0, len(indices) - 1):
            deltas = self.points - position
            return int(np.argmin(np.einsum('ij,ij->i', deltas, deltas)))
        return int(indices[local])

    def project(self, position, hint: Optional[int] = None,
                window: int = 40) -> Tuple[int, float, float]:
        """Project onto the raceline.

        Returns ``(index, s, signed_lateral_error)``, the error positive when
        the car is to the LEFT of the line.
        """
        position = np.asarray(position, dtype=float)
        index = self.nearest_index(position, hint=hint, window=window)
        count = len(self.s)
        best = None
        for start in (index - 1, index):
            a = self.points[start % count]
            b = self.points[(start + 1) % count]
            segment = b - a
            length_sq = float(segment @ segment)
            if length_sq < 1e-12:
                continue
            t = float(np.clip((position - a) @ segment / length_sq, 0.0, 1.0))
            foot = a + t * segment
            distance = float(np.hypot(*(position - foot)))
            if best is None or distance < best[0]:
                lateral = float(np.cross(segment / np.sqrt(length_sq), position - a))
                s_a = self.s[start % count]
                s_b = self.length if start + 1 == count else self.s[(start + 1) % count]
                best = (distance, (s_a + t * (s_b - s_a)) % self.length, lateral,
                        start % count)
        if best is None:
            return index, float(self.s[index]), 0.0
        return best[3], best[1], best[2]

    def lookahead_point(self, position, distance: float,
                        hint: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """First point on the raceline exactly `distance` ahead of `position`.

        The exact circle/segment intersection is used rather than the nearest
        sample: at 0.1 m spacing and 40 Hz, snapping to samples injects a
        visible steering ripple.
        """
        position = np.asarray(position, dtype=float)
        count = len(self.s)
        start = int(self.nearest_index(position, hint=hint) if hint is None else hint)
        start %= count
        horizon = int(min(count, max(8, distance / max(self.spacing, 1e-3) * 3 + 8)))
        previous = self.points[start]
        for step in range(1, horizon + 1):
            index = (start + step) % count
            current = self.points[index]
            if float(np.hypot(*(current - position))) >= distance:
                point = _segment_circle(previous, current, position, distance)
                return point, index, float(np.hypot(*(point - position)))
            previous = current
        index = (start + horizon) % count
        point = self.points[index]
        return point.copy(), index, float(np.hypot(*(point - position)))

    def speed_at(self, s: float) -> float:
        return float(np.interp(s % self.length, self._s_wrapped, self._vx_wrapped))

    def accel_at(self, s: float) -> float:
        return float(np.interp(s % self.length, self._s_wrapped, self._ax_wrapped))

    def curvature_at(self, s: float) -> float:
        return float(np.interp(s % self.length, self._s_wrapped, self._kappa_wrapped))

    def heading_at(self, s: float) -> float:
        return float(wrap_angle(np.interp(s % self.length, self._s_wrapped,
                                          self._psi_wrapped)))

    def mean_abs_curvature(self, s: float, start: float, end: float) -> float:
        """Mean |kappa| over the raceline between `s+start` and `s+end` metres."""
        if end <= start:
            return abs(self.curvature_at(s + start))
        count = max(3, int((end - start) / max(self.spacing, 1e-3)))
        samples = np.linspace(s + start, s + end, count) % self.length
        return float(np.mean(np.abs(np.interp(samples, self._s_wrapped,
                                              self._kappa_wrapped))))

    def mean_curvature(self, s: float, window: float) -> float:
        """SIGNED mean curvature over `window` metres ahead of `s`.

        The unsigned version answers "how tight is it here"; this one answers
        "which way, on average" - what a lookahead point actually follows.
        """
        if window <= 0.0:
            return self.curvature_at(s)
        count = max(3, int(window / max(self.spacing, 1e-3)))
        samples = np.linspace(s, s + window, count) % self.length
        return float(np.mean(np.interp(samples, self._s_wrapped,
                                       self._kappa_wrapped)))

    def sample(self, s) -> np.ndarray:
        """``(N, 5)`` array of ``(x, y, psi, kappa, vx)`` at arc lengths `s`."""
        s = np.asarray(s, dtype=float) % self.length
        return np.column_stack((
            np.interp(s, self._s_wrapped, self._x_wrapped),
            np.interp(s, self._s_wrapped, self._y_wrapped),
            wrap_angle(np.interp(s, self._s_wrapped, self._psi_wrapped)),
            np.interp(s, self._s_wrapped, self._kappa_wrapped),
            np.interp(s, self._s_wrapped, self._vx_wrapped)))

    def point_at(self, s: float) -> np.ndarray:
        row = self.sample([s])[0]
        return np.array([row[0], row[1]])

    def corridor_at(self, s: float) -> Tuple[float, float]:
        """``(left, right)`` distance to the mapped walls at arc length `s`."""
        index = int(round((s % self.length) / max(self.spacing, 1e-6))) % len(self.s)
        return float(self.w_left[index]), float(self.w_right[index])


def _segment_circle(a: np.ndarray, b: np.ndarray, centre: np.ndarray,
                    radius: float) -> np.ndarray:
    """Intersection of segment a->b with the circle of `radius` around centre.

    Returns the far intersection when one exists, otherwise the segment end -
    the caller only ever asks about segments that straddle the circle.
    """
    direction = b - a
    offset = a - centre
    aa = float(direction @ direction)
    if aa < 1e-12:
        return b.copy()
    bb = 2.0 * float(offset @ direction)
    cc = float(offset @ offset) - radius * radius
    discriminant = bb * bb - 4.0 * aa * cc
    if discriminant < 0.0:
        return b.copy()
    root = np.sqrt(discriminant)
    t = max((-bb + root) / (2.0 * aa), (-bb - root) / (2.0 * aa))
    return a + float(np.clip(t, 0.0, 1.0)) * direction


def wrap_s(delta, length: float):
    """Arc-length difference wrapped to (-length/2, length/2]."""
    return (np.asarray(delta, dtype=float) + length / 2.0) % length - length / 2.0


def offset_curvature(d, d1, d2, kappa, kappa_rate):
    """Exact curvature of the line displaced by `d` from a base of `kappa`."""
    one_minus = 1.0 - d * kappa
    scale_sq = one_minus * one_minus + d1 * d1
    return ((one_minus * (kappa * one_minus + d2)
             + d1 * (2.0 * d1 * kappa + d * kappa_rate))
            / np.maximum(scale_sq, 1e-9) ** 1.5)


def curvature_dd_bounds(d, d1, kappa, kappa_rate, limit_left: float,
                        limit_right: float):
    """``(lo, hi)`` on d'' such that the offset curvature stays inside lock.

    The relation above is linear in d'', so the sign-aware lock inverts into
    a per-sample box: d'' = (k_e G^1.5 - X) / omega - k omega, with
    omega = 1 - d k, G = omega^2 + d'^2 and X = d'(2 d' k + d k').  Evaluated
    on the previous iterate of the profile, which is the one linearisation
    in the whole solve.
    """
    one_minus = np.maximum(1.0 - d * kappa, 0.1)
    scale_15 = (one_minus * one_minus + d1 * d1) ** 1.5
    cross = d1 * (2.0 * d1 * kappa + d * kappa_rate)
    turn = kappa * one_minus
    return ((-limit_right * scale_15 - cross) / one_minus - turn,
            (limit_left * scale_15 - cross) / one_minus - turn)


@dataclass
class Target:
    """The obstacle the planner is working on, in the line's own frame."""

    obstacle_id: int
    x: float
    y: float
    radius: float
    s: float
    d: float
    ahead: float
    confirmed: bool = True


@dataclass
class AvoidancePlan:
    target: Target
    side: int                 # +1 evading left of the line, -1 right
    d_apex: float             # deepest point of the profile, signed
    s_start: float            # where the profile leaves the global line
    s_apex: float             # trailing edge of the pass - commitment point
    s_end: float              # where it has returned
    line: Raceline
    # ``(3, N)`` of (d, d', d'') on the raceline grid, for grafting a replan
    # onto the manoeuvre the car is already driving.
    profile: Optional[np.ndarray] = None
    # Which OTHER obstacles were folded into the tube when this was built;
    # if that set changes the geometry changed even though the target stood
    # still, and the plan wants rebuilding.
    others: tuple = ()


class AvoidancePlanner:
    """Confirmed static obstacles in, the line the controller should track out."""

    def __init__(self, raceline: Raceline, vehicle: VehicleParams,
                 config: AvoidanceConfig, track=None):
        self.raceline = raceline
        self.vehicle = vehicle
        self.config = config
        self.track = track
        self._kappa_rate = np.gradient(np.concatenate(
            [raceline.kappa, raceline.kappa[:1]]))[:-1] / max(raceline.spacing, 1e-6)
        self._ops_cache = {}
        self._tube_hi, self._tube_lo = self._tube()
        self._base_wall_slack = self._wall_slack_of_base()
        self.reset()

    def reset(self) -> None:
        self.plan: Optional[AvoidancePlan] = None
        self.reason = 'clear'
        self._hint: Optional[int] = None
        self.progress_s = 0.0
        self._outside = 0

    @property
    def active(self) -> bool:
        return self.plan is not None

    @property
    def line(self) -> Raceline:
        return self.plan.line if self.plan is not None else self.raceline

    # ------------------------------------------------------------------ main
    def update(self, obstacles: List, position, speed: float) -> Raceline:
        """Return the line to track: the global one, or an evasion of it.

        `obstacles` is everything the tracker holds, not just what it is
        confident about.  Starting a manoeuvre and finishing one are different
        questions: a plan may only be STARTED for an obstacle the tracker has
        confirmed, and once started it is kept while that obstacle is tracked
        at all.  Judging both with the same test is what made the car plan an
        evasion and abandon it a third of a second later - the estimate of a
        stationary cone walks as the car closes on it, the confirmation test
        failed on that walk, and the plan went with it.
        """
        config = self.config
        index, s_car, lateral = self.raceline.project(position, hint=self._hint)
        self._hint = index
        self.progress_s = s_car
        if not config.enable_avoidance:
            self.plan, self.reason = None, 'off'
            return self.raceline

        released = False
        if self.plan is not None:
            self._release(s_car, obstacles)
            released = self.plan is None
        if self.plan is not None and self._committed(s_car):
            # Past the obstacle the car is alongside it; snapping back to the
            # global line there is the one thing worse than a plan built on a
            # stale estimate.  The narrowed e-stop corridor is what watches
            # this stretch.
            self.reason = 'avoiding'
            return self.plan.line
        if (self.plan is not None and not self._started(s_car)
                and not self._standing_plan_clears(obstacles)):
            # The estimates have walked out from under the plan (a stationary
            # cone's estimate moves 0.8 m as the car closes, and a second one
            # can be confirmed anywhere).  A line that no longer clears what
            # is tracked must not keep being published on the strength of a
            # check it passed against other numbers.  ONLY before the
            # turn-out begins: dropping the plan with the car mid-pass hands
            # the controller the global line, which aims it at the very
            # obstacle it was passing - measured, that is how a clean pass
            # became a 0.13 m hit.  Once started, replans graft and the
            # narrowed e-stop corridor is the backstop.
            self.plan, self.reason = None, 'blocked'
        target = self._pick(obstacles, s_car)
        if target is None:
            # `passed` and `lost` are events; they survive the cycle they
            # happened on and become `clear` on the next one.
            if self.plan is None and not released:
                self.reason = 'clear'
            return self.line
        if self._needs_replan(target, s_car, obstacles):
            plan = self._build(target, s_car, speed, obstacles, lateral)
            if plan is not None:
                self.plan = plan
                self._outside = 0
            # A rebuild that fails does NOT cancel the plan that is already
            # standing: it was checked against the map when it was made, the
            # car is on it, and the health check above retires it the moment
            # it stops clearing what is actually tracked.
        if self.plan is not None:
            self.reason = 'avoiding'
        return self.line

    def _standing_plan_clears(self, obstacles) -> bool:
        """Does the STANDING plan still clear everything the tracker holds?

        Re-judged every cycle before commitment (0.5 ms): a plan is a claim
        about the current obstacle estimates, not about the ones it was built
        from.
        """
        plan = self.plan
        if plan.profile is None:
            return True
        window = np.abs(plan.profile[0]) > 1e-3
        if not np.any(window):
            return True
        return self._clears(plan.line, window, obstacles)

    def _pick(self, obstacles, s_car: float) -> Optional[Target]:
        """The obstacle to work on: the one already being avoided, or a new one.

        Tracks churn: the tracker forgets an obstacle the moment it leaves
        the field of view for a second, and re-acquires it under a new id -
        measured mid-pass on the real car.  A new id standing where the
        plan's obstacle stood IS that obstacle, and the plan adopts it;
        judged by id alone, the standing plan is orphaned and every rebuild
        falls into the fresh-plan path, which refuses at close range.
        """
        targets = self._targets(obstacles, s_car)
        if self.plan is not None:
            held = [target for target in targets
                    if target.obstacle_id == self.plan.target.obstacle_id]
            if held:
                return held[0]
            close_in = float(wrap_s(self.plan.target.s - s_car,
                                    self.raceline.length)) < 1.5
            if self._started(s_car) or close_in:
                # Mid-manoeuvre - or too close for the fresh-plan path to
                # run at all (it refuses under a metre of run-up, and a
                # standing plan whose id churned there would otherwise fly
                # unrefreshed while its obstacle's estimate grows).  Adopt
                # the CLOSEST match, and only when identity is unambiguous -
                # two competition boxes may legally stand 0.30 m apart, and
                # adopting the nearer-AHEAD one re-aimed a live plan 0.5 m
                # wrong in review.
                spaced = sorted(
                    (math.hypot(target.x - self.plan.target.x,
                                target.y - self.plan.target.y), index)
                    for index, target in enumerate(targets))
                if spaced and spaced[0][0] < 0.3 \
                        and (len(spaced) < 2
                             or spaced[1][0] > 2.0 * spaced[0][0]):
                    found = targets[spaced[0][1]]
                    self.plan.target.obstacle_id = found.obstacle_id
                    return found
        confirmed = [target for target in targets if target.confirmed]
        return confirmed[0] if confirmed else None

    # ------------------------------------------------------------- selection
    def _targets(self, obstacles, s_car: float) -> List[Target]:
        """Blocking obstacles ahead, nearest first.

        The obstacle a plan is already committed to keeps its place here on
        looser terms - it may be closer than `stand_off` by now, and its
        estimate may have wandered out of the blocking band - because having
        decided to go round something is not a decision to re-take every
        25 ms.
        """
        config = self.config
        length = self.raceline.length
        current = self.plan.target.obstacle_id if self.plan is not None else None
        block = self._disc_radius() + config.block_margin
        targets = []
        for obstacle in obstacles:
            held = obstacle.id == current
            _, s_obs, d_obs = self.raceline.project(obstacle.position)
            ahead = (s_obs - s_car) % length
            if held:
                # Anything from just behind the car to the far end of the
                # manoeuvre still counts as the thing being avoided.
                if wrap_s(s_obs - s_car, length) < -config.stand_off:
                    continue
            elif ahead < config.stand_off or ahead > config.plan_horizon:
                continue
            # Does it actually stand in the way?  An obstacle beside the line
            # is not a reason to leave the line.  `block` books the disc
            # cover's radius - what `_clears` will measure - not the bare
            # half-width.
            elif abs(d_obs) > obstacle.radius + block:
                continue
            targets.append(Target(obstacle_id=obstacle.id, x=obstacle.x,
                                  y=obstacle.y, radius=obstacle.radius,
                                  s=s_obs, d=d_obs, ahead=ahead,
                                  confirmed=bool(getattr(obstacle, 'confirmed',
                                                         True))))
        targets.sort(key=lambda target: target.ahead)
        return targets

    def _span_from(self, anchor: float, s_car: float) -> float:
        return float(wrap_s(s_car - anchor, self.raceline.length))

    def _started(self, s_car: float) -> bool:
        """Within the manoeuvre, measured from its own start - BOUNDED.

        A half-plane test ("anywhere in the half-lap after s_start") reads a
        relocalisation glitch as progress: during an escape shuffle beside
        the hairpin the projection jumped across a wall to s = 18.7, a
        half-plane `passed` fired eleven metres past the manoeuvre's end,
        and the plan the car was about to use was thrown away.
        """
        plan = self.plan
        extent = (self._span_from(plan.s_start, plan.s_end)
                  % self.raceline.length)
        along = self._span_from(plan.s_start, s_car)
        return bool(0.0 <= along <= extent + 1.0)

    def _committed(self, s_car: float) -> bool:
        plan = self.plan
        extent = (self._span_from(plan.s_apex, plan.s_end)
                  % self.raceline.length)
        along = self._span_from(plan.s_apex, s_car)
        return bool(0.0 <= along <= extent + 1.0)

    def _release(self, s_car: float, obstacles) -> None:
        """Give the global line back once the manoeuvre is behind the car.

        Or, before it has begun, once the obstacle has stopped being TRACKED -
        not merely stopped being confirmed.  Once the turn-out is under way
        the plan is seen through whatever the tracker does: the cone leaves
        the field of view long before the car is past it.
        """
        plan = self.plan
        length = self.raceline.length
        past = wrap_s(s_car - plan.s_end, length)
        if 0.0 <= past <= 5.0:
            self.plan, self.reason = None, 'passed'
            return
        # Outside the manoeuvre in EITHER direction, and not just for a
        # cycle: one projection glitch across a wall must not retire a live
        # plan (that was the bug the bounded windows fixed), but a car that
        # STAYS outside the plan's stretch of track has no business
        # publishing its evasion line - without this, a plan whose exit was
        # missed is immortal while its obstacle stays tracked.
        before = self._span_from(plan.s_start, s_car)
        inside = ((-self.config.window_back_fast - 1.0) <= before
                  and past <= 5.0)
        self._outside = 0 if inside else self._outside + 1
        if self._outside >= 8:
            self.plan, self.reason = None, 'stale'
            return
        if self._started(s_car):
            return
        # "Still tracked" is a question about the OBSTACLE, not about the
        # id: the tracker forgets a box the moment it leaves the view for a
        # second and re-acquires it under a new number - measured mid-run -
        # and a plan dropped for that re-numbering is a deep pass thrown
        # away half a metre before the box.
        if not any(obstacle.id == plan.target.obstacle_id
                   or math.hypot(obstacle.x - plan.target.x,
                                 obstacle.y - plan.target.y) < 0.6
                   for obstacle in obstacles):
            self.plan, self.reason = None, 'lost'

    def _needs_replan(self, target: Target, s_car: float, obstacles) -> bool:
        """A path rebuilt around a jittering centroid jitters with it."""
        plan = self.plan
        if plan is None or plan.target.obstacle_id != target.obstacle_id:
            return True
        moved = math.hypot(target.x - plan.target.x, target.y - plan.target.y)
        if (moved > self.config.replan_tolerance
                or abs(target.radius - plan.target.radius)
                > self.config.replan_tolerance):
            return True
        # The tube the plan was solved in also depends on the OTHER obstacles
        # that were folded into it; a second cone confirmed 3 m on changes
        # the geometry without the target moving a millimetre.
        return self._other_signature(target, obstacles) != plan.others

    def _other_signature(self, target: Target, obstacles) -> tuple:
        """What the fold saw: each other obstacle in the SOLVE window, with
        its position quantised at the fold's own scale.  Id alone misses a
        second cone whose estimate walked a metre sideways; a lap-wide band
        forces rebuilds for cones the fold never touches.
        """
        config = self.config
        length = self.raceline.length
        horizon = config.window_back_fast + config.window_out
        names = []
        for obstacle in obstacles:
            if obstacle.id == target.obstacle_id:
                continue
            _, s_obs, d_obs = self.raceline.project(obstacle.position)
            if abs(wrap_s(s_obs - target.s, length)) <= horizon:
                names.append((obstacle.id, round(s_obs / 0.25),
                              round(d_obs / 0.25)))
        return tuple(sorted(names))

    # ------------------------------------------------------------------ tube
    def _tube(self):
        """``(E_hi, E_lo)`` per raceline sample: the offsets the LINE may take.

        From the raceline's own width columns, tightened by the map's distance
        field marched along the normal (the field knows about walls that are
        not on the width ray - at a corner it is stricter), less the sagitta
        of the DISC COVER through the local curvature (the outermost disc
        centre stands off the path by reach^2 |k| / 2 in a corner, and the
        final gate measures it there - this pre-books what `_clears_walls`
        will charge, so solutions are not refused after the fact).  Clamped
        so that zero is always inside: the base line itself must always be
        an admissible profile.
        """
        base = self.raceline
        clear = self._line_clearance()
        hi = np.maximum(base.w_left - clear, 0.0)
        lo = -np.maximum(base.w_right - clear, 0.0)
        if self.track is not None:
            steps = np.arange(0.05, 3.0 + 1e-9, 0.05)
            normal = np.column_stack([-np.sin(base.psi), np.cos(base.psi)])
            for sign in (1.0, -1.0):
                points = (base.points[None, :, :]
                          + (sign * steps)[:, None, None] * normal[None, :, :])
                clearance = np.asarray(self.track.clearance(
                    points[:, :, 0].ravel(), points[:, :, 1].ravel())
                ).reshape(len(steps), -1)
                bad = clearance < clear
                first = np.where(bad.any(axis=0), bad.argmax(axis=0),
                                 len(steps))
                reach = first * 0.05
                if sign > 0:
                    hi = np.minimum(hi, reach)
                else:
                    lo = np.maximum(lo, -reach)
        # Towards the inside of a turn the parallel curve steepens as
        # k/(1 - dk): cap the tube at the offset where a HELD offset reaches
        # the (de-rated) lock, and well short of the centre of curvature,
        # where `_offset_line` refuses outright.
        kappa = base.kappa
        for sign, lock in ((1.0, self.vehicle.curvature_towards(1.0)),
                           (-1.0, self.vehicle.curvature_towards(-1.0))):
            inside = sign * kappa > 0.1
            if not np.any(inside):
                continue
            steep = np.abs(kappa[inside])
            allowed = np.maximum(lock * self.config.lock_fraction, steep)
            limit = np.minimum(1.0 / steep - 1.0 / allowed, 0.85 / steep)
            if sign > 0:
                hi[inside] = np.minimum(hi[inside], np.maximum(limit, 0.0))
            else:
                lo[inside] = np.maximum(lo[inside], -np.maximum(limit, 0.0))
        sag = self._disc_reach() ** 2 * np.abs(base.kappa) / 2.0
        return (np.maximum(hi - sag, 0.0), np.minimum(lo + sag, 0.0))

    def _wall_slack_of_base(self) -> np.ndarray:
        """Room the BASE line's own body keeps from the walls, per sample.

        The reference `_speed_caps` charges wall proximity against: an
        evasion is only slowed for a wall where it stands closer to it than
        the racing line - which is driven flat out - already does.
        """
        base = self.raceline
        if self.track is None:
            return np.full(len(base.s), np.inf)
        radius, offsets = self._body_discs()
        cos, sin = np.cos(base.psi), np.sin(base.psi)
        slack = np.full(len(base.s), np.inf)
        for offset in offsets:
            clearance = np.asarray(self.track.clearance(
                base.points[:, 0] + offset * cos,
                base.points[:, 1] + offset * sin))
            slack = np.minimum(slack, clearance - radius)
        return slack

    # ---------------------------------------------------------------- spline
    def _ops(self, knots: int, spacing: float, step: float):
        """Evaluation operators of the moment-basis spline.

        The unknowns are the profile's second derivatives ``M_0..M_knots`` at
        evenly spaced knots; d'' is piecewise linear in between, and d', d
        are its exact first and second integrals (with the window-start value
        and slope carried as affine terms by the caller).  Returns
        ``(D0, D1, D2, d0_end, d1_end, tau)`` evaluating d, d', d'' at
        arc-length offsets ``tau`` = 0, step, 2 step, ... spacing*knots.
        """
        key = (knots, round(spacing, 6), round(step, 6))
        if key in self._ops_cache:
            return self._ops_cache[key]
        count = knots + 1
        tau = np.arange(0.0, knots * spacing + 1e-9, step)
        j = np.minimum((tau / spacing).astype(int), knots - 1)
        f = tau / spacing - j
        eye = np.eye(count)
        D2 = (1.0 - f)[:, None] * eye[j] + f[:, None] * eye[j + 1]
        D1k = np.zeros((count, count))
        D0k = np.zeros((count, count))
        for a in range(knots):
            D1k[a + 1] = D1k[a] + spacing * 0.5 * (eye[a] + eye[a + 1])
            D0k[a + 1] = (D0k[a] + spacing * D1k[a]
                          + spacing * spacing * (2.0 * eye[a] + eye[a + 1]) / 6.0)
        D1 = D1k[j] + spacing * ((f - f * f / 2.0)[:, None] * eye[j]
                                 + (f * f / 2.0)[:, None] * eye[j + 1])
        D0 = (D0k[j] + spacing * f[:, None] * D1k[j]
              + spacing * spacing * ((f * f / 2.0 - f ** 3 / 6.0)[:, None] * eye[j]
                                     + (f ** 3 / 6.0)[:, None] * eye[j + 1]))
        self._ops_cache[key] = (D0, D1, D2, D0k[knots], D1k[knots], tau)
        return self._ops_cache[key]

    @staticmethod
    def _admm(factor, sigma, q, A, lo, up, rho, iterations: int, warm=None):
        """min 1/2 x'Px + q'x  s.t.  lo <= Ax <= up  (OSQP-style ADMM).

        `factor` is the Cholesky factor of ``P + sigma I + A' diag(rho) A``,
        which does not change between the curvature re-linearisations - only
        the bounds move - so the caller factorises once per solve.  Bounded
        work: at most `iterations` sweeps, early exit on the primal residual.
        Returns ``(x, z, y, used)``.
        """
        from scipy.linalg import cho_solve
        alpha = 1.6
        if warm is not None:
            x, z, y = (item.copy() for item in warm)
        else:
            x = np.zeros(A.shape[1])
            z = A @ x
            y = np.zeros(len(lo))
        used = 0
        for used in range(1, iterations + 1):
            rhs = sigma * x - q + A.T @ (rho * z - y)
            x_new = cho_solve(factor, rhs, check_finite=False)
            step = A @ x_new
            x = alpha * x_new + (1.0 - alpha) * x
            mixed = alpha * step + (1.0 - alpha) * z
            z_new = np.clip(mixed + y / rho, lo, up)
            y = y + rho * (mixed - z_new)
            z = z_new
            if used % 25 == 0:
                residual = np.max(np.abs(A @ x - z))
                if residual < 1e-3:
                    break
        return x, z, y, used

    # -------------------------------------------------------------- assembly
    def _build(self, target: Target, s_car: float, speed: float,
               obstacles, lateral: float = 0.0) -> Optional[AvoidancePlan]:
        """Solve for a profile round `target`, or explain why not."""
        config = self.config
        base = self.raceline
        length = base.length
        budget = [max(int(config.qp_iterations), 50)]
        tried = []

        graft = (self.plan is not None
                 and self.plan.target.obstacle_id == target.obstacle_id
                 and self.plan.profile is not None
                 and self._started(s_car))
        sides = self._sides(target, s_car, tried)
        others = self._other_signature(target, obstacles)
        for attempt, side in enumerate(sides):
            # Each side gets a fair share of what is left, or the first can
            # starve the second and a compute limit reads as geometry.
            share = budget[0] // (len(sides) - attempt)
            spent = [share]
            profile = self._solve(target, s_car, speed, side, obstacles,
                                  graft, spent, tried, lateral)
            budget[0] -= share - spent[0]
            if profile is None:
                continue
            d_full, d1_full, d2_full, s_start, s_end = profile
            window = np.abs(d_full) > 1e-3
            if not np.any(window):
                tried.append('no_room')
                continue
            # The finished geometry, from the profile arrays alone - the
            # speed caps need it BEFORE the line is built, so the line is
            # built exactly once (its speed profile is the expensive part).
            one_minus = 1.0 - d_full * base.kappa
            if float(np.min(one_minus)) < 0.1:
                tried.append('unsafe')
                continue
            points = self._displace(d_full)
            psi = base.psi + np.arctan2(d1_full, one_minus)
            kappa_e = offset_curvature(d_full, d1_full, d2_full,
                                       base.kappa, self._kappa_rate)
            # The error the car is CARRYING right now decays over the next
            # second or two of travel, and until it has, the slack the plan
            # thinks it has is not all there.
            err0 = abs(lateral - float(d_full[
                int(round(s_car / base.spacing)) % len(base.s)]))
            fade_s = np.asarray(wrap_s(base.s[window] - s_car, length))
            inherited = np.where(
                fade_s >= 0.0,
                err0 * np.exp(-fade_s / max(1.5 * max(speed, 1.0), 1.0)),
                0.0)
            # The error at a sample is driven by the sharpest bend being
            # negotiated there - the BASE line's, when that is sharper than
            # the offset path's own: passing outside a near-lock apex, the
            # offset curvature reads a gentle 0.6 while the pursuit is
            # cutting across a 1.1, and the caps sized on 0.6 under-predicted
            # the undershoot 3x (measured, 0.50 box at the hairpin apex).
            bend_e = np.maximum(np.abs(kappa_e), np.abs(base.kappa))
            caps = self._speed_caps(points[window], psi[window],
                                    bend_e[window], window, d_full[window],
                                    obstacles, inherited)
            cap_full = np.full(len(base.s), np.inf)
            cap_full[window] = caps
            line = self._offset_line(d_full, d1_full, d2_full, target.s,
                                     cap=cap_full)
            if line is None:
                tried.append('unsafe')
                continue
            if not self._steerable(line, window):
                tried.append('too_tight')
                continue
            if not self._clears_walls(line, window):
                tried.append('unsafe')
                continue
            # The target's own clearance failing is want of room; another
            # obstacle in the way is a blocked corridor.  They are different
            # problems and the log has to say which happened.
            themselves = [obstacle for obstacle in obstacles
                          if obstacle.id == target.obstacle_id]
            around = [obstacle for obstacle in obstacles
                      if obstacle.id != target.obstacle_id]
            if not self._clears(line, window, themselves):
                tried.append('no_room')
                continue
            if not self._clears(line, window, around):
                tried.append('blocked')
                continue
            trailing = (target.s + self._disc_reach()) % length
            deep = int(np.argmax(np.abs(d_full)))
            # `s_start` is where the manoeuvre actually LEAVES the line, not
            # where its solve window happened to begin: a window anchored at
            # the car whose profile hugs zero for three metres has not
            # started anything yet, and the release logic (a ghost obstacle
            # may cancel an un-started plan) must keep meaning that.
            ordered = (np.flatnonzero(
                np.abs(d_full[(int(round(s_start / self.raceline.spacing))
                               + np.arange(len(d_full)))
                              % len(d_full)]) > 0.03))
            if len(ordered):
                liftoff = (s_start + float(ordered[0])
                           * self.raceline.spacing) % length
            else:
                liftoff = s_start
            return AvoidancePlan(
                target=target, side=side, d_apex=float(d_full[deep]),
                s_start=liftoff, s_apex=trailing,
                s_end=s_end, line=line,
                profile=np.vstack([d_full, d1_full, d2_full]),
                others=others)
        if tried:
            self.reason = self._why(tried)
        return None

    def _sides(self, target: Target, s_car: float, tried) -> list:
        """Sides worth trying, the one with more usable room first.

        First apply the hard physical-opening gate (`min_passage_width`), then
        judge the body with the permissive clearance (`lateral_margin_min`).
        A side that cannot give both has no profile to find.

        The side of a STARTED manoeuvre is never re-decided - the car is in
        the middle of it.  Before the turn-out begins the choice stays open:
        an estimate that walks across the line wants the other side, and
        freezing early left the old plan aimed 0.6 m wrong.
        """
        if (self.plan is not None
                and self.plan.target.obstacle_id == target.obstacle_id
                and self._started(s_car)):
            return [self.plan.side]
        held = (self.plan.side
                if self.plan is not None
                and self.plan.target.obstacle_id == target.obstacle_id
                else 0)
        scores = {}
        for side in (1, -1):
            if (self._passage_width(target, side) + 1e-9
                    < self.config.min_passage_width):
                continue
            slack = self._keyhole_slack(target, side)
            if slack < 0.0:
                continue
            # Prefer the outside of the local turn at equal room: the
            # curvature budget is asymmetric there.  And prefer the side the
            # standing plan already chose: before the turn-out begins the
            # choice is still open, but flipping it on centimetres of
            # centroid jitter builds a different plan every cycle.
            lean = -0.06 * side * self.raceline.mean_curvature(target.s, 3.0)
            scores[side] = slack + lean + (0.3 if side == held else 0.0)
        if not scores:
            tried.append('no_room')
            return []
        return sorted(scores, key=lambda side: -scores[side])

    def _passage_width(self, target: Target, side: int) -> float:
        """Narrowest physical obstacle-edge-to-wall opening on one side.

        Raceline corridor columns are distances from the base line to the
        mapped walls.  Across the obstacle's longitudinal footprint, subtract
        its circular cross-section from the selected wall distance.  Include
        the exact centre section explicitly so a coarse raceline grid cannot
        overestimate the opening by missing the obstacle's widest section.

        This is deliberately separate from `_keyhole_slack`: this answers
        "is the real opening at least the configured minimum?", while the
        keyhole answers "can this car, its disc cover, and its safety margins
        actually trace a steerable path through it?".
        """
        side = 1 if side >= 0 else -1
        base = self.raceline
        delta = wrap_s(base.s - target.s, base.length)
        band = np.abs(delta) <= target.radius
        widths = base.w_left if side > 0 else base.w_right
        gaps = []
        if np.any(band):
            half = np.sqrt(np.maximum(target.radius ** 2
                                      - delta[band] ** 2, 0.0))
            gaps.extend((widths[band] - side * target.d - half).tolist())
        centre_width = base.corridor_at(target.s)[0 if side > 0 else 1]
        gaps.append(centre_width - side * target.d - target.radius)
        return float(np.min(gaps))

    def _keyhole_slack(self, target: Target, side: int) -> float:
        """Worst-sample slack between the pass requirement and the tube."""
        config = self.config
        length = self.raceline.length
        clearance = target.radius + self._disc_radius() + config.lateral_margin_min
        reach = clearance + self._disc_reach()
        delta = wrap_s(self.raceline.s - target.s, length)
        band = np.abs(delta) <= reach
        if not np.any(band):
            return -1.0
        flat = np.maximum(np.abs(delta[band]) - self._disc_reach(), 0.0)
        need = target.d + side * np.sqrt(
            np.maximum(clearance ** 2 - flat ** 2, 0.0))
        tube = self._tube_hi[band] if side > 0 else self._tube_lo[band]
        return float(np.min(side * (tube - need)))

    def _solve(self, target: Target, s_car: float, speed: float, side: int,
               obstacles, graft: bool, budget, tried, lateral: float = 0.0):
        """One QP: the smoothest profile in the tube that makes the pass.

        Returns ``(d, d', d'', s_start, s_end)`` on the full raceline grid,
        or None with a reason appended to `tried`.
        """
        config = self.config
        base = self.raceline
        length, h = base.length, base.spacing
        reference = max(speed, base.speed_at(target.s), 0.5)
        back = (config.window_back_fast if reference >= config.window_fast_speed
                else config.window_back)
        # SIGNED distance to the target: the estimate of a stationary cone
        # walks as the car closes on it, and once it walks behind the car a
        # `% length` here would call it a whole lap ahead and solve a 40 m
        # window every cycle.  Behind the car there is nothing left to plan.
        ahead = float(wrap_s(target.s - s_car, length))
        start_gap = 0.2
        if graft:
            if ahead < 0.3:
                tried.append('too_late')
                return None
            s_from = s_car
            d_start = self._profile_at(s_car)
        else:
            if ahead < 1.0:
                tried.append('too_late')
                return None
            # The window starts AT THE CAR, from where the car actually is:
            # a fresh plan after a stop-and-back-out can find the car half a
            # metre off the line, and a profile that starts at d = 0 just
            # ahead of it demands an impossible snap back before the
            # manoeuvre even begins.  On a normal approach `lateral` is
            # centimetres and this is the same window as before.
            run_up = min(back, max(ahead - start_gap, 0.0))
            s_from = (s_car + max(ahead - start_gap - run_up, 0.0)) % length
            index_car = int(round((s_from % length) / h)) % len(base.s)
            inside = float(np.clip(lateral, self._tube_lo[index_car],
                                   self._tube_hi[index_car]))
            d_start = (inside if abs(inside) > 0.05 else 0.0, 0.0, 0.0)
        s_from = (round(s_from / h) * h) % length
        span = float(wrap_s(target.s - s_from, length)) + config.window_out
        span = min(math.ceil(span / 0.5) * 0.5,
                   back + config.window_out + 1.0,
                   math.floor(0.4 * length / 0.5) * 0.5)
        if span < 1.0:
            tried.append('too_late')
            return None
        knots = max(4, int(round(span / config.qp_knot_spacing)))
        spacing = span / knots
        step = config.qp_sample_spacing

        D0, D1, D2, d0_end, d1_end, tau = self._ops(knots, spacing, step)
        count = knots + 1
        arcs = s_from + tau
        idx = (np.round((arcs % length) / h).astype(int)) % len(base.s)
        # The constraints are sampled every `step` but the finished line is
        # judged on the raceline's own finer grid, whose curvature peaks can
        # fall BETWEEN constraint samples.  Each sample therefore stands for
        # its whole neighbourhood - enough neighbours either side that the
        # pooled windows tile the grid - and the curvature box is taken as
        # the INTERSECTION over those neighbours, so it holds at every grid
        # point the sample covers, whichever way each of them bends.
        half = max(1, int(math.ceil(step / (2.0 * h))))
        spread = np.stack([(idx + shift) % len(base.s)
                           for shift in range(-half, half + 1)])
        kappas = base.kappa[spread]
        kappa_rates = self._kappa_rate[spread]
        d_a, dp_a, dd_a = d_start
        affine = d_a + dp_a * tau

        # The tube for THIS solve: the precomputed one, shrunk by the solver
        # tolerance, with every other tracked obstacle folded in as a keyhole
        # exclusion on its roomier side.
        hi = np.min(self._tube_hi[spread], axis=0) - config.tube_margin
        lo = np.max(self._tube_lo[spread], axis=0) + config.tube_margin
        hi, lo, folded = self._fold(hi, lo, arcs, target, obstacles, tried)
        if hi is None:
            return None
        if graft:
            # The car is not ON the profile it is grafting to - `lateral` is
            # where it actually is.  Book the difference against the tube
            # over the next second of travel, fading out as the controller
            # takes it out, so the re-solved profile leaves room for the
            # error to decay instead of assuming it already has.
            error = lateral - d_a
            fade = np.clip(1.0 - tau / max(1.0, speed), 0.0, 1.0)
            if error > 0.0:
                hi = hi - error * fade
            else:
                lo = lo - error * fade

        # The pass: each body disc keeps `lateral_margin` (comfort; the gate
        # holds out for `lateral_margin_min`) from the obstacle.  Plus the
        # sagitta: the discs sit along the BODY, and through a curve a disc
        # a reach ahead stands closer to things beside the path than the
        # path point does - the Frenet lateral difference alone under-books
        # the true gap by reach^2 |k| / 2, which at the 0.84 1/m left-hander
        # is the 3 cm between passing and being refused by the gate.
        sagitta = (self._disc_reach() ** 2
                   * np.abs(base.kappa[idx]) / 2.0)
        # AIM as if the obstacle were the biggest the competition runs: a
        # box first seen obliquely round a corner reads 0.21-0.23 m of
        # radius and IS a 0.35 - measured, and the pass planned for the
        # small reading hit the big box.  The sigma slack hands the extra
        # depth back wherever the corridor cannot give it, and the gate
        # still judges the finished line against the TRACKED radius, so
        # nothing that used to fit is refused - roomy passes just run wide
        # of what the silhouette might be hiding.
        clearance = (max(target.radius, config.obstacle_max_radius)
                     + self._disc_radius() + config.lateral_margin)
        disc = self._disc_reach()
        delta = np.abs(wrap_s(arcs - target.s, length))
        flat = np.maximum(delta - disc, 0.0)
        band = flat < clearance
        if not np.any(band):
            tried.append('too_late')
            return None
        required = target.d + side * (np.sqrt(
            np.maximum(clearance ** 2 - flat ** 2, 0.0)) + sagitta)

        # Rows: 4 equalities (M0, M_end, d(end), d'(end)), the tube, the
        # d'' box, the pass band (with one shared slack), slack >= 0.
        samples = len(tau)
        n = count + 1
        Z0 = np.zeros((samples, n))
        Z0[:, :count] = D0
        Z2 = np.zeros((samples, n))
        Z2[:, :count] = D2
        eq = np.zeros((4, n))
        eq[0, 0] = 1.0
        eq[1, count - 1] = 1.0
        eq[2, :count] = d0_end
        eq[3, :count] = d1_end
        eq_val = np.array([dd_a, 0.0, -(d_a + dp_a * span), -dp_a])
        pass_rows = Z0[band] * side
        pass_rows[:, count] = 1.0
        slack_row = np.zeros((1, n))
        slack_row[0, count] = 1.0
        A = np.vstack([eq, Z0, Z2, pass_rows, slack_row])
        n_band = int(band.sum())
        lo_vec = np.concatenate([eq_val, lo - affine,
                                 np.full(samples, -np.inf),
                                 side * required[band] - side * affine[band],
                                 [0.0]])
        up_vec = np.concatenate([eq_val, hi - affine,
                                 np.full(samples, np.inf),
                                 np.full(n_band, np.inf), [np.inf]])
        # Tube rows another obstacle folded into carry obstacle-grade weight:
        # the depth cost pulls the profile toward zero, and at 3.0 the fold
        # loses that fight by the few millimetres `_clears` then refuses.
        tube_rho = np.full(samples, 3.0)
        tube_rho[folded] = 30.0
        rho = np.concatenate([np.full(4, 1e3), tube_rho,
                              np.full(samples, 1.0), np.full(n_band, 30.0),
                              [1.0]])
        weight = step
        P = 2.0 * weight * (Z2.T @ Z2) + 2.0 * 0.05 * weight * (Z0.T @ Z0)
        P[count, count] += 2.0 * 800.0
        q = 2.0 * 0.05 * weight * (Z0[:, :count].T @ affine)
        q = np.concatenate([q, [0.0]])

        from scipy.linalg import cho_factor
        sigma = 1e-6
        factor = cho_factor(P + sigma * np.eye(n) + (A.T * rho) @ A,
                            lower=True, check_finite=False)
        limit_left = self.vehicle.curvature_towards(1.0) * config.lock_fraction
        limit_right = self.vehicle.curvature_towards(-1.0) * config.lock_fraction
        d_bar = affine.copy()
        dp_bar = np.full(samples, dp_a)
        warm = None
        x = None
        for round_ in range(max(int(config.qp_relinearizations), 1)):
            box_lo = np.full(samples, -np.inf)
            box_hi = np.full(samples, np.inf)
            for row in range(spread.shape[0]):
                row_lo, row_hi = curvature_dd_bounds(
                    d_bar, dp_bar, kappas[row], kappa_rates[row],
                    limit_left, limit_right)
                box_lo = np.maximum(box_lo, row_lo)
                box_hi = np.minimum(box_hi, row_hi)
            lo_vec[4 + samples:4 + 2 * samples] = box_lo
            up_vec[4 + samples:4 + 2 * samples] = box_hi
            # The graft's starting d'' is an EQUALITY; if it falls outside
            # the box the two constraints fight and the solve stalls.  The
            # controller feels d and d' continuity; d'' may be clipped.
            lo_vec[0] = up_vec[0] = float(np.clip(dd_a, box_lo[0], box_hi[0]))
            if budget[0] <= 0:
                tried.append('budget')
                return None
            if warm is not None:
                # The box rows moved from under their duals; a stale dual on
                # a moved bound is why a second solve can need its full
                # allowance after the first converged in a handful.
                warm[2][4 + samples:4 + 2 * samples] = 0.0
            allowed = min(budget[0], 150)
            x, z, y, used = self._admm(factor, sigma, q, A, lo_vec, up_vec,
                                       rho, allowed, warm)
            budget[0] -= used
            warm = (x, z, y)
            moments = x[:count]
            d_bar = affine + D0 @ moments
            dp_bar = dp_a + D1 @ moments
        moments = x[:count]

        # A WILDLY under-converged solve is a refusal, not a plan.  Small
        # residuals are the gates' business - every tube component is a
        # conservative estimate of something `_steerable` or `_clears_walls`
        # verifies exactly on the finished line, so a profile a few
        # centimetres over the shrunk tube is judged on its merits.  A
        # profile 15 cm outside it is not a near-miss, it is noise, and
        # putting it through the gates just spends their time.
        along = Z0[:, :count] @ moments + affine
        overshoot = float(np.max(np.maximum(along - hi, lo - along)))
        if overshoot > 0.15:
            tried.append('budget')
            return None

        # Evaluate the SAME spline analytically on the raceline's own grid.
        D0g, D1g, D2g, _, _, tau_g = self._ops(knots, spacing, h)
        grid = len(tau_g)
        i0 = int(round(s_from / h)) % len(base.s)
        idx_g = (i0 + np.arange(grid)) % len(base.s)
        d_full = np.zeros(len(base.s))
        d1_full = np.zeros(len(base.s))
        d2_full = np.zeros(len(base.s))
        d_full[idx_g] = d_a + dp_a * tau_g + D0g @ moments
        d1_full[idx_g] = dp_a + D1g @ moments
        d2_full[idx_g] = D2g @ moments
        s_start = s_from
        if graft:
            # Behind the car the OLD profile is still the line being driven;
            # leaving it at zero puts a 0.7 m step into the published line at
            # the car itself, and the e-stop corridor watches that line.
            # The graft matched (d, d', d'') at `s_from`, so old-then-new is
            # one C2 profile.
            standing = self.plan.profile
            start_i = int(round(self.plan.s_start / h)) % len(base.s)
            count_back = int(round(
                (wrap_s(s_from - self.plan.s_start, length)) / h))
            if 0 < count_back < len(base.s):
                behind = (start_i + np.arange(count_back)) % len(base.s)
                d_full[behind] = standing[0, behind]
                d1_full[behind] = standing[1, behind]
                d2_full[behind] = standing[2, behind]
                s_start = self.plan.s_start
        return d_full, d1_full, d2_full, s_start, (s_from + span) % length

    def _fold(self, hi, lo, arcs, target: Target, obstacles, tried):
        """Clip the tube round every OTHER tracked obstacle in the window.

        Each is excluded on whichever side of it has more tube room; if the
        fold empties the tube anywhere, that obstacle's side is flipped once,
        and if it is still empty the corridor genuinely is blocked.
        """
        config = self.config
        length = self.raceline.length
        # Booked a shade beyond the gate's own floor: a solution that sits ON
        # the folded boundary is a millimetre from `_clears` refusing it, and
        # an ADMM answer is only ever within tolerance of where it was asked
        # to be.  The extra centimetre and a half is what lets a thread
        # between two obstacles come out the gates' side of the line.
        margin = (self._disc_radius() + config.lateral_margin_min
                  + config.tube_margin / 2.0)
        folded = np.zeros(len(arcs), dtype=bool)
        for obstacle in obstacles:
            if obstacle.id == target.obstacle_id:
                continue
            _, s_obs, d_obs = self.raceline.project(obstacle.position)
            clearance = obstacle.radius + margin
            reach = clearance + self._disc_reach()
            delta = np.abs(wrap_s(arcs - s_obs, length))
            band = delta < reach
            if not np.any(band):
                continue
            flat = np.maximum(delta[band] - self._disc_reach(), 0.0)
            index = (np.round((arcs[band] % length)
                              / self.raceline.spacing).astype(int)
                     % len(self.raceline.s))
            sagitta = (self._disc_reach() ** 2
                       * np.abs(self.raceline.kappa[index]) / 2.0)
            need = np.sqrt(np.maximum(clearance ** 2 - flat ** 2, 0.0)) + sagitta
            room_left = float(np.min(hi[band])) - d_obs
            # The room below is worst where the LOWER bound is highest.
            room_right = d_obs - float(np.max(lo[band]))
            for side in ((1, -1) if room_left >= room_right else (-1, 1)):
                hi_try, lo_try = hi.copy(), lo.copy()
                if side > 0:
                    lo_try[band] = np.maximum(lo_try[band], d_obs + need)
                else:
                    hi_try[band] = np.minimum(hi_try[band], d_obs - need)
                if np.all(hi_try >= lo_try - 1e-9):
                    hi, lo = hi_try, lo_try
                    folded |= band
                    break
            else:
                tried.append('blocked')
                return None, None, None
        return hi, lo, folded

    def _profile_at(self, s: float):
        """``(d, d', d'')`` of the STANDING plan at arc length `s`."""
        profile = self.plan.profile
        index = int(round((s % self.raceline.length) / self.raceline.spacing))
        index %= profile.shape[1]
        return (float(profile[0, index]), float(profile[1, index]),
                float(profile[2, index]))

    @staticmethod
    def _why(tried) -> str:
        """One reason out of everything that was tried.

        When the attempts failed differently, say so: "no room on the side
        with space, blocked on the other" is a different problem from either
        half alone.
        """
        order = ('no_room', 'too_late', 'too_tight', 'blocked', 'unsafe',
                 'budget')
        seen = [reason for reason in order if reason in tried]
        return '+'.join(seen) if seen else tried[-1]

    # -------------------------------------------------------------- geometry
    def _displace(self, d) -> np.ndarray:
        base = self.raceline
        normal = np.column_stack([-np.sin(base.psi), np.cos(base.psi)])
        return base.points + np.asarray(d)[:, None] * normal

    def _offset_line(self, d, slope, second, s_apex: float,
                     cap=None) -> Optional[Raceline]:
        """The displaced closed line, with its own heading, curvature, speed.

        `cap` is an optional per-sample speed ceiling on top of the grip
        limit - the slack-matched crawl of `_speed_caps` - applied before
        the ramps so braking into it and pulling away from it stay feasible.
        """
        base = self.raceline
        config, vehicle = self.config, self.vehicle
        one_minus = 1.0 - d * base.kappa
        if float(np.min(one_minus)) < 0.1:
            return None                     # offset past the centre of curvature
        points = self._displace(d)
        psi = base.psi + np.arctan2(slope, one_minus)
        scale = np.hypot(one_minus, slope)
        kappa = ((one_minus * (base.kappa * one_minus + second)
                  + slope * (2.0 * slope * base.kappa + d * self._kappa_rate))
                 / np.maximum(scale, 1e-6) ** 3)

        step = scale * base.spacing
        speed = np.minimum(base.vx, np.sqrt(
            vehicle.max_lateral_accel * config.grip_margin
            / np.maximum(np.abs(kappa), 1e-3)))
        if cap is not None:
            speed = np.minimum(speed, cap)
        if config.pass_speed > 0.0:
            beside = np.abs(wrap_s(base.s - s_apex, base.length)) <= config.pass_window
            speed[beside] = np.minimum(speed[beside], config.pass_speed)
        speed = _limit_ramps(speed, step, vehicle.max_accel * config.grip_margin,
                             vehicle.max_decel * config.grip_margin)
        accel = (np.roll(speed, -1) ** 2 - speed ** 2) / (2.0 * np.maximum(step, 1e-6))

        data = np.empty((len(base.s), 9))
        data[:, S] = base.s
        data[:, X], data[:, Y] = points[:, 0], points[:, 1]
        data[:, PSI], data[:, KAPPA] = psi, kappa
        data[:, VX], data[:, AX] = speed, accel
        data[:, W_LEFT] = base.w_left - d
        data[:, W_RIGHT] = base.w_right + d
        return Raceline(data, name='avoidance')

    # ----------------------------------------------------------------- gates
    def _steerable(self, line: Raceline, window) -> bool:
        """Can the car actually drive this?

        Judged at EVERY sample against the lock available in the direction
        that sample turns - the servo reaches +0.51 rad left and -0.48 right,
        and a manoeuvre bends both ways, so checking only the largest
        excursion against its own side's limit passes paths whose opposite
        bend the servo cannot deliver.  The base line's own demand at the
        same sample is allowed too, since the car is demonstrably driving
        that - per sample, so licence earned at the hairpin is not spent on
        the straight.
        """
        if not np.any(window):
            return True
        kappa = line.kappa[window]
        limit = np.maximum(
            np.where(kappa >= 0.0, self.vehicle.curvature_towards(1.0),
                     self.vehicle.curvature_towards(-1.0)),
            np.abs(self.raceline.kappa[window]))
        return bool(np.all(np.abs(kappa) <= limit))

    def _clears_walls(self, line: Raceline, window) -> bool:
        """Does the finished manoeuvre keep the BODY off the walls?

        Checked at each disc of the cover, not at the point the body is
        centred on: through a corner the nose and tail swing wide of the path,
        and a line whose own points clear a wall can still put a corner into
        it.  This is the gate the margin promise rests on, so it is the one
        that looks at the whole car.
        """
        if self.track is None or not np.any(window):
            return True
        points = line.points[window]
        headings = line.psi[window]
        radius, offsets = self._body_discs()
        floor = radius + self.config.wall_body_margin
        cos, sin = np.cos(headings), np.sin(headings)
        for offset in offsets:
            clearance = np.asarray(self.track.clearance(
                points[:, 0] + offset * cos, points[:, 1] + offset * sin))
            if float(np.min(clearance)) < floor:
                return False
        return True

    def _clears(self, line: Raceline, window, obstacles) -> bool:
        """Does the evasion line clear EVERY obstacle, not just the target?"""
        if not np.any(window):
            return True
        config = self.config
        points = line.points[window]
        for obstacle in obstacles:
            # The car is YAWED here - it is on an evasion line - so the
            # distance that matters is from the obstacle to the body, not to
            # the point the body is centred on.  Cover the rectangle with
            # discs along its own axis, the way the harness measures it, and
            # the answer is right at every heading without anyone having to
            # remember a cosine.
            gap = self._body_gap(points, line.psi[window], obstacle)
            if gap < obstacle.radius + config.lateral_margin_min:
                return False
        return True

    def _speed_caps(self, points, headings, kappa, window, d,
                    obstacles, inherited=None) -> np.ndarray:
        """Per-sample speed ceilings that make the achieved slack honest.

        Tracking error grows with speed and path curvature - see the model
        in `_cap_from_slack` - so the speed at each sample is capped where
        the predicted error still fits inside its obstacle clearance.  Wall
        room is geometric validation only by default: a nearby mapped side
        wall does not lower speed.  The optional legacy wall cap can be
        enabled explicitly, but the car configuration leaves it off.  The
        result is floored at `min_pass_speed`, where the predicted error is
        millimetres and the e-stop still stands behind everything.
        """
        del d
        config = self.config
        radius, offsets = self._body_discs()
        cos, sin = np.cos(headings), np.sin(headings)
        wall = np.full(len(points), np.inf)
        obstruction = np.full(len(points), np.inf)
        for offset in offsets:
            x = points[:, 0] + offset * cos
            y = points[:, 1] + offset * sin
            if self.track is not None and config.enable_wall_speed_cap:
                wall = np.minimum(
                    wall, np.asarray(self.track.clearance(x, y)) - radius)
            for obstacle in obstacles:
                obstruction = np.minimum(
                    obstruction, np.hypot(x - obstacle.x, y - obstacle.y)
                    - radius - obstacle.radius)
        if inherited is not None:
            obstruction = obstruction - inherited
            if config.enable_wall_speed_cap:
                wall = wall - inherited
        # A frozen estimate still carries the bias it froze with - measured
        # ~0.1 m on the 0.50 m box at the occluded apex, which let the cap
        # allow 2 m/s through a slack that was not really there and cost a
        # 6 mm flank graze.  Book it against the obstacle slack only; walls
        # come from the map, not the tracker.
        obstruction = obstruction - config.pass_obstacle_reserve
        caps = self._cap_from_slack(obstruction, kappa)
        if config.enable_wall_speed_cap:
            wall_caps = self._cap_from_slack(wall, kappa)
            unchanged = wall >= self._base_wall_slack[window] - 0.01
            wall_caps[unchanged] = np.maximum(
                wall_caps[unchanged], self.raceline.vx[window][unchanged])
            caps = np.minimum(caps, wall_caps)
        caps = np.maximum(caps, config.min_pass_speed)
        # Reach each cap BEFORE its thin spot, not at it: the drivetrain
        # answers a braking command ~0.3 s late (the 0.14 s actuation delay
        # plus the VESC's own lag), so the commanded speed must already be
        # down a lag-distance earlier or the car arrives carrying the speed
        # it was told to shed.  A backward rolling minimum over ~0.8 m does
        # that; `_limit_ramps` then makes the braking into it feasible.
        pull = max(1, int(round(0.8 / max(self.raceline.spacing, 1e-3))))
        pulled = caps.copy()
        for ahead in range(1, pull + 1):
            pulled[:-ahead] = np.minimum(pulled[:-ahead], caps[ahead:])
        return pulled

    def _cap_from_slack(self, slack, kappa) -> np.ndarray:
        """Invert the tracking-error model for the speed a slack affords.

        e_p95 ~ quad |kappa| v^2 + lin v + floor: the quadratic limb is the
        path curvature times the delay-squared dynamics, so a gentle bump on
        a straight is barely capped while the same speed through a 1.4 1/m S
        is; the linear and constant limbs are what is left at zero curvature
        (localisation and the controller's own floor).  Fitted to closed-loop
        measurements: 0.19 m at 2.4 m/s through the hairpin S (|k| ~ 1.4),
        0.09 m at ~5 m/s on a straight bump (|k| ~ 0.2), 0.021 m at a
        0.74 m/s crawl, and consistent with the racing line's own 0.11 m at
        7.5 m/s - and it must NOT cap the racing line itself anywhere, which
        is the standing consistency test.
        """
        config = self.config
        budget = np.maximum(
            config.pass_error_factor * (slack - config.pass_error_keep)
            - config.pass_error_floor, 1e-4)
        # Near the lock the model must balloon: a car with the wheel on the
        # stop has no authority left to take error out, and the linear-
        # tracking assumption under the fit breaks down.  Measured: a pass
        # planned at |k| = 1.3 (90 % of the right lock) was commanded 1.5 m/s
        # by the plain model and the car converged INTO the box with the
        # steering saturated the whole way.
        lock = np.where(np.asarray(kappa) >= 0.0,
                        self.vehicle.curvature_towards(1.0),
                        self.vehicle.curvature_towards(-1.0))
        inflation = np.clip(lock / np.maximum(lock - np.abs(kappa), 1e-3),
                            1.0, 7.5)
        bend = np.abs(kappa) * max(config.pass_error_quad, 1e-6) * inflation
        lin = max(config.pass_error_lin, 1e-6)
        straight = bend < 1e-9
        with np.errstate(divide='ignore', invalid='ignore'):
            caps = (-lin + np.sqrt(lin * lin + 4.0 * bend * budget)) \
                / (2.0 * bend)
        caps[straight] = budget[straight] / lin
        return caps

    def _body_gap(self, points: np.ndarray, headings: np.ndarray,
                  obstacle) -> float:
        """Closest approach of the CAR BODY to an obstacle, over a path.

        Three discs covering the 0.58 x 0.31 m rectangle - one at the centre
        and one at each third - which is the same cover `tools/sim_harness.py`
        measures wall clearance with, so the planner books what the harness
        checks.  Anchored on the path point: the pose a raceline carries is
        the point the body is centred on.
        """
        radius, offsets = self._body_discs()
        cos, sin = np.cos(headings), np.sin(headings)
        closest = np.inf
        for offset in offsets:
            gap = np.hypot(points[:, 0] + offset * cos - obstacle.x,
                           points[:, 1] + offset * sin - obstacle.y)
            closest = min(closest, float(np.min(gap)))
        return closest - radius

    def _line_clearance(self) -> float:
        """Map clearance the line's own points must keep.

        The body plus the gap the config asks for.  Every sizing step works in
        centre-of-body terms, so this is what they compare against; the final
        gate in `_clears_walls` then re-checks the finished manoeuvre at each
        disc, where a curve puts the nose and tail further out than the point
        the body is centred on.
        """
        return self._disc_radius() + self.config.wall_body_margin

    def _disc_radius(self) -> float:
        radius, _ = self._body_discs()
        return radius

    def _body_discs(self):
        """``(radius, offsets)`` of a disc cover of the car.

        `body_disc_count` discs along the body axis, each of radius
        hypot(segment/2, half_width): the fewer the discs, the fatter each
        one has to be to cover the rectangle's corners, and the fatter the
        model the wider the gaps it refuses.  Three books the 0.28 m car as
        0.34 m; nine books it as 0.29 and lets it take the 0.30 m gaps it
        physically fits through.
        """
        config = self.config
        count = max(1, int(config.body_disc_count))
        segment = 2.0 * config.ego_half_length / count
        radius = math.hypot(segment / 2.0, config.ego_half_width)
        offsets = tuple(-config.ego_half_length + segment * (index + 0.5)
                        for index in range(count))
        return radius, offsets

    def _disc_reach(self) -> float:
        """How far the outermost disc's CENTRE sits from the path point."""
        _, offsets = self._body_discs()
        return max(abs(offset) for offset in offsets)


def _limit_ramps(speed: np.ndarray, step: np.ndarray, accel: float,
                 brake: float) -> np.ndarray:
    """Make a speed profile reachable: brake into every cap, accelerate out.

    Two passes round the closed loop, because a cap on the last sample has to
    be able to reach back across the start/finish line.
    """
    speed = np.asarray(speed, dtype=float).copy()
    count = len(speed)
    for _ in range(2):
        for index in range(count - 1, -1, -1):
            ahead = speed[(index + 1) % count]
            speed[index] = min(speed[index],
                               math.sqrt(ahead * ahead + 2.0 * brake * step[index]))
    for _ in range(2):
        for index in range(count):
            behind = speed[(index - 1) % count]
            speed[index] = min(speed[index],
                               math.sqrt(behind * behind
                                         + 2.0 * accel * step[(index - 1) % count]))
    return speed


def path_ahead(line: Raceline, position, distance: float,
               spacing: float = 0.15) -> np.ndarray:
    """The line the car is about to drive, from the car itself, in map frame.

    Starts at the car - not at its projection - so that a corridor built on it
    is a corridor the CAR sweeps, and any lateral error the controller still
    has to take out is part of the shape.
    """
    position = np.asarray(position, dtype=float)
    _, s_car, _ = line.project(position)
    steps = max(2, int(distance / max(spacing, 1e-3)))
    samples = line.sample(s_car + np.linspace(0.0, distance, steps))[:, :2]
    return np.vstack([position, samples])
