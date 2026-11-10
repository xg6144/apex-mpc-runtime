"""Scan-based obstacle detection and tracking against a known static map.

Both modules are ROS-independent.  ``MapDistanceField`` turns the occupancy
map into a Euclidean clearance field used to reject wall returns during
detection and to keep MPC rollouts away from walls.  ``ObstacleTracker``
clusters the remaining lidar points and tracks position and velocity so the
controller can predict opponent motion over its horizon.
"""

import re
from pathlib import Path
from typing import List, Optional

import numpy as np


def extrapolate_obstacles(obstacles: np.ndarray, age: float,
                          maximum_age: float) -> np.ndarray:
    """Advance tracked centres to the consumer time without runaway drift.

    Scan processing may run in parallel with the controller.  This helper
    compensates the resulting bounded snapshot age for moving rows while
    leaving the immutable producer array untouched.  Negative ages (clock
    quantisation) become zero and delayed frames never extrapolate beyond the
    caller's explicit safety cap.
    """
    rows = np.asarray(obstacles, dtype=float).reshape(-1, 5)
    dt = float(np.clip(age, 0.0, max(float(maximum_age), 0.0)))
    if dt <= 0.0 or rows.shape[0] == 0:
        return rows
    moving = np.any(rows[:, 2:4] != 0.0, axis=1)
    if not np.any(moving):
        return rows
    predicted = rows.copy()
    predicted[moving, 0:2] += predicted[moving, 2:4] * dt
    return predicted


def consolidate_static_obstacles(obstacles: np.ndarray, merge_gap: float = 0.05,
                                 static_speed: float = 0.10) -> np.ndarray:
    """Merge overlapping static discs without losing covered geometry.

    Lidar fragmentation can represent one cone/body with several adjacent
    tracks.  Each merge uses the exact enclosing circle of the two input
    circles, so every point covered before consolidation remains covered.
    Moving rows are intentionally left untouched because their multiple discs
    encode a vehicle capsule and must retain their individual predictions.
    """
    rows = np.asarray(obstacles, dtype=float).reshape(-1, 5)
    if rows.shape[0] < 2 or merge_gap < 0.0:
        return rows.copy()

    moving = np.hypot(rows[:, 2], rows[:, 3]) > static_speed
    result = [row.copy() for row in rows[moving]]
    pending = [row.copy() for row in rows[~moving]]
    changed = True
    while changed:
        changed = False
        for first in range(len(pending)):
            for second in range(first + 1, len(pending)):
                a = pending[first]
                b = pending[second]
                dx = float(b[0] - a[0])
                dy = float(b[1] - a[1])
                distance = float(np.hypot(dx, dy))
                if distance > a[4] + b[4] + merge_gap:
                    continue
                if a[4] >= distance + b[4]:
                    merged = a.copy()
                elif b[4] >= distance + a[4]:
                    merged = b.copy()
                else:
                    radius = 0.5 * (distance + a[4] + b[4])
                    fraction = 0.0 if distance <= 1e-12 else \
                        (radius - a[4]) / distance
                    weight = max(float(a[4] + b[4]), 1e-9)
                    merged = np.array([
                        a[0] + fraction * dx,
                        a[1] + fraction * dy,
                        (a[4] * a[2] + b[4] * b[2]) / weight,
                        (a[4] * a[3] + b[4] * b[3]) / weight,
                        radius,
                    ])
                pending[first] = merged
                del pending[second]
                changed = True
                break
            if changed:
                break
    result.extend(pending)
    return np.asarray(result, dtype=float).reshape(-1, 5)


class MapDistanceField:
    """Euclidean distance (m) from any world point to the nearest wall."""

    def __init__(self, image: np.ndarray, resolution: float, origin_x: float,
                 origin_y: float, free_threshold: int = 250):
        from scipy.ndimage import distance_transform_edt

        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError('map image must be 2-D grayscale')
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.height, self.width = image.shape
        free = image >= free_threshold
        self._distance = distance_transform_edt(free) * self.resolution

    @classmethod
    def from_map_yaml(cls, yaml_path) -> 'MapDistanceField':
        """Load from a ROS map yaml (pgm image next to it)."""
        from PIL import Image

        source = Path(yaml_path).expanduser()
        text = source.read_text()
        resolution = float(re.search(r'resolution:\s*([0-9.eE+-]+)', text).group(1))
        origin = re.search(r'origin:\s*\[([^\]]+)\]', text).group(1).split(',')
        image_name = re.search(r'image:\s*(\S+)', text).group(1)
        image = np.array(Image.open(source.parent / image_name).convert('L'))
        return cls(image, resolution, float(origin[0]), float(origin[1]))

    def with_static_discs(self, discs, influence: float = 1.0) -> 'MapDistanceField':
        """Return a field whose clearance also counts frozen static discs.

        ``discs`` rows are (x, y, radius) in map coordinates.  The distance to
        a disc is analytic (||p - c|| - r), so the composite needs no EDT
        recomputation: each disc is min-stamped into a copy of the base grid
        inside its influence window.  Beyond ``influence`` metres of the disc
        surface the stamp is skipped - the solver's wall costs only activate
        below ~0.4 m clearance, so larger values never contribute.  The base
        field is not modified; callers keep it to rebuild composites.
        """
        discs = np.asarray(discs, dtype=float).reshape(-1, 3)
        clone = MapDistanceField.__new__(MapDistanceField)
        clone.resolution = self.resolution
        clone.origin_x = self.origin_x
        clone.origin_y = self.origin_y
        clone.height = self.height
        clone.width = self.width
        distance = self._distance.copy()
        for x, y, radius in discs:
            if not np.isfinite([x, y, radius]).all() or radius < 0.0:
                raise ValueError('static discs must be finite with r >= 0')
            reach = radius + influence
            # Inverse of the clearance() mapping: cell centre coordinates.
            col_low = int(np.floor((x - reach - self.origin_x)
                                   / self.resolution - 0.5))
            col_high = int(np.ceil((x + reach - self.origin_x)
                                   / self.resolution - 0.5)) + 1
            row_low = int(np.floor(self.height
                                   - (y + reach - self.origin_y)
                                   / self.resolution - 0.5))
            row_high = int(np.ceil(self.height
                                   - (y - reach - self.origin_y)
                                   / self.resolution - 0.5)) + 1
            col_low = max(col_low, 0)
            row_low = max(row_low, 0)
            col_high = min(col_high, self.width)
            row_high = min(row_high, self.height)
            if col_low >= col_high or row_low >= row_high:
                continue
            cols = np.arange(col_low, col_high)
            rows = np.arange(row_low, row_high)
            cell_x = self.origin_x + (cols + 0.5) * self.resolution
            cell_y = self.origin_y + (self.height - rows - 0.5) * self.resolution
            disc_distance = np.hypot(cell_x[None, :] - x, cell_y[:, None] - y)
            np.minimum(
                distance[row_low:row_high, col_low:col_high],
                np.maximum(disc_distance - radius, 0.0),
                out=distance[row_low:row_high, col_low:col_high])
        clone._distance = distance
        return clone

    def clearance(self, x, y) -> np.ndarray:
        """Bilinear wall clearance at world coordinates; 0 outside the map."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        column = (x - self.origin_x) / self.resolution - 0.5
        row = self.height - (y - self.origin_y) / self.resolution - 0.5
        c0 = np.floor(column).astype(int)
        r0 = np.floor(row).astype(int)
        fc = column - c0
        fr = row - r0
        valid = (c0 >= 0) & (c0 < self.width - 1) & (r0 >= 0) & (r0 < self.height - 1)
        c0c = np.clip(c0, 0, self.width - 2)
        r0c = np.clip(r0, 0, self.height - 2)
        d = self._distance
        interpolated = (
            d[r0c, c0c] * (1 - fr) * (1 - fc)
            + d[r0c, c0c + 1] * (1 - fr) * fc
            + d[r0c + 1, c0c] * fr * (1 - fc)
            + d[r0c + 1, c0c + 1] * fr * fc
        )
        return np.where(valid, interpolated, 0.0)


class Obstacle:
    """Tracked obstacle state in map coordinates."""

    __slots__ = ('x', 'y', 'vx', 'vy', 'radius', 'age', 'missed', 'trace')

    def __init__(self, x, y, radius):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.radius = float(radius)
        self.age = 1
        self.missed = 0
        # Recent matched positions, for the bake stability test.
        self.trace = [(self.x, self.y)]

    def as_row(self):
        return (self.x, self.y, self.vx, self.vy, self.radius)


class ObstacleTracker:
    """Cluster non-map lidar returns and track them with an alpha-beta filter."""

    def __init__(self, wall_field: MapDistanceField,
                 detection_range: float = 8.0,
                 wall_filter_margin: float = 0.12,
                 wall_center_gate: float = 0.0,
                 path=None,
                 path_keep_distance: float = 0.30,
                 path_keep_wall_floor: float = 0.0,
                 cluster_gap: float = 0.25,
                 min_points: int = 3,
                 max_radius: float = 0.45,
                 min_radius: float = 0.08,
                 association_gate: float = 0.8,
                 position_alpha: float = 0.5,
                 velocity_beta: float = 0.25,
                 static_speed: float = 0.35,
                 max_missed: int = 8,
                 confirm_age: int = 2,
                 output_coast: int = 5,
                 max_track_speed: float = 8.0,
                 merge_gap: float = 0.05,
                 bake_enabled: bool = False,
                 bake_age: int = 20,
                 bake_position_std: float = 0.05,
                 bake_radius_margin: float = 0.05,
                 bake_absent_scans: int = 10):
        self.wall_field = wall_field
        self.detection_range = detection_range
        self.wall_filter_margin = wall_filter_margin
        self.wall_center_gate = wall_center_gate
        self.path_keep_distance = max(float(path_keep_distance), 0.0)
        self.path_keep_wall_floor = max(float(path_keep_wall_floor), 0.0)
        self.cluster_gap = cluster_gap
        self.min_points = min_points
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.association_gate = association_gate
        self.position_alpha = position_alpha
        self.velocity_beta = velocity_beta
        self.static_speed = static_speed
        self.max_missed = max_missed
        self.confirm_age = confirm_age
        self.output_coast = output_coast
        self.max_track_speed = max_track_speed
        self.merge_gap = float(merge_gap)
        self.bake_enabled = bool(bake_enabled)
        self.bake_age = max(int(bake_age), 1)
        self.bake_position_std = max(float(bake_position_std), 0.0)
        self.bake_radius_margin = max(float(bake_radius_margin), 0.0)
        self.bake_absent_scans = max(int(bake_absent_scans), 1)
        # Frozen discs live in their own registry, decoupled from track
        # lifetime: a track dies within ~0.2 s of the obstacle leaving the
        # lidar's view, but out-of-view is not evidence of removal.  Each
        # entry is [x, y, radius, consecutive-absent-scan count].
        self._baked = []  # type: List[list]
        # Incremented whenever the frozen-disc set changes, so the controller
        # knows when to rebuild the composite wall field.
        self.bake_generation = 0
        self.last_raw_count = 0
        self._tracks: List[Obstacle] = []
        self._last_time: Optional[float] = None
        self._path_tree = None
        self._path_corridor = None
        # LaserScan geometry is constant for a sensor.  Caching beam
        # trigonometry removes two 1080-element trig passes from every
        # control cycle; the key still handles a sensor/reconfiguration
        # changing its scan layout at runtime.
        self._scan_geometry_key = None
        self._scan_cos = np.empty(0, dtype=float)
        self._scan_sin = np.empty(0, dtype=float)
        self.set_path(path)

    _TRACE_LENGTH = 20

    def baked_discs(self) -> np.ndarray:
        """Frozen (x, y, radius) rows in the persistent static registry."""
        discs = [entry[:3] for entry in self._baked]
        return np.asarray(discs, dtype=float).reshape(-1, 3)

    def _promote_tracks(self) -> None:
        """Freeze stable static tracks into the world-fixed registry.

        Promotion requires a currently matched, velocity-snapped track whose
        recent matched positions stayed within ``bake_position_std`` (peak
        deviation catches jitter, endpoint displacement catches a slow crawl
        below the snap threshold).  A promoted track is removed: its returns
        are suppressed at cluster level from then on, so it would only decay
        into a duplicate.
        """
        survivors = []
        for track in self._tracks:
            if (track.age < self.bake_age or track.missed != 0
                    or track.vx != 0.0 or track.vy != 0.0
                    or len(track.trace) < min(self.bake_age,
                                              self._TRACE_LENGTH)):
                survivors.append(track)
                continue
            trace = np.asarray(track.trace, dtype=float)
            mean = trace.mean(axis=0)
            deviation = float(np.max(np.hypot(
                trace[:, 0] - mean[0], trace[:, 1] - mean[1])))
            drift = float(np.hypot(
                trace[-1, 0] - trace[0, 0], trace[-1, 1] - trace[0, 1]))
            if (deviation > self.bake_position_std
                    or drift > self.bake_position_std):
                survivors.append(track)
                continue
            if (self.wall_center_gate > 0.0
                    and float(self.wall_field.clearance(mean[0], mean[1]))
                    <= self.wall_center_gate
                    and not bool(self._path_keep(mean[0], mean[1]))):
                survivors.append(track)
                continue
            # The wall cost demands ego_disc_radius + wall_margin beyond the
            # stamped surface, slightly less than the obstacle cost's
            # ego_disc_radius + obstacle_margin; the radius margin keeps the
            # baked representation at least as conservative.
            self._baked.append([float(mean[0]), float(mean[1]),
                                float(track.radius + self.bake_radius_margin),
                                0])
            self.bake_generation += 1
        self._tracks = survivors

    def _monitor_baked(self, ranges: np.ndarray, angle_min: float,
                       angle_increment: float, laser_x: float, laser_y: float,
                       ego_yaw: float) -> None:
        """Drop a frozen disc only on confirmed in-view absence.

        Out-of-view (behind the ego, beyond detection range) is no evidence:
        the disc persists so a lap around the track cannot churn the wall
        field.  When the disc's angular span is visible and the raw ranges
        consistently pass BEYOND the disc, the object is gone (or moved - the
        mover shows up as a fresh track from its new returns) and the disc is
        removed after ``bake_absent_scans`` consecutive confirmations.
        """
        if not self._baked:
            return
        beam_count = ranges.shape[0]
        kept = []
        for entry in self._baked:
            dx = entry[0] - laser_x
            dy = entry[1] - laser_y
            distance = float(np.hypot(dx, dy))
            if distance >= self.detection_range or distance < 0.10:
                entry[3] = 0
                kept.append(entry)
                continue
            bearing = np.arctan2(np.sin(np.arctan2(dy, dx) - ego_yaw),
                                 np.cos(np.arctan2(dy, dx) - ego_yaw))
            centre = (float(bearing) - angle_min) / angle_increment
            # Look only through the disc's core so edge beams that graze the
            # surface cannot fake a see-through.
            half = np.arcsin(min(0.7 * entry[2] / distance, 1.0)) \
                / angle_increment
            low = int(np.floor(centre - half))
            high = int(np.ceil(centre + half)) + 1
            if low < 0 or high > beam_count:
                entry[3] = 0
                kept.append(entry)
                continue
            span = ranges[low:high]
            finite = span[np.isfinite(span) & (span > 0.05)]
            if (finite.size >= max(1, span.shape[0] // 2)
                    and float(finite.min()) > distance + entry[2] + 0.30):
                entry[3] += 1
            else:
                entry[3] = 0
            if entry[3] >= self.bake_absent_scans:
                self.bake_generation += 1
            else:
                kept.append(entry)
        self._baked = kept

    def set_path(self, path) -> None:
        """Protect the driven corridor from the map-wall rejection filter.

        A large wall filter is needed on the real car because localization
        error lets wall returns leak through the clean map.  Applying that
        filter blindly also erases a real cone or vehicle in a tight corner,
        where the raceline itself can be less than the filter margin from a
        wall.  Keep returns inside a narrow corridor around the driven path;
        mapped walls remain outside that corridor while an object blocking the
        vehicle necessarily intersects it.

        ``path`` may be a Raceline-like object exposing ``x`` and ``y`` or an
        ``(N, 2)`` array.  Keeping this module duck-typed avoids a dependency on
        the MPC implementation.
        """
        self._path_tree = None
        self._path_corridor = None
        if path is None or self.path_keep_distance <= 0.0:
            return
        if hasattr(path, 'x') and hasattr(path, 'y'):
            points = np.column_stack((path.x, path.y))
        else:
            points = np.asarray(path, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise ValueError('path must contain at least two (x, y) points')
        if not np.all(np.isfinite(points)):
            raise ValueError('path points must be finite')
        from scipy.spatial import cKDTree
        self._path_tree = cKDTree(points)
        # A tree query per scalar cluster dominated real scan processing.
        # Rasterise the fixed raceline corridor once onto the map grid so all
        # subsequent near-path tests are O(N) integer lookups.  Inflate by a
        # half cell diagonal: nearest-cell quantisation may create a few
        # conservative extra detections, but can never blind the tracker to a
        # blocker that the exact KD-tree test would have retained.
        columns = np.arange(self.wall_field.width, dtype=float)
        rows = np.arange(self.wall_field.height, dtype=float)
        cell_x = self.wall_field.origin_x + (
            columns + 0.5) * self.wall_field.resolution
        cell_y = self.wall_field.origin_y + (
            self.wall_field.height - rows - 0.5) * self.wall_field.resolution
        grid_x, grid_y = np.meshgrid(cell_x, cell_y)
        distance, _ = self._path_tree.query(
            np.column_stack((grid_x.ravel(), grid_y.ravel())), k=1)
        quantisation = self.wall_field.resolution / np.sqrt(2.0)
        self._path_corridor = (
            distance.reshape(self.wall_field.height, self.wall_field.width)
            <= self.path_keep_distance + quantisation)

    def _near_path(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self._path_corridor is None:
            return np.zeros(np.broadcast(x, y).shape, dtype=bool)
        bx, by = np.broadcast_arrays(x, y)
        column = np.rint(
            (bx - self.wall_field.origin_x) / self.wall_field.resolution - 0.5
        ).astype(int)
        row = np.rint(
            self.wall_field.height
            - (by - self.wall_field.origin_y) / self.wall_field.resolution
            - 0.5
        ).astype(int)
        valid = ((column >= 0) & (column < self.wall_field.width)
                 & (row >= 0) & (row < self.wall_field.height))
        result = np.zeros(bx.shape, dtype=bool)
        if np.any(valid):
            result[valid] = self._path_corridor[row[valid], column[valid]]
        return result

    def _path_keep(self, x, y, clearance=None) -> np.ndarray:
        """Return the safe near-path exception to the broad wall filter.

        The exception exists so a real obstacle on the driven corridor is not
        erased by the car's conservative wall margin.  It must never revive a
        return on the mapped wall itself: PF/map jitter made that old behavior
        generate persistent on-line phantom obstacles.  The small clearance
        floor separates wall-shell returns from an object protruding into the
        route.
        """
        if clearance is None:
            clearance = self.wall_field.clearance(x, y)
        clearance = np.asarray(clearance, dtype=float)
        return ((clearance >= self.path_keep_wall_floor)
                & self._near_path(x, y))

    def reset(self) -> None:
        if self._baked:
            self.bake_generation += 1
        self._baked = []
        self._tracks = []
        self._last_time = None
        self.last_raw_count = 0

    def update(self, timestamp: float, ranges: np.ndarray, angle_min: float,
               angle_increment: float, ego_x: float, ego_y: float, ego_yaw: float,
               laser_offset: float = 0.275) -> np.ndarray:
        """Ingest one scan; return confirmed obstacles as an (M, 5) array.

        Rows are (x, y, vx, vy, radius).  The ego pose must correspond to the
        scan time; the lidar sits ``laser_offset`` metres ahead of it.
        """
        laser_x = ego_x + laser_offset * np.cos(ego_yaw)
        laser_y = ego_y + laser_offset * np.sin(ego_yaw)
        points = self._scan_points(ranges, angle_min, angle_increment,
                                   ego_x, ego_y, ego_yaw, laser_offset)
        clusters = self._cluster(points, laser_x, laser_y)
        if self.wall_center_gate > 0.0 and clusters:
            # A cluster whose (depth-pushed) centre lies inside the wall band
            # is wall structure the map missed - a leaned barrier or a
            # bystander at the track edge.  The solver's wall costs already
            # keep the car out of that band; tracking it as an obstacle only
            # stalls the planner (bag mpc_20260724_165451: 79% of tracked
            # obstacles sat within 0.45 m of a mapped wall, costing ~2 s/lap).
            rows = np.asarray(clusters, dtype=float).reshape(-1, 3)
            clearance = self.wall_field.clearance(rows[:, 0], rows[:, 1])
            keep = clearance > self.wall_center_gate
            keep |= self._path_keep(rows[:, 0], rows[:, 1], clearance)
            clusters = [cluster for cluster, wanted in zip(clusters, keep)
                        if wanted]
        if self.bake_enabled and self._baked:
            # Returns from a baked object are already represented by its
            # frozen disc; clustering and tracking them again would only
            # rebuild a duplicate track that re-promotes every lap.  The
            # 0.30 m margin absorbs the viewpoint migration of the
            # depth-pushed centroid (rule obstacles sit >= 1 m apart, so a
            # genuinely new object cannot hide inside the margin).
            discs = self.baked_discs()
            if clusters:
                rows = np.asarray(clusters, dtype=float).reshape(-1, 3)
                clearance = np.min(
                    np.hypot(rows[:, None, 0] - discs[None, :, 0],
                             rows[:, None, 1] - discs[None, :, 1])
                    - discs[None, :, 2], axis=1)
                clusters = [cluster for cluster, value in zip(
                    clusters, clearance) if value > 0.30]
        dt = 0.0 if self._last_time is None else max(timestamp - self._last_time, 0.0)
        self._last_time = timestamp
        self._associate(clusters, dt)
        if self.bake_enabled:
            self._promote_tracks()
            self._monitor_baked(np.asarray(ranges, dtype=float), angle_min,
                                angle_increment, laser_x, laser_y, ego_yaw)
        # Keep briefly coasting tracks in the output: dropping an occluded or
        # momentarily unmatched opponent would make the MPC re-accelerate
        # straight into it.
        confirmed = []
        baked = [(entry[0], entry[1], 0.0, 0.0, entry[2])
                 for entry in self._baked]
        for track in self._tracks:
            if track.age < self.confirm_age or track.missed > self.output_coast:
                continue
            confirmed.append(track.as_row())
            speed = float(np.hypot(track.vx, track.vy))
            if speed > self.static_speed and track.age >= 2 * self.confirm_age:
                # A moving obstacle is a vehicle: cover its length with a
                # second disc ahead along the velocity direction (capsule).
                # Fresh tracks skip this - their velocity is not trustworthy
                # yet and a phantom capsule can distort the plan.
                unit_x = track.vx / speed
                unit_y = track.vy / speed
                confirmed.append((track.x + 0.30 * unit_x,
                                  track.y + 0.30 * unit_y,
                                  track.vx, track.vy, track.radius))
        raw = np.array(confirmed, dtype=float).reshape(-1, 5)
        self.last_raw_count = int(raw.shape[0]) + len(baked)
        consolidated = consolidate_static_obstacles(
            raw, merge_gap=self.merge_gap,
            static_speed=min(self.static_speed, 0.10))
        if not baked:
            return consolidated
        return np.vstack((
            np.asarray(baked, dtype=float).reshape(-1, 5), consolidated))

    def _scan_points(self, ranges, angle_min, angle_increment,
                     ego_x, ego_y, ego_yaw, laser_offset) -> np.ndarray:
        ranges = np.asarray(ranges, dtype=float)
        geometry_key = (ranges.size, float(angle_min), float(angle_increment))
        if geometry_key != self._scan_geometry_key:
            angles = angle_min + np.arange(ranges.size) * angle_increment
            self._scan_cos = np.cos(angles)
            self._scan_sin = np.sin(angles)
            self._scan_geometry_key = geometry_key
        valid = np.isfinite(ranges) & (ranges > 0.05) & (ranges < self.detection_range)
        if not np.any(valid):
            return np.empty((0, 2))
        r = ranges[valid]
        cosine = np.cos(ego_yaw)
        sine = np.sin(ego_yaw)
        local_x = laser_offset + r * self._scan_cos[valid]
        local_y = r * self._scan_sin[valid]
        px = ego_x + cosine * local_x - sine * local_y
        py = ego_y + sine * local_x + cosine * local_y
        clearance = self.wall_field.clearance(px, py)
        off_wall = clearance > self.wall_filter_margin
        # Do not let a conservative real-car wall margin blind the vehicle to
        # an object occupying its actual route through a tight corner.  The
        # floor prevents mapped-wall returns from using that exception.
        off_wall |= self._path_keep(px, py, clearance)
        return np.column_stack((px[off_wall], py[off_wall]))

    def _cluster(self, points: np.ndarray, laser_x: float,
                 laser_y: float) -> List[np.ndarray]:
        """Split consecutive scan points into clusters at range discontinuities."""
        clusters = []
        if points.shape[0] == 0:
            return clusters
        breaks = np.flatnonzero(
            np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1])) > self.cluster_gap)
        start = 0
        for stop in list(breaks + 1) + [points.shape[0]]:
            chunk = points[start:stop]
            start = stop
            if chunk.shape[0] < self.min_points:
                continue
            center = chunk.mean(axis=0)
            spread = np.hypot(chunk[:, 0] - center[0], chunk[:, 1] - center[1])
            radius = float(np.max(spread))
            if radius > self.max_radius:
                # Anything off the wall this large is a real object seen at an
                # angle (e.g. an opponent's rear plus side).  Splitting keeps
                # it in the output; dropping it would blind the controller at
                # close range, exactly when it matters most.
                pieces = int(np.ceil(radius / (0.5 * self.max_radius)))
                for piece in np.array_split(chunk, pieces):
                    if piece.shape[0] >= self.min_points:
                        clusters.append(self._make_disc(piece, laser_x, laser_y))
                continue
            clusters.append(self._make_disc(chunk, laser_x, laser_y))
        return clusters

    def _make_disc(self, chunk: np.ndarray, laser_x: float,
                   laser_y: float) -> np.ndarray:
        center = chunk.mean(axis=0)
        spread = np.hypot(chunk[:, 0] - center[0], chunk[:, 1] - center[1])
        radius = float(np.max(spread))
        # The lidar sees only the near face of an object, so the centroid of
        # the returns sits on its surface.  Push the centre deeper along the
        # viewing ray by roughly the visible half-extent; keep the radius tied
        # to the visible width so narrow gaps stay passable.
        depth = float(np.clip(max(radius, self.min_radius), 0.10, 0.25))
        view = np.array([center[0] - laser_x, center[1] - laser_y])
        norm = float(np.hypot(view[0], view[1]))
        if norm > 1e-6:
            center = center + view / norm * depth
        # Cap the disc radius near the vehicle scale: oversized discs from
        # angled views make every passing lane look blocked, and larger
        # objects are already covered by cluster splitting.
        return np.array([center[0], center[1],
                         min(max(radius, self.min_radius) + 0.06, 0.30)])

    def _associate(self, clusters: List[np.ndarray], dt: float) -> None:
        cluster_rows = np.asarray(clusters, dtype=float).reshape(-1, 3)
        matched = np.zeros(cluster_rows.shape[0], dtype=bool)
        for track in self._tracks:
            predicted_x = track.x + track.vx * dt
            predicted_y = track.y + track.vy * dt
            best = None
            if cluster_rows.shape[0]:
                distance = np.hypot(cluster_rows[:, 0] - predicted_x,
                                    cluster_rows[:, 1] - predicted_y)
                distance[matched] = np.inf
                index = int(np.argmin(distance))
                if distance[index] < self.association_gate:
                    best = index
            if best is None:
                track.missed += 1
                track.x += track.vx * dt
                track.y += track.vy * dt
                continue
            matched[best] = True
            cluster = cluster_rows[best]
            residual_x = cluster[0] - predicted_x
            residual_y = cluster[1] - predicted_y
            track.x = predicted_x + self.position_alpha * residual_x
            track.y = predicted_y + self.position_alpha * residual_y
            if dt > 1e-4:
                track.vx += self.velocity_beta * residual_x / dt
                track.vy += self.velocity_beta * residual_y / dt
            speed = float(np.hypot(track.vx, track.vy))
            if speed < self.static_speed:
                track.vx = 0.0
                track.vy = 0.0
            elif speed > self.max_track_speed:
                scale = self.max_track_speed / speed
                track.vx *= scale
                track.vy *= scale
            track.radius = float(np.clip(
                0.6 * track.radius + 0.4 * float(cluster[2]),
                self.min_radius, 0.30))
            track.age += 1
            track.missed = 0
            track.trace.append((track.x, track.y))
            if len(track.trace) > self._TRACE_LENGTH:
                track.trace.pop(0)
        for index, cluster in enumerate(cluster_rows):
            if not matched[index]:
                self._tracks.append(Obstacle(cluster[0], cluster[1], cluster[2]))
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
