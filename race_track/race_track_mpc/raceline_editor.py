"""ROS-independent speed and geometry editor for a closed raceline."""

import os
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .mpc_core import Raceline


def closed_geometry(x, y):
    """Heading, curvature, arc length and segment lengths of a closed line.

    The first difference is halved so it spans the same step as the second.
    Mismatch the two and curvature comes out a factor of four small, which
    reads as a line that can be driven far faster than it can.
    """
    dx = (np.roll(x, -1) - np.roll(x, 1)) / 2.0
    dy = (np.roll(y, -1) - np.roll(y, 1)) / 2.0
    ddx = np.roll(x, -1) - 2.0 * x + np.roll(x, 1)
    ddy = np.roll(y, -1) - 2.0 * y + np.roll(y, 1)
    curvature = ((dx * ddy - dy * ddx)
                 / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-12))
    segment = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    arc = np.concatenate([[0.0], np.cumsum(segment)[:-1]])
    return np.arctan2(dy, dx), curvature, arc, segment


class RacelineEditorModel:
    """Edit raceline speeds, and optionally the path geometry itself.

    A speed edit changes the authored speed over a cosine-tapered
    neighbourhood and leaves the five geometry columns byte-identical.

    A geometry edit moves x/y over that same taper - never a single station,
    because at the shipped 0.10 m spacing one lone 8 cm step is a 16 1/m
    curvature spike against a line whose real peak is 0.76 - and then rebuilds
    every derived column from the new positions: arc length, heading,
    curvature, speed and acceleration.  Geometry editing stays disabled unless
    a `speed_profile` is supplied, so a caller that only wants the original
    speed-only behaviour gets exactly that.
    """

    def __init__(self, raceline: Raceline, influence_points: int = 12,
                 minimum_speed: float = 0.0,
                 maximum_speed: float = float('inf'),
                 speed_profile: Optional[Callable] = None,
                 clearance: Optional[Callable] = None,
                 minimum_clearance: float = 0.0):
        if influence_points < 2:
            raise ValueError('influence_points must be at least 2')
        if not np.isfinite(minimum_speed) or minimum_speed < 0.0:
            raise ValueError('minimum_speed must be finite and non-negative')
        if np.isnan(maximum_speed) or maximum_speed <= minimum_speed:
            raise ValueError('maximum_speed must be greater than minimum_speed')
        self.influence_points = int(influence_points)
        self.minimum_speed = float(minimum_speed)
        self.maximum_speed = float(maximum_speed)
        self._speed_profile = speed_profile
        self._clearance = clearance
        self.minimum_clearance = float(minimum_clearance)
        self._original = self._from_raceline(raceline)
        self._data = self._original.copy()
        self._original_speed_source = self._original[:, 5].copy()
        self._speed_source = self._original_speed_source.copy()
        self._segment_length = np.hypot(
            np.roll(self._original[:, 1], -1) - self._original[:, 1],
            np.roll(self._original[:, 2], -1) - self._original[:, 2],
        )
        if np.any(self._segment_length < 1e-4):
            raise ValueError('raceline contains a zero-length segment')
        self._history = []  # type: List[tuple]

    @property
    def geometry_enabled(self) -> bool:
        """Geometry edits need a profile function to re-derive speed."""
        return self._speed_profile is not None

    @staticmethod
    def _from_raceline(raceline: Raceline) -> np.ndarray:
        # Store map-frame headings so saved files retain the exact geometry
        # consumed by the running controller.
        return np.column_stack((
            raceline.s,
            raceline.x,
            raceline.y,
            raceline.heading,
            raceline.curvature,
            raceline.speed,
            raceline.acceleration,
        ))

    @property
    def point_count(self) -> int:
        return int(self._data.shape[0])

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    @property
    def raceline(self) -> Raceline:
        return Raceline(self._data)

    def point(self, index: int):
        """Fixed map position used to anchor a speed handle in RViz."""
        self._check_index(index)
        return float(self._data[index, 1]), float(self._data[index, 2])

    def speed(self, index: int) -> float:
        """Authored reference speed currently in force at one point."""
        self._check_index(index)
        return float(self._data[index, 5])

    def requested_speed(self, index: int) -> float:
        """Current blended speed request before acceleration is derived."""
        self._check_index(index)
        return float(self._speed_source[index])

    def _blend_weights(self, index: int) -> np.ndarray:
        """Cosine taper centred on one point, wrapping around the loop."""
        count = self.point_count
        offsets = np.arange(count) - int(index)
        offsets = (offsets + count // 2) % count - count // 2
        distance = np.abs(offsets).astype(float)
        weights = np.zeros(count, dtype=float)
        affected = distance < self.influence_points
        weights[affected] = 0.5 * (
            1.0 + np.cos(np.pi * distance[affected] / self.influence_points))
        return weights

    def set_speed(self, index: int, speed: float) -> Raceline:
        """Retarget speed around one point while leaving geometry untouched."""
        self._check_index(index)
        if not np.isfinite(speed):
            raise ValueError('raceline speed must be finite')
        if speed < 0.0:
            raise ValueError('raceline speed cannot be negative')

        # Clamp the request before blending. This keeps the value shown by
        # the editor identical to the value saved and handed to the MPC.
        target = float(np.clip(
            speed, self.minimum_speed, self.maximum_speed))
        delta = target - self._speed_source[index]
        if abs(delta) < 1e-9:
            return self.raceline
        source = self._speed_source + delta * self._blend_weights(index)
        source = np.clip(source, self.minimum_speed, self.maximum_speed)
        return self._commit(source)

    def scale_speed(self, index: int, factor: float) -> Raceline:
        """Speed edit expressed as a multiple of the current request."""
        self._check_index(index)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError('raceline speed factor must be positive')
        return self.set_speed(index, self._speed_source[index] * float(factor))

    def reset_speed(self, index: int) -> Raceline:
        """Restore the loaded speed at one control point and its blend."""
        self._check_index(index)
        return self.set_speed(index, self._original_speed_source[index])

    def set_position(self, index: int, x: float, y: float) -> Raceline:
        """Move one station in the map plane and re-derive everything.

        The requested displacement is applied through the same cosine taper a
        speed edit uses, so the handle drags its neighbours with it.
        """
        self._check_index(index)
        if not self.geometry_enabled:
            raise ValueError('raceline geometry editing is disabled')
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError('raceline position must be finite')
        shift_x = float(x) - self._data[index, 1]
        shift_y = float(y) - self._data[index, 2]
        if np.hypot(shift_x, shift_y) < 1e-6:
            return self.raceline
        weights = self._blend_weights(index)
        return self._commit_geometry(self._data[:, 1] + weights * shift_x,
                                     self._data[:, 2] + weights * shift_y,
                                     weights > 0.0)

    def reset_position(self, index: int) -> Raceline:
        """Pull one station's neighbourhood back onto the loaded geometry."""
        self._check_index(index)
        if not self.geometry_enabled:
            raise ValueError('raceline geometry editing is disabled')
        weights = self._blend_weights(index)
        moved = weights > 0.0
        if not np.any(np.abs(self._data[moved, 1:3]
                             - self._original[moved, 1:3]) > 1e-9):
            return self.raceline
        x = self._data[:, 1] + weights * (self._original[:, 1] - self._data[:, 1])
        y = self._data[:, 2] + weights * (self._original[:, 2] - self._data[:, 2])
        return self._commit_geometry(x, y, moved)

    def _commit_geometry(self, x, y, touched) -> Raceline:
        """Rebuild all seven columns from new positions, guarding clearance."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        heading, curvature, arc, segment = closed_geometry(x, y)
        if np.any(segment < 1e-4):
            raise ValueError('raceline edit collapsed a segment to zero length')
        if self._clearance is not None:
            # Only the tapered window moved, so only it can newly touch a
            # wall.  Checking the whole lap costs 50 ms against 4.5 ms here,
            # and this runs on the executor thread the controller shares.
            window = np.asarray(touched, dtype=bool)
            gap = np.asarray(self._clearance(x[window], y[window],
                                            heading[window]), dtype=float)
            if gap.size and float(gap.min()) < self.minimum_clearance:
                raise ValueError(
                    'edit leaves {:.3f} m of body clearance, under the '
                    '{:.2f} m floor'.format(float(gap.min()),
                                            self.minimum_clearance))
        count = self.point_count
        # The profile function is a LIMITER, not a generator: Raceline's
        # _safe_speed_array starts from the authored vx column and only ever
        # cuts it down, flooring at minimum_reference_speed.  Seeding the
        # column with zeros therefore collapsed the whole lap to that floor -
        # one 5 cm handle drag turned a 2.4-8.5 m/s profile into a flat
        # 1.6 m/s.  Seed the moved window at the ceiling instead, exactly as
        # tools/rebuild_raceline.py does, so the new curvature is what decides
        # its speed and a corner that got wider is allowed to get faster.
        # Untouched stations keep their authored speed, so a hand-dialled
        # corner elsewhere survives a geometry drag; the braking/acceleration
        # passes inside the profile still run over the whole loop.
        ceiling = (self.maximum_speed if np.isfinite(self.maximum_speed)
                   else float(np.max(self._original[:, 5])) * 4.0 + 1.0)
        seed = self._speed_source.copy()
        seed[np.asarray(touched, dtype=bool)] = ceiling
        rebuilt = np.column_stack((
            arc, x, y, heading, curvature, seed, np.zeros(count)))
        speed = np.asarray(self._speed_profile(rebuilt), dtype=float)
        if speed.shape != (count,) or not np.all(np.isfinite(speed)):
            raise ValueError('raceline speed profile is invalid')
        rebuilt[:, 5] = np.clip(speed, self.minimum_speed, self.maximum_speed)
        speed_squared = rebuilt[:, 5] ** 2
        rebuilt[:, 6] = 0.25 * (
            (speed_squared - np.roll(speed_squared, 1)) / np.roll(segment, 1)
            + (np.roll(speed_squared, -1) - speed_squared) / segment)
        if not np.all(np.isfinite(rebuilt)):
            raise ValueError('raceline geometry edit produced non-finite values')
        self._history.append((self._data.copy(), self._speed_source.copy()))
        self._data = rebuilt
        self._speed_source = rebuilt[:, 5].copy()
        self._segment_length = segment
        return Raceline(rebuilt)

    def _refresh_segments(self) -> None:
        """Segment lengths follow whatever geometry _data currently holds."""
        self._segment_length = np.hypot(
            np.roll(self._data[:, 1], -1) - self._data[:, 1],
            np.roll(self._data[:, 2], -1) - self._data[:, 2])

    def _commit(self, speed_source: np.ndarray) -> Raceline:
        """Apply and record one speed edit as a single undo step."""
        rebuilt = self._with_speed(speed_source)
        self._history.append((self._data.copy(), self._speed_source.copy()))
        self._data = rebuilt
        self._speed_source = np.asarray(speed_source, dtype=float).copy()
        return Raceline(rebuilt)

    def undo(self) -> Raceline:
        if not self._history:
            raise ValueError('there is no raceline edit to undo')
        data, speed_source = self._history.pop()
        self._data = data
        self._speed_source = speed_source
        # A geometry edit may be what is being undone, so the cached segment
        # lengths the acceleration column is built from have to follow.
        self._refresh_segments()
        return Raceline(data)

    def reset(self) -> Raceline:
        if not np.array_equal(self._data, self._original):
            self._history.append((self._data.copy(), self._speed_source.copy()))
        self._data = self._original.copy()
        self._speed_source = self._original_speed_source.copy()
        self._refresh_segments()
        return Raceline(self._data)

    def save(self, path) -> Path:
        """Atomically save the edited line with an explicit closing row."""
        destination = Path(path).expanduser()
        if not destination.name:
            raise ValueError('raceline save path is empty')
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                'raceline save directory does not exist: {}'.format(
                    destination.parent))

        raceline = self.raceline
        rows = np.vstack((self._data, self._data[0]))
        rows[-1, 0] = raceline.total_length
        temporary = None  # type: Optional[str]
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', prefix='.' + destination.name + '.',
                    suffix='.tmp', dir=str(destination.parent),
                    delete=False) as output:
                temporary = output.name
                np.savetxt(
                    output,
                    rows,
                    delimiter='; ',
                    fmt='%.9f',
                    header=(
                        'race_track_mpc raceline editor\n'
                        's_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2'
                    ),
                )
            os.replace(temporary, str(destination))
        except Exception:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            raise
        return destination

    def _check_index(self, index: int) -> None:
        if index < 0 or index >= self.point_count:
            raise IndexError('raceline point index is out of range')

    def _with_speed(self, speed_source: np.ndarray) -> np.ndarray:
        """Replace only vx/ax; the five geometry columns stay byte-identical."""
        source = np.asarray(speed_source, dtype=float)
        if source.shape != self._data[:, 5].shape:
            raise ValueError('raceline speed profile has the wrong shape')
        if not np.all(np.isfinite(source)) or np.any(source < 0.0):
            raise ValueError('raceline speed profile is invalid')

        rebuilt = self._data.copy()
        rebuilt[:, 5] = np.clip(
            source, self.minimum_speed, self.maximum_speed)
        speed_squared = rebuilt[:, 5] ** 2
        previous_dv2 = speed_squared - np.roll(speed_squared, 1)
        next_dv2 = np.roll(speed_squared, -1) - speed_squared
        rebuilt[:, 6] = 0.25 * (
            previous_dv2 / np.roll(self._segment_length, 1)
            + next_dv2 / self._segment_length)
        if not np.all(np.isfinite(rebuilt[:, 5:7])):
            raise ValueError('raceline speed edit produced non-finite values')
        return rebuilt
