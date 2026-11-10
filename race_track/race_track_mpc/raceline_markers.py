"""RViz interactive-marker adapter for live raceline speed and path editing."""

import math
from pathlib import Path
from typing import Callable

import numpy as np
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from visualization_msgs.msg import (
    InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback,
    Marker)

from .mpc_core import MPCConfig, Raceline
from .raceline_editor import RacelineEditorModel


class RacelineMarkerEditor:
    """Expose raceline edits through RViz interactive markers.

    Each handle carries a vertical control for speed and, when a
    speed_profile is supplied, a planar control that moves the line
    itself. Vertical drags set speed; planar drags move the path and
    re-derive arc length, heading, curvature, speed and acceleration.
    """

    def __init__(self, node, raceline: Raceline, config: MPCConfig,
                 on_raceline: Callable[[Raceline], None], frame_id: str,
                 control_stride: int = 8, influence_points: int = 12,
                 marker_scale: float = 0.16, save_path: str = '',
                 speed_height_scale: float = 0.10,
                 speed_step: float = 0.5,
                 speed_profile: Callable = None,
                 clearance: Callable = None,
                 minimum_clearance: float = 0.0):
        if control_stride < 1:
            raise ValueError('raceline editor control_stride must be positive')
        if marker_scale <= 0.0:
            raise ValueError('raceline editor marker_scale must be positive')
        if speed_height_scale <= 0.0:
            raise ValueError(
                'raceline editor speed_height_scale must be positive')
        if speed_step <= 0.0:
            raise ValueError('raceline editor speed_step must be positive')

        self._node = node
        self._on_raceline = on_raceline
        self._frame_id = str(frame_id)
        self._control_stride = int(control_stride)
        self._marker_scale = float(marker_scale)
        self._save_path = str(Path(save_path).expanduser()) if save_path else ''
        self._speed_height_scale = float(speed_height_scale)
        self._speed_step = float(speed_step)
        self._closed = False

        # Keep the editor and the actuator-facing controller on the same
        # ceiling. Curvature/braking feasibility remains a runtime MPC layer;
        # it must not silently rewrite the authored CSV speed profile.
        #
        # Passing speed_profile is what turns map X/Y editing on: a moved
        # station changes curvature, and a curvature the old speed column
        # cannot honour is exactly how a hand edit drives into a wall.
        self._model = RacelineEditorModel(
            raceline,
            influence_points=influence_points,
            minimum_speed=0.0,
            maximum_speed=config.max_speed,
            speed_profile=speed_profile,
            clearance=clearance,
            minimum_clearance=minimum_clearance,
        )
        self._indices = list(range(0, self._model.point_count,
                                   self._control_stride))
        self._server = InteractiveMarkerServer(
            node, '/mpc/raceline_editor')
        self._menu = MenuHandler()
        self._menu.insert('Undo last speed edit', callback=self._undo_feedback)
        self._menu.insert('Reset loaded speed profile',
                          callback=self._reset_feedback)
        self._menu.insert('Save speed profile CSV', callback=self._save_feedback)
        speed_menu = self._menu.insert('Speed')
        self._menu.insert(
            'Faster (+{:.2f} m/s)'.format(self._speed_step), parent=speed_menu,
            callback=self._speed_up_feedback)
        self._menu.insert(
            'Slower (-{:.2f} m/s)'.format(self._speed_step), parent=speed_menu,
            callback=self._speed_down_feedback)
        self._menu.insert('Restore loaded speed', parent=speed_menu,
                          callback=self._speed_reset_feedback)
        if self._model.geometry_enabled:
            geometry_menu = self._menu.insert('Geometry')
            self._menu.insert('Restore loaded geometry here',
                              parent=geometry_menu,
                              callback=self._geometry_reset_feedback)

        for index in self._indices:
            self._insert_marker(index)
        self._server.applyChanges()

        self._undo_service = node.create_service(
            Trigger, '/mpc/raceline_editor/undo', self._undo_service_callback)
        self._reset_service = node.create_service(
            Trigger, '/mpc/raceline_editor/reset', self._reset_service_callback)
        self._save_service = node.create_service(
            Trigger, '/mpc/raceline_editor/save', self._save_service_callback)
        self._speed_subscription = node.create_subscription(
            Float32MultiArray, '/mpc/raceline_editor/set_speed',
            self._set_speed_message, 1)

    @property
    def handle_count(self) -> int:
        return len(self._indices)

    @property
    def save_path(self) -> str:
        return self._save_path

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()

    def _speed_height(self, speed: float) -> float:
        """Marker height above the floor that stands for a reference speed."""
        return 0.04 + float(speed) * self._speed_height_scale

    def _height_speed(self, height: float) -> float:
        return max((float(height) - 0.04) / self._speed_height_scale, 0.0)

    def _speed_color(self, speed: float):
        """Blue (slowest) to red (fastest) across the current profile."""
        speeds = self._model.raceline.speed
        low = float(np.min(speeds))
        high = float(np.max(speeds))
        if high - low < 1e-6:
            fraction = 0.5
        else:
            fraction = (float(speed) - low) / (high - low)
        fraction = min(max(fraction, 0.0), 1.0)
        return fraction, 0.35 * (1.0 - abs(2.0 * fraction - 1.0)), 1.0 - fraction

    def _make_marker(self, index: int) -> InteractiveMarker:
        x, y = self._model.point(index)
        speed = self._model.speed(index)
        marker = InteractiveMarker()
        marker.header.frame_id = self._frame_id
        marker.name = self._marker_name(index)
        marker.description = '{} {} - {:.2f} m/s'.format(
            'Point' if self._model.geometry_enabled else 'Speed',
            index, speed)
        marker.scale = 3.0 * self._marker_scale
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = self._speed_height(speed)
        marker.pose.orientation.w = 1.0

        visual = Marker()
        visual.type = Marker.SPHERE
        visual.scale.x = self._marker_scale
        visual.scale.y = self._marker_scale
        visual.scale.z = self._marker_scale
        red, green, blue = self._speed_color(speed)
        visual.color.r = red
        visual.color.g = green
        visual.color.b = blue
        visual.color.a = 0.95

        # Local X is rotated onto map Z, so dragging this control changes only
        # the speed the handle stands for.
        speed_control = InteractiveMarkerControl()
        speed_control.name = 'set_speed_vertical_only'
        speed_control.orientation.w = math.sqrt(0.5)
        speed_control.orientation.y = math.sqrt(0.5)
        speed_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        speed_control.always_visible = True
        speed_control.markers.append(visual)
        marker.controls.append(speed_control)

        if self._model.geometry_enabled:
            # Same axis, but MOVE_PLANE moves in the plane PERPENDICULAR to
            # it: map X/Y. Without a speed_profile this control is absent and
            # the marker stays vertical-only, exactly as it always was.
            move_control = InteractiveMarkerControl()
            move_control.name = 'set_position_map_plane'
            move_control.orientation.w = math.sqrt(0.5)
            move_control.orientation.y = math.sqrt(0.5)
            move_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
            move_control.always_visible = False
            marker.controls.append(move_control)
        return marker

    def _insert_marker(self, index: int) -> None:
        marker = self._make_marker(index)
        self._server.insert(marker, feedback_callback=self._feedback)
        self._menu.apply(self._server, marker.name)

    @staticmethod
    def _marker_name(index: int) -> str:
        return 'speed_{:04d}'.format(index)

    @staticmethod
    def _marker_index(name: str) -> int:
        return int(name.rsplit('_', 1)[1])

    def _feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        if feedback.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return
        try:
            index = self._marker_index(feedback.marker_name)
            x, y = self._model.point(index)
            moved = math.hypot(feedback.pose.position.x - x,
                               feedback.pose.position.y - y)
            current_height = self._speed_height(self._model.speed(index))
            raised = abs(feedback.pose.position.z - current_height)
            if moved >= 1e-3 and self._model.geometry_enabled:
                # A planar drag re-derives speed from the new curvature, so
                # whatever height the handle was left at is not a speed
                # request - it is discarded and the marker resyncs.
                self._apply(self._model.set_position(
                    index, feedback.pose.position.x, feedback.pose.position.y))
                self._node.get_logger().info(
                    'Raceline point {} moved {:.3f} m, speed re-derived to '
                    '{:.2f} m/s'.format(index, moved, self._model.speed(index)))
                return
            if raised < 1e-3:
                # Re-inserting also pins X/Y if a third-party client sent a
                # malformed pose, or if geometry editing is switched off.
                self._sync_markers()
                return
            requested = self._height_speed(feedback.pose.position.z)
            self._apply(self._model.set_speed(index, requested))
            self._node.get_logger().info(
                'Raceline speed {} requested {:.2f} m/s, applied {:.2f} m/s'.format(
                    index, requested, self._model.speed(index)))
        except (IndexError, ValueError) as error:
            self._node.get_logger().warn(
                'Raceline edit rejected: {}'.format(error))
            self._sync_markers()

    def _apply(self, raceline: Raceline) -> None:
        self._on_raceline(raceline)
        self._sync_markers()

    def _sync_markers(self) -> None:
        # Re-insert because height, colour, and label all follow the profile.
        # This runs on mouse-up or a menu/topic action, never in the MPC loop.
        for index in self._indices:
            self._insert_marker(index)
        self._server.applyChanges()

    def _undo(self):
        self._apply(self._model.undo())
        return 'restored previous speed edit'

    def _reset(self):
        self._apply(self._model.reset())
        return 'restored speed profile loaded at startup'

    def _save(self):
        if not self._save_path:
            raise ValueError('raceline_editor_save_path is empty')
        destination = self._model.save(self._save_path)
        return 'saved speed profile to {}'.format(destination)

    def _run_action(self, action) -> None:
        try:
            message = action()
            self._node.get_logger().info('Raceline editor: {}'.format(message))
        except (OSError, ValueError, IndexError) as error:
            self._node.get_logger().warn(
                'Raceline editor action failed: {}'.format(error))
            self._sync_markers()

    def _step_speed(self, index: int, step: float) -> str:
        requested = max(self._model.requested_speed(index) + step, 0.0)
        self._apply(self._model.set_speed(index, requested))
        return 'speed {} requested {:.2f} m/s, applied {:.2f} m/s'.format(
            index, requested, self._model.speed(index))

    def _reset_geometry(self, index: int) -> str:
        self._apply(self._model.reset_position(index))
        x, y = self._model.point(index)
        return 'geometry {} restored to {:.3f}, {:.3f}'.format(index, x, y)

    def _geometry_reset_feedback(self, feedback) -> None:
        self._run_action(
            lambda: self._reset_geometry(self._feedback_index(feedback)))

    def _reset_speed(self, index: int) -> str:
        self._apply(self._model.reset_speed(index))
        return 'speed {} restored to {:.2f} m/s'.format(
            index, self._model.speed(index))

    def _set_speed(self, index: int, speed: float) -> str:
        self._apply(self._model.set_speed(index, speed))
        return 'speed {} requested {:.2f} m/s, applied {:.2f} m/s'.format(
            index, speed, self._model.speed(index))

    def _feedback_index(self, feedback) -> int:
        return self._marker_index(feedback.marker_name)

    def _speed_up_feedback(self, feedback) -> None:
        self._run_action(
            lambda: self._step_speed(self._feedback_index(feedback),
                                     self._speed_step))

    def _speed_down_feedback(self, feedback) -> None:
        self._run_action(
            lambda: self._step_speed(self._feedback_index(feedback),
                                     -self._speed_step))

    def _speed_reset_feedback(self, feedback) -> None:
        self._run_action(
            lambda: self._reset_speed(self._feedback_index(feedback)))

    def _set_speed_message(self, message) -> None:
        values = list(message.data)
        if len(values) != 2:
            self._node.get_logger().warn(
                'raceline_editor/set_speed expects [index, speed]')
            return
        self._run_action(
            lambda: self._set_speed(int(values[0]), float(values[1])))

    def _undo_feedback(self, _) -> None:
        self._run_action(self._undo)

    def _reset_feedback(self, _) -> None:
        self._run_action(self._reset)

    def _save_feedback(self, _) -> None:
        self._run_action(self._save)

    @staticmethod
    def _service_result(response, action):
        try:
            response.message = action()
            response.success = True
        except (OSError, ValueError) as error:
            response.message = str(error)
            response.success = False
        return response

    def _undo_service_callback(self, _, response):
        return self._service_result(response, self._undo)

    def _reset_service_callback(self, _, response):
        return self._service_result(response, self._reset)

    def _save_service_callback(self, _, response):
        return self._service_result(response, self._save)
