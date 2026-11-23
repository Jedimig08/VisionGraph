"""Estimate the carriage pose from calibrated marker ID 4 detections.

This is the vision side of the polargraph pipeline.  It consumes the existing
:class:`marker_tracking.MarkerTracker` result, applies the calibration
homography to marker 4's live corners, and reports the carriage centre and
horizontal inclination in drawing-area millimetres and degrees.

Calibration and fresh detections are preferred.  A gated continuous track may
substitute a predicted position only for a short, flagged grace period, so a
stale position can never silently drive a movement.  This module does not
command motors.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time

import numpy as np

import subpixel_refiner

DEFAULT_MARKER_ID = 4
DEFAULT_ANGLE_PERIOD_DEGREES = 180.0


def wrap_angle(degrees, period=360.0):
    """Wrap an angle into ``[-period / 2, period / 2)``.

    A period of 180 degrees is useful for a square marker because a horizontal
    edge stays horizontal after a half turn.
    """
    period = float(period)
    if not math.isfinite(period) or period <= 0:
        raise ValueError("Angle period must be a finite positive number")
    value = float(degrees)
    if not math.isfinite(value):
        raise ValueError("Angle must be finite")
    return (value + period / 2.0) % period - period / 2.0


def angle_error_degrees(measured, target=0.0, period=DEFAULT_ANGLE_PERIOD_DEGREES):
    """Shortest signed rotation from ``target`` to ``measured``."""
    return wrap_angle(float(measured) - float(target), period)


def marker_axis_angle_degrees(corners, period=DEFAULT_ANGLE_PERIOD_DEGREES):
    """Return the inclination of the marker's first edge in degrees.

    ArUco returns corners in a stable order, so corner 0 -> corner 1 is the
    marker's top edge.  Opposite edges are averaged to reduce corner noise.
    """
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Marker corners must be four finite 2D points")
    axis = (points[1] - points[0]) + (points[2] - points[3])
    if float(np.linalg.norm(axis)) < 1e-9:
        raise ValueError("Marker corners are degenerate")
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    return wrap_angle(angle, period)


def polygon_area(points):
    """Shoelace area of a 2D polygon without depending on OpenCV."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3 or not np.isfinite(points).all():
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


class CircularAngleSmoother:
    """Circular moving average that handles wrap-around correctly."""

    def __init__(self, window=5, period_degrees=DEFAULT_ANGLE_PERIOD_DEGREES):
        if int(window) < 1:
            raise ValueError("Smoothing window must be at least one sample")
        if not math.isfinite(float(period_degrees)) or period_degrees <= 0:
            raise ValueError("Angle period must be a finite positive number")
        self.window = int(window)
        self.period_degrees = float(period_degrees)
        self._samples = deque(maxlen=self.window)

    def reset(self):
        self._samples.clear()

    def add(self, angle_degrees):
        """Add a sample and return the smoothed angle, or ``None`` if ambiguous."""
        self._samples.append(wrap_angle(angle_degrees, self.period_degrees))
        return self.value

    @property
    def value(self):
        if not self._samples:
            return None
        phase = (
            2.0
            * math.pi
            * np.asarray(self._samples, dtype=np.float64)
            / self.period_degrees
        )
        mean_vector = np.mean(np.exp(1j * phase))
        if abs(mean_vector) < 1e-12:
            return None
        angle = (
            float(np.angle(mean_vector))
            * self.period_degrees
            / (2.0 * math.pi)
        )
        return wrap_angle(angle, self.period_degrees)


@dataclass(frozen=True)
class PoseEstimatorConfig:
    """Physical and filtering settings for carriage pose estimation."""

    drawing_width_mm: float
    drawing_height_mm: float
    output_size: tuple
    marker_id: int = DEFAULT_MARKER_ID
    min_reliability: float = 0.5
    min_marker_area_px: float = 100.0
    angle_smoothing_window: int = 5
    angle_period_degrees: float = DEFAULT_ANGLE_PERIOD_DEGREES
    smoothing_reset_seconds: float = 0.5
    allow_predicted_pose: bool = True
    max_prediction_age_seconds: float = 0.3
    subpixel: subpixel_refiner.SubpixelRefinerConfig = field(
        default_factory=subpixel_refiner.SubpixelRefinerConfig
    )

    def __post_init__(self):
        if (
            not math.isfinite(self.drawing_width_mm)
            or not math.isfinite(self.drawing_height_mm)
            or self.drawing_width_mm <= 0
            or self.drawing_height_mm <= 0
        ):
            raise ValueError("Drawing-area dimensions must be finite and positive")
        width, height = (int(value) for value in self.output_size)
        if width < 2 or height < 2:
            raise ValueError("Corrected output size must be at least 2x2 pixels")
        if not 0.0 <= float(self.min_reliability) <= 1.0:
            raise ValueError("Minimum reliability must be between 0 and 1")
        if float(self.min_marker_area_px) < 0:
            raise ValueError("Minimum marker area cannot be negative")
        if int(self.angle_smoothing_window) < 1:
            raise ValueError("Smoothing window must be at least one sample")
        if (
            not math.isfinite(float(self.angle_period_degrees))
            or self.angle_period_degrees <= 0
        ):
            raise ValueError("Angle period must be a finite positive number")
        if float(self.smoothing_reset_seconds) < 0:
            raise ValueError("Smoothing reset interval cannot be negative")
        if (
            not math.isfinite(float(self.max_prediction_age_seconds))
            or self.max_prediction_age_seconds < 0
        ):
            raise ValueError("Maximum prediction age cannot be negative")
        if not isinstance(self.subpixel, subpixel_refiner.SubpixelRefinerConfig):
            raise TypeError("subpixel must be a SubpixelRefinerConfig")


@dataclass(frozen=True)
class PoseEstimate:
    """A carriage pose in the corrected drawing frame, or a rejection reason."""

    valid: bool
    reason: str = ""
    marker_id: int = DEFAULT_MARKER_ID
    x_px: float | None = None
    y_px: float | None = None
    x_mm: float | None = None
    y_mm: float | None = None
    angle_deg: float | None = None
    raw_angle_deg: float | None = None
    reliability: float = 0.0
    inside_drawing_area: bool = False
    refined: bool = False
    fit_error_px: float | None = None
    refinement_reason: str = ""
    predicted: bool = False
    prediction_age_seconds: float = 0.0
    corners_px: np.ndarray | None = field(default=None, repr=False)


class PoseEstimator:
    """Turn ``MarkerTracker`` results into a smoothed carriage pose.

    Coordinates use the corrected image orientation: origin at the top-left
    of the drawing area, X to the right and Y downwards.  A later planner
    stage can map these to G-code coordinates without changing this module.
    """

    def __init__(self, config):
        if not isinstance(config, PoseEstimatorConfig):
            raise TypeError("config must be a PoseEstimatorConfig")
        self.config = config
        self._smoother = CircularAngleSmoother(
            window=config.angle_smoothing_window,
            period_degrees=config.angle_period_degrees,
        )
        self._last_valid_time = None

    def reset(self):
        self._smoother.reset()
        self._last_valid_time = None

    def _reject(self, reason, reliability=0.0):
        return PoseEstimate(
            valid=False,
            reason=reason,
            marker_id=self.config.marker_id,
            reliability=float(reliability),
        )

    def _corrected_corners(self, corners, homography):
        points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("invalid marker corners")
        transform = np.asarray(homography, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(transform).all():
            raise ValueError("invalid calibration homography")
        homogeneous = np.hstack((points, np.ones((4, 1))))
        projected = homogeneous @ transform.T
        scale = projected[:, 2]
        if np.any(np.abs(scale) < 1e-12):
            raise ValueError("calibration homography is degenerate")
        return projected[:, :2] / scale[:, None]

    def update(self, tracker_result, frame=None, now=None):
        """Update from a ``TrackerResult`` and return a ``PoseEstimate``.

        When ``frame`` is supplied and subpixel refinement is enabled, the
        marker-4 corners are refined from edge lines before the homography is
        applied.  Refinement failures fall back to the ArUco corners.
        """
        now = time.monotonic() if now is None else float(now)
        config = self.config
        marker_id = config.marker_id

        if tracker_result is None:
            return self._reject("no tracker result")
        if not getattr(tracker_result, "calibrated", False):
            return self._reject("calibration not ready")

        reliability_map = getattr(tracker_result, "reliability", {}) or {}
        reliability = float(reliability_map.get(marker_id, 0.0))

        if reliability < config.min_reliability:
            return self._reject("marker detection unreliable", reliability)

        live = getattr(tracker_result, "live", {}) or {}
        live_corners = getattr(tracker_result, "live_corners", {}) or {}
        tracks = getattr(tracker_result, "tracks", None) or {}
        track = tracks.get(marker_id) if hasattr(tracks, "get") else None

        measured_corners = None
        predicted = False
        prediction_age = 0.0
        if track is not None:
            if getattr(track, "is_live", False):
                measured_corners = getattr(track, "corners", None)
            elif (
                config.allow_predicted_pose
                and getattr(track, "corners", None) is not None
                and float(getattr(track, "age", 0.0))
                <= config.max_prediction_age_seconds
            ):
                measured_corners = track.corners
                predicted = True
                prediction_age = float(getattr(track, "age", 0.0))
            else:
                reason = (
                    "marker not detected live"
                    if not config.allow_predicted_pose
                    else "predicted marker position expired"
                )
                return self._reject(reason, reliability)
        elif marker_id in live:
            measured_corners = live_corners.get(marker_id)
            if measured_corners is None:
                return self._reject("marker corners unavailable", reliability)
        else:
            return self._reject("marker not detected live", reliability)

        if measured_corners is None:
            return self._reject("marker corners unavailable", reliability)
        refined = False
        fit_error = None
        refinement_reason = ""
        if config.subpixel.enabled:
            if frame is None:
                refinement_reason = "no frame supplied"
            else:
                refinement = subpixel_refiner.refine_marker_corners(
                    frame,
                    measured_corners,
                    config.subpixel,
                )
                if refinement.success:
                    measured_corners = refinement.corners
                    refined = True
                    fit_error = refinement.rms_error_px
                else:
                    refinement_reason = refinement.reason
        else:
            refinement_reason = "subpixel refinement disabled"

        try:
            corrected = self._corrected_corners(
                measured_corners,
                tracker_result.calibration,
            )
        except ValueError:
            return self._reject("invalid marker geometry", reliability)

        area = polygon_area(corrected)
        if area < config.min_marker_area_px:
            return self._reject("marker too small in corrected view", reliability)

        try:
            raw_angle = marker_axis_angle_degrees(
                corrected,
                period=config.angle_period_degrees,
            )
        except ValueError:
            return self._reject("degenerate marker orientation", reliability)

        if (
            self._last_valid_time is not None
            and now - self._last_valid_time > config.smoothing_reset_seconds
        ):
            self._smoother.reset()

        smoothed_angle = self._smoother.add(raw_angle)
        if smoothed_angle is None:
            smoothed_angle = raw_angle
        self._last_valid_time = now

        centre = corrected.mean(axis=0)
        width, height = (float(value) for value in config.output_size)
        x_px = float(centre[0])
        y_px = float(centre[1])
        x_mm = x_px / (width - 1.0) * config.drawing_width_mm
        y_mm = y_px / (height - 1.0) * config.drawing_height_mm
        inside = (
            0.0 <= x_px <= width - 1.0 and 0.0 <= y_px <= height - 1.0
        )

        return PoseEstimate(
            valid=True,
            marker_id=marker_id,
            x_px=x_px,
            y_px=y_px,
            x_mm=x_mm,
            y_mm=y_mm,
            angle_deg=smoothed_angle,
            raw_angle_deg=raw_angle,
            reliability=reliability,
            inside_drawing_area=inside,
            refined=refined,
            fit_error_px=fit_error,
            refinement_reason=refinement_reason,
            predicted=predicted,
            prediction_age_seconds=prediction_age,
            corners_px=corrected,
        )
