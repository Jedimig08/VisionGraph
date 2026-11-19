"""Subpixel edge-line refinement for a detected square marker.

ArUco already refines corner locations, but a marker corner is a single point
and is sensitive to noise.  This module samples the marker's four outer edges,
locates each black/white transition with subpixel gradient interpolation, fits
a straight line to each edge, and intersects adjacent lines to recover refined
corners.  The refined corners are intended for measurements such as centre and
inclination; a failed refinement simply falls back to the ArUco corners.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class SubpixelRefinerConfig:
    """Sampling and acceptance settings for edge-line refinement."""

    enabled: bool = True
    samples_per_edge: int = 24
    search_radius_px: float = 2.5
    scan_step_px: float = 0.25
    min_gradient: float = 15.0
    min_inliers_per_edge: int = 6
    max_fit_error_px: float = 1.5
    max_corner_shift_px: float = 3.0
    edge_margin: float = 0.12
    min_edge_length_px: float = 8.0

    def __post_init__(self):
        if int(self.samples_per_edge) < 4:
            raise ValueError("At least four samples per edge are required")
        if not math.isfinite(self.search_radius_px) or self.search_radius_px <= 0:
            raise ValueError("Search radius must be a finite positive number")
        if not math.isfinite(self.scan_step_px) or self.scan_step_px <= 0:
            raise ValueError("Scan step must be a finite positive number")
        if not math.isfinite(self.min_gradient) or self.min_gradient < 0:
            raise ValueError("Minimum gradient cannot be negative")
        if not 2 <= int(self.min_inliers_per_edge) <= int(self.samples_per_edge):
            raise ValueError("Minimum inliers must be between 2 and samples_per_edge")
        if not math.isfinite(self.max_fit_error_px) or self.max_fit_error_px <= 0:
            raise ValueError("Maximum fit error must be a finite positive number")
        if not math.isfinite(self.max_corner_shift_px) or self.max_corner_shift_px <= 0:
            raise ValueError("Maximum corner shift must be a finite positive number")
        if not 0.0 <= float(self.edge_margin) < 0.5:
            raise ValueError("Edge margin must be in [0, 0.5)")
        if not math.isfinite(self.min_edge_length_px) or self.min_edge_length_px <= 0:
            raise ValueError("Minimum edge length must be a finite positive number")


@dataclass(frozen=True)
class SubpixelRefinement:
    success: bool
    corners: np.ndarray | None = None
    rms_error_px: float | None = None
    edge_rms_px: tuple = ()
    max_corner_shift_px: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class _FittedLine:
    point: np.ndarray
    direction: np.ndarray
    rms_error_px: float
    inlier_fraction: float


def _failure(reason):
    return SubpixelRefinement(success=False, reason=reason)


def _to_grayscale(frame):
    image = np.asarray(frame)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Frame must be grayscale, BGR, or BGRA")
    return np.asarray(gray, dtype=np.float32)


def _bilinear(image, x, y):
    height, width = image.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (y >= 0.0)
        & (x <= width - 1.0)
        & (y <= height - 1.0)
    )
    clipped_x = np.clip(x, 0.0, width - 1.0)
    clipped_y = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(clipped_x).astype(np.int32)
    y0 = np.floor(clipped_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = clipped_x - x0
    fy = clipped_y - y0
    top = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy, valid


def _scan_offsets(config):
    reach = max(1, int(math.ceil(config.search_radius_px / config.scan_step_px)))
    return np.linspace(
        -config.search_radius_px,
        config.search_radius_px,
        2 * reach + 1,
    )


def _edge_points(gray, p0, p1, config):
    edge = p1 - p0
    length = float(np.linalg.norm(edge))
    if length < config.min_edge_length_px:
        return None, "marker edge too short"

    tangent = edge / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    fractions = np.linspace(
        config.edge_margin,
        1.0 - config.edge_margin,
        config.samples_per_edge,
    )
    base = p0[None, :] + fractions[:, None] * edge[None, :]
    offsets = _scan_offsets(config)
    spacing = float(offsets[1] - offsets[0])
    xs = base[:, 0, None] + offsets[None, :] * normal[0]
    ys = base[:, 1, None] + offsets[None, :] * normal[1]
    intensities, valid = _bilinear(gray, xs, ys)

    points = []
    for sample_index in range(base.shape[0]):
        if not np.all(valid[sample_index]):
            continue
        gradient = np.gradient(intensities[sample_index], offsets)
        magnitude = np.abs(gradient)
        candidates = []
        for peak_index in range(1, len(offsets) - 1):
            value = magnitude[peak_index]
            if value < config.min_gradient:
                continue
            if value >= magnitude[peak_index - 1] and value >= magnitude[peak_index + 1]:
                candidates.append(peak_index)
        if not candidates:
            continue
        peak_index = min(candidates, key=lambda index: abs(offsets[index]))
        delta = 0.0
        if 0 < peak_index < len(offsets) - 1:
            left = magnitude[peak_index - 1]
            centre = magnitude[peak_index]
            right = magnitude[peak_index + 1]
            denominator = left - 2.0 * centre + right
            if abs(denominator) > 1e-9:
                delta = 0.5 * (left - right) / denominator
                delta = float(np.clip(delta, -0.5, 0.5))
        offset = offsets[peak_index] + delta * spacing
        points.append(base[sample_index] + offset * normal)

    if not points:
        return None, "no sharp edge transitions found"
    return np.asarray(points, dtype=np.float64), ""


def _fit_line(points, config):
    if len(points) < config.min_inliers_per_edge:
        return None
    centroid = points.mean(axis=0)
    centred = points - centroid
    if float(np.linalg.norm(centred)) < 1e-9:
        return None
    _, _, right_vectors = np.linalg.svd(centred, full_matrices=False)
    direction = right_vectors[0]
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)

    residuals = np.abs(centred @ normal)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    sigma = 1.4826 * mad
    cutoff = max(0.5, min(config.max_fit_error_px, median + 3.0 * sigma))
    inlier_mask = residuals <= cutoff
    inliers = points[inlier_mask]
    if len(inliers) < config.min_inliers_per_edge:
        return None

    centroid = inliers.mean(axis=0)
    centred = inliers - centroid
    _, _, right_vectors = np.linalg.svd(centred, full_matrices=False)
    direction = right_vectors[0]
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    residuals = np.abs(centred @ normal)
    rms_error = float(np.sqrt(np.mean(residuals**2)))
    if rms_error > config.max_fit_error_px:
        return None

    return _FittedLine(
        point=centroid,
        direction=direction,
        rms_error_px=rms_error,
        inlier_fraction=len(inliers) / len(points),
    )


def _cross2d(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _intersect(first, second):
    denominator = _cross2d(first.direction, second.direction)
    if abs(denominator) < 1e-6:
        return None
    offset = second.point - first.point
    distance = _cross2d(offset, second.direction) / denominator
    point = first.point + distance * first.direction
    if not np.isfinite(point).all():
        return None
    return point


def _is_convex(points):
    if not np.isfinite(points).all():
        return False
    signs = []
    for index in range(4):
        previous = points[index] - points[(index - 1) % 4]
        following = points[(index + 1) % 4] - points[index]
        signs.append(_cross2d(previous, following))
    return all(value > 1e-9 for value in signs) or all(
        value < -1e-9 for value in signs
    )


def refine_marker_corners(frame, corners, config=None):
    """Refine four marker corners from subpixel edge lines.

    Returns a :class:`SubpixelRefinement`.  On any failure the caller should
    keep using the original ArUco corners.
    """
    config = SubpixelRefinerConfig() if config is None else config
    if not isinstance(config, SubpixelRefinerConfig):
        raise TypeError("config must be a SubpixelRefinerConfig")
    if not config.enabled:
        return _failure("subpixel refinement disabled")

    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return _failure("invalid marker corners")
    try:
        gray = _to_grayscale(frame)
    except (TypeError, ValueError):
        return _failure("invalid frame")

    lines = []
    edge_errors = []
    for index in range(4):
        edge_points, reason = _edge_points(
            gray,
            points[index],
            points[(index + 1) % 4],
            config,
        )
        if edge_points is None:
            return _failure(reason)
        line = _fit_line(edge_points, config)
        if line is None:
            return _failure("edge line fit failed")
        lines.append(line)
        edge_errors.append(line.rms_error_px)

    refined = np.empty((4, 2), dtype=np.float64)
    for index in range(4):
        corner = _intersect(lines[(index - 1) % 4], lines[index])
        if corner is None:
            return _failure("adjacent edge lines are parallel")
        refined[index] = corner

    if not _is_convex(refined):
        return _failure("refined corners are not convex")

    shifts = np.linalg.norm(refined - points, axis=1)
    max_shift = float(shifts.max())
    if max_shift > config.max_corner_shift_px:
        return _failure("refined corners moved too far")

    return SubpixelRefinement(
        success=True,
        corners=refined,
        rms_error_px=float(np.sqrt(np.mean(np.square(edge_errors)))),
        edge_rms_px=tuple(float(value) for value in edge_errors),
        max_corner_shift_px=max_shift,
    )
