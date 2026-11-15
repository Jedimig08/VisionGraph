"""ArUco detection, marker memory, and multi-frame surface calibration."""

from collections import deque
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np


EXPECTED_MARKER_IDS = (0, 1, 2, 3, 4)
CALIBRATION_CORNER_IDS = (0, 1, 3, 2)
OBSERVATION_WINDOW = 5
MINIMUM_OBSERVATIONS = 3
MEMORY_SECONDS = 2.0
RELIABILITY_WINDOW = 100


def output_size_for_area(width, height, long_edge=1000):
    """Scale area dimensions to pixels while preserving their aspect ratio."""
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("Drawing-area dimensions must be finite positive numbers")
    if long_edge <= 0:
        raise ValueError("Output long edge must be a positive number of pixels")
    longest_dimension = max(width, height)
    return (
        max(1, round(width / longest_dimension * long_edge)),
        max(1, round(height / longest_dimension * long_edge)),
    )


def configure_aruco_params(cv2_module=cv2):
    """Create detector parameters supported by both OpenCV ArUco APIs."""
    aruco = cv2_module.aruco
    params = (
        aruco.DetectorParameters()
        if hasattr(aruco, "DetectorParameters")
        else aruco.DetectorParameters_create()
    )
    settings = {
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 41,
        "adaptiveThreshWinSizeStep": 10,
        "adaptiveThreshConstant": 5,
        "minMarkerPerimeterRate": 0.02,
        "maxMarkerPerimeterRate": 4.0,
        "polygonalApproxAccuracyRate": 0.03,
        "errorCorrectionRate": 0.5,
        "cornerRefinementMethod": aruco.CORNER_REFINE_SUBPIX,
        "cornerRefinementWinSize": 5,
        "cornerRefinementMaxIterations": 30,
        "cornerRefinementMinAccuracy": 0.1,
    }
    for name, value in settings.items():
        if hasattr(params, name):
            setattr(params, name, value)
    return params


def merge_marker_ids(base_ids, additional_ids):
    """Append marker IDs while preserving the original ID dtype and shape."""
    base = np.asarray(base_ids)
    additional = np.asarray(additional_ids)
    if base.size == 0:
        return additional.astype(additional.dtype, copy=False).reshape(-1, 1)
    return np.concatenate(
        (base.reshape(-1), additional.reshape(-1).astype(base.dtype))
    ).reshape(-1, 1)


class ArucoMarkerDetector:
    """Try progressively enhanced images until all expected IDs are found."""

    def __init__(self, cv2_module=cv2, expected_ids=EXPECTED_MARKER_IDS):
        self.cv2 = cv2_module
        self.expected_ids = set(expected_ids)
        self.dictionary = cv2_module.aruco.getPredefinedDictionary(
            cv2_module.aruco.DICT_4X4_50
        )
        self.parameters = configure_aruco_params(cv2_module)
        self.detector = (
            cv2_module.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2_module.aruco, "ArucoDetector")
            else None
        )
        self.clahe = cv2_module.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _detect(self, image):
        if self.detector is not None:
            return self.detector.detectMarkers(image)
        return self.cv2.aruco.detectMarkers(
            image,
            self.dictionary,
            parameters=self.parameters,
        )

    def _processing_variants(self, frame):
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        yield gray
        yield self.clahe.apply(gray)
        for method in (self.cv2.ADAPTIVE_THRESH_MEAN_C, self.cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
            yield self.cv2.adaptiveThreshold(
                gray,
                255,
                method,
                self.cv2.THRESH_BINARY,
                31,
                5,
            )

    def _candidate_area(self, corners):
        points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        return abs(float(self.cv2.contourArea(points)))

    def detect(self, frame):
        detections = {}
        rejected = []
        seen_rejected = []
        for image in self._processing_variants(frame):
            corners, ids, rejected_candidates = self._detect(image)
            for corner in rejected_candidates:
                if not any(
                    self._same_region(corner, existing)
                    for existing in seen_rejected
                ):
                    seen_rejected.append(corner)
            rejected = seen_rejected

            if ids is not None:
                for corner, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
                    marker_id = int(marker_id)
                    if marker_id not in self.expected_ids:
                        continue
                    current = detections.get(marker_id)
                    if current is None or self._candidate_area(corner) > self._candidate_area(current):
                        detections[marker_id] = corner

            if self.expected_ids.issubset(detections):
                break

        ids = np.asarray(sorted(detections), dtype=np.int32).reshape(-1, 1)
        detected_corners = [detections[int(marker_id)] for marker_id in ids.reshape(-1)]
        return detected_corners, ids if ids.size else None, len(rejected)

    def _same_region(self, first, second):
        first_box = self.cv2.boundingRect(first)
        second_box = self.cv2.boundingRect(second)
        x1, y1, width1, height1 = first_box
        x2, y2, width2, height2 = second_box
        overlap_width = max(0, min(x1 + width1, x2 + width2) - max(x1, x2))
        overlap_height = max(0, min(y1 + height1, y2 + height2) - max(y1, y2))
        overlap = overlap_width * overlap_height
        smaller = min(width1 * height1, width2 * height2)
        return smaller > 0 and overlap / smaller > 0.5


@dataclass
class MarkerMemory:
    center: np.ndarray
    corners: np.ndarray
    last_seen: float


@dataclass
class TrackerResult:
    live: dict
    live_corners: dict
    remembered: dict
    missing: tuple
    reliability: dict
    rejected_count: int
    calibrated: bool
    calibration_observations: dict
    calibration: np.ndarray | None
    output_size: tuple


class MarkerTracker:
    """Track live and remembered markers and calibrate from stable observations."""

    def __init__(
        self,
        output_size,
        cv2_module=cv2,
        expected_ids=EXPECTED_MARKER_IDS,
        corner_ids=CALIBRATION_CORNER_IDS,
        memory_seconds=MEMORY_SECONDS,
        observation_window=OBSERVATION_WINDOW,
        minimum_observations=MINIMUM_OBSERVATIONS,
    ):
        self.cv2 = cv2_module
        self.expected_ids = tuple(expected_ids)
        self.corner_ids = tuple(corner_ids)
        self.calibration_ids = self.corner_ids
        self.output_size = output_size
        self.memory_seconds = memory_seconds
        self.observation_window = observation_window
        self.minimum_observations = minimum_observations
        self.memory = {}
        self.observations = {
            marker_id: deque(maxlen=observation_window)
            for marker_id in self.expected_ids
        }
        self.corner_observations = {
            marker_id: deque(maxlen=observation_window)
            for marker_id in self.expected_ids
        }
        self.reliability_samples = {
            marker_id: deque(maxlen=RELIABILITY_WINDOW)
            for marker_id in self.expected_ids
        }
        self.movement_samples = {
            marker_id: deque(maxlen=observation_window)
            for marker_id in self.calibration_ids
        }
        self.calibration = None
        self.calibration_sources = None
        self.calibration_output_size = output_size

    @staticmethod
    def _center(corners):
        return np.asarray(corners, dtype=np.float32).reshape(-1, 2).mean(axis=0)

    def _stable_position(self, marker_id, consistency_radius):
        samples = [
            sample for sample in self.observations[marker_id] if sample is not None
        ]
        if len(samples) < self.minimum_observations:
            return None
        positions = np.asarray(samples, dtype=np.float32)
        median = np.median(positions, axis=0)
        distances = np.linalg.norm(positions - median, axis=1)
        inliers = positions[distances <= consistency_radius]
        if len(inliers) < self.minimum_observations:
            return None
        return np.median(inliers, axis=0)

    def _stable_corners(self, marker_id, consistency_radius):
        samples = [
            (position, corners)
            for position, corners in zip(
                self.observations[marker_id],
                self.corner_observations[marker_id],
            )
            if position is not None and corners is not None
        ]
        if len(samples) < self.minimum_observations:
            return None
        positions = np.asarray([position for position, _ in samples], dtype=np.float32)
        median = np.median(positions, axis=0)
        inlier_indices = np.flatnonzero(
            np.linalg.norm(positions - median, axis=1) <= consistency_radius
        )
        if len(inlier_indices) < self.minimum_observations:
            return None
        inlier_corners = np.asarray(
            [samples[index][1] for index in inlier_indices],
            dtype=np.float32,
        )
        return np.median(inlier_corners, axis=0)

    def _make_calibration(self, frame_shape):
        diagonal = math.hypot(frame_shape[1], frame_shape[0])
        consistency_radius = max(6.0, diagonal * 0.005)
        positions = {
            marker_id: self._stable_position(marker_id, consistency_radius)
            for marker_id in self.corner_ids
        }
        if any(position is None for position in positions.values()):
            return

        marker_corners = {
            marker_id: self._stable_corners(marker_id, consistency_radius)
            for marker_id in self.corner_ids
        }
        if any(corners is None for corners in marker_corners.values()):
            return

        surface_center = np.mean(
            [positions[marker_id] for marker_id in self.corner_ids],
            axis=0,
        )
        source = []
        for marker_id in self.corner_ids:
            marker_center = positions[marker_id]
            outward_direction = marker_center - surface_center
            vertices = np.asarray(
                marker_corners[marker_id],
                dtype=np.float32,
            ).reshape(-1, 2)
            outer_vertex = np.argmax(
                (vertices - marker_center) @ outward_direction
            )
            source.append(vertices[outer_vertex])
        source = np.asarray(source, dtype=np.float32)
        width, height = self.output_size
        destination_corners = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        homography = self.cv2.getPerspectiveTransform(
            source,
            np.asarray(destination_corners, dtype=np.float32),
        )
        if (
            not np.isfinite(homography).all()
            or abs(float(np.linalg.det(homography))) < 1e-12
        ):
            return

        self.calibration = homography
        self.calibration_output_size = self.output_size
        self.calibration_sources = {
            marker_id: positions[marker_id].copy()
            for marker_id in self.calibration_ids
        }

    def _invalidate_if_moved(self, live, live_corners, frame_shape):
        diagonal = math.hypot(frame_shape[1], frame_shape[0])
        movement_threshold = max(12.0, diagonal * 0.02)
        for marker_id in self.calibration_ids:
            center = live.get(marker_id)
            if center is None:
                self.movement_samples[marker_id].append(False)
            else:
                distance = np.linalg.norm(
                    center - self.calibration_sources[marker_id]
                )
                self.movement_samples[marker_id].append(
                    distance > movement_threshold
                )
            if sum(self.movement_samples[marker_id]) >= self.minimum_observations:
                self.calibration = None
                self.calibration_sources = None
                self.calibration_output_size = self.output_size
                for samples in self.observations.values():
                    samples.clear()
                for samples in self.corner_observations.values():
                    samples.clear()
                for samples in self.movement_samples.values():
                    samples.clear()
                for observed_id, position in live.items():
                    if observed_id in self.observations:
                        self.observations[observed_id].append(position.copy())
                        self.corner_observations[observed_id].append(
                            live_corners[observed_id].copy()
                        )
                return True
        return False

    def update(self, corners, ids, rejected_count, frame_shape, now=None):
        now = time.monotonic() if now is None else now
        live = {}
        live_corners = {}
        if ids is not None:
            for marker_corners, marker_id in zip(
                corners, np.asarray(ids).reshape(-1)
            ):
                marker_id = int(marker_id)
                if marker_id not in self.observations:
                    continue
                center = self._center(marker_corners)
                live[marker_id] = center
                live_corners[marker_id] = np.asarray(marker_corners).copy()
                self.memory[marker_id] = MarkerMemory(
                    center=center.copy(),
                    corners=np.asarray(marker_corners).copy(),
                    last_seen=now,
                )

        for marker_id in self.expected_ids:
            self.reliability_samples[marker_id].append(marker_id in live)
            self.observations[marker_id].append(
                live[marker_id].copy() if marker_id in live else None
            )
            self.corner_observations[marker_id].append(
                live_corners[marker_id].copy() if marker_id in live else None
            )

        for marker_id, remembered in list(self.memory.items()):
            if now - remembered.last_seen > self.memory_seconds:
                del self.memory[marker_id]

        moved = False
        if self.calibration is not None:
            moved = self._invalidate_if_moved(live, live_corners, frame_shape)
        if self.calibration is None and not moved:
            self._make_calibration(frame_shape)

        missing = tuple(
            marker_id for marker_id in self.expected_ids if marker_id not in live
        )
        remembered = {
            marker_id: self.memory[marker_id]
            for marker_id in missing
            if marker_id in self.memory
        }
        reliability = {
            marker_id: (
                sum(samples) / len(samples) if samples else 0.0
            )
            for marker_id, samples in self.reliability_samples.items()
        }
        observation_counts = {
            marker_id: sum(sample is not None for sample in samples)
            for marker_id, samples in self.observations.items()
        }
        return TrackerResult(
            live=live,
            live_corners=live_corners,
            remembered=remembered,
            missing=missing,
            reliability=reliability,
            rejected_count=rejected_count,
            calibrated=self.calibration is not None,
            calibration_observations=observation_counts,
            calibration=self.calibration,
            output_size=(
                self.calibration_output_size
                if self.calibration is not None
                else self.output_size
            ),
        )
