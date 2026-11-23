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


@dataclass(frozen=True)
class DetectionConfig:
    """Rules for choosing one candidate per marker across processing variants."""

    max_area_ratio: float = 1.8
    min_compactness: float = 0.5

    def __post_init__(self):
        if (
            not math.isfinite(float(self.max_area_ratio))
            or float(self.max_area_ratio) < 1.0
        ):
            raise ValueError("Maximum area ratio must be at least one")
        if (
            not math.isfinite(float(self.min_compactness))
            or not 0.0 <= float(self.min_compactness) <= 1.0
        ):
            raise ValueError("Minimum compactness must be between 0 and 1")


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


def _candidate_area(corners, cv2_module):
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return abs(float(cv2_module.contourArea(points)))


def _candidate_compactness(corners, cv2_module):
    """Isoperimetric quotient: 1.0 for a circle, about 0.79 for a square."""
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return 0.0
    area = _candidate_area(points, cv2_module)
    perimeter = float(
        cv2_module.arcLength(points.reshape(-1, 1, 2), True)
    )
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def choose_detection_candidates(candidates, cv2_module, config=None):
    """Pick the most plausible corners per ID instead of the largest contour.

    A threshold variant can fuse the marker border with the bright background
    and report an oversized quadrilateral.  Across variants the true marker
    size is the one that repeats, so candidates close to their ID's median area
    win; an oversized outlier is discarded.  Ties favour the tighter, more
    compact contour and the variant that appeared first (plain grayscale).
    """
    config = DetectionConfig() if config is None else config
    detections = {}
    for marker_id, options in candidates.items():
        if not options:
            continue
        areas = [
            _candidate_area(corner, cv2_module) for corner in options
        ]
        median_area = float(np.median(np.asarray(areas, dtype=np.float64)))
        scored = []
        for corner, area in zip(options, areas):
            compactness = _candidate_compactness(corner, cv2_module)
            if compactness < config.min_compactness:
                continue
            if area > median_area * config.max_area_ratio:
                continue
            scored.append(
                (abs(area - median_area), -compactness, float(area), corner)
            )
        if not scored:
            scored = [
                (
                    float("inf"),
                    -_candidate_compactness(corner, cv2_module),
                    float(area),
                    corner,
                )
                for corner, area in zip(options, areas)
            ]
        scored.sort(key=lambda item: item[:3])
        detections[marker_id] = scored[0][3]
    return detections


class ArucoMarkerDetector:
    """Try progressively enhanced images until all expected IDs are found."""

    def __init__(self, cv2_module=cv2, expected_ids=EXPECTED_MARKER_IDS, config=None):
        self.cv2 = cv2_module
        self.expected_ids = set(expected_ids)
        self.config = DetectionConfig() if config is None else config
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
        candidates = {}
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
                    candidates.setdefault(marker_id, []).append(
                        np.asarray(corner).copy()
                    )

            if self.expected_ids.issubset(candidates):
                break

        config = getattr(self, "config", None)
        detections = choose_detection_candidates(candidates, self.cv2, config)
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
class MarkerTrack:
    """Continuous state for one of the fixed set of expected marker IDs."""

    marker_id: int
    center: np.ndarray
    corners: np.ndarray
    velocity: np.ndarray
    last_seen: float
    age: float = 0.0
    is_live: bool = True


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
    calibration_confirmed: bool
    calibration_pending: bool
    tracked_ids: tuple
    tracks: dict
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
        association_radius_rate=0.05,
        velocity_smoothing=0.5,
        max_velocity_rate=2.0,
        prediction_window_seconds=1.0,
        carriage_id=4,
    ):
        if not math.isfinite(float(association_radius_rate)) or association_radius_rate <= 0:
            raise ValueError("Association radius rate must be positive")
        if not 0.0 <= float(velocity_smoothing) <= 1.0:
            raise ValueError("Velocity smoothing must be between 0 and 1")
        if not math.isfinite(float(max_velocity_rate)) or max_velocity_rate <= 0:
            raise ValueError("Maximum velocity rate must be positive")
        if (
            not math.isfinite(float(prediction_window_seconds))
            or prediction_window_seconds < 0
        ):
            raise ValueError("Prediction window cannot be negative")
        self.cv2 = cv2_module
        self.expected_ids = tuple(expected_ids)
        self.corner_ids = tuple(corner_ids)
        self.calibration_ids = self.corner_ids
        self.carriage_id = int(carriage_id)
        self.output_size = output_size
        self.memory_seconds = memory_seconds
        self.observation_window = observation_window
        self.minimum_observations = minimum_observations
        self.association_radius_rate = float(association_radius_rate)
        self.velocity_smoothing = float(velocity_smoothing)
        self.max_velocity_rate = float(max_velocity_rate)
        self.prediction_window_seconds = float(prediction_window_seconds)
        self.memory = {}
        self.tracks = {marker_id: None for marker_id in self.expected_ids}
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
        self.calibration_confirmed = False
        self.locked_calibration = None
        self.locked_sources = None
        self.locked_output_size = output_size

    @property
    def tracked_ids(self):
        """Only the carriage matters once the human has locked calibration."""
        if self.calibration_confirmed:
            return (self.carriage_id,)
        return self.expected_ids

    def confirm_calibration(self):
        """Lock the current homography after human inspection.

        Returns False when no calibration exists yet.  Corner markers stay
        detected afterwards, but only to notice movement that unlocks again.
        """
        if self.calibration is None or self.calibration_sources is None:
            return False
        self.calibration_confirmed = True
        self.locked_calibration = self.calibration.copy()
        self.locked_sources = {
            marker_id: position.copy()
            for marker_id, position in self.calibration_sources.items()
        }
        self.locked_output_size = self.calibration_output_size
        for marker_id in self.corner_ids:
            self.memory.pop(marker_id, None)
            self.tracks[marker_id] = None
        return True

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
        if self.calibration_confirmed:
            return
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
                self.calibration_confirmed = False
                self.locked_calibration = None
                self.locked_sources = None
                self.locked_output_size = self.output_size
                for corner_id in self.corner_ids:
                    self.memory.pop(corner_id, None)
                    self.tracks[corner_id] = None
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

    def _update_tracks(self, live, live_corners, frame_shape, now, tracked_ids=None):
        """Associate fresh detections by ID and coast tracks through misses.

        A detection is accepted only when it lands near the position predicted
        from the track's recent velocity, so a marker cannot teleport.  Missed
        detections leave the track at its predicted position with ``is_live``
        false, which consumers may only use for a short, flagged grace period.
        """
        tracked_ids = self.expected_ids if tracked_ids is None else tracked_ids
        diagonal = math.hypot(frame_shape[1], frame_shape[0])
        radius = max(20.0, diagonal * self.association_radius_rate)
        max_speed = diagonal * self.max_velocity_rate
        for marker_id in tracked_ids:
            track = self.tracks.get(marker_id)
            raw_center = live.get(marker_id)
            raw_corners = live_corners.get(marker_id)

            predicted_center = None
            dt = 0.0
            if track is not None:
                dt = now - track.last_seen
                if 0.0 < dt <= self.prediction_window_seconds:
                    predicted_center = track.center + track.velocity * dt
                else:
                    predicted_center = track.center.copy()

            accepted = raw_center is not None and (
                track is None
                or predicted_center is None
                or float(np.linalg.norm(raw_center - predicted_center)) <= radius
            )

            if accepted:
                center = np.asarray(raw_center, dtype=np.float32).copy()
                corners = np.asarray(raw_corners).copy()
                if track is not None and dt > 1e-3:
                    instantaneous = (center - predicted_center) / dt
                else:
                    instantaneous = np.zeros(2, dtype=np.float32)
                if track is not None:
                    velocity = (
                        self.velocity_smoothing * track.velocity
                        + (1.0 - self.velocity_smoothing) * instantaneous
                    )
                else:
                    velocity = np.zeros(2, dtype=np.float32)
                speed = float(np.linalg.norm(velocity))
                if speed > max_speed and speed > 0.0:
                    velocity = velocity * (max_speed / speed)
                self.tracks[marker_id] = MarkerTrack(
                    marker_id=marker_id,
                    center=center,
                    corners=corners,
                    velocity=velocity,
                    last_seen=now,
                    age=0.0,
                    is_live=True,
                )
            elif track is not None:
                center = predicted_center.copy()
                displacement = center - track.center
                self.tracks[marker_id] = MarkerTrack(
                    marker_id=marker_id,
                    center=center,
                    corners=track.corners + displacement,
                    velocity=track.velocity,
                    last_seen=track.last_seen,
                    age=max(0.0, now - track.last_seen),
                    is_live=False,
                )

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
                if marker_id in self.tracked_ids:
                    self.memory[marker_id] = MarkerMemory(
                        center=center.copy(),
                        corners=np.asarray(marker_corners).copy(),
                        last_seen=now,
                    )

        for marker_id in self.tracked_ids:
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

        self._update_tracks(
            live,
            live_corners,
            frame_shape,
            now,
            tracked_ids=self.tracked_ids,
        )

        moved = False
        if self.calibration is not None:
            moved = self._invalidate_if_moved(live, live_corners, frame_shape)
        if self.calibration is None and not moved:
            self._make_calibration(frame_shape)

        missing = tuple(
            marker_id for marker_id in self.tracked_ids if marker_id not in live
        )
        remembered = {
            marker_id: self.memory[marker_id]
            for marker_id in missing
            if marker_id in self.memory
        }
        reliability = {
            marker_id: (
                sum(self.reliability_samples[marker_id])
                / len(self.reliability_samples[marker_id])
                if self.reliability_samples[marker_id]
                else 0.0
            )
            for marker_id in self.tracked_ids
        }
        observation_counts = {
            marker_id: sum(
                sample is not None for sample in self.observations[marker_id]
            )
            for marker_id in self.tracked_ids
        }
        tracks = {
            marker_id: self.tracks[marker_id]
            for marker_id in self.tracked_ids
            if self.tracks.get(marker_id) is not None
        }
        return TrackerResult(
            live=live,
            live_corners=live_corners,
            remembered=remembered,
            tracks=tracks,
            missing=missing,
            reliability=reliability,
            rejected_count=rejected_count,
            calibrated=self.calibration is not None,
            calibration_observations=observation_counts,
            calibration=self.calibration,
            calibration_confirmed=self.calibration_confirmed,
            calibration_pending=(
                self.calibration is not None and not self.calibration_confirmed
            ),
            tracked_ids=self.tracked_ids,
            output_size=(
                self.calibration_output_size
                if self.calibration is not None
                else self.output_size
            ),
        )
