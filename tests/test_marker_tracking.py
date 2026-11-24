import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

import marker_tracking


class MarkerTrackerTests(unittest.TestCase):
    frame_shape = (1000, 2000, 3)
    centers = {
        0: (100, 100),
        1: (1900, 100),
        3: (1900, 900),
        2: (100, 900),
        4: (1750, 500),
    }

    @staticmethod
    def marker_corners(center):
        x, y = center
        return np.asarray(
            [[[x - 10, y - 10], [x + 10, y - 10], [x + 10, y + 10], [x - 10, y + 10]]],
            dtype=np.float32,
        )

    @classmethod
    def outer_marker_corners(cls, centers, marker_ids=(0, 1, 3, 2)):
        surface_center = np.mean(
            [centers[marker_id] for marker_id in marker_ids],
            axis=0,
        )
        outer_corners = []
        for marker_id in marker_ids:
            marker_center = np.asarray(centers[marker_id], dtype=np.float32)
            vertices = cls.marker_corners(centers[marker_id]).reshape(-1, 2)
            outward_direction = marker_center - surface_center
            outer_index = np.argmax((vertices - marker_center) @ outward_direction)
            outer_corners.append(vertices[outer_index])
        return np.asarray(outer_corners, dtype=np.float32).reshape(-1, 1, 2)

    def frame_detections(self, marker_ids, centers=None):
        centers = centers or self.centers
        ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1)
        corners = [self.marker_corners(centers[marker_id]) for marker_id in marker_ids]
        return corners, ids

    def test_calibrates_from_consistent_observations_across_frames(self):
        output_size = marker_tracking.output_size_for_area(600, 400)
        tracker = marker_tracking.MarkerTracker(output_size=output_size)
        result = None
        observations = (
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3, 4),
            (1, 2, 3, 4),
            (1, 2, 3, 4),
        )

        for frame_index, marker_ids in enumerate(observations):
            corners, ids = self.frame_detections(marker_ids)
            result = tracker.update(
                corners,
                ids,
                0,
                self.frame_shape,
                now=frame_index / 10,
            )

        self.assertTrue(result.calibrated)
        self.assertEqual(
            result.calibration_observations,
            {0: 3, 1: 4, 2: 4, 3: 4, 4: 3},
        )
        self.assertNotIn(4, tracker.calibration_sources)
        self.assertEqual(result.output_size, output_size)

        mapped = cv2.perspectiveTransform(
            self.outer_marker_corners(self.centers),
            result.calibration,
        ).reshape(-1, 2)
        np.testing.assert_allclose(
            mapped,
            [
                [0, 0],
                [output_size[0] - 1, 0],
                [output_size[0] - 1, output_size[1] - 1],
                [0, output_size[1] - 1],
            ],
            atol=1,
        )
        source_corners = np.asarray(
            [
                self.marker_corners(self.centers[marker_id]).reshape(-1, 2)
                for marker_id in (0, 1, 3, 2)
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        mapped_corners = cv2.perspectiveTransform(
            source_corners,
            result.calibration,
        ).reshape(-1, 2)
        self.assertTrue(np.all(mapped_corners[:, 0] >= -1e-4))
        self.assertTrue(np.all(mapped_corners[:, 0] <= result.output_size[0] - 1 + 1e-4))
        self.assertTrue(np.all(mapped_corners[:, 1] >= -1e-4))
        self.assertTrue(np.all(mapped_corners[:, 1] <= result.output_size[1] - 1 + 1e-4))

    def test_output_size_scales_to_long_edge_and_preserves_aspect_ratio(self):
        self.assertEqual(marker_tracking.output_size_for_area(600, 400), (1000, 667))
        with self.assertRaises(ValueError):
            marker_tracking.output_size_for_area(0, 400)
        with self.assertRaises(ValueError):
            marker_tracking.output_size_for_area(600, 400, 0)

    def test_remembered_markers_are_not_live_and_expire(self):
        tracker = marker_tracking.MarkerTracker(
            output_size=marker_tracking.output_size_for_area(2, 1)
        )
        corners, ids = self.frame_detections([0])
        first = tracker.update(corners, ids, 2, self.frame_shape, now=0)

        self.assertEqual(set(first.live), {0})
        self.assertEqual(len(first.missing), 4)
        self.assertNotIn(0, first.remembered)
        self.assertFalse(first.calibrated)

        absent = tracker.update([], None, 1, self.frame_shape, now=1)
        self.assertNotIn(0, absent.live)
        self.assertIn(0, absent.remembered)
        self.assertFalse(absent.calibrated)
        self.assertEqual(absent.reliability[0], 0.5)

        expired = tracker.update([], None, 0, self.frame_shape, now=2.1)
        self.assertNotIn(0, expired.remembered)

    def test_track_follows_motion_and_coasts_through_a_brief_dropout(self):
        tracker = marker_tracking.MarkerTracker(
            output_size=marker_tracking.output_size_for_area(2, 1)
        )
        corners, ids = self.frame_detections([4])
        first = tracker.update(corners, ids, 0, self.frame_shape, now=0.0)
        self.assertTrue(first.tracks[4].is_live)
        self.assertEqual(first.tracks[4].age, 0.0)

        moved = dict(self.centers)
        moved[4] = (self.centers[4][0] + 10, self.centers[4][1])
        corners, ids = self.frame_detections([4], moved)
        second = tracker.update(corners, ids, 0, self.frame_shape, now=0.1)
        self.assertTrue(second.tracks[4].is_live)
        self.assertGreater(second.tracks[4].velocity[0], 0.0)

        missing = tracker.update([], None, 0, self.frame_shape, now=0.2)
        track = missing.tracks[4]
        self.assertFalse(track.is_live)
        self.assertAlmostEqual(track.age, 0.1)
        self.assertGreater(track.center[0], moved[4][0])
        self.assertLess(track.center[0], moved[4][0] + 20)
        self.assertEqual(track.corners.shape, corners[0].shape)

        later = tracker.update([], None, 0, self.frame_shape, now=0.3)
        self.assertGreater(later.tracks[4].center[0], track.center[0])

    def test_track_rejects_a_teleporting_detection(self):
        tracker = marker_tracking.MarkerTracker(
            output_size=marker_tracking.output_size_for_area(2, 1)
        )
        corners, ids = self.frame_detections([4])
        tracker.update(corners, ids, 0, self.frame_shape, now=0.0)

        jumped = dict(self.centers)
        jumped[4] = (200, 200)
        corners, ids = self.frame_detections([4], jumped)
        result = tracker.update(corners, ids, 0, self.frame_shape, now=0.1)

        self.assertIn(4, result.live)
        track = result.tracks[4]
        self.assertFalse(track.is_live)
        self.assertAlmostEqual(track.center[0], self.centers[4][0])
        self.assertAlmostEqual(track.center[1], self.centers[4][1])

    def test_tracks_are_only_created_for_expected_ids(self):
        tracker = marker_tracking.MarkerTracker(
            output_size=marker_tracking.output_size_for_area(2, 1)
        )
        unexpected = np.asarray([[9]], dtype=np.int32)
        result = tracker.update(
            self.marker_corners((100, 100)),
            unexpected,
            0,
            self.frame_shape,
            now=0.0,
        )
        self.assertEqual(result.tracks, {})

    def make_calibrated_tracker(self):
        tracker = marker_tracking.MarkerTracker(
            output_size=marker_tracking.output_size_for_area(600, 400)
        )
        for frame_index in range(3):
            corners, ids = self.frame_detections(marker_tracking.EXPECTED_MARKER_IDS)
            result = tracker.update(
                corners,
                ids,
                0,
                self.frame_shape,
                now=frame_index / 10,
            )
        return tracker, result

    def test_confirmation_requires_a_ready_calibration(self):
        tracker = marker_tracking.MarkerTracker(
            output_size=marker_tracking.output_size_for_area(600, 400)
        )
        self.assertFalse(tracker.confirm_calibration())

        tracker, result = self.make_calibrated_tracker()
        self.assertTrue(result.calibration_pending)
        self.assertFalse(result.calibration_confirmed)
        self.assertTrue(tracker.confirm_calibration())

    def test_confirming_locks_homography_and_limits_tracking_to_carriage(self):
        tracker, result = self.make_calibrated_tracker()
        locked = result.calibration.copy()
        self.assertTrue(tracker.confirm_calibration())

        moved = dict(self.centers)
        moved[4] = (1600, 500)
        moved[0] = (150, 150)
        corners, ids = self.frame_detections(marker_tracking.EXPECTED_MARKER_IDS, moved)
        after = tracker.update(corners, ids, 0, self.frame_shape, now=0.4)

        self.assertTrue(after.calibration_confirmed)
        self.assertFalse(after.calibration_pending)
        np.testing.assert_allclose(after.calibration, locked)
        self.assertEqual(after.tracked_ids, (4,))
        self.assertEqual(set(after.reliability), {4})
        self.assertEqual(set(after.tracks), {4})
        self.assertEqual(set(after.calibration_observations), {4})
        self.assertEqual(after.missing, ())

    def test_corner_movement_unlocks_and_requires_a_new_confirmation(self):
        tracker, _result = self.make_calibrated_tracker()
        self.assertTrue(tracker.confirm_calibration())

        moved = dict(self.centers)
        moved[0] = (250, 250)
        for frame_index in range(3, 9):
            corners, ids = self.frame_detections(
                marker_tracking.EXPECTED_MARKER_IDS,
                moved,
            )
            result = tracker.update(
                corners,
                ids,
                0,
                self.frame_shape,
                now=frame_index / 10,
            )

        self.assertFalse(result.calibration_confirmed)
        self.assertTrue(result.calibration_pending)
        self.assertIsNotNone(result.calibration)
        self.assertEqual(set(result.tracked_ids), set(marker_tracking.EXPECTED_MARKER_IDS))
        self.assertTrue(tracker.confirm_calibration())

    def test_calibration_is_invalidated_after_consistent_marker_movement(self):
        tracker = marker_tracking.MarkerTracker(
            output_size=marker_tracking.output_size_for_area(2, 1)
        )
        for frame_index in range(3):
            corners, ids = self.frame_detections(marker_tracking.EXPECTED_MARKER_IDS)
            result = tracker.update(
                corners,
                ids,
                0,
                self.frame_shape,
                now=frame_index / 10,
            )
        self.assertTrue(result.calibrated)

        moved_centers = dict(self.centers)
        moved_centers[0] = (250, 250)
        for frame_index in range(3, 6):
            corners, ids = self.frame_detections(
                marker_tracking.EXPECTED_MARKER_IDS,
                moved_centers,
            )
            result = tracker.update(
                corners,
                ids,
                0,
                self.frame_shape,
                now=frame_index / 10,
            )

        self.assertFalse(result.calibrated)

    def test_id_four_is_not_used_to_calibrate_or_invalidate(self):
        tracker = marker_tracking.MarkerTracker(
            output_size=marker_tracking.output_size_for_area(2, 1)
        )
        for frame_index in range(3):
            corners, ids = self.frame_detections(marker_tracking.EXPECTED_MARKER_IDS)
            result = tracker.update(
                corners,
                ids,
                0,
                self.frame_shape,
                now=frame_index / 10,
            )
        self.assertTrue(result.calibrated)

        moved_centers = dict(self.centers)
        moved_centers[4] = (1200, 650)
        for frame_index in range(3, 6):
            corners, ids = self.frame_detections(
                marker_tracking.EXPECTED_MARKER_IDS,
                moved_centers,
            )
            result = tracker.update(
                corners,
                ids,
                0,
                self.frame_shape,
                now=frame_index / 10,
            )

        self.assertTrue(result.calibrated)

    def test_four_corner_markers_calibrate_without_id_four(self):
        tracker = marker_tracking.MarkerTracker(
            output_size=marker_tracking.output_size_for_area(600, 400)
        )
        for frame_index in range(3):
            corners, ids = self.frame_detections((0, 1, 3, 2))
            result = tracker.update(
                corners,
                ids,
                0,
                self.frame_shape,
                now=frame_index / 10,
            )

        self.assertTrue(result.calibrated)

    def test_perspective_skew_is_rectified_from_four_corner_centers(self):
        skewed_centers = {
            0: (300, 100),
            1: (1700, 220),
            3: (1550, 850),
            2: (200, 900),
            4: (1800, 500),
        }
        output_size = marker_tracking.output_size_for_area(600, 400)
        tracker = marker_tracking.MarkerTracker(output_size=output_size)
        for frame_index in range(3):
            corners, ids = self.frame_detections((0, 1, 3, 2), skewed_centers)
            result = tracker.update(
                corners,
                ids,
                0,
                self.frame_shape,
                now=frame_index / 10,
            )

        self.assertTrue(result.calibrated)
        source_corners = self.outer_marker_corners(skewed_centers)
        mapped = cv2.perspectiveTransform(
            source_corners,
            result.calibration,
        ).reshape(-1, 2)
        np.testing.assert_allclose(
            mapped,
            [
                [0, 0],
                [output_size[0] - 1, 0],
                [output_size[0] - 1, output_size[1] - 1],
                [0, output_size[1] - 1],
            ],
            atol=1,
        )


class CalibrationPersistenceTests(MarkerTrackerTests):
    def saved_payload(self):
        tracker, result = self.make_calibrated_tracker()
        payload = tracker.export_calibration()
        payload["drawing_area"] = [600.0, 400.0]
        payload["resolution"] = [2000, 1000]
        return tracker, result, payload

    def test_export_returns_nothing_without_a_calibration(self):
        tracker = marker_tracking.MarkerTracker(output_size=(1000, 667))
        self.assertIsNone(tracker.export_calibration())

    def test_import_restores_a_locked_calibration_without_observations(self):
        _tracker, result, payload = self.saved_payload()
        restored = marker_tracking.MarkerTracker(output_size=(10, 10))

        self.assertTrue(restored.import_calibration(payload))
        after = restored.update([], None, 0, self.frame_shape, now=0.0)

        self.assertTrue(after.calibrated)
        self.assertTrue(after.calibration_confirmed)
        self.assertFalse(after.calibration_pending)
        np.testing.assert_allclose(after.calibration, result.calibration)
        self.assertEqual(after.output_size, result.output_size)
        self.assertEqual(after.tracked_ids, (4,))

    def test_imported_calibration_still_invalidates_when_a_corner_moves(self):
        restored = marker_tracking.MarkerTracker(output_size=(1000, 667))
        _tracker, _result, payload = self.saved_payload()
        restored.import_calibration(payload)

        moved = dict(self.centers)
        moved[0] = (250, 250)
        after = None
        for frame_index in range(3):
            corners, ids = self.frame_detections(
                marker_tracking.EXPECTED_MARKER_IDS, moved
            )
            after = restored.update(
                corners, ids, 0, self.frame_shape, now=frame_index / 10
            )

        self.assertFalse(after.calibrated)
        self.assertFalse(after.calibration_confirmed)

    def test_malformed_calibration_payloads_are_rejected(self):
        _tracker, _result, payload = self.saved_payload()

        wrong_shape = copy.deepcopy(payload)
        wrong_shape["homography"] = [[1, 0, 0], [0, 1, 0]]
        with self.assertRaises(ValueError):
            marker_tracking.normalize_calibration(wrong_shape)

        non_finite = copy.deepcopy(payload)
        non_finite["homography"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            marker_tracking.normalize_calibration(non_finite)

        singular = copy.deepcopy(payload)
        singular["homography"] = np.zeros((3, 3)).tolist()
        with self.assertRaises(ValueError):
            marker_tracking.normalize_calibration(singular)

        missing_corner = copy.deepcopy(payload)
        del missing_corner["sources"]["0"]
        with self.assertRaises(ValueError):
            marker_tracking.normalize_calibration(missing_corner)

        bad_area = copy.deepcopy(payload)
        bad_area["drawing_area"] = [0, 400]
        with self.assertRaises(ValueError):
            marker_tracking.normalize_calibration(bad_area)

        unsupported_version = copy.deepcopy(payload)
        unsupported_version["version"] = 99
        with self.assertRaises(ValueError):
            marker_tracking.normalize_calibration(unsupported_version)


class DetectorTests(unittest.TestCase):
    def test_merges_ids_across_processing_methods_without_duplicates(self):
        detector = marker_tracking.ArucoMarkerDetector.__new__(
            marker_tracking.ArucoMarkerDetector
        )
        detector.cv2 = cv2
        detector.expected_ids = {0, 1}
        detector._processing_variants = lambda _frame: [0, 1, 2]
        marker0 = MarkerTrackerTests.marker_corners((20, 20))
        larger_marker0 = MarkerTrackerTests.marker_corners((20, 20)) * 2
        marker1 = MarkerTrackerTests.marker_corners((80, 80))
        detector._detect = mock.Mock(
            side_effect=[
                ([marker0], np.asarray([[0]]), []),
                (
                    [larger_marker0, marker1],
                    np.asarray([[0], [1]]),
                    [],
                ),
            ]
        )

        corners, ids, rejected_count = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        np.testing.assert_array_equal(ids, [[0], [1]])
        self.assertEqual(len(corners), 2)
        self.assertEqual(rejected_count, 0)
        self.assertEqual(detector._detect.call_count, 2)

    def test_inflated_threshold_candidate_is_rejected(self):
        detector = marker_tracking.ArucoMarkerDetector.__new__(
            marker_tracking.ArucoMarkerDetector
        )
        detector.cv2 = cv2
        detector.expected_ids = {0, 1}
        detector._processing_variants = lambda _frame: [0, 1]
        compact = MarkerTrackerTests.marker_corners((20, 20))
        inflated = MarkerTrackerTests.marker_corners((20, 20)) * 4
        marker1 = MarkerTrackerTests.marker_corners((80, 80))
        detector._detect = mock.Mock(
            side_effect=[
                ([compact], np.asarray([[0]]), []),
                ([inflated, marker1], np.asarray([[0], [1]]), []),
            ]
        )

        corners, ids, _rejected = detector.detect(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        np.testing.assert_array_equal(ids, [[0], [1]])
        np.testing.assert_allclose(corners[0], compact)

    def test_detection_config_validates_limits(self):
        with self.assertRaises(ValueError):
            marker_tracking.DetectionConfig(max_area_ratio=0.9)
        with self.assertRaises(ValueError):
            marker_tracking.DetectionConfig(min_compactness=1.2)

    def test_subpixel_refinement_and_threshold_tuning_are_set(self):
        params = marker_tracking.configure_aruco_params()

        self.assertEqual(
            params.cornerRefinementMethod,
            cv2.aruco.CORNER_REFINE_SUBPIX,
        )
        self.assertEqual(params.adaptiveThreshWinSizeMin, 3)
        self.assertEqual(params.adaptiveThreshWinSizeMax, 41)
        self.assertEqual(params.adaptiveThreshWinSizeStep, 10)

    def test_detector_parameters_are_configured_with_legacy_opencv_api(self):
        parameters = SimpleNamespace(
            adaptiveThreshWinSizeMin=0,
            adaptiveThreshWinSizeMax=0,
            adaptiveThreshWinSizeStep=0,
            adaptiveThreshConstant=0,
            minMarkerPerimeterRate=0.0,
            maxMarkerPerimeterRate=0.0,
            polygonalApproxAccuracyRate=0.0,
            errorCorrectionRate=0.0,
            cornerRefinementMethod=0,
            cornerRefinementWinSize=0,
            cornerRefinementMaxIterations=0,
            cornerRefinementMinAccuracy=0.0,
        )
        aruco = SimpleNamespace(
            DetectorParameters_create=lambda: parameters,
            CORNER_REFINE_SUBPIX=1,
        )
        legacy_cv2 = SimpleNamespace(aruco=aruco)

        configured = marker_tracking.configure_aruco_params(legacy_cv2)

        self.assertIs(configured, parameters)
        self.assertEqual(configured.cornerRefinementMethod, 1)
        self.assertEqual(configured.adaptiveThreshWinSizeMax, 41)

    def test_detects_required_markers_under_normal_and_dim_lighting(self):
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        canvas = np.full((800, 800), 210, dtype=np.uint8)
        generate_marker = getattr(aruco, "generateImageMarker", None)
        if generate_marker is None:
            generate_marker = aruco.drawMarker

        positions = {
            0: (100, 100),
            1: (700, 100),
            3: (700, 700),
            2: (100, 700),
            4: (400, 400),
        }
        marker_size = 100
        half_size = marker_size // 2
        for marker_id, (center_x, center_y) in positions.items():
            marker = generate_marker(dictionary, marker_id, marker_size)
            canvas[
                center_y - half_size:center_y + half_size,
                center_x - half_size:center_x + half_size,
            ] = marker

        detector = marker_tracking.ArucoMarkerDetector()
        lighting_frames = (
            cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(
                np.clip(canvas.astype(np.float32) * 0.65 + 30, 0, 255).astype(
                    np.uint8
                ),
                cv2.COLOR_GRAY2BGR,
            ),
        )
        for frame in lighting_frames:
            _corners, ids, _rejected = detector.detect(frame)
            self.assertEqual(set(np.asarray(ids).reshape(-1)), set(positions))


if __name__ == "__main__":
    unittest.main()
