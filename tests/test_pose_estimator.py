import math
import unittest
from types import SimpleNamespace

import numpy as np

import pose_estimator


def rotated_square(center=(0.0, 0.0), half_size=20.0, angle_degrees=0.0):
    """Return ArUco-ordered corners of a square rotated by angle_degrees."""
    radians = math.radians(angle_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    local = np.asarray(
        [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.asarray(center, dtype=np.float64)


def tracker_result(
    corners,
    *,
    calibration=None,
    live=None,
    reliability=1.0,
    calibrated=True,
):
    centre = np.asarray(corners, dtype=np.float64).mean(axis=0)
    return SimpleNamespace(
        calibrated=calibrated,
        calibration=np.eye(3) if calibration is None else calibration,
        live={4: centre} if live is None else live,
        live_corners={4: corners} if live is None else {},
        reliability={4: reliability},
        output_size=(1001, 501),
    )


def marker_track(corners, *, is_live=False, age=0.0):
    centre = np.asarray(corners, dtype=np.float64).reshape(-1, 2).mean(axis=0)
    return SimpleNamespace(
        marker_id=4,
        center=centre,
        corners=corners,
        velocity=np.zeros(2),
        last_seen=0.0,
        age=age,
        is_live=is_live,
    )


def result_with_track(corners, *, is_live=False, age=0.0):
    result = tracker_result(corners, live={})
    result.tracks = {4: marker_track(corners, is_live=is_live, age=age)}
    return result


class AngleHelperTests(unittest.TestCase):
    def test_wrap_angle_handles_half_turns_and_wrap_around(self):
        self.assertAlmostEqual(pose_estimator.wrap_angle(0, 180), 0)
        self.assertAlmostEqual(pose_estimator.wrap_angle(179, 180), -1)
        self.assertAlmostEqual(pose_estimator.wrap_angle(-179, 180), 1)
        self.assertAlmostEqual(pose_estimator.wrap_angle(181, 180), 1)
        self.assertAlmostEqual(pose_estimator.wrap_angle(90, 180), -90)
        self.assertAlmostEqual(pose_estimator.wrap_angle(-270, 360), 90)

    def test_angle_error_uses_shortest_rotation(self):
        self.assertAlmostEqual(pose_estimator.angle_error_degrees(10), 10)
        self.assertAlmostEqual(
            pose_estimator.angle_error_degrees(179, -179, period=360),
            -2,
        )
        self.assertAlmostEqual(
            pose_estimator.angle_error_degrees(-1, 1, period=180),
            -2,
        )

    def test_marker_axis_angle_tracks_corner_order(self):
        for angle in (0.0, 10.0, -10.0, 30.0, 190.0):
            corners = rotated_square(angle_degrees=angle)
            measured = pose_estimator.marker_axis_angle_degrees(corners, period=180)
            self.assertAlmostEqual(measured, pose_estimator.wrap_angle(angle, 180))

    def test_degenerate_corners_are_rejected(self):
        with self.assertRaises(ValueError):
            pose_estimator.marker_axis_angle_degrees(
                np.zeros((4, 2)),
                period=180,
            )
        with self.assertRaises(ValueError):
            pose_estimator.marker_axis_angle_degrees(
                np.zeros((3, 2)),
                period=180,
            )


class CircularAngleSmootherTests(unittest.TestCase):
    def test_smoothing_is_circular_across_the_wrap_point(self):
        smoother = pose_estimator.CircularAngleSmoother(window=4, period_degrees=360)
        smoother.add(179)
        value = smoother.add(-179)
        self.assertAlmostEqual(abs(value), 180.0, places=9)

    def test_square_marker_half_turn_is_horizontal(self):
        smoother = pose_estimator.CircularAngleSmoother(window=4, period_degrees=180)
        smoother.add(179)
        value = smoother.add(-179)
        self.assertAlmostEqual(value, 0.0, places=9)

    def test_reset_and_empty_value(self):
        smoother = pose_estimator.CircularAngleSmoother(window=2, period_degrees=180)
        self.assertIsNone(smoother.value)
        smoother.add(20)
        self.assertAlmostEqual(smoother.value, 20)
        smoother.reset()
        self.assertIsNone(smoother.value)

    def test_window_must_be_positive(self):
        with self.assertRaises(ValueError):
            pose_estimator.CircularAngleSmoother(window=0)


class PoseEstimatorTests(unittest.TestCase):
    def make_estimator(self, **overrides):
        settings = dict(
            drawing_width_mm=100.0,
            drawing_height_mm=50.0,
            output_size=(1001, 501),
            angle_smoothing_window=1,
        )
        settings.update(overrides)
        return pose_estimator.PoseEstimator(
            pose_estimator.PoseEstimatorConfig(**settings)
        )

    def test_pose_is_reported_in_pixels_millimetres_and_degrees(self):
        estimator = self.make_estimator()
        corners = rotated_square((500.0, 250.0), 40.0, 10.0)

        pose = estimator.update(tracker_result(corners), now=0.0)

        self.assertTrue(pose.valid)
        self.assertAlmostEqual(pose.x_px, 500.0)
        self.assertAlmostEqual(pose.y_px, 250.0)
        self.assertAlmostEqual(pose.x_mm, 50.0)
        self.assertAlmostEqual(pose.y_mm, 25.0)
        self.assertAlmostEqual(pose.angle_deg, 10.0)
        self.assertAlmostEqual(pose.raw_angle_deg, 10.0)
        self.assertTrue(pose.inside_drawing_area)
        self.assertEqual(pose.corners_px.shape, (4, 2))

    def test_calibration_homography_is_applied_to_corners(self):
        estimator = self.make_estimator()
        homography = np.asarray(
            [[1.0, 0.0, 400.0], [0.0, 1.0, 150.0], [0.0, 0.0, 1.0]]
        )
        corners = rotated_square((100.0, 100.0), 20.0, 5.0)

        pose = estimator.update(
            tracker_result(corners, calibration=homography),
            now=0.0,
        )

        self.assertTrue(pose.valid)
        self.assertAlmostEqual(pose.x_px, 500.0)
        self.assertAlmostEqual(pose.y_px, 250.0)
        self.assertAlmostEqual(pose.angle_deg, 5.0)

    def test_angle_smoothing_reduces_noise_and_resets_after_a_gap(self):
        estimator = self.make_estimator(angle_smoothing_window=3)
        first = estimator.update(
            tracker_result(rotated_square((500, 250), 40, 0)),
            now=0.0,
        )
        second = estimator.update(
            tracker_result(rotated_square((500, 250), 40, 10)),
            now=0.1,
        )
        resumed = estimator.update(
            tracker_result(rotated_square((500, 250), 40, 20)),
            now=2.0,
        )

        self.assertAlmostEqual(first.angle_deg, 0.0)
        self.assertAlmostEqual(second.angle_deg, 5.0)
        self.assertAlmostEqual(resumed.angle_deg, 20.0)

    def test_pose_outside_the_drawing_area_is_flagged(self):
        estimator = self.make_estimator()
        pose = estimator.update(
            tracker_result(rotated_square((1100, 250), 40, 0)),
            now=0.0,
        )

        self.assertTrue(pose.valid)
        self.assertFalse(pose.inside_drawing_area)

    def test_missing_or_unreliable_detections_are_rejected(self):
        estimator = self.make_estimator(min_reliability=0.6)
        corners = rotated_square((500, 250), 40, 0)

        uncalibrated = estimator.update(
            tracker_result(corners, calibrated=False, calibration=None),
            now=0.0,
        )
        missing = estimator.update(
            tracker_result(corners, live={}),
            now=0.0,
        )
        unreliable = estimator.update(
            tracker_result(corners, reliability=0.2),
            now=0.0,
        )

        self.assertFalse(uncalibrated.valid)
        self.assertFalse(missing.valid)
        self.assertFalse(unreliable.valid)
        self.assertIn("unreliable", unreliable.reason)

    def test_predicted_pose_is_flagged_within_the_grace_period(self):
        estimator = self.make_estimator()
        corners = rotated_square((500, 250), 40, 10)

        pose = estimator.update(
            result_with_track(corners, is_live=False, age=0.2),
            now=0.2,
        )

        self.assertTrue(pose.valid)
        self.assertTrue(pose.predicted)
        self.assertAlmostEqual(pose.prediction_age_seconds, 0.2)
        self.assertAlmostEqual(pose.x_px, 500.0)
        self.assertAlmostEqual(pose.angle_deg, 10.0)

    def test_expired_or_disabled_predictions_are_rejected(self):
        corners = rotated_square((500, 250), 40, 0)
        estimator = self.make_estimator()

        expired = estimator.update(
            result_with_track(corners, age=0.5),
            now=0.5,
        )
        disabled = self.make_estimator(allow_predicted_pose=False).update(
            result_with_track(corners, age=0.1),
            now=0.1,
        )

        self.assertFalse(expired.valid)
        self.assertIn("expired", expired.reason)
        self.assertFalse(disabled.valid)
        self.assertIn("not detected", disabled.reason)

    def test_live_track_takes_priority_over_prediction(self):
        estimator = self.make_estimator()
        corners = rotated_square((500, 250), 40, 0)

        pose = estimator.update(
            result_with_track(corners, is_live=True, age=0.5),
            now=0.5,
        )

        self.assertTrue(pose.valid)
        self.assertFalse(pose.predicted)
        self.assertEqual(pose.prediction_age_seconds, 0.0)

    def test_degenerate_or_too_small_markers_are_rejected(self):
        estimator = self.make_estimator(min_marker_area_px=50.0)
        degenerate = estimator.update(
            tracker_result(np.zeros((4, 2))),
            now=0.0,
        )
        too_small = estimator.update(
            tracker_result(rotated_square((500, 250), 2.0, 0)),
            now=0.0,
        )

        self.assertFalse(degenerate.valid)
        self.assertFalse(too_small.valid)

    def test_non_finite_homography_is_rejected(self):
        estimator = self.make_estimator()
        bad_homography = np.eye(3)
        bad_homography[0, 0] = np.nan

        pose = estimator.update(
            tracker_result(
                rotated_square((500, 250), 40, 0),
                calibration=bad_homography,
            ),
            now=0.0,
        )

        self.assertFalse(pose.valid)

    def test_config_validates_physical_and_filter_settings(self):
        with self.assertRaises(ValueError):
            pose_estimator.PoseEstimatorConfig(
                drawing_width_mm=0,
                drawing_height_mm=50,
                output_size=(100, 50),
            )
        with self.assertRaises(ValueError):
            pose_estimator.PoseEstimatorConfig(
                drawing_width_mm=100,
                drawing_height_mm=50,
                output_size=(1, 50),
            )
        with self.assertRaises(ValueError):
            pose_estimator.PoseEstimatorConfig(
                drawing_width_mm=100,
                drawing_height_mm=50,
                output_size=(100, 50),
                min_reliability=1.5,
            )
        with self.assertRaises(ValueError):
            pose_estimator.PoseEstimatorConfig(
                drawing_width_mm=100,
                drawing_height_mm=50,
                output_size=(100, 50),
                angle_period_degrees=0,
            )


if __name__ == "__main__":
    unittest.main()
