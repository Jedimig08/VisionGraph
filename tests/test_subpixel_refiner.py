import math
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

import pose_estimator
import subpixel_refiner


def rotated_square(center=(0.0, 0.0), half_size=20.0, angle_degrees=0.0):
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


def render_filled_square(
    shape,
    center,
    half_size,
    angle_degrees,
    supersample=8,
    border_padding=255,
):
    height, width = shape
    high_resolution = np.full(
        (height * supersample, width * supersample),
        border_padding,
        dtype=np.uint8,
    )
    corners = rotated_square(center, half_size, angle_degrees)
    high_resolution_corners = (
        corners * supersample + (supersample - 1) / 2.0
    )
    cv2.fillPoly(
        high_resolution,
        [np.round(high_resolution_corners).astype(np.int32)],
        0,
    )
    return cv2.resize(
        high_resolution,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def corner_errors(measured, truth):
    return np.linalg.norm(np.asarray(measured) - np.asarray(truth), axis=1)


class SubpixelRefinerTests(unittest.TestCase):
    def test_refines_corners_angle_and_centre_beyond_corner_jitter(self):
        centre = (200.3, 190.7)
        angle = 7.3
        truth = rotated_square(centre, 55.2, angle)
        frame = render_filled_square((400, 400), centre, 55.2, angle)
        jitter = np.asarray(
            [[0.6, -0.4], [-0.7, 0.5], [0.4, 0.7], [-0.5, -0.6]],
            dtype=np.float64,
        )
        initial = truth + jitter

        result = subpixel_refiner.refine_marker_corners(frame, initial)

        self.assertTrue(result.success, result.reason)
        self.assertLess(result.rms_error_px, 0.6)
        initial_error = float(np.sqrt(np.mean(np.square(corner_errors(initial, truth)))))
        refined_error = float(
            np.sqrt(np.mean(np.square(corner_errors(result.corners, truth))))
        )
        self.assertLess(refined_error, initial_error * 0.6)
        self.assertAlmostEqual(
            pose_estimator.marker_axis_angle_degrees(result.corners, 180),
            angle,
            delta=0.3,
        )
        centre_error = np.linalg.norm(result.corners.mean(axis=0) - np.asarray(centre))
        self.assertLess(centre_error, 0.35)

    def test_fractional_centre_shift_is_preserved(self):
        first_centre = (200.0, 200.0)
        second_centre = (200.5, 199.5)
        first = subpixel_refiner.refine_marker_corners(
            render_filled_square((400, 400), first_centre, 55.2, 7.3),
            rotated_square(first_centre, 55.2, 7.3),
        )
        second = subpixel_refiner.refine_marker_corners(
            render_filled_square((400, 400), second_centre, 55.2, 7.3),
            rotated_square(second_centre, 55.2, 7.3),
        )

        self.assertTrue(first.success, first.reason)
        self.assertTrue(second.success, second.reason)
        measured_shift = second.corners.mean(axis=0) - first.corners.mean(axis=0)
        np.testing.assert_allclose(measured_shift, [0.5, -0.5], atol=0.1)

    def test_bgr_frames_are_supported(self):
        centre = (200.0, 200.0)
        truth = rotated_square(centre, 55.2, 0.0)
        gray = render_filled_square((400, 400), centre, 55.2, 0.0)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result = subpixel_refiner.refine_marker_corners(bgr, truth)

        self.assertTrue(result.success, result.reason)

    def test_featureless_frame_falls_back_with_a_reason(self):
        centre = (100.0, 100.0)
        result = subpixel_refiner.refine_marker_corners(
            np.full((200, 200), 255, dtype=np.uint8),
            rotated_square(centre, 40.0, 0.0),
        )

        self.assertFalse(result.success)
        self.assertTrue(result.reason)

    def test_disabled_and_invalid_inputs_are_rejected(self):
        frame = np.full((200, 200), 255, dtype=np.uint8)
        corners = rotated_square((100.0, 100.0), 40.0, 0.0)

        disabled = subpixel_refiner.refine_marker_corners(
            frame,
            corners,
            subpixel_refiner.SubpixelRefinerConfig(enabled=False),
        )
        invalid = subpixel_refiner.refine_marker_corners(
            frame,
            np.zeros((3, 2)),
        )

        self.assertFalse(disabled.success)
        self.assertIn("disabled", disabled.reason)
        self.assertFalse(invalid.success)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            subpixel_refiner.SubpixelRefinerConfig(samples_per_edge=2)
        with self.assertRaises(ValueError):
            subpixel_refiner.SubpixelRefinerConfig(search_radius_px=0)
        with self.assertRaises(ValueError):
            subpixel_refiner.SubpixelRefinerConfig(min_inliers_per_edge=30)
        with self.assertRaises(ValueError):
            subpixel_refiner.SubpixelRefinerConfig(edge_margin=0.5)


class PoseEstimatorRefinementIntegrationTests(unittest.TestCase):
    def make_estimator(self):
        return pose_estimator.PoseEstimator(
            pose_estimator.PoseEstimatorConfig(
                drawing_width_mm=400.0,
                drawing_height_mm=400.0,
                output_size=(401, 401),
                angle_smoothing_window=1,
                min_marker_area_px=100.0,
            )
        )

    def tracker_result(self, corners):
        return SimpleNamespace(
            calibrated=True,
            calibration=np.eye(3),
            live={4: np.asarray(corners, dtype=np.float64).mean(axis=0)},
            live_corners={4: np.asarray(corners, dtype=np.float64)},
            reliability={4: 1.0},
            output_size=(401, 401),
        )

    def test_pose_estimator_uses_refined_corners_when_frame_is_supplied(self):
        centre = (200.3, 190.7)
        angle = 7.3
        truth = rotated_square(centre, 55.2, angle)
        frame = render_filled_square((400, 400), centre, 55.2, angle)
        initial = truth + np.asarray(
            [[0.6, -0.4], [-0.7, 0.5], [0.4, 0.7], [-0.5, -0.6]],
            dtype=np.float64,
        )
        result = self.tracker_result(initial)

        raw = self.make_estimator().update(result, frame=None, now=0.0)
        refined = self.make_estimator().update(result, frame=frame, now=0.0)

        self.assertTrue(raw.valid)
        self.assertFalse(raw.refined)
        self.assertEqual(raw.refinement_reason, "no frame supplied")
        self.assertTrue(refined.valid)
        self.assertTrue(refined.refined)
        self.assertIsNotNone(refined.fit_error_px)
        self.assertLess(refined.fit_error_px, 0.6)
        self.assertLess(
            abs(refined.angle_deg - angle),
            abs(raw.angle_deg - angle),
        )
        self.assertAlmostEqual(refined.angle_deg, angle, delta=0.3)
        self.assertAlmostEqual(refined.x_mm, centre[0], delta=0.35)
        self.assertAlmostEqual(refined.y_mm, centre[1], delta=0.35)

    def test_refinement_failure_falls_back_to_aruco_corners(self):
        truth = rotated_square((200.0, 200.0), 55.2, 4.0)
        result = self.tracker_result(truth)

        pose = self.make_estimator().update(
            result,
            frame=np.full((400, 400), 255, dtype=np.uint8),
            now=0.0,
        )

        self.assertTrue(pose.valid)
        self.assertFalse(pose.refined)
        self.assertTrue(pose.refinement_reason)
        self.assertAlmostEqual(pose.angle_deg, 4.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
