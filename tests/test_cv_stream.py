import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

import cv_stream


class MarkerIdMergeTests(unittest.TestCase):
    def test_merge_normalizes_mixed_id_shapes(self):
        base_ids = np.array([[3], [8]], dtype=np.int32)
        additional_ids = np.array([13], dtype=np.int64)

        merged = cv_stream.merge_marker_ids(base_ids, additional_ids)

        self.assertEqual(merged.shape, (3, 1))
        self.assertEqual(merged.dtype, np.int32)
        np.testing.assert_array_equal(merged, [[3], [8], [13]])

    def test_drawing_dimensions_are_required_and_positive(self):
        with mock.patch("builtins.input", side_effect=["0", "600", "400"]):
            self.assertEqual(cv_stream.choose_drawing_dimensions(), (600, 400))

    def test_full_frame_display_skips_calibration_warp(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        result = SimpleNamespace(calibration=np.eye(3), output_size=(4, 2))
        with (
            mock.patch.object(cv_stream, "draw_tracking_status"),
            mock.patch.object(cv_stream.cv2, "warpPerspective") as warp,
        ):
            display_frame = cv_stream.prepare_display_frame(
                frame,
                result,
                show_full_frame=True,
            )

        warp.assert_not_called()
        self.assertEqual(display_frame.shape, frame.shape)
        self.assertIsNot(display_frame, frame)

    def test_corrected_display_keeps_calibration_warp(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        result = SimpleNamespace(calibration=np.eye(3), output_size=(4, 2))
        corrected = np.zeros((2, 4, 3), dtype=np.uint8)
        with (
            mock.patch.object(cv_stream, "draw_tracking_status"),
            mock.patch.object(
                cv_stream.cv2,
                "warpPerspective",
                return_value=corrected,
            ) as warp,
        ):
            display_frame = cv_stream.prepare_display_frame(
                frame,
                result,
                show_full_frame=False,
            )

        warp.assert_called_once()
        self.assertIs(display_frame, corrected)

    def test_corrected_frame_is_warped_before_tracking_overlays(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        result = SimpleNamespace(calibration=np.eye(3), output_size=(8, 6))
        corrected = np.zeros((6, 8, 3), dtype=np.uint8)
        with (
            mock.patch.object(
                cv_stream.cv2,
                "warpPerspective",
                return_value=corrected,
            ) as warp,
            mock.patch.object(cv_stream, "draw_tracking_status") as draw,
        ):
            display_frame = cv_stream.prepare_display_frame(
                frame,
                result,
                show_full_frame=False,
            )

        self.assertIs(display_frame, corrected)
        warp.assert_called_once_with(frame, result.calibration, result.output_size)
        draw.assert_called_once_with(corrected, result, result.calibration)

    def test_warp_perspective_changes_source_image_coordinates(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[2, 3] = (255, 255, 255)
        homography = np.asarray(
            [[1, 0, 2], [0, 1, 1], [0, 0, 1]],
            dtype=np.float32,
        )
        result = SimpleNamespace(
            calibration=homography,
            output_size=(12, 12),
        )
        with mock.patch.object(cv_stream, "draw_tracking_status"):
            corrected = cv_stream.prepare_display_frame(
                frame,
                result,
                show_full_frame=False,
            )

        np.testing.assert_array_equal(corrected[3, 5], (255, 255, 255))
        self.assertEqual(corrected.shape, (12, 12, 3))

    def test_confirm_key_reports_tracker_state(self):
        tracker = mock.Mock()
        tracker.confirm_calibration.return_value = True
        self.assertIn("locked", cv_stream.confirm_calibration_key(tracker))

        tracker.confirm_calibration.return_value = False
        tracker.calibration_confirmed = True
        self.assertIn("already locked", cv_stream.confirm_calibration_key(tracker))

        tracker.calibration_confirmed = False
        self.assertIn("not ready", cv_stream.confirm_calibration_key(tracker))

    def test_locked_display_draws_only_the_carriage_marker(self):
        corners = np.zeros((1, 4, 2), dtype=np.float32)
        result = SimpleNamespace(
            live={0: np.array([10.0, 10.0]), 4: np.array([100.0, 100.0])},
            live_corners={0: corners, 4: corners},
            remembered={},
            tracks={},
            missing=(),
            reliability={4: 1.0},
            rejected_count=0,
            calibrated=True,
            calibration_observations={4: 3},
            calibration_confirmed=True,
            calibration_pending=False,
            tracked_ids=(4,),
        )
        frame = np.zeros((800, 1200, 3), dtype=np.uint8)

        with mock.patch.object(cv_stream.cv2.aruco, "drawDetectedMarkers") as draw:
            cv_stream.draw_tracking_status(frame, result)

        np.testing.assert_array_equal(
            np.asarray(draw.call_args[0][2]).reshape(-1),
            [4],
        )

    def test_pose_status_is_drawn_when_a_pose_is_supplied(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        result = SimpleNamespace(calibration=None, output_size=(4, 2))
        pose = SimpleNamespace(valid=False, reason="marker not detected live")
        with (
            mock.patch.object(cv_stream, "draw_tracking_status"),
            mock.patch.object(cv_stream, "draw_pose_status") as draw_pose,
        ):
            display_frame = cv_stream.prepare_display_frame(
                frame,
                result,
                show_full_frame=True,
                pose=pose,
            )

        draw_pose.assert_called_once_with(display_frame, pose)

    def test_open_stream_retries_until_camera_is_available(self):
        unavailable_stream = mock.Mock()
        unavailable_stream.isOpened.return_value = False
        available_stream = mock.Mock()
        available_stream.isOpened.return_value = True
        with (
            mock.patch.object(
                cv_stream.cv2,
                "VideoCapture",
                side_effect=[unavailable_stream, available_stream],
            ) as video_capture,
            mock.patch.object(cv_stream.time, "sleep") as sleep,
        ):
            result = cv_stream.open_stream("http://camera/stream")

        self.assertIs(result, available_stream)
        unavailable_stream.release.assert_called_once_with()
        sleep.assert_called_once_with(cv_stream.STREAM_RETRY_SECONDS)
        self.assertEqual(video_capture.call_count, 2)


if __name__ == "__main__":
    unittest.main()