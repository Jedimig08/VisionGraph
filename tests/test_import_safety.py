import importlib
import threading
import unittest
from unittest import mock

import camera_select
import cv_stream
import discovery


class ImportSafetyTests(unittest.TestCase):
    def test_discovery_import_does_not_prompt_or_wait(self):
        with (
            mock.patch("builtins.input", side_effect=AssertionError("prompted")),
            mock.patch.object(
                threading.Event,
                "wait",
                side_effect=AssertionError("waited for discovery"),
            ),
        ):
            importlib.reload(discovery)

    def test_stream_import_does_not_select_camera_or_open_window(self):
        with (
            mock.patch.object(camera_select, "select_camera", side_effect=AssertionError),
            mock.patch.object(cv_stream.cv2, "namedWindow", side_effect=AssertionError),
        ):
            importlib.reload(cv_stream)


if __name__ == "__main__":
    unittest.main()