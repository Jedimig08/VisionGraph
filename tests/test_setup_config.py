import unittest
from unittest import mock

import camera_select
import setup_config


class SetupConfigTests(unittest.TestCase):
    def test_validate_dimensions(self):
        self.assertEqual(setup_config.validate_dimensions(600, 400), (600.0, 400.0))
        for width, height in ((0, 400), (600, -1), ("x", 400)):
            with self.assertRaises((ValueError, TypeError)):
                setup_config.validate_dimensions(width, height)

    def test_resolution_helpers(self):
        self.assertIsNone(setup_config.parse_resolution(""))
        self.assertEqual(setup_config.parse_resolution("640x480"), (640, 480))
        self.assertEqual(setup_config.format_resolution((640, 480)), "640x480")
        with self.assertRaises(ValueError):
            setup_config.parse_resolution("abc")

    def test_select_camera_by_id_and_auto(self):
        cameras = [
            {"id": "0", "facing": "Back", "type": "Wide"},
            {"id": "2", "facing": "Back", "type": "UltraWide"},
        ]
        self.assertEqual(setup_config.select_camera(cameras, "2")["id"], "2")
        with mock.patch.object(camera_select, "select_camera", return_value=cameras[1]):
            self.assertEqual(setup_config.select_camera(cameras)["id"], "2")
        with self.assertRaises(RuntimeError):
            setup_config.select_camera([], "0")
        with self.assertRaises(RuntimeError):
            setup_config.select_camera(cameras, "9")

    def test_select_resolution_uses_supported_override(self):
        camera = {
            "supportedResolutions": [
                {"width": 640, "height": 480},
                {"width": 1280, "height": 720},
            ]
        }
        self.assertEqual(
            setup_config.select_resolution(camera, (1280, 720)), (1280, 720)
        )
        with mock.patch.object(
            camera_select, "choose_resolution", return_value=(640, 480)
        ) as choose:
            self.assertEqual(
                setup_config.select_resolution(camera, (999, 999)), (640, 480)
            )
            choose.assert_called_once_with(camera, 999, 999)

    def test_output_size_uses_marker_tracking(self):
        self.assertEqual(setup_config.output_size_for_area(600, 400), (1000, 667))


if __name__ == "__main__":
    unittest.main()
