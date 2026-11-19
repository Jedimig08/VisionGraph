import unittest
import urllib.error
from unittest import mock

import camera_select


class CameraSelectTests(unittest.TestCase):
    def test_get_cameras_selects_hub_before_requesting_cameras(self):
        with (
            mock.patch.object(
                camera_select.discovery,
                "choose_hub",
            ) as choose_hub,
            mock.patch.object(
                camera_select.discovery,
                "app_url",
                side_effect=lambda: (
                    "http://hub.local"
                    if choose_hub.called
                    else "no url"
                ),
            ),
            mock.patch.object(
                camera_select.urllib.request,
                "urlopen",
                return_value=mock.mock_open(read_data="[]").return_value,
            ) as urlopen,
            mock.patch.object(camera_select.discovery, "remember_hub_url"),
        ):
            self.assertEqual(camera_select.get_cameras(), [])

        choose_hub.assert_called_once_with()
        urlopen.assert_called_once_with(
            "http://hub.local/api/cameras",
            timeout=camera_select.HUB_HTTP_TIMEOUT_SECONDS,
        )

    def test_unreachable_saved_hub_rediscovery_retries_until_reachable(self):
        unreachable = urllib.error.URLError(OSError(113, "No route to host"))
        response = mock.mock_open(read_data="[]").return_value
        with (
            mock.patch.object(camera_select.discovery, "choose_hub"),
            mock.patch.object(camera_select.discovery, "app_url", return_value="http://hub.local"),
            mock.patch.object(
                camera_select.discovery,
                "saved_url_selected",
                True,
            ),
            mock.patch.object(
                camera_select.urllib.request,
                "urlopen",
                side_effect=[unreachable, unreachable, response],
            ) as urlopen,
            mock.patch.object(camera_select.discovery, "discover") as discover,
            mock.patch.object(camera_select.discovery, "remember_hub_url"),
            mock.patch.object(camera_select.time, "sleep") as sleep,
        ):
            self.assertEqual(camera_select.get_cameras(), [])

        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(discover.call_count, 2)
        sleep.assert_called_once_with(camera_select.HUB_RETRY_SECONDS)

    def test_unreachable_discovered_hub_is_retried(self):
        unreachable = urllib.error.URLError(OSError(113, "No route to host"))
        response = mock.mock_open(read_data="[]").return_value
        with (
            mock.patch.object(camera_select.discovery, "choose_hub"),
            mock.patch.object(camera_select.discovery, "app_url", return_value="http://hub.local"),
            mock.patch.object(
                camera_select.discovery,
                "saved_url_selected",
                False,
            ),
            mock.patch.object(
                camera_select.urllib.request,
                "urlopen",
                side_effect=[unreachable, response],
            ) as urlopen,
            mock.patch.object(camera_select.discovery, "discover") as discover,
            mock.patch.object(camera_select.discovery, "remember_hub_url"),
            mock.patch.object(camera_select.time, "sleep") as sleep,
        ):
            self.assertEqual(camera_select.get_cameras(), [])

        self.assertEqual(urlopen.call_count, 2)
        discover.assert_called_once_with()
        sleep.assert_not_called()

    def test_select_camera_prefers_back_wide_camera(self):
        cameras = [
            {"id": "front", "facing": "Front", "type": "Wide"},
            {"id": "rear", "facing": "Back", "type": "Wide"},
        ]

        self.assertEqual(camera_select.select_camera(cameras)["id"], "rear")

    def test_choose_starting_camera_defaults_to_recommended_camera(self):
        cameras = [
            {"id": "front", "facing": "Front", "type": "Wide"},
            {"id": "rear", "facing": "Back", "type": "Wide"},
        ]

        with mock.patch("builtins.input", return_value=""):
            selected = camera_select.choose_starting_camera(cameras)

        self.assertEqual(selected["id"], "rear")

    def test_choose_starting_camera_returns_user_selection(self):
        cameras = [
            {"id": "front", "facing": "Front", "type": "Wide"},
            {"id": "rear", "facing": "Back", "type": "Wide"},
        ]

        with mock.patch("builtins.input", return_value="1"):
            selected = camera_select.choose_starting_camera(cameras)

        self.assertEqual(selected["id"], "front")

    def test_resolution_steps_are_unique_and_largest_first(self):
        camera = {
            "supportedResolutions": [
                {"width": 640, "height": 480},
                {"width": 1280, "height": 720},
                {"width": 640, "height": 480},
            ]
        }

        self.assertEqual(
            camera_select.choose_resolution_steps(camera),
            [(1280, 720), (640, 480)],
        )


if __name__ == "__main__":
    unittest.main()