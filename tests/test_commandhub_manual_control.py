import unittest
from unittest import mock

import commandhub_manual_control as control


class FakeScreen:
    def __init__(self, keys, size=(24, 80)):
        self.keys = iter(keys)
        self.size = size
        self.lines = {}

    def erase(self):
        self.lines.clear()

    def addnstr(self, row, col, text, limit):
        self.lines[row] = text[:limit]

    def getmaxyx(self):
        return self.size

    def refresh(self):
        pass

    def keypad(self, enabled):
        pass

    def getch(self):
        return next(self.keys)


class CommandHubManualControlTests(unittest.TestCase):
    def setUp(self):
        self.speeds = [
            {"up": 180, "down": 100},
            {"up": 150, "down": 90},
        ]

    def test_stop_commands_zero_every_pin(self):
        self.assertEqual(
            control.stop_commands(),
            ["P 32 0", "P 33 0", "P 25 0", "P 26 0"],
        )

    def test_drive_commands_pwm_one_pin_and_zero_the_other(self):
        self.assertEqual(
            control.drive_commands(0, "up", self.speeds),
            ["P 32 180", "P 33 0"],
        )
        self.assertEqual(
            control.drive_commands(1, "down", self.speeds),
            ["P 25 0", "P 26 90"],
        )

    def test_send_pwm_posts_raw_commands_to_bluetooth_send(self):
        hub = control.CommandHub("http://phone.local:8080/")
        response = mock.mock_open(read_data=b"OK").return_value

        with mock.patch.object(control.urllib.request, "urlopen", return_value=response) as urlopen:
            hub.send_pwm(["P 32 180", "P 33 0"])

        self.assertEqual(urlopen.call_count, 2)
        first, second = (call.args[0] for call in urlopen.call_args_list)
        self.assertEqual(first.full_url, "http://phone.local:8080/bluetooth-send")
        self.assertEqual(first.data, b"P 32 180")
        self.assertEqual(first.get_method(), "POST")
        self.assertEqual(second.data, b"P 33 0")

    def test_connect_posts_mac_address(self):
        hub = control.CommandHub("http://phone.local:8080")
        response = mock.mock_open(read_data=b"Connected").return_value

        with mock.patch.object(control.urllib.request, "urlopen", return_value=response) as urlopen:
            result = hub.connect("AA:BB:CC:DD:EE:FF")

        self.assertEqual(result, "Connected")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://phone.local:8080/bluetooth-connect")
        self.assertEqual(request.data, b"AA:BB:CC:DD:EE:FF")

    def test_transmit_reports_error_without_raising(self):
        hub = mock.Mock()
        hub.send_pwm.side_effect = OSError("no route")
        state = control.ControlState(self.speeds, "http://phone.local:8080", "Connected to ESP32")

        self.assertFalse(control.transmit(state, hub, ["P 32 0"], "stopped"))
        self.assertIn("no route", state.error)
        self.assertIn("FAILED", state.message)

    def test_keys_drive_selected_motor_and_quit_stops_all(self):
        hub = mock.Mock()
        state = control.ControlState(self.speeds, "http://phone.local:8080", "Connected to ESP32")
        keys = [
            control.curses.KEY_UP,
            control.curses.KEY_RIGHT,
            control.curses.KEY_DOWN,
            ord(" "),
            ord("q"),
        ]

        with mock.patch.object(control.curses, "curs_set"):
            control.control_loop(FakeScreen(keys), hub, state)

        self.assertEqual(
            [call.args[0] for call in hub.send_pwm.call_args_list],
            [
                ["P 32 180", "P 33 0"],
                ["P 25 0", "P 26 90"],
                control.stop_commands(),
                control.stop_commands(),
            ],
        )



    def test_connect_bluetooth_requires_success_response(self):
        hub = mock.Mock()
        hub.connect.return_value = "Connected to ESP32"
        self.assertEqual(
            control.connect_bluetooth(hub, "AA:BB:CC:DD:EE:FF"),
            "Connected to ESP32",
        )

        hub.connect.return_value = "Connection failed: Bluetooth is disabled"
        with self.assertRaises(RuntimeError):
            control.connect_bluetooth(hub, "AA:BB:CC:DD:EE:FF")

    def test_failed_connection_never_starts_key_controls(self):
        with (
            mock.patch.object(control.discovery, "choose_hub"),
            mock.patch.object(control.discovery, "app_url", return_value="http://phone.local:8080"),
            mock.patch.object(control, "prompt_speeds") as prompt_speeds,
            mock.patch.object(control, "select_device") as select_device,
            mock.patch.object(
                control,
                "connect_bluetooth",
                side_effect=RuntimeError("Connection failed: Device not found"),
            ),
            mock.patch.object(control.CommandHub, "send_pwm") as send_pwm,
            mock.patch.object(control.curses, "wrapper") as wrapper,
        ):
            with self.assertRaises(SystemExit):
                control.main(["--device", "AA:BB:CC:DD:EE:FF"])

        prompt_speeds.assert_not_called()
        select_device.assert_not_called()
        send_pwm.assert_not_called()
        wrapper.assert_not_called()

    def test_successful_connection_starts_key_controls(self):
        with (
            mock.patch.object(control.discovery, "choose_hub"),
            mock.patch.object(control.discovery, "app_url", return_value="http://phone.local:8080"),
            mock.patch.object(control.discovery, "remember_hub_url"),
            mock.patch.object(control, "prompt_speeds", return_value=self.speeds),
            mock.patch.object(control, "select_device") as select_device,
            mock.patch.object(control, "connect_bluetooth", return_value="Connected to ESP32"),
            mock.patch.object(control.CommandHub, "send_pwm") as send_pwm,
            mock.patch.object(control.curses, "wrapper") as wrapper,
        ):
            control.main(["--device", "AA:BB:CC:DD:EE:FF"])

        select_device.assert_not_called()
        send_pwm.assert_called_once_with(control.stop_commands())
        wrapper.assert_called_once()

    def test_get_paired_devices_reads_json_and_filters_incomplete_entries(self):
        hub = mock.Mock()
        hub.get_json.return_value = [
            {"name": "ESP32", "address": "AA:BB:CC:DD:EE:FF"},
            {"name": "Nameless"},
            {"name": "Other", "address": "11:22:33:44:55:66"},
        ]

        self.assertEqual(
            control.get_paired_devices(hub),
            [
                {"name": "ESP32", "address": "AA:BB:CC:DD:EE:FF"},
                {"name": "Other", "address": "11:22:33:44:55:66"},
            ],
        )
        hub.get_json.assert_called_once_with("/api/bluetooth/devices")

    def test_choose_device_uses_numbered_selection(self):
        devices = [
            {"name": "ESP32", "address": "AA:BB:CC:DD:EE:FF"},
            {"name": "HC-05", "address": "11:22:33:44:55:66"},
        ]

        with mock.patch("builtins.input", return_value="2"):
            selected = control.choose_device(devices)
        self.assertEqual(selected["address"], "11:22:33:44:55:66")

        with mock.patch("builtins.input", return_value=""):
            selected = control.choose_device(devices)
        self.assertEqual(selected["address"], "AA:BB:CC:DD:EE:FF")

    def test_choose_device_exits_when_none_are_paired(self):
        with self.assertRaises(SystemExit):
            control.choose_device([])

    def test_device_menu_is_used_without_device_flag(self):
        devices = [
            {"name": "ESP32", "address": "AA:BB:CC:DD:EE:FF"},
            {"name": "HC-05", "address": "11:22:33:44:55:66"},
        ]
        hub = mock.Mock()
        hub.get_json.return_value = devices

        with mock.patch("builtins.input", return_value="2"):
            selected = control.select_device(hub)

        self.assertEqual(selected["address"], "11:22:33:44:55:66")


    def test_main_connects_chosen_device_from_menu(self):
        with (
            mock.patch.object(control.discovery, "choose_hub"),
            mock.patch.object(control.discovery, "app_url", return_value="http://phone.local:8080"),
            mock.patch.object(control.discovery, "remember_hub_url"),
            mock.patch.object(control, "prompt_speeds", return_value=self.speeds),
            mock.patch.object(
                control,
                "select_device",
                return_value={"name": "ESP32", "address": "AA:BB:CC:DD:EE:FF"},
            ) as select_device,
            mock.patch.object(control, "connect_bluetooth", return_value="Connected to ESP32") as connect,
            mock.patch.object(control.CommandHub, "send_pwm"),
            mock.patch.object(control.curses, "wrapper"),
        ):
            control.main([])

        select_device.assert_called_once()
        connect.assert_called_once()
        self.assertEqual(connect.call_args.args[1], "AA:BB:CC:DD:EE:FF")

    def test_small_terminal_shows_resize_message_without_drawing_controls(self):
        state = control.ControlState(self.speeds, "http://phone.local:8080", "Connected to ESP32")
        screen = FakeScreen([], size=(10, 40))

        self.assertFalse(control.draw(screen, state))
        self.assertEqual(screen.lines[0], control.TOO_SMALL_MESSAGE[:39])
        self.assertNotIn(control.HELP_LINE, screen.lines.values())

    def test_keys_are_ignored_until_terminal_is_resized(self):
        class ResizingScreen(FakeScreen):
            def getch(self):
                key = next(self.keys)
                if key == control.curses.KEY_RESIZE:
                    self.size = (24, 80)
                return key

        hub = mock.Mock()
        state = control.ControlState(self.speeds, "http://phone.local:8080", "Connected to ESP32")
        screen = ResizingScreen(
            [
                control.curses.KEY_UP,
                ord("q"),
                control.curses.KEY_RESIZE,
                control.curses.KEY_DOWN,
                ord("q"),
            ],
            size=(10, 40),
        )

        with mock.patch.object(control.curses, "curs_set"):
            control.control_loop(screen, hub, state)

        self.assertEqual(
            [call.args[0] for call in hub.send_pwm.call_args_list],
            [
                ["P 32 0", "P 33 100"],
                control.stop_commands(),
            ],
        )

    def test_draw_clips_long_lines_to_screen_width(self):
        state = control.ControlState(
            self.speeds,
            "http://phone.local:8080",
            "Connected to a very long Bluetooth device name that does not fit",
        )
        screen = FakeScreen([], size=(24, 80))

        self.assertTrue(control.draw(screen, state))
        self.assertIn(control.HELP_LINE, screen.lines.values())
        for line in screen.lines.values():
            self.assertLessEqual(len(line), 79)


if __name__ == "__main__":
    unittest.main()
