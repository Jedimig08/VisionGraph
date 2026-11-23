#!/usr/bin/env python3
"""Manually drive the carriage from a terminal through CommandHub Bluetooth.

Human-in-the-loop bench tool. It uses the same CommandHub discovery and HTTP
API as the camera client, but sends only the keypresses you make: it is not
part of the CV/pose pipeline and never moves the carriage on its own.

CommandHub endpoints used:

    GET  /api/bluetooth/devices  paired devices for the numbered selection menu
    POST /bluetooth-send     raw body "P <pin> <0-255>" (the app appends "\n")
    POST /bluetooth-connect  raw body "<MAC address>" (required before any key)

Requires the phone to be running CommandHub on the same Wi-Fi, Bluetooth to be
enabled in CommandHub's privacy settings, and the ESP32/Arduino to be paired.
The connection must succeed before the key controls are started.

Wiring (edit MOTORS only if the H-bridge is wired differently):

    Motor 1: IN1 = pin 32 (up),   IN2 = pin 33 (down)
    Motor 2: IN3 = pin 25 (up),   IN4 = pin 26 (down)

Run:

    python commandhub_manual_control.py
    python commandhub_manual_control.py --device XX:XX:XX:XX:XX:XX
"""

import argparse
import curses
import json
import urllib.error
import urllib.request

import discovery


HUB_HTTP_TIMEOUT_SECONDS = 2.0

MIN_TERMINAL_ROWS = 12
MIN_TERMINAL_COLS = 76
HELP_LINE = "<-/-> select motor   up/down drive selected motor   space STOP ALL   q quit"
TOO_SMALL_MESSAGE = "Terminal too small — resize to at least 76x12; waiting for resize..."

# name, "up" pin, "down" pin
MOTORS = (
    ("Motor 1", 32, 33),
    ("Motor 2", 25, 26),
)

DIRECTIONS = ("up", "down")


def pwm_commands(pin_values):
    return [f"P {pin} {value}" for pin, value in pin_values]


def stop_commands():
    pin_values = []
    for _name, up_pin, down_pin in MOTORS:
        pin_values.extend(((up_pin, 0), (down_pin, 0)))
    return pwm_commands(pin_values)


def drive_commands(motor_index, direction, speeds):
    _name, up_pin, down_pin = MOTORS[motor_index]
    speed = speeds[motor_index][direction]
    if direction == "up":
        return pwm_commands(((up_pin, speed), (down_pin, 0)))
    return pwm_commands(((up_pin, 0), (down_pin, speed)))


class CommandHub:
    """Thin client for the CommandHub endpoints this tool needs."""

    def __init__(self, base_url, timeout=HUB_HTTP_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_text(self, path, text):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=text.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    def get_json(self, path):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def send_pwm(self, commands):
        for command in commands:
            self.post_text("/bluetooth-send", command)

    def connect(self, address):
        return self.post_text("/bluetooth-connect", address)


def connect_bluetooth(hub, address):
    result = hub.connect(address)
    if not result.startswith("Connected to "):
        raise RuntimeError(f"Bluetooth connection failed: {result}")
    return result


def get_paired_devices(hub):
    devices = hub.get_json("/api/bluetooth/devices")
    if not isinstance(devices, list):
        raise RuntimeError("CommandHub returned an unexpected paired-device list")
    return [
        {"name": str(device.get("name", "")), "address": str(device.get("address", ""))}
        for device in devices
        if isinstance(device, dict) and device.get("address")
    ]


def choose_device(devices):
    if not devices:
        raise SystemExit(
            "CommandHub reported no paired Bluetooth devices.\n"
            "Enable Bluetooth in its privacy settings and pair the ESP32/Arduino "
            "with the phone, then run this tool again."
        )

    default = devices[0]
    print("Available paired Bluetooth devices:")
    for index, device in enumerate(devices, start=1):
        print(f"  {index}. {device['name']} ({device['address']})")

    while True:
        try:
            choice = input(
                f"Choose device [1-{len(devices)}, Enter for {default['name']}]: "
            ).strip()
        except EOFError:
            return default
        if not choice:
            return default
        try:
            index = int(choice) - 1
        except ValueError:
            index = -1
        if 0 <= index < len(devices):
            return devices[index]
        print(
            f"Enter a number from 1 to {len(devices)}, "
            f"or press Enter for {default['name']}."
        )


def select_device(hub):
    try:
        devices = get_paired_devices(hub)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise SystemExit(
            f"Could not get paired Bluetooth devices from CommandHub: {exc}\n"
            "Check that CommandHub is updated, Bluetooth is enabled in its privacy "
            "settings, and the phone is on the same Wi-Fi."
        ) from exc
    return choose_device(devices)


def prompt_speeds():
    print("=== MANUAL CARRIAGE CONTROL VIA COMMANDHUB BLUETOOTH ===")
    print("Human-in-the-loop tool: no camera, pose or autonomous code is used.\n")
    speeds = []
    for name, up_pin, down_pin in MOTORS:
        print(f"{name}: up pin {up_pin}, down pin {down_pin}")
        motor_speeds = {}
        for direction in DIRECTIONS:
            while True:
                try:
                    raw = input(f"  {name} {direction} speed (0-255): ").strip()
                except EOFError as exc:
                    raise SystemExit("Speeds are required; no motors were driven.") from exc
                try:
                    value = int(raw)
                except ValueError:
                    print("    Enter a whole number from 0 to 255.")
                    continue
                if not 0 <= value <= 255:
                    print("    Enter a value from 0 to 255.")
                    continue
                motor_speeds[direction] = value
                break
        speeds.append(motor_speeds)
    print()
    return speeds


class ControlState:
    def __init__(self, speeds, hub_url, device):
        self.speeds = speeds
        self.hub_url = hub_url
        self.device = device
        self.selected = 0
        self.directions = [None] * len(MOTORS)
        self.last_commands = []
        self.message = "Ready."
        self.error = ""


def select_motor(state, step):
    state.selected = (state.selected + step) % len(MOTORS)
    state.message = f"Selected {MOTORS[state.selected][0]}."


def transmit(state, hub, commands, message):
    state.last_commands = commands
    try:
        hub.send_pwm(commands)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        state.error = f"CommandHub send failed: {exc}"
        state.message = f"{message} (FAILED)"
        return False
    state.error = ""
    state.message = message
    return True


def drive(state, hub, direction):
    commands = drive_commands(state.selected, direction, state.speeds)
    if transmit(state, hub, commands, f"{MOTORS[state.selected][0]} driving {direction}."):
        state.directions[state.selected] = direction


def stop_all(state, hub):
    ok = transmit(state, hub, stop_commands(), "ALL MOTORS STOPPED.")
    state.directions = [None] * len(MOTORS)
    return ok


def add_line(stdscr, row, text):
    height, width = stdscr.getmaxyx()
    if row < 0 or row >= height or width < 2 or not text:
        return
    try:
        stdscr.addnstr(row, 0, text, min(len(text), width - 1))
    except curses.error:
        pass


def draw(stdscr, state):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < MIN_TERMINAL_ROWS or width < MIN_TERMINAL_COLS:
        add_line(stdscr, 0, TOO_SMALL_MESSAGE)
        stdscr.refresh()
        return False

    add_line(stdscr, 0, "MANUAL CARRIAGE CONTROL — COMMANDHUB BLUETOOTH BRIDGE")
    add_line(stdscr, 1, f"Hub: {state.hub_url}   BT: {state.device}")
    add_line(stdscr, 3, "Motor     Pins         Up speed  Down speed  State")

    for index, (name, up_pin, down_pin) in enumerate(MOTORS):
        marker = ">" if index == state.selected else " "
        direction = state.directions[index]
        status = direction.upper() if direction else "stopped"
        line = (
            f"{marker} {name:<9} {up_pin}/{down_pin:<11} "
            f"{state.speeds[index]['up']:<9} {state.speeds[index]['down']:<11} {status}"
        )
        add_line(stdscr, 4 + index, line)

    row = 4 + len(MOTORS) + 1
    add_line(stdscr, row, state.message)
    add_line(stdscr, row + 1, "Last sent: " + ", ".join(state.last_commands))
    if state.error:
        add_line(stdscr, row + 2, state.error)
    add_line(stdscr, row + 4, HELP_LINE)
    stdscr.refresh()
    return True


def control_loop(stdscr, hub, state):
    curses.curs_set(0)
    stdscr.keypad(True)

    try:
        while True:
            ready = draw(stdscr, state)
            key = stdscr.getch()
            if not ready:
                continue

            if key in (ord("q"), 27):
                break
            if key == curses.KEY_LEFT:
                select_motor(state, -1)
            elif key == curses.KEY_RIGHT:
                select_motor(state, 1)
            elif key == curses.KEY_UP:
                drive(state, hub, "up")
            elif key == curses.KEY_DOWN:
                drive(state, hub, "down")
            elif key in (ord(" "), curses.KEY_BACKSPACE):
                stop_all(state, hub)
    finally:
        stop_all(state, hub)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Manually drive the carriage through CommandHub's Bluetooth bridge."
    )
    parser.add_argument(
        "--device",
        metavar="MAC",
        help="MAC address of the paired ESP32/Arduino (prompted for if omitted)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    discovery.choose_hub()
    hub = CommandHub(discovery.app_url())

    address = args.device or select_device(hub)["address"]
    try:
        device = connect_bluetooth(hub, address)
    except (RuntimeError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"Could not connect CommandHub to {address}: {exc}") from exc
    print(f"CommandHub Bluetooth connect: {device}")

    speeds = prompt_speeds()
    state = ControlState(speeds, hub.base_url, device)
    if not stop_all(state, hub):
        raise SystemExit(
            f"CommandHub at {hub.base_url} did not accept commands: {state.error}\n"
            "Check that CommandHub is running, Bluetooth is enabled in its privacy "
            "settings, and an ESP32/Arduino is paired and connected."
        )
    discovery.remember_hub_url()

    try:
        curses.wrapper(control_loop, hub, state)
    except KeyboardInterrupt:
        pass

    print("Stopped all motors and exited manual control.")


if __name__ == "__main__":
    main()
