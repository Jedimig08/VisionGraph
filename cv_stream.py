"""Run live ArUco tracking and multi-frame surface calibration."""

import os
import math
import time
from collections import deque

os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import camera_select
import marker_tracking

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"


FPS_WINDOW_SECONDS = 3.0
FPS_LEVELS = (15, 12, 10, 8, 5)
CALIBRATION_RETRY_SECONDS = 15.0
STREAM_RETRY_SECONDS = 3.0

configure_aruco_params = marker_tracking.configure_aruco_params
merge_marker_ids = marker_tracking.merge_marker_ids


def choose_starting_resolution(resolutions):
    print("Available starting resolutions (largest first):")
    for option, (option_width, option_height) in enumerate(resolutions, start=1):
        print(f"  {option}. {option_width}x{option_height}")

    while True:
        try:
            choice = input(
                f"Choose starting resolution [1-{len(resolutions)}, Enter for largest]: "
            ).strip()
        except EOFError:
            return 0

        if not choice:
            return 0
        try:
            index = int(choice) - 1
        except ValueError:
            index = -1
        if 0 <= index < len(resolutions):
            return index
        print(f"Enter a number from 1 to {len(resolutions)}, or press Enter for the largest.")


def choose_drawing_dimensions():
    dimensions = []
    for label in ("width", "height"):
        while True:
            try:
                value = float(input(f"Drawing-area {label}: ").strip())
            except EOFError as exc:
                raise RuntimeError("Drawing-area dimensions are required for calibration") from exc
            except ValueError:
                print("Enter a positive number.")
                continue
            if math.isfinite(value) and value > 0:
                dimensions.append(value)
                break
            print("Enter a finite positive number.")

    return tuple(dimensions)


def open_stream(stream_url):
    while True:
        stream = cv2.VideoCapture(stream_url)
        if stream.isOpened():
            break
        stream.release()
        print(
            f"Unable to open camera stream: {stream_url}. "
            f"Retrying in {STREAM_RETRY_SECONDS:g} seconds..."
        )
        time.sleep(STREAM_RETRY_SECONDS)

    if stream.set(cv2.CAP_PROP_BUFFERSIZE, 1):
        print("Capture backend accepted a one-frame buffer request")
    else:
        print("Capture backend does not support setting the buffer size")

    autofocus_property = getattr(cv2, "CAP_PROP_AUTOFOCUS", None)
    if autofocus_property is not None:
        if stream.set(autofocus_property, 0):
            print("Disabled camera autofocus to keep focus consistent")
        else:
            print("Capture backend does not support locking camera focus")

    exposure_property = getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", None)
    if exposure_property is not None:
        if stream.set(exposure_property, 0.25):
            print("Requested manual exposure to keep brightness consistent")
        else:
            print("Capture backend does not support locking camera exposure")
    return stream


def draw_tracking_status(frame, result, homography=None):
    def transform_points(points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if homography is not None:
            points = cv2.perspectiveTransform(points, homography)
        return points.reshape(-1, 2)

    if result.live:
        live_ids = np.asarray(sorted(result.live), dtype=np.int32).reshape(-1, 1)
        live_corners = [
            transform_points(result.live_corners[marker_id]).reshape(1, -1, 2)
            for marker_id in sorted(result.live)
        ]
        cv2.aruco.drawDetectedMarkers(frame, live_corners, live_ids)
        for marker_id, center in result.live.items():
            point = tuple(np.round(transform_points([center])[0]).astype(int))
            cv2.putText(
                frame,
                f"ID {marker_id} live",
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    for marker_id, memory in result.remembered.items():
        center = tuple(np.round(transform_points([memory.center])[0]).astype(int))
        cv2.circle(frame, center, 8, (0, 165, 255), 2)
        age = max(0.0, time.monotonic() - memory.last_seen)
        cv2.putText(
            frame,
            f"ID {marker_id} remembered {age:.1f}s",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            2,
        )

    lines = [
        "Detected: " + (", ".join(map(str, sorted(result.live))) or "none"),
        "Missing: " + (", ".join(map(str, result.missing)) or "none"),
        "Remembered: " + (", ".join(map(str, sorted(result.remembered))) or "none"),
        f"Rejected ArUco candidates: {result.rejected_count}",
        "Detection reliability: "
        + " ".join(
            f"{marker_id}:{result.reliability[marker_id] * 100:.0f}%"
            for marker_id in marker_tracking.EXPECTED_MARKER_IDS
        ),
        (
            "Calibration: ready"
            if result.calibrated
            else "Calibration observations: "
            + " ".join(
                f"{marker_id}:{result.calibration_observations[marker_id]}/"
                f"{marker_tracking.MINIMUM_OBSERVATIONS}"
                for marker_id in marker_tracking.EXPECTED_MARKER_IDS
            )
        ),
    ]
    for line_index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, 30 + line_index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def prepare_display_frame(frame, result, show_full_frame):
    homography = None
    if not show_full_frame and result.calibration is not None:
        homography = result.calibration
        display_frame = cv2.warpPerspective(
            frame,
            homography,
            result.output_size,
        )
    else:
        display_frame = frame.copy()
    draw_tracking_status(display_frame, result, homography)
    return display_frame


def main():
    area_width, area_height = choose_drawing_dimensions()
    output_size = marker_tracking.output_size_for_area(
        area_width,
        area_height,
    )
    print(
        f"Drawing-area aspect ratio: {area_width:g}x{area_height:g}; "
        f"corrected area size: {output_size[0]}x{output_size[1]} pixels "
        "(longest edge 1000 pixels, plus corner-marker margins)"
    )
    cameras = camera_select.get_cameras()
    camera = camera_select.choose_starting_camera(cameras)
    resolutions = camera_select.choose_resolution_steps(camera)
    resolution_index = choose_starting_resolution(resolutions)
    width, height = resolutions[resolution_index]
    fps_index = 0
    requested_fps = FPS_LEVELS[fps_index]

    detector = marker_tracking.ArucoMarkerDetector()
    tracker = marker_tracking.MarkerTracker(output_size=output_size)
    stream_url = camera_select.stream_url(camera, width, height, requested_fps)
    print(
        f"Selected camera {camera['id']} "
        f"({camera.get('facing')} {camera.get('type')}) "
        f"at {width}x{height} @ {requested_fps} FPS"
    )
    print(f"Connecting to {stream_url}...")
    cap = open_stream(stream_url)

    cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True
    show_full_frame = False
    frame_times = deque()
    slow_windows = 0
    last_fps_evaluation = time.monotonic()
    calibration_started = time.monotonic()
    last_resolution_retry = calibration_started

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Couldn't read stream; reconnecting...")
                cap.release()
                cap = open_stream(stream_url)
                tracker = marker_tracking.MarkerTracker(output_size=output_size)
                frame_times.clear()
                now = time.monotonic()
                last_fps_evaluation = now
                calibration_started = now
                last_resolution_retry = now
                slow_windows = 0
                continue

            frame_time = time.monotonic()
            frame_times.append(frame_time)
            while frame_times and frame_times[0] < frame_time - FPS_WINDOW_SECONDS:
                frame_times.popleft()
            stream_fps = None
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed >= FPS_WINDOW_SECONDS * 0.8:
                    stream_fps = (len(frame_times) - 1) / elapsed

            corners, ids, rejected_count = detector.detect(frame)
            result = tracker.update(
                corners,
                ids,
                rejected_count,
                frame.shape,
                now=frame_time,
            )

            if not result.calibrated and frame_time - calibration_started >= CALIBRATION_RETRY_SECONDS:
                if resolution_index > 0 and frame_time - last_resolution_retry >= CALIBRATION_RETRY_SECONDS:
                    resolution_index -= 1
                    width, height = resolutions[resolution_index]
                    print(
                        f"Calibration is still incomplete; trying higher resolution "
                        f"{width}x{height} before reducing FPS"
                    )
                    cap.release()
                    stream_url = camera_select.stream_url(
                        camera, width, height, requested_fps
                    )
                    cap = open_stream(stream_url)
                    tracker = marker_tracking.MarkerTracker(output_size=output_size)
                    frame_times.clear()
                    last_fps_evaluation = time.monotonic()
                    calibration_started = time.monotonic()
                    last_resolution_retry = calibration_started
                    slow_windows = 0
                    continue
                calibration_started = frame_time

            if (
                stream_fps is not None
                and frame_time - last_fps_evaluation >= FPS_WINDOW_SECONDS
                and (result.calibrated or resolution_index == 0)
            ):
                if stream_fps < requested_fps * 0.75:
                    slow_windows += 1
                else:
                    slow_windows = 0
                last_fps_evaluation = frame_time

                if slow_windows >= 2:
                    if fps_index + 1 < len(FPS_LEVELS):
                        fps_index += 1
                        requested_fps = FPS_LEVELS[fps_index]
                        print(
                            f"Stream is slow; keeping {width}x{height} resolution "
                            f"and reducing requested FPS to {requested_fps}"
                        )
                        cap.release()
                        stream_url = camera_select.stream_url(
                            camera, width, height, requested_fps
                        )
                        cap = open_stream(stream_url)
                        frame_times.clear()
                        last_fps_evaluation = time.monotonic()
                    else:
                        print(
                            f"Stream remains slow at {requested_fps} requested FPS"
                        )
                    slow_windows = 0

            display_frame = prepare_display_frame(
                frame,
                result,
                show_full_frame,
                result.output_size,
            )
            view_label = (
                "View: full camera frame"
                if show_full_frame or result.calibration is None
                else "View: calibrated area"
            )
            cv2.putText(
                display_frame,
                view_label,
                (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if stream_fps is not None:
                cv2.putText(
                    display_frame,
                    f"Live rate: {stream_fps:.1f} FPS",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Stream", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    "Stream",
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            if key == ord("v"):
                show_full_frame = not show_full_frame
            if key == ord("r"):
                tracker = marker_tracking.MarkerTracker(output_size=output_size)
                calibration_started = time.monotonic()
                print("Cleared marker history; collecting calibration observations again")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
