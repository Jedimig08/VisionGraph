"""Run live ArUco tracking and multi-frame surface calibration."""

import os
import math
import time
from collections import deque

os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import calibration_store
import camera_select
import marker_tracking
import pose_estimator

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"


FPS_WINDOW_SECONDS = 3.0
FPS_LEVELS = (15, 12, 10, 8, 5)
CALIBRATION_RETRY_SECONDS = 15.0
STREAM_RETRY_SECONDS = 3.0
PREDICTED_DISPLAY_SECONDS = 0.3

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


def make_tracker(output_size, calibration=None):
    tracker = marker_tracking.MarkerTracker(output_size=output_size)
    if calibration is not None:
        tracker.import_calibration(calibration)
    return tracker


def make_pose_estimator(area_width, area_height, output_size):
    return pose_estimator.PoseEstimator(
        pose_estimator.PoseEstimatorConfig(
            drawing_width_mm=area_width,
            drawing_height_mm=area_height,
            output_size=output_size,
        )
    )


def calibration_detail(payload):
    """Describe a saved payload for the selection menu, or mark it unusable."""
    try:
        normalized = marker_tracking.normalize_calibration(payload)
    except (ValueError, TypeError):
        return " (unusable)"
    width, height = normalized["drawing_area"]
    return (
        f" (drawing area {width:g}x{height:g} mm, "
        f"cutout {normalized['output_size'][0]}x{normalized['output_size'][1]})"
    )


def prompt_calibration_name(names, default_name=None, none_label="Cancel"):
    """Show a numbered catalog and return the chosen name, or None for ``none``."""
    print("Saved calibrations:")
    for index, name in enumerate(names, start=1):
        print(f"  {index}. {name}{calibration_detail(calibration_store.load(name))}")
    print(f"  0. {none_label}")

    default_index = (
        names.index(default_name) + 1 if default_name in names else 1 if names else 0
    )
    while True:
        try:
            choice = input(
                f"Choose calibration [0-{len(names)}, Enter for "
                f"{default_index}]: "
            ).strip()
        except EOFError:
            choice = ""
        if not choice:
            return names[default_index - 1] if default_index else None
        if choice in ("0", "n", "no"):
            return None
        try:
            index = int(choice)
        except ValueError:
            index = -1
        if 1 <= index <= len(names):
            return names[index - 1]
        print(f"Enter a number from 0 to {len(names)}.")


def choose_startup_calibration():
    """Prompt for a saved calibration before any recalculation starts."""
    try:
        names = calibration_store.names()
    except (OSError, ValueError) as exc:
        print(f"Could not read saved calibrations: {exc}")
        return None
    if not names:
        return None
    try:
        last = calibration_store.last_name()
    except (OSError, ValueError):
        last = None
    name = prompt_calibration_name(
        names,
        default_name=last,
        none_label="Start a fresh calibration",
    )
    if name is None:
        return None
    try:
        normalized = marker_tracking.normalize_calibration(
            calibration_store.load(name)
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"Saved calibration '{name}' is unusable ({exc}); ignoring it")
        return None
    try:
        calibration_store.mark_used(name)
    except OSError:
        pass
    print(f"Using saved calibration '{name}'")
    return name, normalized


def prompt_new_calibration_name(default_name=None):
    suggestion = default_name or "calibration"
    try:
        answer = input(
            f"Save calibration as [Enter for '{suggestion}']: "
        ).strip()
    except EOFError:
        answer = ""
    return answer or suggestion


def save_calibration_key(tracker, area_width, area_height, width, height, name):
    """Persist the current homography, cutout, and capture settings by name."""
    payload = tracker.export_calibration()
    if payload is None:
        return "Calibration is not ready to save yet", None, None
    payload["drawing_area"] = [area_width, area_height]
    payload["resolution"] = [width, height]
    try:
        normalized = marker_tracking.normalize_calibration(payload)
        destination = calibration_store.save(name, payload)
    except (OSError, ValueError) as exc:
        return f"Could not save calibration: {exc}", None, None
    return (
        f"Calibration saved as '{name}' to {destination}",
        name,
        normalized,
    )


def load_calibration_key(tracker, resolution=None):
    """Pick a saved calibration from the catalog and restore it."""
    try:
        names = calibration_store.names()
    except (OSError, ValueError) as exc:
        return f"Could not read saved calibrations: {exc}", None, None
    if not names:
        return "No saved calibrations found", None, None
    try:
        last = calibration_store.last_name()
    except (OSError, ValueError):
        last = None
    name = prompt_calibration_name(
        names,
        default_name=last,
        none_label="Cancel",
    )
    if name is None:
        return "Load cancelled", None, None
    try:
        normalized = marker_tracking.normalize_calibration(
            calibration_store.load(name)
        )
    except (OSError, ValueError, TypeError) as exc:
        return f"Could not load saved calibration '{name}': {exc}", None, None
    saved_resolution = normalized["resolution"]
    if (
        resolution is not None
        and saved_resolution is not None
        and saved_resolution != tuple(resolution)
    ):
        return (
            "Saved calibration uses a different camera resolution; "
            "restart to apply it",
            None,
            None,
        )
    try:
        tracker.import_calibration(normalized)
    except (ValueError, TypeError) as exc:
        return f"Could not load saved calibration '{name}': {exc}", None, None
    try:
        calibration_store.mark_used(name)
    except OSError:
        pass
    return (
        f"Loaded saved calibration '{name}': homography and cutout locked; "
        "press R to recalibrate",
        name,
        normalized,
    )


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


def confirm_calibration_key(tracker):
    """Lock a ready calibration and return the status message to print."""
    if tracker.confirm_calibration():
        return (
            "Calibration confirmed: homography locked, "
            "tracking carriage marker only"
        )
    if getattr(tracker, "calibration_confirmed", False):
        return "Calibration is already locked; press R to recalibrate"
    return "Calibration is not ready to confirm yet"


def draw_tracking_status(frame, result, homography=None):
    def transform_points(points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if homography is not None:
            points = cv2.perspectiveTransform(points, homography)
        return points.reshape(-1, 2)

    tracked_ids = tuple(getattr(result, "tracked_ids", ()))
    visible_live = {
        marker_id: center
        for marker_id, center in result.live.items()
        if not tracked_ids or marker_id in tracked_ids
    }
    if visible_live:
        live_ids = np.asarray(sorted(visible_live), dtype=np.int32).reshape(-1, 1)
        live_corners = [
            transform_points(result.live_corners[marker_id]).reshape(1, -1, 2)
            for marker_id in sorted(visible_live)
        ]
        cv2.aruco.drawDetectedMarkers(frame, live_corners, live_ids)
        for marker_id, center in visible_live.items():
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
        if tracked_ids and marker_id not in tracked_ids:
            continue
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

    tracks = getattr(result, "tracks", {}) or {}
    predicted = {
        marker_id: track
        for marker_id, track in tracks.items()
        if (not tracked_ids or marker_id in tracked_ids)
        and not getattr(track, "is_live", True)
        and 0.0 <= float(getattr(track, "age", 0.0)) <= PREDICTED_DISPLAY_SECONDS
    }
    for marker_id, track in predicted.items():
        center = tuple(np.round(transform_points([track.center])[0]).astype(int))
        cv2.circle(frame, center, 6, (255, 0, 255), 1)
        cv2.putText(
            frame,
            f"ID {marker_id} predicted {track.age:.1f}s",
            (center[0] + 10, center[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

    lines = [
        "Detected: " + (", ".join(map(str, sorted(result.live))) or "none"),
        "Missing: " + (", ".join(map(str, result.missing)) or "none"),
        "Remembered: " + (", ".join(map(str, sorted(result.remembered))) or "none"),
        "Predicted: "
        + (
            ", ".join(
                f"{marker_id}:{track.age:.1f}s"
                for marker_id, track in sorted(predicted.items())
            )
            or "none"
        ),
        f"Rejected ArUco candidates: {result.rejected_count}",
        "Detection reliability: "
        + " ".join(
            f"{marker_id}:{result.reliability[marker_id] * 100:.0f}%"
            for marker_id in sorted(result.reliability)
        ),
        (
            "Calibration: LOCKED (S save, L load, R recalibrate)"
            if getattr(result, "calibration_confirmed", False)
            else "Calibration: PROVISIONAL - C confirm, S save"
            if result.calibrated
            else "Calibration observations: "
            + " ".join(
                f"{marker_id}:{result.calibration_observations[marker_id]}/"
                f"{marker_tracking.MINIMUM_OBSERVATIONS}"
                for marker_id in sorted(result.calibration_observations)
            )
        ),
    ]
    if getattr(result, "calibration_confirmed", False):
        lines.append("Tracking: carriage marker only")
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


def draw_pose_status(frame, estimate):
    if estimate is None:
        return
    if estimate.valid:
        if estimate.refined:
            corner_source = f"subpixel edges ({estimate.fit_error_px:.2f} px fit)"
        else:
            corner_source = f"raw ArUco corners ({estimate.refinement_reason})"
        lines = [
            (
                f"Carriage: X={estimate.x_mm:.2f} mm  Y={estimate.y_mm:.2f} mm  "
                f"angle={estimate.angle_deg:+.2f} deg"
            ),
            (
                f"Carriage pixels: ({estimate.x_px:.1f}, {estimate.y_px:.1f})  "
                f"reliability={estimate.reliability * 100:.0f}%"
            ),
            f"Corner source: {corner_source}",
        ]
        if getattr(estimate, "predicted", False):
            lines.append(
                "PREDICTED pose "
                f"({getattr(estimate, 'prediction_age_seconds', 0.0):.2f}s since detection)"
            )
        if not estimate.inside_drawing_area:
            lines.append("Warning: carriage is outside the drawing area")
        colour = (0, 255, 0) if not getattr(estimate, "predicted", False) else (0, 165, 255)
    else:
        lines = [f"Carriage pose unavailable: {estimate.reason}"]
        colour = (0, 0, 255)

    for line_index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, 255 + line_index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )


def prepare_display_frame(frame, result, show_full_frame, pose=None):
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
    draw_pose_status(display_frame, pose)
    return display_frame


def main():
    startup = choose_startup_calibration()
    if startup is not None:
        calibration_name, calibration = startup
        area_width, area_height = calibration["drawing_area"]
        output_size = calibration["output_size"]
    else:
        calibration_name = None
        calibration = None
        area_width, area_height = choose_drawing_dimensions()
        output_size = marker_tracking.output_size_for_area(
            area_width,
            area_height,
        )
    print(
        f"Drawing-area aspect ratio: {area_width:g}x{area_height:g}; "
        f"corrected area size: {output_size[0]}x{output_size[1]} pixels "
        "(longest edge 1000 pixels)"
    )
    cameras = camera_select.get_cameras()
    camera = camera_select.choose_starting_camera(cameras)
    resolutions = camera_select.choose_resolution_steps(camera)
    if calibration is not None:
        saved_resolution = calibration.get("resolution")
        if saved_resolution in resolutions:
            resolution_index = resolutions.index(saved_resolution)
            print(
                f"Using saved camera resolution {saved_resolution[0]}x"
                f"{saved_resolution[1]}"
            )
        else:
            print("Saved calibration resolution is unavailable; recalibrating")
            calibration_name = None
            calibration = None
    if calibration is None:
        resolution_index = choose_starting_resolution(resolutions)
    width, height = resolutions[resolution_index]
    fps_index = 0
    requested_fps = FPS_LEVELS[fps_index]

    detector = marker_tracking.ArucoMarkerDetector()
    tracker = make_tracker(output_size, calibration)
    if calibration is not None:
        print("Loaded saved homography and cutout (locked); press R to recalibrate")
    pose = make_pose_estimator(area_width, area_height, output_size)
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
                reusable = (
                    calibration
                    if calibration is not None
                    and calibration.get("resolution") == (width, height)
                    else None
                )
                tracker = make_tracker(output_size, reusable)
                pose.reset()
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
            pose_result = pose.update(result, frame=frame, now=frame_time)

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
                    calibration_name = None
                    calibration = None
                    tracker = make_tracker(output_size)
                    pose.reset()
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
                pose_result,
            )
            if show_full_frame or result.calibration is None:
                view_label = "View: full camera frame"
            elif getattr(result, "calibration_confirmed", False):
                view_label = "View: calibrated area (locked"
                if calibration_name:
                    view_label += f": {calibration_name}"
                view_label += ")"
            else:
                view_label = "View: calibrated area"
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
            if key == ord("c"):
                print(confirm_calibration_key(tracker))
            if key == ord("s"):
                name = prompt_new_calibration_name(calibration_name)
                message, saved_name, payload = save_calibration_key(
                    tracker,
                    area_width,
                    area_height,
                    width,
                    height,
                    name,
                )
                print(message)
                if payload is not None:
                    calibration_name = saved_name
                    calibration = payload
            if key == ord("l"):
                message, loaded_name, payload = load_calibration_key(
                    tracker, (width, height)
                )
                print(message)
                if payload is not None:
                    saved_area = payload["drawing_area"]
                    if (
                        saved_area != (area_width, area_height)
                        or payload["output_size"] != output_size
                    ):
                        area_width, area_height = saved_area
                        output_size = payload["output_size"]
                        pose = make_pose_estimator(
                            area_width, area_height, output_size
                        )
                    calibration_name = loaded_name
                    calibration = payload
                    pose.reset()
                    calibration_started = time.monotonic()
            if key == ord("r"):
                tracker = make_tracker(output_size)
                calibration_name = None
                calibration = None
                pose.reset()
                calibration_started = time.monotonic()
                print(
                    "Cleared calibration lock and marker history; "
                    "collecting observations again"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
