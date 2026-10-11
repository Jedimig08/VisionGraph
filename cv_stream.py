import os
import time
from collections import deque

os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import camera_select

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"


def configure_aruco_params(low_contrast=False):
    params = (
        cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters")
        else cv2.aruco.DetectorParameters_create()
    )
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 0.8
    params.polygonalApproxAccuracyRate = 0.03

    if low_contrast:
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 31
        params.adaptiveThreshConstant = 9
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 0.9

    return params


def choose_starting_resolution(resolutions):
    print("Available starting resolutions:")
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


camera = camera_select.select_camera()
resolutions = camera_select.choose_resolution_steps(camera)
current_resolution_index = choose_starting_resolution(resolutions)
width, height = resolutions[current_resolution_index]

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
if hasattr(cv2.aruco, "DetectorParameters_create"):
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_detector = None
else:
    aruco_params = configure_aruco_params(False)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

url = camera_select.stream_url(camera, width, height, camera_select.MODERATE_FPS)

print(
    f"Selected camera {camera['id']} "
    f"({camera.get('facing')} {camera.get('type')}) "
    f"at {width}x{height} @ {camera_select.MODERATE_FPS} FPS"
)
print(f"Connecting to {url}...")

def open_stream(stream_url):
    stream = cv2.VideoCapture(stream_url)
    if stream.set(cv2.CAP_PROP_BUFFERSIZE, 1):
        print("Capture backend accepted a one-frame buffer request")
    else:
        print("Capture backend does not support setting the buffer size")
    return stream


cap = open_stream(url)
cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
fullscreen = True
low_contrast_mode = False
missed_frames = 0
stable_detected_frames = 0
manual_resolution_prompt = False
prompt_target_index = None
last_good_corners = None
last_good_ids = None
marker_hold_frames = 0
HOLD_MARKER_FRAMES = 3
FPS_WINDOW_SECONDS = 3.0
MIN_STREAM_FPS = camera_select.MODERATE_FPS * 0.8
PROMPT_COOLDOWN_SECONDS = 10.0
frame_times = deque()
stream_fps = None
prompt_reason = None
prompt_snooze_until = 0.0


def detect_aruco(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if aruco_detector is not None:
        return aruco_detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)


def apply_aruco_mode(force_low_contrast=False):
    global aruco_params, aruco_detector
    aruco_params = configure_aruco_params(force_low_contrast)
    if aruco_detector is not None:
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't read stream")
        break

    frame_time = time.monotonic()
    frame_times.append(frame_time)
    while frame_times and frame_times[0] < frame_time - FPS_WINDOW_SECONDS:
        frame_times.popleft()
    if len(frame_times) >= 2:
        elapsed = frame_times[-1] - frame_times[0]
        if elapsed >= FPS_WINDOW_SECONDS * 0.8:
            stream_fps = (len(frame_times) - 1) / elapsed

    corners, ids, rejected = detect_aruco(frame)
    if ids is not None:
        stable_detected_frames += 1
        missed_frames = 0
        low_contrast_mode = False
        if prompt_reason != "network":
            manual_resolution_prompt = False
            prompt_target_index = None
            prompt_reason = None
        last_good_corners = corners
        last_good_ids = ids
        marker_hold_frames = 0
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            center = corners[i][0].mean(axis=0).astype(int)
            cv2.putText(
                frame,
                f"ID {marker_id}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
    else:
        stable_detected_frames = 0
        missed_frames += 1

        if last_good_corners is not None and last_good_ids is not None:
            marker_hold_frames += 1
            if marker_hold_frames <= HOLD_MARKER_FRAMES:
                cv2.aruco.drawDetectedMarkers(frame, last_good_corners, last_good_ids)
                for i, marker_id in enumerate(last_good_ids.flatten()):
                    center = last_good_corners[i][0].mean(axis=0).astype(int)
                    cv2.putText(
                        frame,
                        f"ID {marker_id}",
                        (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
            else:
                last_good_corners = None
                last_good_ids = None
                marker_hold_frames = 0

        if missed_frames >= 15 and not low_contrast_mode:
            low_contrast_mode = True
            apply_aruco_mode(True)
            print("Low-contrast ArUco detection enabled")

        if (
            missed_frames >= 30
            and current_resolution_index < len(resolutions) - 1
            and not manual_resolution_prompt
            and time.monotonic() >= prompt_snooze_until
        ):
            prompt_target_index = current_resolution_index + 1
            prompt_width, prompt_height = resolutions[prompt_target_index]
            manual_resolution_prompt = True
            prompt_reason = "markers"
            print(
                f"Marker detection is weak. Press y to lower resolution to "
                f"{prompt_width}x{prompt_height}, or n to keep the current stream."
            )

    if (
        stream_fps is not None
        and stream_fps < MIN_STREAM_FPS
        and current_resolution_index < len(resolutions) - 1
        and not manual_resolution_prompt
        and time.monotonic() >= prompt_snooze_until
    ):
        prompt_target_index = current_resolution_index + 1
        prompt_width, prompt_height = resolutions[prompt_target_index]
        manual_resolution_prompt = True
        prompt_reason = "network"
        print(
            f"Live frame rate is {stream_fps:.1f} FPS; press y to lower resolution to "
            f"{prompt_width}x{prompt_height}, or n to keep the current stream."
        )

    if stream_fps is not None:
        cv2.putText(
            frame,
            f"Live rate: {stream_fps:.1f} FPS",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if manual_resolution_prompt and prompt_target_index is not None:
        prompt_width, prompt_height = resolutions[prompt_target_index]
        reason = "Stream is slow. " if prompt_reason == "network" else "Markers are not being detected. "
        overlay = reason + "Downgrade to " + str(prompt_width) + "x" + str(prompt_height) + "? Press Y / N"
        cv2.putText(
            frame,
            overlay,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Stream", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("y") and manual_resolution_prompt and prompt_target_index is not None:
        target_index = prompt_target_index
        target_width, target_height = resolutions[target_index]
        target_url = camera_select.stream_url(camera, target_width, target_height, camera_select.MODERATE_FPS)
        print(f"User accepted: lowering resolution to {target_width}x{target_height}")
        cap.release()
        cap = open_stream(target_url)
        if cap.isOpened():
            current_resolution_index = target_index
            width = target_width
            height = target_height
            url = target_url
            manual_resolution_prompt = False
            prompt_target_index = None
            prompt_reason = None
            prompt_snooze_until = 0.0
            missed_frames = 0
            low_contrast_mode = False
            frame_times.clear()
            stream_fps = None
            apply_aruco_mode(False)
        else:
            print("Lower resolution failed; keeping the current stream")
            cap = open_stream(url)
            manual_resolution_prompt = False
            prompt_target_index = None
            prompt_reason = None
            prompt_snooze_until = time.monotonic() + PROMPT_COOLDOWN_SECONDS
    elif key == ord("n") and manual_resolution_prompt:
        print("User kept the current resolution")
        manual_resolution_prompt = False
        prompt_target_index = None
        prompt_reason = None
        prompt_snooze_until = time.monotonic() + PROMPT_COOLDOWN_SECONDS

    if key == ord("q"):
        break
    if key == ord("f"):
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            "Stream",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )

cap.release()
cv2.destroyAllWindows()