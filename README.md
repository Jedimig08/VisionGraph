# VisionGraph
A low-cost camera-feedback polargraph controlled by a phone and ESP32.

## Concept

VisionGraph uses a camera to track markers for a polargraph. In the current
prototype, the phone's CommandHub provides camera discovery and video, while
the host PC runs the OpenCV ArUco detection and displays the live stream.

The ESP32 is intended to act as a simple motor controller. Polargraph
kinematics and motor-command integration are not implemented in this client.

## Architecture

Host PC
└── OpenCV ArUco detection and live display
        │ HTTP camera list and video stream
        ↓
Phone running CommandHub
└── Camera access
        │
        ↓
      ESP32 (planned motor control)

## Run

Install dependencies with `python -m pip install -r requirements.txt`, then run
`python cv_stream.py`. The client uses the last saved CommandHub URL when
available; otherwise it discovers a local CommandHub service. It selects the
best reported camera by default, prompting you to choose a camera and then a
starting resolution. Press Enter at either prompt to accept the recommended
camera or largest resolution. Press `f` to toggle fullscreen, `r` to clear
marker history and recalibrate, `v` to toggle between the full camera frame and
the calibrated area, or `q` to quit. The calibrated area is shown by default.
If the camera stream is temporarily unavailable, the client keeps retrying
every three seconds and reconnects if the stream drops.
At startup, enter the drawing-area width and height. The corrected image
preserves their aspect ratio with its longest edge scaled to 1000 pixels. The
outward-facing corners of the four corner markers align exactly with the four
corners of the corrected view.

## Marker Tracking and Calibration

The client uses the `DICT_4X4_50` ArUco dictionary. IDs 0, 1, 3, and 2 define
the surface corners clockwise from the top-left; the homography maps the
outward-facing corner of each marker directly to a corner of the corrected
view. ID 4 is detected and tracked as an additional reference but is not
assumed to be at the center or used to calculate the homography. Each corner
marker must be consistently observed in at least 3 of the latest 5 frames.

Calibration observations are collected across frames, so all four corner
markers do not need to appear together. Each incoming frame is warped to the
top-down view, then live detections and tracking overlays are drawn at their
transformed positions. Remembered positions expire after two seconds and are
never used to calibrate. Movement of a corner marker invalidates calibration.
The app defaults to the largest camera resolution, tries higher resolutions if
calibration remains incomplete, and never lowers resolution automatically.
Requested FPS is reduced only when the stream is persistently running slowly.
The camera backend's autofocus and exposure controls are requested when
available; the app reports when the stream backend cannot lock them.

Run tests with `python -m unittest discover -s tests`.

## CommandHub API Contract

This client expects CommandHub to expose these HTTP endpoints:

- `GET /api/cameras` returns a JSON array of camera objects. Each object must
  include `id`; camera scoring also uses `facing`, `type`, and `isPhysical`,
  while resolution selection uses `supportedResolutions`, an array of objects
  with integer `width` and `height` fields.
- `GET /stream/{id}?res={width}x{height}&fps={fps}` returns a live video stream
  readable by OpenCV. The client starts at 15 FPS and may lower its FPS
  request when the stream is persistently slow; it keeps the selected
  resolution.

The CommandHub URL is discovered over local zeroconf HTTP service discovery or
loaded from `~/.config/visiongraph/last_hub_url`. This documents the client-side
contract; the endpoint implementation belongs to the CommandHub project.

## Status

Early development.