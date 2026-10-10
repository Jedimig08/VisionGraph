"""Select the best camera for the overhead polargraph view.

Queries the CommandHub ``/api/cameras`` endpoint and chooses the camera
that is most suitable, together with a moderate frame rate and resolution.
It only performs selection; the stream itself is opened in ``cv_stream``.
"""

import json
import urllib.request

import discovery

MODERATE_FPS = 15
MODERATE_WIDTH = 640
MODERATE_HEIGHT = 480

_TYPE_SCORES = {
    "UltraWide": 20,
    "Wide": 50,
    "Telephoto": 10,
}


def get_cameras():
    """Return the list of CameraInfo dicts reported by the CommandHub."""
    url = f"{discovery.app_url()}/api/cameras"
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def score_camera(camera):
    """Higher is better. Prefers the back main (Wide) camera."""
    score = 0

    if camera.get("facing") == "Back":
        score += 100
    elif camera.get("facing") == "External":
        score += 40

    camera_type = camera.get("type", "")
    for name, points in _TYPE_SCORES.items():
        if name in camera_type:
            score += points
            break

    if camera.get("isPhysical", True):
        score += 5

    max_pixels = max(
        (res["width"] * res["height"] for res in camera.get("supportedResolutions", [])),
        default=0,
    )
    score += max_pixels / 1_000_000

    return score


def select_camera(cameras=None):
    """Choose the best available camera, or raise if none are reported."""
    if cameras is None:
        cameras = get_cameras()
    if not cameras:
        raise RuntimeError("CommandHub reported no cameras")
    return max(cameras, key=score_camera)


def choose_resolution(camera, width=MODERATE_WIDTH, height=MODERATE_HEIGHT):
    """Pick the supported resolution closest to the requested moderate size."""
    resolutions = camera.get("supportedResolutions", [])
    if not resolutions:
        return width, height

    target = width * height
    best = min(resolutions, key=lambda res: abs(res["width"] * res["height"] - target))
    return best["width"], best["height"]
