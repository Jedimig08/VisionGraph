"""Select the best camera for the overhead polargraph view.

Queries the CommandHub ``/api/cameras`` endpoint and chooses the camera
that is most suitable, together with a moderate frame rate and resolution.
It only performs selection; the stream itself is opened in ``cv_stream``.
"""

import json
import time
import urllib.request

import discovery

MODERATE_FPS = 15
MODERATE_WIDTH = 640
MODERATE_HEIGHT = 480
HUB_HTTP_TIMEOUT_SECONDS = 5.0
HUB_RETRY_SECONDS = 3.0

_TYPE_SCORES = {
    "UltraWide": 20,
    "Wide": 50,
    "Telephoto": 10,
}


def _fetch_cameras(url):
    with urllib.request.urlopen(url, timeout=HUB_HTTP_TIMEOUT_SECONDS) as response:
        return json.load(response)


def get_cameras():
    """Return the list of CameraInfo dicts reported by the CommandHub."""
    discovery.choose_hub()
    url = f"{discovery.app_url()}/api/cameras"
    try:
        cameras = _fetch_cameras(url)
    except (urllib.error.URLError, TimeoutError) as error:
        source = "Saved" if discovery.saved_url_selected else "Discovered"
        print(f"{source} CommandHub URL did not respond ({error}); searching for the hub...")
        while True:
            discovery.discover()
            url = f"{discovery.app_url()}/api/cameras"
            try:
                cameras = _fetch_cameras(url)
                break
            except (urllib.error.URLError, TimeoutError) as error:
                print(
                    f"CommandHub was discovered at {url} but is not reachable "
                    f"({error}).\n"
                    "Check that the phone is on the same Wi-Fi, the CommandHub "
                    "app is running, and VPN or guest-network isolation is not "
                    f"blocking the connection. Retrying in {HUB_RETRY_SECONDS:g} "
                    "seconds..."
                )
                time.sleep(HUB_RETRY_SECONDS)

    discovery.remember_hub_url()
    return cameras


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


def choose_starting_camera(cameras):
    """Prompt for a camera, defaulting to the highest-scoring camera."""
    if not cameras:
        raise RuntimeError("CommandHub reported no cameras")

    default_camera = select_camera(cameras)
    print("Available cameras:")
    for index, camera in enumerate(cameras, start=1):
        details = " ".join(
            value for value in (camera.get("facing"), camera.get("type")) if value
        )
        print(f"  {index}. {camera['id']} ({details})")

    while True:
        try:
            choice = input(
                f"Choose camera [1-{len(cameras)}, Enter for recommended "
                f"{default_camera['id']}]: "
            ).strip()
        except EOFError:
            return default_camera

        if not choice:
            return default_camera
        try:
            index = int(choice) - 1
        except ValueError:
            index = -1
        if 0 <= index < len(cameras):
            return cameras[index]
        print(f"Enter a number from 1 to {len(cameras)}, or press Enter for the recommended camera.")


def choose_resolution(camera, width=MODERATE_WIDTH, height=MODERATE_HEIGHT):
    """Pick the supported resolution closest to the requested moderate size."""
    resolutions = camera.get("supportedResolutions", [])
    if not resolutions:
        return width, height

    target = width * height
    best = min(resolutions, key=lambda res: abs(res["width"] * res["height"] - target))
    return best["width"], best["height"]


def choose_resolution_steps(camera, width=MODERATE_WIDTH, height=MODERATE_HEIGHT):
    """Return supported resolutions from best quality to lower quality."""
    resolutions = camera.get("supportedResolutions", [])
    if not resolutions:
        return [(width, height)]

    unique = sorted(
        {
            (int(res["width"]), int(res["height"]))
            for res in resolutions
            if int(res.get("width", 0)) > 0 and int(res.get("height", 0)) > 0
        },
        key=lambda res: res[0] * res[1],
        reverse=True,
    )

    requested = (width, height)
    if requested not in unique:
        unique.append(requested)
    unique = sorted(set(unique), key=lambda res: res[0] * res[1], reverse=True)
    return unique


def stream_url(camera, width=MODERATE_WIDTH, height=MODERATE_HEIGHT, fps=MODERATE_FPS):
    """Build the camera stream URL for the selected camera and resolution."""
    return f"{discovery.app_url()}/stream/{camera['id']}?res={width}x{height}&fps={fps}"
