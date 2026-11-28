import math

import camera_select
import marker_tracking


def validate_dimensions(width, height):
    width = float(width)
    height = float(height)
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("Drawing area dimensions must be finite and positive")
    return width, height


def parse_resolution(value):
    if not value:
        return None
    parts = str(value).lower().replace("x", ":").split(":")
    if len(parts) != 2:
        raise ValueError("resolution must look like 640x480")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")
    return width, height


def format_resolution(resolution):
    if not resolution:
        return ""
    return f"{resolution[0]}x{resolution[1]}"


def select_camera(cameras, camera_id=None):
    if not cameras:
        raise RuntimeError("CommandHub reported no cameras")
    if camera_id:
        for camera in cameras:
            if str(camera.get("id")) == str(camera_id):
                return camera
        raise RuntimeError(f"camera {camera_id} is not available")
    return camera_select.select_camera(cameras)


def select_resolution(camera, resolution=None):
    if not resolution:
        return camera_select.choose_resolution(
            camera,
            camera_select.MODERATE_WIDTH,
            camera_select.MODERATE_HEIGHT,
        )
    width, height = resolution
    supported = {
        (int(entry["width"]), int(entry["height"]))
        for entry in camera.get("supportedResolutions", [])
    }
    if supported and (width, height) not in supported:
        return camera_select.choose_resolution(camera, width, height)
    return width, height


def output_size_for_area(width, height):
    return marker_tracking.output_size_for_area(width, height)
