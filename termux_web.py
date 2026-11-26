import argparse
import json
import threading
import time
from collections import deque

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

import calibration_store
import camera_select
import discovery
import marker_tracking
import pose_estimator

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisionGraph Termux</title>
<style>
body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 16px; }
h1 { font-size: 1.2rem; color: #38bdf8; }
img { width: 100%; max-width: 900px; border: 1px solid #334155; }
pre { background: #1e293b; padding: 12px; overflow-x: auto; border-radius: 8px; }
button { padding: 8px 12px; margin: 4px 4px 4px 0; background: #0284c7; color: white; border: 0; border-radius: 6px; cursor: pointer; }
input { padding: 8px; background: #0f172a; color: white; border: 1px solid #475569; border-radius: 6px; }
.row { margin: 12px 0; }
#message { color: #38bdf8; min-height: 1.2em; }
#freshness { font-weight: bold; margin: 8px 0; }
</style>
</head>
<body>
<h1>VisionGraph Termux</h1>
<img id="frame" src="frame.jpg" alt="camera">
<div class="row">
<button onclick="action('calibration/confirm', {})">Confirm calibration</button>
<button onclick="action('calibration/clear', {})">Clear / recalibrate</button>
<button onclick="action('view', {mode: 'calibrated'})">Calibrated view</button>
<button onclick="action('view', {mode: 'raw'})">Raw view</button>
</div>
<div class="row">
<input id="calName" placeholder="calibration name">
<button onclick="action('calibration/save', {name: document.getElementById('calName').value})">Save</button>
<button onclick="action('calibration/load', {name: document.getElementById('calName').value})">Load</button>
</div>
<div id="message"></div>
<div id="freshness">no updates</div>
<h3>Pose</h3><pre id="pose">loading...</pre>
<h3>Detection</h3><pre id="diagnostics">loading...</pre>
<h3>Status</h3><pre id="status">loading...</pre>
<script>
const fps = {{ fps }};
function get(path) {
  return fetch(path + '?t=' + Date.now()).then(r => r.json());
}
function markStale(updated) {
  const el = document.getElementById('freshness');
  if (!updated) { el.textContent = 'no updates'; return; }
  const age = Date.now() / 1000 - updated;
  el.textContent = age > 3 / fps ? 'STALE (' + age.toFixed(1) + 's old)' : 'live (' + age.toFixed(1) + 's ago)';
}
function refresh() {
  document.getElementById('frame').src = 'frame.jpg?t=' + Date.now();
  get('api/pose').then(d => {
    document.getElementById('pose').textContent = JSON.stringify(d, null, 2);
  });
  get('api/diagnostics').then(d => {
    document.getElementById('diagnostics').textContent = JSON.stringify(d, null, 2);
  });
  get('api/status').then(d => {
    document.getElementById('status').textContent = JSON.stringify(d, null, 2);
    markStale(d.updated_at);
  });
}
function action(path, body) {
  fetch('api/' + path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  }).then(r => r.json()).then(d => {
    document.getElementById('message').textContent = d.message || JSON.stringify(d);
    refresh();
  });
}
refresh();
setInterval(refresh, 1000 / fps);
</script>
</body>
</html>
"""


class Processor:
    def __init__(self, area_width, area_height, fps, view_mode="calibrated", calibration=None):
        self.area_width = float(area_width)
        self.area_height = float(area_height)
        self.fps = float(fps)
        self.view_mode = view_mode
        self.output_size = marker_tracking.output_size_for_area(self.area_width, self.area_height)
        self.detector = marker_tracking.ArucoMarkerDetector()
        self.tracker = self._new_tracker(calibration)
        self.pose = self._new_pose()
        self.calibration_name = None
        self.camera = None
        self.resolution = None
        self.stream_url = ""
        self.last_error = None
        self.frame_times = deque()
        self.action_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.running = True
        self.jpeg = None
        self.snapshot = self._empty_snapshot()
        self.diagnostics = self._empty_diagnostics()

    def _new_tracker(self, calibration=None):
        tracker = marker_tracking.MarkerTracker(output_size=self.output_size)
        if calibration is not None:
            tracker.import_calibration(calibration)
        return tracker

    def _new_pose(self):
        return pose_estimator.PoseEstimator(
            pose_estimator.PoseEstimatorConfig(
                drawing_width_mm=self.area_width,
                drawing_height_mm=self.area_height,
                output_size=self.output_size,
            )
        )

    def _empty_snapshot(self):
        return {
            "valid": False,
            "reason": "starting",
            "x_mm": None,
            "y_mm": None,
            "angle_deg": None,
            "raw_angle_deg": None,
            "reliability": 0.0,
            "predicted": False,
            "prediction_age_seconds": 0.0,
            "inside_drawing_area": False,
            "refined": False,
            "fit_error_px": None,
            "calibrated": False,
            "calibration_pending": False,
            "calibration_confirmed": False,
            "timestamp": None,
            "updated_at": None,
        }

    def _empty_diagnostics(self):
        return {
            "detected_ids": [],
            "missing_ids": [],
            "remembered_ids": [],
            "tracked_ids": [],
            "calibration_observations": {},
            "rejected_count": 0,
            "reliability": {},
            "calibration_pending": False,
            "calibration_confirmed": False,
            "calibrated": False,
        }

    def _result_diagnostics(self, result):
        return {
            "detected_ids": sorted(int(value) for value in result.live),
            "missing_ids": sorted(int(value) for value in result.missing),
            "remembered_ids": sorted(int(value) for value in result.remembered),
            "tracked_ids": sorted(int(value) for value in result.tracked_ids),
            "calibration_observations": {
                str(key): int(value)
                for key, value in result.calibration_observations.items()
            },
            "rejected_count": int(result.rejected_count),
            "reliability": {
                str(key): float(value) for key, value in result.reliability.items()
            },
            "calibration_pending": bool(result.calibration_pending),
            "calibration_confirmed": bool(result.calibration_confirmed),
            "calibrated": bool(result.calibrated),
        }

    def _error_snapshot(self, reason):
        snapshot = self._empty_snapshot()
        snapshot["reason"] = reason
        snapshot["updated_at"] = time.time()
        return snapshot

    def configure_stream(self):
        cameras = camera_select.get_cameras()
        self.camera = camera_select.select_camera(cameras)
        self.resolution = camera_select.choose_resolution(
            self.camera,
            camera_select.MODERATE_WIDTH,
            camera_select.MODERATE_HEIGHT,
        )
        self.stream_url = camera_select.stream_url(
            self.camera,
            self.resolution[0],
            self.resolution[1],
            self.fps,
        )
        return self.stream_url

    def open_capture(self):
        capture = cv2.VideoCapture(self.stream_url)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open stream {self.stream_url}")
        return capture

    def run(self, capture):
        interval = 1.0 / self.fps
        while self.running:
            started = time.monotonic()
            ret, frame = capture.read()
            if not ret or frame is None:
                with self.state_lock:
                    self.last_error = "camera stream ended"
                    self.snapshot = self._error_snapshot("stream unavailable")
                capture.release()
                time.sleep(min(interval, 1.0))
                while self.running:
                    try:
                        capture = self.open_capture()
                        break
                    except Exception as exc:
                        with self.state_lock:
                            self.last_error = str(exc)
                        time.sleep(1.0)
                continue
            now = time.monotonic()
            try:
                with self.action_lock:
                    corners, ids, rejected = self.detector.detect(frame)
                    result = self.tracker.update(
                        corners, ids, rejected, frame.shape, now=now
                    )
                    estimate = self.pose.update(result, frame=frame, now=now)
                    display = self._display_frame(frame, result)
                    ok, encoded = cv2.imencode(".jpg", display)
                    jpeg = encoded.tobytes() if ok else None
                    diagnostics = self._result_diagnostics(result)
                self.frame_times.append(now)
                while self.frame_times and self.frame_times[0] < now - 5.0:
                    self.frame_times.popleft()
                with self.state_lock:
                    self.jpeg = jpeg
                    self.snapshot = self._pose_snapshot(estimate, result, now)
                    self.diagnostics = diagnostics
                    self.last_error = None
            except Exception as exc:
                with self.state_lock:
                    self.last_error = str(exc)
                    self.snapshot = self._error_snapshot(str(exc))
                    self.diagnostics = self._empty_diagnostics()
                time.sleep(interval)
                continue
            delay = interval - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)

    def _display_frame(self, frame, result):
        if self.view_mode == "calibrated" and result.calibration is not None:
            return cv2.warpPerspective(frame, result.calibration, result.output_size)
        return frame.copy()

    def _pose_snapshot(self, estimate, result, now):
        return {
            "valid": bool(estimate.valid),
            "reason": estimate.reason,
            "x_mm": estimate.x_mm,
            "y_mm": estimate.y_mm,
            "angle_deg": estimate.angle_deg,
            "raw_angle_deg": estimate.raw_angle_deg,
            "reliability": estimate.reliability,
            "predicted": bool(estimate.predicted),
            "prediction_age_seconds": estimate.prediction_age_seconds,
            "inside_drawing_area": bool(estimate.inside_drawing_area),
            "refined": bool(estimate.refined),
            "fit_error_px": estimate.fit_error_px,
            "calibrated": bool(result.calibrated),
            "calibration_pending": bool(result.calibration_pending),
            "calibration_confirmed": bool(result.calibration_confirmed),
            "timestamp": now,
            "updated_at": time.time(),
        }

    def pose_json(self):
        with self.state_lock:
            return dict(self.snapshot)

    def diagnostics_json(self):
        with self.state_lock:
            data = dict(self.diagnostics)
            data["last_error"] = self.last_error
            data["updated_at"] = self.snapshot.get("updated_at")
            return data

    def status_json(self):
        with self.state_lock:
            elapsed = list(self.frame_times)
            actual_fps = (len(elapsed) - 1) / (elapsed[-1] - elapsed[0]) if len(elapsed) >= 2 else 0.0
            return {
                "camera_id": self.camera["id"] if self.camera else None,
                "camera_facing": self.camera.get("facing") if self.camera else None,
                "camera_type": self.camera.get("type") if self.camera else None,
                "resolution": list(self.resolution) if self.resolution else None,
                "target_fps": self.fps,
                "actual_fps": actual_fps,
                "stream_url": self.stream_url,
                "stream_connected": self.last_error is None,
                "last_error": self.last_error,
                "view_mode": self.view_mode,
                "calibration_name": self.calibration_name,
                "drawing_area": [self.area_width, self.area_height],
                "output_size": list(self.output_size),
                "updated_at": self.snapshot.get("updated_at"),
            }

    def frame_bytes(self):
        with self.state_lock:
            return self.jpeg

    def confirm_calibration(self):
        with self.action_lock:
            confirmed = self.tracker.confirm_calibration()
        return {"confirmed": confirmed, "message": "calibration locked" if confirmed else "calibration not ready"}

    def clear_calibration(self):
        with self.action_lock:
            self.tracker = self._new_tracker()
            self.pose = self._new_pose()
            self.calibration_name = None
        return {"message": "calibration cleared; collecting observations again"}

    def calibration_payload(self):
        with self.action_lock:
            payload = self.tracker.export_calibration()
        if payload is None:
            return None
        payload["drawing_area"] = [self.area_width, self.area_height]
        payload["resolution"] = list(self.resolution) if self.resolution else None
        return payload

    def save_calibration(self, name):
        payload = self.calibration_payload()
        if payload is None:
            return {"saved": False, "message": "calibration is not ready to save"}
        try:
            marker_tracking.normalize_calibration(payload)
            destination = calibration_store.save(name, payload)
        except (OSError, ValueError, TypeError) as exc:
            return {"saved": False, "message": f"could not save calibration: {exc}"}
        with self.state_lock:
            self.calibration_name = name
        return {"saved": True, "message": f"calibration saved as '{name}' to {destination}"}

    def load_calibration(self, name):
        try:
            payload = calibration_store.load(name)
            normalized = marker_tracking.normalize_calibration(payload)
        except (OSError, ValueError, TypeError) as exc:
            return {"loaded": False, "message": f"could not load calibration: {exc}"}
        if payload is None:
            return {"loaded": False, "message": f"no saved calibration named '{name}'"}
        saved_resolution = normalized.get("resolution")
        if saved_resolution and self.resolution and tuple(saved_resolution) != tuple(self.resolution):
            return {
                "loaded": False,
                "message": "saved calibration uses a different camera resolution",
            }
        with self.action_lock:
            self.tracker.import_calibration(normalized)
            self.pose = self._new_pose()
            self.calibration_name = name
        calibration_store.mark_used(name)
        return {"loaded": True, "message": f"loaded calibration '{name}'"}

    def calibration_names(self):
        try:
            return {"names": calibration_store.names()}
        except (OSError, ValueError) as exc:
            return {"names": [], "message": str(exc)}

    def set_view(self, mode):
        if mode not in ("raw", "calibrated"):
            return {"message": "view mode must be raw or calibrated"}
        with self.action_lock:
            self.view_mode = mode
        return {"message": f"view mode set to {mode}"}


def create_app(processor):
    app = Flask(__name__)

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return render_template_string(PAGE, fps=processor.fps)

    @app.get("/frame.jpg")
    def frame():
        jpeg = processor.frame_bytes()
        if jpeg is None:
            return Response(status=503)
        return Response(jpeg, mimetype="image/jpeg")

    @app.get("/api/pose")
    def pose():
        return jsonify(processor.pose_json())

    @app.get("/api/status")
    def status():
        return jsonify(processor.status_json())

    @app.get("/api/diagnostics")
    def diagnostics():
        return jsonify(processor.diagnostics_json())

    @app.get("/api/calibration/names")
    def calibration_names():
        return jsonify(processor.calibration_names())

    @app.post("/api/calibration/confirm")
    def confirm():
        return jsonify(processor.confirm_calibration())

    @app.post("/api/calibration/clear")
    def clear():
        return jsonify(processor.clear_calibration())

    @app.post("/api/calibration/save")
    def save():
        name = request.get_json(silent=True).get("name", "") if request.get_json(silent=True) else ""
        if not name.strip():
            return jsonify({"saved": False, "message": "a calibration name is required"}), 400
        return jsonify(processor.save_calibration(name.strip()))

    @app.post("/api/calibration/load")
    def load():
        name = request.get_json(silent=True).get("name", "") if request.get_json(silent=True) else ""
        if not name.strip():
            return jsonify({"loaded": False, "message": "a calibration name is required"}), 400
        return jsonify(processor.load_calibration(name.strip()))

    @app.post("/api/view")
    def view():
        payload = request.get_json(silent=True) or {}
        return jsonify(processor.set_view(payload.get("mode", "")))

    return app


def main():
    parser = argparse.ArgumentParser(description="VisionGraph web viewer for Termux")
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("fps must be positive")

    discovery.use_static_url("http://127.0.0.1:8080")
    processor = Processor(args.width, args.height, args.fps)
    processor.configure_stream()
    capture = processor.open_capture()
    thread = threading.Thread(target=processor.run, args=(capture,), daemon=True)
    thread.start()
    create_app(processor).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
