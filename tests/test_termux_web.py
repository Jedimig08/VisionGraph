import json
import unittest
from types import SimpleNamespace

import termux_web


class FakeProcessor:
    fps = 5.0

    def pose_json(self):
        return {
            "valid": True,
            "reason": "",
            "x_mm": 12.5,
            "y_mm": 30.0,
            "angle_deg": 1.25,
            "raw_angle_deg": 1.3,
            "reliability": 0.9,
            "predicted": False,
            "prediction_age_seconds": 0.0,
            "inside_drawing_area": True,
            "refined": False,
            "fit_error_px": None,
            "calibrated": True,
            "calibration_pending": False,
            "calibration_confirmed": True,
            "timestamp": 100.0,
        }

    def status_json(self):
        return {"target_fps": 5.0, "actual_fps": 4.8, "view_mode": "calibrated"}

    def frame_bytes(self):
        return b"jpeg-bytes"

    def confirm_calibration(self):
        return {"confirmed": True, "message": "calibration locked"}

    def clear_calibration(self):
        return {"message": "calibration cleared"}

    def save_calibration(self, name):
        return {"saved": True, "message": f"saved {name}"}

    def load_calibration(self, name):
        return {"loaded": True, "message": f"loaded {name}"}

    def calibration_names(self):
        return {"names": ["one", "two"]}

    def set_view(self, mode):
        return {"message": f"view mode set to {mode}"}


class TermuxWebTests(unittest.TestCase):
    def setUp(self):
        self.app = termux_web.create_app(FakeProcessor())
        self.client = self.app.test_client()

    def test_page_has_controls_and_poll_rate(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Confirm calibration", body)
        self.assertIn("frame.jpg", body)
        self.assertIn("const fps = 5.0;", body)
        self.assertIn("setInterval(refresh, 1000 / fps)", body)

    def test_pose_api_fields(self):
        data = json.loads(self.client.get("/api/pose").get_data(as_text=True))
        self.assertEqual(data["x_mm"], 12.5)
        self.assertEqual(data["y_mm"], 30.0)
        self.assertEqual(data["angle_deg"], 1.25)
        self.assertIn("calibration_confirmed", data)
        self.assertIn("prediction_age_seconds", data)

    def test_status_and_calibration_names_api(self):
        status = json.loads(self.client.get("/api/status").get_data(as_text=True))
        self.assertEqual(status["actual_fps"], 4.8)
        names = json.loads(self.client.get("/api/calibration/names").get_data(as_text=True))
        self.assertEqual(names["names"], ["one", "two"])

    def test_frame_route(self):
        response = self.client.get("/frame.jpg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertEqual(response.data, b"jpeg-bytes")

    def test_control_routes(self):
        result = self.client.post("/api/calibration/confirm").get_json()
        self.assertTrue(result["confirmed"])
        self.assertEqual(
            self.client.post("/api/calibration/clear").get_json()["message"],
            "calibration cleared",
        )
        self.assertEqual(
            self.client.post(
                "/api/calibration/save", json={"name": "test"}
            ).get_json()["message"],
            "saved test",
        )
        self.assertEqual(
            self.client.post(
                "/api/calibration/load", json={"name": "test"}
            ).get_json()["message"],
            "loaded test",
        )
        self.assertEqual(
            self.client.post("/api/view", json={"mode": "raw"}).get_json()["message"],
            "view mode set to raw",
        )

    def test_empty_name_is_rejected(self):
        self.assertEqual(self.client.post("/api/calibration/save", json={"name": ""}).status_code, 400)
        self.assertEqual(self.client.post("/api/calibration/load", json={}).status_code, 400)

    def test_processor_output_size_and_snapshot_schema(self):
        processor = termux_web.Processor(400, 300, 5)
        self.assertEqual(processor.output_size, (1000, 750))
        estimate = SimpleNamespace(
            valid=True,
            reason="",
            x_mm=10.0,
            y_mm=20.0,
            angle_deg=0.5,
            raw_angle_deg=0.6,
            reliability=0.8,
            predicted=False,
            prediction_age_seconds=0.0,
            inside_drawing_area=True,
            refined=False,
            fit_error_px=None,
        )
        result = SimpleNamespace(
            calibrated=True,
            calibration_pending=False,
            calibration_confirmed=True,
        )
        snapshot = processor._pose_snapshot(estimate, result, 42.0)
        self.assertEqual(snapshot["timestamp"], 42.0)
        self.assertTrue(snapshot["valid"])
        self.assertTrue(snapshot["calibrated"])


if __name__ == "__main__":
    unittest.main()
