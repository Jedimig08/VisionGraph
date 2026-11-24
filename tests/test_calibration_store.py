import tempfile
import unittest
from pathlib import Path

import calibration_store


class CalibrationStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nested" / "calibrations.json"

    @staticmethod
    def payload(label):
        return {
            "version": 1,
            "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "label": label,
        }

    def test_save_creates_parent_directories_and_load_returns_payload(self):
        saved = calibration_store.save("bench", self.payload("a"), self.path)

        self.assertEqual(saved, self.path)
        self.assertEqual(calibration_store.load("bench", self.path), self.payload("a"))

    def test_names_lists_profiles_alphabetically(self):
        calibration_store.save("zeta", self.payload("z"), self.path)
        calibration_store.save("alpha", self.payload("a"), self.path)

        self.assertEqual(calibration_store.names(self.path), ["alpha", "zeta"])

    def test_last_name_tracks_most_recent_save_and_mark_used(self):
        calibration_store.save("first", self.payload("1"), self.path)
        calibration_store.save("second", self.payload("2"), self.path)
        self.assertEqual(calibration_store.last_name(self.path), "second")

        self.assertTrue(calibration_store.mark_used("first", self.path))
        self.assertEqual(calibration_store.last_name(self.path), "first")
        self.assertFalse(calibration_store.mark_used("missing", self.path))

    def test_delete_removes_only_the_named_profile(self):
        calibration_store.save("keep", self.payload("k"), self.path)
        calibration_store.save("drop", self.payload("d"), self.path)

        self.assertTrue(calibration_store.delete("drop", self.path))
        self.assertEqual(calibration_store.names(self.path), ["keep"])
        self.assertIsNone(calibration_store.load("drop", self.path))
        self.assertFalse(calibration_store.delete("drop", self.path))

    def test_load_returns_none_when_nothing_was_saved(self):
        self.assertIsNone(calibration_store.load("bench", self.path))
        self.assertEqual(calibration_store.names(self.path), [])

    def test_blank_names_are_rejected(self):
        with self.assertRaises(ValueError):
            calibration_store.save("   ", self.payload("x"), self.path)

    def test_load_rejects_malformed_json(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("not json", encoding="utf-8")

        with self.assertRaises(ValueError):
            calibration_store.load("bench", self.path)
