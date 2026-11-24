"""Persist named surface calibrations between VisionGraph runs."""

import json
from pathlib import Path

CALIBRATIONS_FILE = Path.home() / ".config" / "visiongraph" / "calibrations.json"
CATALOG_VERSION = 1


def calibration_path(path=None):
    return CALIBRATIONS_FILE if path is None else Path(path)


def _load_json(source):
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Saved calibrations file is not valid JSON: {source}") from exc


def _read_catalog(path=None):
    source = calibration_path(path)
    data = _load_json(source)
    if data is None:
        return {"version": CATALOG_VERSION, "last": None, "profiles": {}}
    if not isinstance(data, dict):
        raise ValueError(f"Saved calibrations file is malformed: {source}")
    profiles = data.get("profiles", data)
    if not isinstance(profiles, dict):
        raise ValueError(f"Saved calibrations file is malformed: {source}")
    last = data.get("last")
    return {
        "version": CATALOG_VERSION,
        "last": last if isinstance(last, str) and last in profiles else None,
        "profiles": profiles,
    }


def _write_catalog(catalog, path=None):
    destination = calibration_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _validate_name(name):
    if not isinstance(name, str):
        raise ValueError("Calibration name must be text")
    name = name.strip()
    if not name:
        raise ValueError("Calibration name cannot be empty")
    return name


def names(path=None):
    """Return the saved calibration names, sorted alphabetically."""
    return sorted(_read_catalog(path)["profiles"])


def last_name(path=None):
    """Return the most recently saved or loaded calibration name, if any."""
    return _read_catalog(path)["last"]


def load(name, path=None):
    """Return the saved payload for a name, or None when it does not exist."""
    return _read_catalog(path)["profiles"].get(name)


def save(name, payload, path=None):
    """Store a payload under a name and remember it as the last used."""
    name = _validate_name(name)
    catalog = _read_catalog(path)
    catalog["profiles"][name] = payload
    catalog["last"] = name
    return _write_catalog(catalog, path)


def mark_used(name, path=None):
    """Remember a name as the last used one without rewriting its payload."""
    catalog = _read_catalog(path)
    if name not in catalog["profiles"]:
        return False
    catalog["last"] = name
    _write_catalog(catalog, path)
    return True


def delete(name, path=None):
    """Remove a stored calibration by name; return whether it existed."""
    catalog = _read_catalog(path)
    if name not in catalog["profiles"]:
        return False
    del catalog["profiles"][name]
    if catalog["last"] == name:
        catalog["last"] = None
    _write_catalog(catalog, path)
    return True
