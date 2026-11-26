from pathlib import Path

LOCAL_HUB_URL = "http://127.0.0.1:8080"

hub_url = LOCAL_HUB_URL
LAST_HUB_FILE = Path.home() / ".config" / "visiongraph" / "last_hub_url"
saved_url_selected = False


def app_url():
    return hub_url


def use_static_url(url):
    global hub_url
    if not url.startswith(("http://", "https://")):
        raise ValueError("CommandHub URL must start with http:// or https://")
    hub_url = url.rstrip("/")


def remember_hub_url(url=None):
    return


def discover():
    return


def choose_hub():
    return
