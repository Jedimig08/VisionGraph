from zeroconf import Zeroconf, ServiceBrowser
from threading import Event
from pathlib import Path

hub_url = "no url"
hub_found = Event()
LAST_HUB_FILE = Path.home() / ".config" / "visiongraph" / "last_hub_url"
saved_url_selected = False


def load_last_hub_url():
    try:
        url = LAST_HUB_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return url if url.startswith(("http://", "https://")) else None


def remember_hub_url(url=None):
    url = url or hub_url
    if not url.startswith(("http://", "https://")):
        return
    LAST_HUB_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_HUB_FILE.write_text(url, encoding="utf-8")

def app_url():
    return hub_url

class MyListener:

    def add_service(self, zc, type_, name):
        global hub_url
        info = zc.get_service_info(type_, name)

        if info:
            addresses = info.parsed_addresses()

            if addresses:
                ip = addresses[0]

                print(f"{name}")
                print(f"IP: {ip}")
                print(f"Port: {info.port}")

                url = f"http://{ip}:{info.port}"
                print(url)

                normalized_name = name.casefold().replace("-", "")
                if "commandhub" in normalized_name:
                    hub_url = url
                    print(f"Set hub_url to {hub_url}")
                    hub_found.set()

    def remove_service(self, zc, type_, name):
        pass

    def update_service(self, zc, type_, name):
        pass


def discover():
    global hub_url, saved_url_selected
    hub_url = "no url"
    saved_url_selected = False
    hub_found.clear()
    zc = Zeroconf()
    browser = ServiceBrowser(
        zc,
        "_http._tcp.local.",
        MyListener()
    )
    print("Searching for local HTTP services...")
    try:
        hub_found.wait()
    finally:
        zc.close()


def choose_hub():
    global hub_url, saved_url_selected
    saved_url = load_last_hub_url()
    if saved_url:
        hub_url = saved_url
        saved_url_selected = True
        print(f"Using saved CommandHub URL: {hub_url}")
        return
    discover()