from threading import Event
from pathlib import Path

Zeroconf = None
ServiceBrowser = None

hub_url = "no url"
static_hub_url = None
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
    if static_hub_url:
        return
    url = url or hub_url
    if not url.startswith(("http://", "https://")):
        return
    LAST_HUB_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_HUB_FILE.write_text(url, encoding="utf-8")

def app_url():
    return static_hub_url or hub_url

def use_static_url(url):
    global static_hub_url, hub_url
    if not url.startswith(("http://", "https://")):
        raise ValueError("CommandHub URL must start with http:// or https://")
    static_hub_url = url.rstrip("/")
    hub_url = static_hub_url

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
    global hub_url, saved_url_selected, Zeroconf, ServiceBrowser
    if static_hub_url:
        return
    if Zeroconf is None or ServiceBrowser is None:
        from zeroconf import Zeroconf as ZeroconfClass, ServiceBrowser as ServiceBrowserClass
        Zeroconf = ZeroconfClass
        ServiceBrowser = ServiceBrowserClass
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
    if static_hub_url:
        return
    saved_url = load_last_hub_url()
    if saved_url:
        hub_url = saved_url
        saved_url_selected = True
        print(f"Using saved CommandHub URL: {hub_url}")
        return
    discover()