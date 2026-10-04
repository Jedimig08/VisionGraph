from zeroconf import Zeroconf, ServiceBrowser
from threading import Event

hub_url = "no url"
hub_found = Event()

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

                if "command-hub" in name:
                    hub_url = url
                    print(f"Set hub_url to {hub_url}")
                    hub_found.set()

    def remove_service(self, zc, type_, name):
        pass

    def update_service(self, zc, type_, name):
        pass


def discover():
    zc = Zeroconf()
    ServiceBrowser(
        zc,
        "_http._tcp.local.",
        MyListener()
    )
    print("Searching for local HTTP services...")
    try:
        hub_found.wait()
    finally:
        zc.close()

# Run discovery once on import until the hub is found
discover()