from zeroconf import Zeroconf, ServiceBrowser
import time

hub_url = "no url"

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

                if name == "command-hub._http._tcp.local.":
                    hub_url = url


zeroconf = Zeroconf()

browser = ServiceBrowser(
    zeroconf,
    "_http._tcp.local.",
    MyListener()
)

print("Searching for local HTTP services...")

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    zeroconf.close()