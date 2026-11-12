import unittest
from types import SimpleNamespace
from unittest import mock

import discovery


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        discovery.hub_url = "no url"
        discovery.hub_found.clear()

    def tearDown(self):
        discovery.hub_url = "no url"
        discovery.hub_found.clear()

    def test_discover_waits_until_commandhub_is_found(self):
        info = SimpleNamespace(
            parsed_addresses=lambda: ["192.168.1.20"],
            port=8080,
        )
        zeroconf = mock.Mock()
        zeroconf.get_service_info.return_value = info
        listeners = []

        def start_browser(_zeroconf, _service_type, listener):
            listeners.append(listener)
            return mock.Mock()

        def wait_for_hub(*args, **kwargs):
            self.assertEqual(args, ())
            self.assertEqual(kwargs, {})
            listeners[0].add_service(
                zeroconf,
                "_http._tcp.local.",
                "printer._http._tcp.local.",
            )
            self.assertFalse(discovery.hub_found.is_set())
            listeners[0].add_service(
                zeroconf,
                "_http._tcp.local.",
                "CommandHub._http._tcp.local.",
            )
            return discovery.hub_found.is_set()

        with (
            mock.patch.object(discovery, "Zeroconf", return_value=zeroconf),
            mock.patch.object(discovery, "ServiceBrowser", side_effect=start_browser),
            mock.patch.object(discovery.hub_found, "wait", side_effect=wait_for_hub) as wait,
        ):
            discovery.discover()

        wait.assert_called_once_with()
        self.assertEqual(discovery.app_url(), "http://192.168.1.20:8080")
        zeroconf.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
