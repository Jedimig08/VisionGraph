import unittest

import discovery


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        discovery.hub_url = discovery.LOCAL_HUB_URL

    def tearDown(self):
        discovery.hub_url = discovery.LOCAL_HUB_URL

    def test_default_url_is_local_commandhub(self):
        self.assertEqual(discovery.app_url(), "http://127.0.0.1:8080")

    def test_choose_hub_and_discover_are_no_ops(self):
        discovery.choose_hub()
        discovery.discover()
        self.assertEqual(discovery.app_url(), "http://127.0.0.1:8080")

    def test_use_static_url_normalizes_trailing_slash(self):
        discovery.use_static_url("http://127.0.0.1:8080/")
        self.assertEqual(discovery.app_url(), "http://127.0.0.1:8080")

    def test_use_static_url_rejects_non_http(self):
        with self.assertRaises(ValueError):
            discovery.use_static_url("javascript:alert(1)")

    def test_remember_hub_url_does_not_write(self):
        discovery.remember_hub_url("http://example.com")
        self.assertEqual(discovery.app_url(), "http://127.0.0.1:8080")


if __name__ == "__main__":
    unittest.main()
