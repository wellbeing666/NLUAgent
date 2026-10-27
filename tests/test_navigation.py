"""高德适配器契约测试：全部使用固定响应，不消耗真实接口配额。"""

import base64
import json
import unittest
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from cockpit.navigation import AmapClient, NavigationError, NoRedirect, coordinate, navigation_url, matching_places, distance_m
from cockpit.tools import ToolService

ORIGIN = {"name": "北京南站", "location": "116.378517,39.865246", "id": "TEST_ORIGIN"}
DESTINATION = {"name": "首都机场T3", "location": "116.615095,40.052596", "id": "TEST_DEST"}
SEARCH_RESPONSE = {"status": "1", "pois": [
    {**ORIGIN, "pname": "北京市", "cityname": [], "adname": "丰台区", "address": "测试地址"},
    {"name": "无坐标地点", "location": []},
]}
DRIVING_RESPONSE = {"status": "1", "route": {"paths": [
    {"distance": "48000", "strategy": "高德推荐", "cost": {"duration": "3600", "tolls": "15"},
     "steps": [{"instruction": "沿测试道路向东行驶", "road_name": "测试道路", "step_distance": "1000",
                "polyline": "116.378517,39.865246;116.500000,39.950000;116.615095,40.052596"}]},
    {"distance": "51000", "cost": {"duration": "3900", "tolls": "0"}, "steps": []},
]}}
NEARBY_RESPONSE = {"status": "1", "pois": [
    {"name": "北京南站", "location": "116.450000,39.800000", "id": "TEST_FAR"},
    {"name": "北京南站停车场", "location": "116.397428,39.909230", "id": "TEST_PARKING"},
    {"name": "北京西站", "location": "116.397420,39.909230", "id": "TEST_WRONG"},
    ORIGIN,
]}
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lL0AAAAASUVORK5CYII=')


class FixtureTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, url):
        parsed = urlsplit(url)
        self.calls.append((parsed.path, parse_qs(parsed.query)))
        if parsed.path == "/v3/staticmap":
            return PNG, "image/png"
        response = SEARCH_RESPONSE if parsed.path == "/v3/place/text" else DRIVING_RESPONSE
        if parsed.path == "/v3/place/around":
            response = NEARBY_RESPONSE
        if parsed.path == "/v3/place/text" and "机场" in parse_qs(parsed.query).get("keywords", [""])[0]:
            response = {"status": "1", "pois": [{**DESTINATION, "address": "界面测试终点，不是真实搜索结果"}]}
        return json.dumps(response).encode(), "application/json"


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.transport = FixtureTransport()
        self.client = AmapClient(key="private-test-key", transport=self.transport)

    def test_coordinate_validation_and_rounding(self):
        self.assertEqual(coordinate("116.12345678,39.9"), "116.123457,39.900000")
        for value in [[], True, "nan,30", "181,30", "116,91", "1,2,3", "a,b", None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                coordinate(value)

    def test_search_uses_city_limit_and_skips_unusable_coordinates(self):
        data = self.client.search("南站", "北京")
        self.assertEqual(len(data["places"]), 1)
        self.assertEqual(data["places"][0]["name"], ORIGIN["name"])
        self.assertEqual(self.transport.calls[0][1]["citylimit"], ["true"])
        self.assertEqual(data["coordinate_system"], "GCJ-02")

    def test_invalid_search_never_calls_provider(self):
        for keywords, city in [("", "北京"), ("站", ""), (1, "北京"), ("x" * 101, "北京")]:
            with self.assertRaises(ValueError):
                self.client.search(keywords, city)
        self.assertEqual(self.transport.calls, [])

    def test_v5_driving_units_fields_and_strategy(self):
        data = self.client.driving(ORIGIN, DESTINATION, "no_highway")
        route = data["routes"][0]
        self.assertEqual(route["distance_km"], 48)
        self.assertEqual(route["duration_minutes"], 60)
        self.assertEqual(route["tolls_yuan"], 15)
        self.assertEqual(len(route["geometry"]), 3)
        self.assertNotIn("arrival_battery_percent", route)
        _, params = self.transport.calls[-1]
        self.assertEqual(params["strategy"], ["35"])
        self.assertEqual(params["show_fields"], ["cost,polyline,navi"])
        self.assertEqual(params["destination_id"], ["TEST_DEST"])

    def test_invalid_plan_does_not_call_provider(self):
        for origin, destination, strategy in [(ORIGIN, ORIGIN, "recommended"),
                                              (ORIGIN, DESTINATION, "unknown"),
                                              ({"name": "x", "location": "nan,2"}, DESTINATION, "recommended")]:
            with self.assertRaises(ValueError):
                self.client.driving(origin, destination, strategy)
        self.assertEqual(self.transport.calls, [])

    def test_missing_configuration_never_calls_network(self):
        client = AmapClient(key="", transport=lambda url: self.fail("不应联网"))
        with self.assertRaises(NavigationError) as error:
            client.search("站", "北京")
        self.assertEqual(error.exception.code, "AMAP_NOT_CONFIGURED")

    def test_provider_error_does_not_echo_key_or_raw_response(self):
        for response in [{"status": "0", "infocode": "10001", "info": "private-test-key"}, [], "garbage"]:
            client = AmapClient(key="private-test-key", transport=lambda url: (json.dumps(response).encode(), "application/json"))
            with self.assertRaises(NavigationError) as error:
                client.search("站", "北京")
            self.assertNotIn("private-test-key", str(error.exception))

    def test_network_exceptions_are_redacted(self):
        with patch('cockpit.navigation.build_opener') as opener:
            opener.return_value.open.side_effect = TimeoutError("url?key=private-test-key")
            with self.assertRaises(NavigationError) as error:
                self.client._http("https://restapi.amap.com/?key=private-test-key")
        self.assertEqual(error.exception.code, "AMAP_NETWORK")
        self.assertNotIn("private-test-key", str(error.exception))

    def test_redirect_is_not_followed(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com"))

    def test_incomplete_route_is_error_instead_of_mock_success(self):
        for paths in [[], [{}], [{"distance": "NaN", "cost": {"duration": "60"}}]]:
            response = {"status": "1", "route": {"paths": paths}}
            client = AmapClient(key="test", transport=lambda url: (json.dumps(response).encode(), "application/json"))
            with self.assertRaises(NavigationError):
                client.driving(ORIGIN, DESTINATION)

    def test_navigation_uri_encodes_names_and_translates_strategy(self):
        url = navigation_url({**ORIGIN, "name": "起点 & A"}, DESTINATION, "no_highway", True)
        parsed = urlsplit(url)
        self.assertEqual(parsed.netloc, "uri.amap.com")
        params = parse_qs(parsed.query)
        self.assertEqual(params["policy"], ["3"])
        self.assertEqual(params["callnative"], ["1"])
        self.assertIn("起点 & A", params["from"][0])
        self.assertNotIn("key", params)

    def test_map_sampling_auto_fit_and_no_key_in_response(self):
        plan = self.client.driving(ORIGIN, DESTINATION)
        route = plan["routes"][0]
        route["geometry"] *= 200
        result = self.client.static_map(plan, route)
        params = self.transport.calls[-1][1]
        self.assertLessEqual(len(params["paths"][0].split(":" , 1)[1].split(";")), 80)
        self.assertNotIn("zoom", params)
        self.assertNotIn("location", params)
        self.assertIn("data:image/png;base64,", result["image"])
        self.assertNotIn("private-test-key", json.dumps(result))

    def test_service_real_provenance_private_geometry_and_map_cache(self):
        events = []
        service = ToolService(lambda e, d: events.append((e, d)), navigation_factory=lambda: self.client)
        result = service.execute("plan_navigation_route", {"origin": ORIGIN, "destination": DESTINATION})
        self.assertTrue(result["ok"])
        self.assertFalse(result["simulated"])
        route_id = result["data"]["routes"][0]["route_id"]
        self.assertNotIn("geometry", str(result))
        self.assertNotIn("private-test-key", str(events))
        self.assertEqual(service.navigation_map(route_id), service.navigation_map(route_id))
        self.assertEqual(sum(path == "/v3/staticmap" for path, _ in self.transport.calls), 1)
        self.assertEqual(service.state.battery_percent, 25)
        self.assertTrue(service.execute("get_route_options", {"destination": "机场"})["simulated"])
        result["data"]["origin"]["name"] = "changed"
        self.assertEqual(service.navigation_plan["origin"]["name"], ORIGIN["name"])

    def test_failed_replan_and_reset_invalidate_old_route(self):
        service = ToolService(navigation_factory=lambda: self.client)
        result = service.execute("plan_navigation_route", {"origin": ORIGIN, "destination": DESTINATION})
        old_id = result["data"]["routes"][0]["route_id"]
        self.assertFalse(service.execute("plan_navigation_route", {})["ok"])
        self.assertIsNone(service.navigation_plan)
        with self.assertRaises(ValueError):
            service.navigation_map(old_id)
        service.execute("plan_navigation_route", {"origin": ORIGIN, "destination": DESTINATION})
        service.reset()
        self.assertIsNone(service.navigation_plan)


class CurrentLocationNavigationTests(unittest.TestCase):
    def setUp(self):
        self.transport = FixtureTransport()
        self.client = AmapClient(key='test-only', transport=self.transport)
        self.events = []
        self.service = ToolService(lambda e, d: self.events.append((e, d)), navigation_factory=lambda: self.client)
        self.location = {"name": "北京天安门（模拟）", "location": "116.397428,39.909230"}

    def test_location_is_explicit_simulation_and_not_defaulted(self):
        self.assertIsNone(self.service.vehicle_location)
        result = self.service.execute('get_vehicle_location', {})
        self.assertEqual(result['error']['code'], 'LOCATION_NOT_SET')
        self.service.set_vehicle_location(self.location)
        result = self.service.execute('get_vehicle_location', {})
        self.assertTrue(result['simulated'])
        self.assertEqual(result['data']['source'], 'manual_simulation')
        self.assertEqual(result['data']['coordinate_system'], 'GCJ-02')
        result['data']['name'] = 'changed'
        self.assertEqual(self.service.vehicle_location['name'], self.location['name'])
        self.assertEqual(self.transport.calls, [])

    def test_invalid_position_preserves_existing_position_and_route(self):
        self.service.set_vehicle_location(self.location)
        for value in [{"name": "", "location": "1,2"}, {"name": "x", "location": "NaN,30"},
                      {"name": "x", "location": "181,30"}, {"name": "x", "location": "1,2", "source": "gps"}]:
            with self.assertRaises(ValueError):
                self.service.set_vehicle_location(value)
            self.assertEqual(self.service.vehicle_location['name'], self.location['name'])

    def test_no_location_stops_before_any_network(self):
        result = self.service.execute('navigate_to_destination', {'destination': '北京南站'})
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['code'], 'LOCATION_NOT_SET')
        self.assertEqual(self.transport.calls, [])

    def test_auto_navigation_filters_names_and_facilities_then_sorts_distance(self):
        self.service.set_vehicle_location(self.location)
        result = self.service.execute('navigate_to_destination', {'destination': '北京南站'})
        self.assertTrue(result['ok'])
        self.assertFalse(result['simulated'])
        plan = result['data']
        self.assertEqual(plan['destination']['id'], ORIGIN['id'])
        self.assertEqual(plan['origin']['location'], self.location['location'])
        self.assertTrue(plan['origin_simulated'])
        self.assertTrue(plan['automatic'])
        self.assertEqual(plan['selection']['candidate_count'], 2)
        self.assertEqual(plan['selection']['distance_metric'], 'approx_straight_line')
        self.assertEqual(plan['selection']['search_scope'], 'within_50km')
        self.assertEqual([e for e, _ in self.events], ['tool.call', 'navigation.location', 'navigation.match', 'tool.result'])
        self.assertEqual(self.transport.calls[0][1]['sortrule'], ['distance'])
        self.assertEqual(self.transport.calls[0][1]['radius'], ['50000'])
        self.assertNotIn('geometry', str(plan))
        self.assertNotIn('arrival_battery_percent', str(plan))

    def test_national_fallback_does_not_restrict_to_origin_city(self):
        calls = []
        def transport(url):
            parsed = urlsplit(url)
            calls.append((parsed.path, parse_qs(parsed.query)))
            if parsed.path == '/v3/place/around':
                return b'{"status":"1","pois":[]}', 'application/json'
            return self.transport(url)
        client = AmapClient(key='test-only', transport=transport)
        match = client.nearest('北京南站', DESTINATION['location'])
        self.assertEqual(match['search_scope'], 'national_candidates')
        self.assertEqual(match['selected']['name'], '北京南站')
        self.assertNotIn('city', calls[-1][1])

    def test_no_match_is_not_arbitrary_first_result(self):
        client = AmapClient(key='test', transport=lambda url: (json.dumps(NEARBY_RESPONSE).encode(), 'application/json'))
        with self.assertRaises(NavigationError) as error:
            client.nearest('不存在的门店', self.location['location'])
        self.assertEqual(error.exception.code, 'AMAP_NO_MATCH')

    def test_restaurant_branches_are_chosen_by_distance_not_provider_order(self):
        places = [{"name": "麦当劳(远店)", "location": "116.7,40.2"},
                  {"name": "麦当劳(近店)", "location": "116.4,39.91"}]
        candidates = matching_places(places, '麦当劳', self.location['location'])
        self.assertEqual(candidates[0]['name'], '麦当劳(近店)')
        self.assertGreater(candidates[1]['distance_m'], candidates[0]['distance_m'])
        self.assertEqual(distance_m('116,39', '116,39'), 0)

    def test_station_query_does_not_route_to_hotels_mentions_or_auxiliary_pois(self):
        places = [
            {'name': '速8优选酒店(北京南站前门天坛西门店)', 'location': self.location['location']},
            {'name': '北京南站附近酒店', 'location': self.location['location']},
            {'name': '北京南站北广场', 'location': self.location['location']},
            ORIGIN,
        ]
        candidates = matching_places(places, '北京南站', self.location['location'])
        self.assertEqual([p['name'] for p in candidates], ['北京南站'])

    def test_near_destination_does_not_make_duplicate_route_or_claim_arrival(self):
        self.service.set_vehicle_location({k: ORIGIN[k] for k in ['name', 'location']})
        result = self.service.execute('navigate_to_destination', {'destination': '北京南站'})
        self.assertTrue(result['ok'])
        self.assertEqual(result['data']['status'], 'near_destination')
        self.assertIsNone(self.service.navigation_plan)
        self.assertFalse(any(path == '/v5/direction/driving' for path, _ in self.transport.calls))

    def test_location_change_and_reset_invalidate_maps(self):
        self.service.set_vehicle_location(self.location)
        result = self.service.execute('navigate_to_destination', {'destination': '北京南站'})
        route_id = result['data']['routes'][0]['route_id']
        self.service.set_vehicle_location({'name': '新位置', 'location': '116.5,39.9'})
        self.assertIsNone(self.service.navigation_plan)
        with self.assertRaises(ValueError):
            self.service.navigation_map(route_id)
        self.service.reset()
        self.assertIsNone(self.service.vehicle_location)

    def test_model_cannot_override_origin_in_auto_navigation_tool(self):
        self.service.set_vehicle_location(self.location)
        result = self.service.execute('navigate_to_destination', {'destination': '北京南站', 'origin': DESTINATION})
        self.assertEqual(result['error']['code'], 'INVALID_ARGUMENT')
        self.assertEqual(self.transport.calls, [])


if __name__ == '__main__':
    unittest.main()
