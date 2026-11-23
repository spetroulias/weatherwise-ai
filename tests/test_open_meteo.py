
"""Offline tests for the Open-Meteo functions now located in app.py."""

import unittest
from unittest.mock import patch

import app


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class WeatherClientTests(unittest.TestCase):
    def test_find_city(self):
        payload = {
            "results": [{
                "name": "Athens",
                "latitude": 37.98,
                "longitude": 23.73,
                "timezone": "Europe/Athens",
            }]
        }
        with patch.object(app.requests, "get", return_value=FakeResponse(payload)):
            result = app.find_city("Athens")
        self.assertEqual(result["name"], "Athens")
        self.assertEqual(result["timezone"], "Europe/Athens")

    def test_blank_city_does_not_call_the_api(self):
        with patch.object(app.requests, "get") as get:
            with self.assertRaises(ValueError):
                app.find_city("  ")
            get.assert_not_called()

    def test_unknown_city(self):
        with patch.object(app.requests, "get", return_value=FakeResponse({})):
            with self.assertRaises(ValueError):
                app.find_city("NotAPlace")

    def test_forecast_includes_cloud_cover_and_eleven_days(self):
        payload = {
            "hourly": {
                "time": ["2026-11-20T13:00", "2026-11-20T14:00"],
                "temperature_2m": [20.0, 21.0],
                "apparent_temperature": [19.0, 20.0],
                "precipitation_probability": [10, 20],
                "precipitation": [0.0, 0.1],
                "wind_speed_10m": [5.0, 6.0],
                "cloud_cover": [15, 75],
            }
        }
        with patch.object(app.requests, "get", return_value=FakeResponse(payload)) as get:
            forecast = app.get_forecast(37.98, 23.73)
        self.assertEqual(len(forecast), 2)
        self.assertEqual(forecast.loc[0, "Cloud cover (%)"], 15)
        self.assertEqual(forecast.loc[1, "Rain probability (%)"], 20)
        self.assertEqual(app.FORECAST_DAYS, 11)
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["forecast_days"], 11)
        self.assertEqual(params["timezone"], "auto")
        self.assertIn("cloud_cover", params["hourly"])

    def test_missing_weather_field_is_rejected(self):
        payload = {
            "hourly": {
                "time": ["2026-11-20T13:00"],
                "temperature_2m": [20],
                "apparent_temperature": [19],
                "precipitation_probability": [0],
                "precipitation": [0],
                "wind_speed_10m": [2],
            }
        }
        with patch.object(app.requests, "get", return_value=FakeResponse(payload)):
            with self.assertRaisesRegex(ValueError, "cloud_cover"):
                app.get_forecast(37.98, 23.73)

    def test_empty_forecast_is_rejected(self):
        with patch.object(app.requests, "get", return_value=FakeResponse({"hourly": {}})):
            with self.assertRaisesRegex(ValueError, "no hourly forecast"):
                app.get_forecast(37.98, 23.73)


if __name__ == "__main__":
    unittest.main()
