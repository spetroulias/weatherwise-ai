"""Offline tests for weather summaries, date filtering and Groq integration."""

import json
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import requests

import app


class AppPipelineTests(unittest.TestCase):
    def make_forecast(self):
        return pd.DataFrame({
            "Time": pd.to_datetime([
                "2026-11-21T13:00",
                "2026-11-21T14:00",
                "2026-11-21T16:00",
            ]),
            "Temperature (°C)": [20.0, 21.0, 18.0],
            "Rain probability (%)": [0, 30, 10],
            "Precipitation (mm)": [0.0, 0.5, 0.0],
            "Wind (km/h)": [3.0, 5.0, 4.0],
            "Cloud cover (%)": [10, 70, 20],
        })

    def test_four_hour_summary_uses_available_hours(self):
        result = app.summarize_forecast(self.make_forecast())
        self.assertIn("2026-11-21 13:00 to 2026-11-21 15:00", result)
        self.assertIn("2026-11-21 16:00 to 2026-11-21 17:00", result)
        self.assertIn("average cloud cover 40%", result)
        self.assertIn("maximum cloud cover 70%", result)

    def test_daily_comparison_uses_daily_maximum_wind(self):
        forecast = self.make_forecast()
        next_day = forecast.copy()
        next_day["Time"] += pd.Timedelta(days=1)
        next_day["Wind (km/h)"] = [1.0, 2.0, 3.0]

        result = app.summarize_days(pd.concat([forecast, next_day]))
        self.assertIn("lowest DAILY MAXIMUM wind", result)
        self.assertIn("3.0 km/h on 2026-11-22", result)

    def test_range_end_is_exclusive(self):
        chosen, period = app.select_forecast(self.make_forecast(), {
            "scope": "range",
            "start": "2026-11-21T13:00:00",
            "end": "2026-11-21T16:00:00",
        })
        self.assertEqual(len(chosen), 2)
        self.assertIsNotNone(period)

    def test_invalid_range_falls_back_to_all(self):
        source = self.make_forecast()
        chosen, period = app.select_forecast(source, {
            "scope": "range", "start": "invalid", "end": "invalid"
        })
        self.assertEqual(len(chosen), len(source))
        self.assertIsNone(period)

    def test_query_planner_requests_structured_json(self):
        planned = json.dumps({
            "scope": "range",
            "start": "2026-11-21T12:00:00",
            "end": "2026-11-21T20:00:00",
        })
        with patch.object(app, "groq_chat", return_value=planned) as llm:
            result = app.resolve_filter(
                "fake-key", "tomorrow afternoon",
                datetime.fromisoformat("2026-11-20T10:00:00+02:00"),
                None, [],
            )
        self.assertEqual(result["scope"], "range")
        self.assertTrue(llm.call_args.kwargs["structured"])

    def test_groq_call_uses_json_schema_and_timeout(self):
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "choices": [{"message": {"content": '{"scope":"all","start":null,"end":null}'}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8},
        }
        with patch.object(app.requests, "post", return_value=response) as post:
            result = app.groq_chat(
                "fake-key", [{"role": "user", "content": "tomorrow?"}], structured=True
            )
        self.assertIn('"all"', result)
        self.assertEqual(post.call_args.kwargs["json"]["response_format"]["type"], "json_schema")
        self.assertEqual(post.call_args.kwargs["timeout"], 60)

    def test_rate_limit_shows_friendly_message(self):
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "38"
        error = requests.HTTPError("429", response=response)

        with patch.object(app.st, "warning") as warning:
            app.show_groq_error(error)
        warning.assert_called_once()
        self.assertIn("38 seconds", warning.call_args.args[0])
        self.assertNotIn("https://", warning.call_args.args[0])

    def test_bad_request_shows_safe_english_message(self):
        response = requests.Response()
        response.status_code = 400
        error = requests.HTTPError("400", response=response)

        with patch.object(app.st, "warning") as warning:
            app.show_groq_error(error)
        warning.assert_called_once()
        self.assertIn("could not process", warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
