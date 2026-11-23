"""WeatherWise AI: a weather chatbot using Open-Meteo, Groq and Streamlit."""

import json
import logging
import os
import re
from datetime import datetime
from time import perf_counter, sleep
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st


# APIs and application settings
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
FORECAST_DAYS = 11  # Today and the next ten days
TIMEOUT_SECONDS = 15
REFUSAL = "I can only help with weather forecasts and weather-based activities."

HOURLY_FIELDS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
]

# The planner uses a fixed JSON structure; it never executes generated code.
FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {"type": "string", "enum": ["range", "all", "unrelated"]},
        "start": {"type": ["string", "null"]},
        "end": {"type": ["string", "null"]},
    },
    "required": ["scope", "start", "end"],
    "additionalProperties": False,
}

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


# Open-Meteo ---------------------------------------------------------------

def find_city(city: str) -> dict:
    """Look up coordinates for the first city matching the user's input."""
    city = city.strip()
    if not city:
        raise ValueError("Please enter a city name.")

    response = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    locations = response.json().get("results") or []
    if not locations:
        raise ValueError(f"No location found for '{city}'. Try a more specific name.")
    return locations[0]


def get_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    """Get hourly forecasts and convert Open-Meteo's JSON into a DataFrame."""
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(HOURLY_FIELDS),
            "forecast_days": FORECAST_DAYS,
            "timezone": "auto",
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    hourly = response.json().get("hourly")
    if not hourly or not hourly.get("time"):
        raise ValueError("The weather service returned no hourly forecast.")

    missing = [field for field in HOURLY_FIELDS if field not in hourly]
    if missing:
        raise ValueError(f"Missing expected weather fields: {', '.join(missing)}")

    return pd.DataFrame({
        "Time": pd.to_datetime(hourly["time"]),
        "Temperature (°C)": hourly["temperature_2m"],
        "Feels like (°C)": hourly["apparent_temperature"],
        "Rain probability (%)": hourly["precipitation_probability"],
        "Precipitation (mm)": hourly["precipitation"],
        "Wind (km/h)": hourly["wind_speed_10m"],
        "Cloud cover (%)": hourly["cloud_cover"],
    })


def load_weather(city: str):
    """Get the city and its forecast (cached at the Streamlit entrypoint)."""
    location = find_city(city)
    forecast = get_forecast(location["latitude"], location["longitude"])
    return location, forecast


# Prepare weather data for the LLM ----------------------------------------

def summarize_forecast(forecast: pd.DataFrame) -> str:
    """Summarize hourly measurements in four-hour windows."""
    periods = (forecast.groupby(pd.Grouper(key="Time", freq="4h"))
        .agg(
            first_hour=("Time", "min"),
            last_hour=("Time", "max"),
            min_temp=("Temperature (°C)", "min"),
            max_temp=("Temperature (°C)", "max"),
            rain_chance=("Rain probability (%)", "max"),
            rain_mm=("Precipitation (mm)", "sum"),
            wind=("Wind (km/h)", "max"),
            clouds=("Cloud cover (%)", "mean"),
            max_clouds=("Cloud cover (%)", "max"),
        )
        .dropna(subset=["min_temp", "max_temp"])
    )

    lines = []
    for row in periods.itertuples():
        # Some windows are partial because only future hours are used.
        end = pd.Timestamp(row.last_hour) + pd.Timedelta(hours=1)
        lines.append(
            f"{row.first_hour:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}: "
            f"temperature {row.min_temp:.1f}-{row.max_temp:.1f} C, "
            f"max rain probability {row.rain_chance:.0f}%, "
            f"total precipitation {row.rain_mm:.1f} mm, "
            f"max wind {row.wind:.1f} km/h, "
            f"average cloud cover {row.clouds:.0f}%, "
            f"maximum cloud cover {row.max_clouds:.0f}%"
        )
    return "\n".join(lines)


def summarize_days(forecast: pd.DataFrame) -> str:
    """Use compact daily values for comparisons across several days."""
    daily = (forecast.assign(day=forecast["Time"].dt.date)
        .groupby("day", sort=True)
        .agg(
            hours=("Time", "nunique"),
            min_temp=("Temperature (°C)", "min"),
            max_temp=("Temperature (°C)", "max"),
            rain_chance=("Rain probability (%)", "max"),
            rain_mm=("Precipitation (mm)", "sum"),
            wind=("Wind (km/h)", "max"),
            clouds=("Cloud cover (%)", "mean"),
        ).reset_index()
    )

    lines = []
    for row in daily.itertuples(index=False):
        coverage = "full day" if row.hours == 24 else f"partial: {row.hours} hourly values"
        lines.append(
            f"{row.day:%Y-%m-%d} ({coverage}): "
            f"temperature {row.min_temp:.1f}-{row.max_temp:.1f} C; "
            f"max rain probability {row.rain_chance:.0f}%; "
            f"total precipitation {row.rain_mm:.1f} mm; "
            f"daily MAX wind {row.wind:.1f} km/h; "
            f"average cloud cover {row.clouds:.0f}%"
        )

    # Calculate the lowest daily maximum in Python instead of trusting LLM arithmetic.
    valid = daily.dropna(subset=["wind"])
    if len(valid) > 1:
        lowest = valid["wind"].min()
        days = ", ".join(day.strftime("%Y-%m-%d") for day in valid.loc[valid["wind"] == lowest, "day"])
        lines.append(
            f"CHECKED COMPARISON: lowest DAILY MAXIMUM wind within these selected records "
            f"is {lowest:.1f} km/h on {days}. "
            "A partial day must not be presented as a complete-day comparison."
        )
    return "\n".join(lines)


def forecast_context(selected: pd.DataFrame) -> str:
    """Send shorter daily summaries when a question covers over three days."""
    if selected["Time"].dt.date.nunique() > 3:
        return "DAILY DATA (not hourly readings):\n" + summarize_days(selected)
    return "4-HOUR DATA (may contain partial windows):\n" + summarize_forecast(selected)


# Groq: interpret the question, then generate an answer -------------------

def safe_log_value(value) -> str:
    """Keep API error identifiers safe for logging."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "", str(value or "unknown"))[:80]


def groq_chat(api_key: str, messages: list[dict], *, structured: bool = False) -> str:
    """Call Groq, logging basic metrics and handling short rate-limit delays."""
    payload = {"model": MODEL, "messages": messages}
    if structured:
        payload.update({
            "reasoning_effort": "low",
            "max_completion_tokens": 600,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "forecast_filter", "strict": True, "schema": FILTER_SCHEMA},
            },
        })

    for attempt in range(2):
        started = perf_counter()
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=60,
        )

        if not response.ok:
            # Log provider error identifiers, never user prompts or API keys.
            error = {}
            try:
                body = response.json()
                if isinstance(body, dict) and isinstance(body.get("error"), dict):
                    error = body["error"]
            except ValueError:
                pass
            logger.warning(
                "groq_http_error model=%s planner=%s status=%s type=%s code=%s param=%s request_id=%s",
                MODEL, structured, response.status_code,
                safe_log_value(error.get("type")), safe_log_value(error.get("code")),
                safe_log_value(error.get("param")), safe_log_value(response.headers.get("x-request-id")),
            )
            retry_after = response.headers.get("retry-after", "").strip()
            if response.status_code == 429 and attempt == 0 and retry_after.isdigit() and 0 < int(retry_after) <= 3:
                sleep(int(retry_after))
                continue
            response.raise_for_status()

        data = response.json()
        usage = data.get("usage") or {}
        logger.info(
            "groq_inference model=%s planner=%s latency_s=%.3f prompt_tokens=%s completion_tokens=%s",
            MODEL, structured, perf_counter() - started,
            usage.get("prompt_tokens", "unknown"), usage.get("completion_tokens", "unknown"),
        )
        answer = data["choices"][0]["message"].get("content")
        if not answer:
            raise ValueError("Groq returned an empty response")
        return answer

    raise RuntimeError("Groq request could not be completed")


def resolve_filter(api_key: str, question: str, local_now: datetime, previous_range, recent_messages):
    """First LLM call: turn a question into a time range or a scope decision."""
    previous = f"{previous_range['start']} to {previous_range['end']}" if previous_range else "none"
    history = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in recent_messages[-4:])
    instructions = (
        "Convert weather questions to JSON time filters matching the supplied schema. "
        "Weather questions about other cities or missing variables are in scope. "
        "Use unrelated ONLY for questions unrelated to weather. "
        f"Current city-local time: {local_now.isoformat()}. Previous range: {previous}. "
        "For range, return local ISO 8601 datetimes without offsets; start inclusive, end exclusive. "
        "Use 00:00 to next-day 00:00 for full calendar days. "
        "Morning is 06:00-12:00, afternoon 12:00-18:00, evening 18:00-22:00. "
        "Interpret today, tomorrow, weekdays, dates and time windows naturally. "
        "Use range when the requested period can be inferred. "
        "For time-only follow-ups, reuse the previous date if clear. "
        "Use all for comparisons across all available dates or genuinely unclear dates. "
        "For all/unrelated set start and end to null. Never invent dates or availability."
    )
    result = groq_chat(api_key, [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"Recent conversation:\n{history}\nNew question: {question}"},
    ], structured=True)
    plan = json.loads(result)
    if not isinstance(plan, dict) or plan.get("scope") not in {"range", "all", "unrelated"}:
        raise ValueError("Invalid time filter returned")
    return plan


def select_forecast(upcoming: pd.DataFrame, plan: dict):
    """Apply the planner's dates using Pandas, not code generated by the model."""
    if plan["scope"] != "range":
        return upcoming, None
    try:
        start = datetime.fromisoformat(plan["start"])
        end = datetime.fromisoformat(plan["end"])
        if start.tzinfo or end.tzinfo or end <= start:
            raise ValueError("Invalid interval")
    except (TypeError, ValueError):
        return upcoming, None

    selected = upcoming.loc[(upcoming["Time"] >= start) & (upcoming["Time"] < end)]
    return selected, {"start": start.isoformat(), "end": end.isoformat()}


def build_answer_prompt(place: str, local_now: datetime, selected: pd.DataFrame) -> str:
    """Second LLM call: explain only the relevant forecast, in English."""
    return (
        "You are WeatherWise, a weather-only assistant. Always respond in natural English. "
        "Answer weather questions and weather-dependent activities only. "
        "Questions about other cities or missing variables are still weather questions; "
        "explain limitations instead of refusing them as unrelated. "
        "Use ONLY the supplied forecast data. Do not invent values, cities or dates. "
        "Never give a precise hourly reading if the selected summary cannot support it. "
        "Write dates as DD/MM/YYYY and use 24-hour times: morning 06:00-12:00, "
        "afternoon 12:00-18:00, evening 18:00-22:00. "
        "For numerical comparisons check all supplied values. Compare daily maxima with "
        "daily maxima, not hourly minima, and follow the CHECKED COMPARISON when present. "
        "Treat partial days as partial, not full days. "
        "Respect user preferences, but don't imply very low rain chances make rain likely. "
        "Cloud cover does not directly measure sunshine. "
        "If information is missing, state the limitation. "
        "For a different city, ask the user to select it in the sidebar. "
        "Keep answers concise and avoid invented expressions. "
        "Long-range forecasts are less reliable.\n\n"
        f"Location: {place}\nLocal time: {local_now.isoformat()}\n"
        f"Available forecast:\n{forecast_context(selected)}"
    )


def show_groq_error(error: Exception) -> None:
    """Display safe, English-only messages when Groq cannot complete a request."""
    if isinstance(error, requests.HTTPError) and error.response is not None:
        status = error.response.status_code
        if status == 429:
            delay = error.response.headers.get("retry-after", "").strip()
            if delay.isdigit():
                st.warning(f"The AI service is rate-limited. Try again in about {delay} seconds.")
            else:
                st.warning("The AI service has reached its usage limit. Please try again later.")
            return
        if status in (400, 413, 422):
            st.warning("The AI service could not process this request. Try a shorter or different wording.")
            return
    logger.warning("Groq request failed: %s", type(error).__name__)
    st.error("The AI service is temporarily unavailable. Please try again later.")


# Streamlit interface ------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="WeatherWise AI", page_icon="🌤️")
    st.title("🌤️ WeatherWise AI")
    st.caption("Ask about the weather or weather-dependent activities.")

    for key, default in (("city", "Athens"), ("messages", []), ("last_forecast_range", None)):
        if key not in st.session_state:
            st.session_state[key] = default

    # Changing the sidebar city resets the conversation and previous date filter.
    with st.sidebar:
        city = st.text_input("City", value=st.session_state["city"], max_chars=100)
        if st.button("Use this city", use_container_width=True):
            if not city.strip():
                st.warning("Enter a city.")
            else:
                st.session_state.update(city=city.strip(), messages=[], last_forecast_range=None)
                st.rerun()

    try:
        location, forecast = load_weather(st.session_state["city"])
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        logger.warning("Forecast lookup failed: %s", type(error).__name__)
        st.error("Could not load the weather forecast. Check the city and try again.")
        st.stop()

    place = ", ".join(str(x) for x in (location.get("name"), location.get("country")) if x)
    local_now = datetime.now(ZoneInfo(location.get("timezone") or "UTC"))
    upcoming = forecast.loc[forecast["Time"] >= local_now.replace(tzinfo=None)]

    st.write(f"**Forecast location:** {place}")
    st.caption("Forecast: today and the next 10 days. AI uses 4-hour or daily summaries.")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("e.g. Which day next week is best for running?", max_chars=1000)
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            st.error("The AI service is not configured.")
            st.session_state["messages"].pop()
            return
        if upcoming.empty:
            st.warning("No upcoming forecast data is available.")
            st.session_state["messages"].pop()
            return

        try:
            # Stage 1: the LLM chooses the dates; Python applies the filter.
            with st.spinner("Checking your requested dates..."):
                plan = resolve_filter(key, question, local_now,
                                      st.session_state["last_forecast_range"],
                                      st.session_state["messages"][:-1])

            if plan["scope"] == "unrelated":
                st.markdown(REFUSAL)
                st.session_state["messages"].append({"role": "assistant", "content": REFUSAL})
                return

            selected, period = select_forecast(upcoming, plan)
            if selected.empty:
                st.warning("There is no available forecast for that period. Choose a date within the forecast window.")
                st.session_state["messages"].pop()
                return

            # Stage 2: Groq answers using the selected weather data.
            with st.spinner("Thinking..."):
                answer = groq_chat(key, [
                    {"role": "system", "content": build_answer_prompt(place, local_now, selected)},
                    *st.session_state["messages"][-4:],
                ])
            st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.session_state["last_forecast_range"] = period

        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            show_groq_error(error)
            st.session_state["messages"].pop()


if __name__ == "__main__":
    # Keep the cache wrapper in Python; the production image imports a Nuitka module.
    load_weather = st.cache_data(ttl=900)(load_weather)
    main()
