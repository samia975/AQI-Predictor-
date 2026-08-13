"""
Feature pipeline for the Pearls AQI Predictor project.

What it does:
1. Fetches raw weather + pollutant data for a city from OpenWeather.
2. Computes the real US EPA AQI (0-500) from PM2.5/PM10 as the prediction target.
3. Builds a feature row: time-based features + pollutant/weather readings + AQI change rate.
4. Stores the row in Hopsworks (if configured) OR a local CSV (works with zero setup).

Usage:
    export OPENWEATHER_API_KEY="your_key_here"
    python feature_pipeline.py

Run this on a schedule (hourly) via cron or GitHub Actions once it's working.
"""

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Lahore, Pakistan coordinates
CITY_NAME = "Lahore"
LATITUDE = 31.5497
LONGITUDE = 74.3436

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# Optional Hopsworks config — script falls back to local CSV if these are unset.
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_FEATURE_GROUP = "aqi_features"

LOCAL_CSV_PATH = Path(__file__).parent / "data" / "features.csv"

REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Fetching raw data
# ---------------------------------------------------------------------------

def _get_with_retry(url: str, params: dict) -> dict:
    """GET with a couple of retries — network calls to a public API fail sometimes."""
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            print(f"[warn] request to {url} failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts") from last_error


def fetch_weather(lat: float, lon: float, api_key: str) -> dict:
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    return _get_with_retry(WEATHER_URL, params)


def fetch_pollution(lat: float, lon: float, api_key: str) -> dict:
    params = {"lat": lat, "lon": lon, "appid": api_key}
    return _get_with_retry(POLLUTION_URL, params)


# ---------------------------------------------------------------------------
# AQI calculation (US EPA standard, 0-500 scale)
# ---------------------------------------------------------------------------

# (C_low, C_high, I_low, I_high) breakpoints per EPA AQI technical spec.
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300),
    (425, 504, 301, 400), (505, 604, 401, 500),
]


def _sub_index(concentration: float, breakpoints: list) -> float:
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
    # Above the top breakpoint: clamp to worst category rather than crash.
    return breakpoints[-1][3]


def compute_us_aqi(pm25: float, pm10: float) -> int:
    """US AQI is the max of the individual pollutant sub-indices.
    Simplified to PM2.5 + PM10 (the two dominant drivers of AQI in most South Asian
    cities, Lahore included). Extend with CO/NO2/O3/SO2 sub-indices later if needed."""
    aqi = max(_sub_index(pm25, PM25_BREAKPOINTS), _sub_index(pm10, PM10_BREAKPOINTS))
    return round(aqi)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def get_previous_aqi() -> Optional[float]:
    """Reads the last stored AQI so we can compute a change rate.
    Only reads from the local CSV fallback — if you're fully on Hopsworks,
    swap this to query the feature group's most recent row instead."""
    if not LOCAL_CSV_PATH.exists():
        return None
    try:
        with open(LOCAL_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return float(rows[-1]["aqi"])
    except (ValueError, KeyError, IndexError) as exc:
        print(f"[warn] could not read previous AQI: {exc}")
        return None


def build_feature_row(weather: dict, pollution: dict) -> dict:
    now = datetime.now(timezone.utc)

    components = pollution["list"][0]["components"]
    pm25 = components.get("pm2_5", 0.0)
    pm10 = components.get("pm10", 0.0)
    aqi = compute_us_aqi(pm25, pm10)

    previous_aqi = get_previous_aqi()
    aqi_change_rate = (aqi - previous_aqi) if previous_aqi is not None else 0.0

    return {
        "timestamp": now.isoformat(),
        "hour": now.hour,
        "day": now.day,
        "day_of_week": now.weekday(),
        "month": now.month,
        # pollutants
        "pm2_5": pm25,
        "pm10": pm10,
        "co": components.get("co", 0.0),
        "no2": components.get("no2", 0.0),
        "o3": components.get("o3", 0.0),
        "so2": components.get("so2", 0.0),
        "nh3": components.get("nh3", 0.0),
        # weather
        "temperature": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": weather["wind"]["speed"],
        # target + derived feature
        "aqi": aqi,
        "aqi_change_rate": aqi_change_rate,
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_to_local_csv(row: dict) -> None:
    LOCAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOCAL_CSV_PATH.exists()
    with open(LOCAL_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[ok] wrote feature row to {LOCAL_CSV_PATH}")


def store_to_hopsworks(row: dict) -> None:
    try:
        import hopsworks
        import pandas as pd
    except ImportError:
        print("[warn] hopsworks/pandas not installed — run: pip install hopsworks pandas")
        store_to_local_csv(row)
        return

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    feature_group = fs.get_or_create_feature_group(
        name=HOPSWORKS_FEATURE_GROUP,
        version=1,
        primary_key=["timestamp"],
        description="Lahore AQI features: pollutants, weather, time-based, derived",
    )
    feature_group.insert(pd.DataFrame([row]))
    print(f"[ok] wrote feature row to Hopsworks feature group '{HOPSWORKS_FEATURE_GROUP}'")


def store_features(row: dict) -> None:
    if HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME:
        store_to_hopsworks(row)
    else:
        print("[info] Hopsworks not configured — using local CSV fallback")
        store_to_local_csv(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not OPENWEATHER_API_KEY:
        print("[error] OPENWEATHER_API_KEY environment variable is not set.")
        sys.exit(1)

    print(f"[info] fetching data for {CITY_NAME} ({LATITUDE}, {LONGITUDE})")
    weather = fetch_weather(LATITUDE, LONGITUDE, OPENWEATHER_API_KEY)
    pollution = fetch_pollution(LATITUDE, LONGITUDE, OPENWEATHER_API_KEY)

    row = build_feature_row(weather, pollution)
    print(f"[info] computed AQI={row['aqi']} (change rate: {row['aqi_change_rate']:+.1f})")

    store_features(row)


if __name__ == "__main__":
    main()
