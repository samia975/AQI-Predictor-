"""
Backfill script for the Pearls AQI Predictor project.

Recovers REAL historical air pollution data using OpenWeather's Air Pollution
History endpoint (free tier, data available from Nov 2020 onwards). This fills
the gap left by the Hopsworks materialization outage -- instead of losing the
Aug 14-22 window, we re-fetch the actual pollutant readings for that period
directly from OpenWeather's own archive.

Limitation (documented honestly for the report): OpenWeather's free tier does
not offer historical WEATHER (temperature/humidity/pressure/wind) the way it
offers historical POLLUTION. So backfilled rows use approximate seasonal
averages for Lahore in August for the weather columns, while pollutant columns
and the AQI target are real, retrieved values. This is flagged in the output
CSV via a `backfilled` column so it's fully transparent in the report.

Usage:
    export OPENWEATHER_API_KEY="your_key"
    python backfill_historical.py --start "2026-08-14" --end "2026-08-22"
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_pipeline import compute_us_aqi, LOCAL_CSV_PATH, LATITUDE, LONGITUDE  # noqa: E402

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# Approximate August averages for Lahore -- used only to fill weather columns
# for backfilled rows, since free-tier historical weather isn't available.
# Documented as an approximation; pollutant values and AQI are real.
APPROX_TEMPERATURE = 32.0
APPROX_HUMIDITY = 55.0
APPROX_PRESSURE = 1002.0
APPROX_WIND_SPEED = 2.0


def fetch_pollution_history(start_dt: datetime, end_dt: datetime) -> list:
    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "start": int(start_dt.timestamp()),
        "end": int(end_dt.timestamp()),
        "appid": OPENWEATHER_API_KEY,
    }
    response = requests.get(HISTORY_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()["list"]


def build_row(entry: dict, previous_aqi) -> dict:
    ts = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
    components = entry["components"]
    pm25 = components.get("pm2_5", 0.0)
    pm10 = components.get("pm10", 0.0)
    aqi = compute_us_aqi(pm25, pm10)
    change_rate = (aqi - previous_aqi) if previous_aqi is not None else 0.0

    return {
        "timestamp": ts.isoformat(),
        "hour": ts.hour,
        "day": ts.day,
        "day_of_week": ts.weekday(),
        "month": ts.month,
        "pm2_5": pm25,
        "pm10": pm10,
        "co": components.get("co", 0.0),
        "no2": components.get("no2", 0.0),
        "o3": components.get("o3", 0.0),
        "so2": components.get("so2", 0.0),
        "nh3": components.get("nh3", 0.0),
        "temperature": APPROX_TEMPERATURE,
        "humidity": APPROX_HUMIDITY,
        "pressure": APPROX_PRESSURE,
        "wind_speed": APPROX_WIND_SPEED,
        "aqi": aqi,
        "aqi_change_rate": change_rate,
        "backfilled": True,  # marks approximated-weather rows for transparency
    }


def load_existing_timestamps() -> set:
    if not LOCAL_CSV_PATH.exists():
        return set()
    with open(LOCAL_CSV_PATH, newline="") as f:
        return {row["timestamp"] for row in csv.DictReader(f)}


def append_rows(rows: list) -> None:
    LOCAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOCAL_CSV_PATH.exists()
    fieldnames = list(rows[0].keys())

    # If the CSV already exists but doesn't have the `backfilled` column yet
    # (from earlier live runs), keep those rows compatible by defaulting it.
    if file_exists:
        with open(LOCAL_CSV_PATH, newline="") as f:
            existing_fieldnames = csv.DictReader(f).fieldnames or []
        if "backfilled" not in existing_fieldnames:
            _add_backfilled_column_to_existing_csv()

    with open(LOCAL_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[ok] appended {len(rows)} backfilled rows to {LOCAL_CSV_PATH}")


def _add_backfilled_column_to_existing_csv() -> None:
    with open(LOCAL_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = (reader.fieldnames or []) + ["backfilled"]
    for row in rows:
        row["backfilled"] = False  # these were real live readings, not approximated
    with open(LOCAL_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    if not OPENWEATHER_API_KEY:
        print("[error] OPENWEATHER_API_KEY environment variable is not set.")
        sys.exit(1)

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"[info] fetching pollution history {args.start} to {args.end}")
    entries = fetch_pollution_history(start_dt, end_dt)
    print(f"[info] OpenWeather returned {len(entries)} hourly records")

    existing = load_existing_timestamps()
    rows = []
    previous_aqi = None
    for entry in entries:
        row = build_row(entry, previous_aqi)
        previous_aqi = row["aqi"]
        if row["timestamp"] not in existing:
            rows.append(row)

    if not rows:
        print("[info] no new rows to add (all timestamps already present)")
        return

    append_rows(rows)


if __name__ == "__main__":
    main()
