"""
Training pipeline for the Pearls AQI Predictor project.

What it does:
1. Fetches historical features from the Hopsworks Feature Store.
2. Builds a forecasting target: AQI PREDICTION_HORIZON_HOURS ahead (default: 72h / 3 days).
   If there isn't yet enough historical span to do that safely, it falls back to a
   "nowcast" target (predict AQI right now) purely so you can test that the pipeline
   mechanics work while more data accumulates -- do NOT report nowcast results as your
   final model performance in your report.
3. Trains and evaluates multiple models: Ridge Regression, Random Forest, and (once you
   have enough data) a small neural network.
4. Evaluates with RMSE, MAE, R^2 and picks the best one.
5. Saves the winning model to the Hopsworks Model Registry.

Usage:
    export HOPSWORKS_API_KEY="your_key"
    export HOPSWORKS_PROJECT_NAME="your_project_name"
    python training_pipeline.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

MODEL_REGISTRY_NAME = "aqi_predictor_model"

# Ideal horizon matches "next 3 days" in the project brief. Due to a Hopsworks
# materialization outage (documented in the report), we don't yet have enough
# continuous hourly history for a full 72h-ahead model. Instead of forcing a
# meaningless nowcast, we try progressively shorter horizons and use the longest
# one the available data can actually support.
CANDIDATE_HORIZONS_HOURS = [72, 48, 24, 12, 6, 3, 1]
MIN_USABLE_ROWS = 15  # below this, even a short horizon isn't worth modeling
LOCAL_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "features.csv")

FEATURE_COLUMNS = [
    "hour", "day", "day_of_week", "month",
    "pm2_5", "pm10", "co", "no2", "o3", "so2", "nh3",
    "temperature", "humidity", "pressure", "wind_speed",
]
TARGET_COLUMN = "aqi"

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")


# ---------------------------------------------------------------------------
# Step 1: fetch historical data
# ---------------------------------------------------------------------------

def fetch_from_hopsworks() -> pd.DataFrame:
    import hopsworks

    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
        print("[warn] Hopsworks credentials not set — skipping Hopsworks source")
        return pd.DataFrame()

    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = feature_group.read()
        print(f"[info] fetched {len(df)} rows from Hopsworks feature group '{FEATURE_GROUP_NAME}'")
        return df
    except Exception as exc:
        print(f"[warn] could not fetch from Hopsworks, continuing with local data only: {exc}")
        return pd.DataFrame()


def fetch_from_local_csv() -> pd.DataFrame:
    if not os.path.exists(LOCAL_CSV_PATH):
        print("[info] no local CSV backup found")
        return pd.DataFrame()
    df = pd.read_csv(LOCAL_CSV_PATH)
    print(f"[info] fetched {len(df)} rows from local CSV backup")
    return df


def fetch_training_data() -> pd.DataFrame:
    """Combines both data sources so a partial outage in one doesn't waste the
    rows that made it into the other. Duplicate timestamps (row saved to both
    places successfully) are kept only once."""
    hw_df = fetch_from_hopsworks()
    csv_df = fetch_from_local_csv()

    combined = pd.concat([hw_df, csv_df], ignore_index=True)
    if len(combined) == 0:
        return combined

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], format="ISO8601")
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"[info] combined dataset: {len(combined)} unique rows "
          f"({combined['timestamp'].min()} to {combined['timestamp'].max()})")
    return combined


# ---------------------------------------------------------------------------
# Step 2: build the forecasting target
# ---------------------------------------------------------------------------

def try_horizon(df: pd.DataFrame, horizon_hours: int):
    """Builds a (features, target) pair for a given forecast horizon by shifting
    the AQI column back in time. Assumes roughly hourly cadence; gaps just mean
    fewer usable rows survive the shift, not incorrect ones."""
    combined = df.copy()
    combined["target"] = combined[TARGET_COLUMN].shift(-horizon_hours)
    combined = combined.dropna(subset=["target"])
    return combined[FEATURE_COLUMNS], combined["target"]


def build_training_frame(df: pd.DataFrame):
    """Returns (X, y, horizon_hours, is_real_forecast). Tries the longest forecast
    horizon the data can support, falling back to shorter ones, and finally to a
    nowcast (horizon=0) only as a last resort for pipeline testing."""

    for horizon in CANDIDATE_HORIZONS_HOURS:
        X, y = try_horizon(df, horizon)
        if len(X) >= MIN_USABLE_ROWS:
            print(f"[info] forecasting mode: predicting AQI {horizon}h ahead, {len(X)} usable rows")
            return X, y, horizon, True

    print(
        f"[warn] not enough continuous history for any forecast horizon "
        f"(need {MIN_USABLE_ROWS}+ usable rows). Falling back to NOWCAST mode "
        f"(predicting current AQI) just to test the pipeline. Do not report these metrics "
        f"as final forecasting performance."
    )
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y, 0, False


# ---------------------------------------------------------------------------
# Step 3: train + evaluate models
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series) -> dict:
    results = {}

    if len(X) < 5:
        print("[warn] fewer than 5 rows total — skipping train/test split, "
              "evaluating on training data only (metrics are not meaningful yet).")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Ridge Regression: simple, fast, good baseline ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results["ridge"] = {"model": ridge, "metrics": evaluate(y_test, ridge.predict(X_test))}

    # --- Random Forest: usually the strongest simple option for tabular sensor data ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    results["random_forest"] = {"model": rf, "metrics": evaluate(y_test, rf.predict(X_test))}

    # --- Small neural network: only worth trying once there's real data volume ---
    if len(X_train) >= 100:
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
        mlp.fit(X_train, y_train)
        results["neural_net"] = {"model": mlp, "metrics": evaluate(y_test, mlp.predict(X_test))}
    else:
        print(f"[info] skipping neural network — only {len(X_train)} training rows "
              f"(need 100+ for it to learn anything real)")

    for name, result in results.items():
        m = result["metrics"]
        print(f"[info] {name}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Step 4: pick the winner and save it
# ---------------------------------------------------------------------------

def pick_best_model(results: dict):
    best_name = min(results, key=lambda name: results[name]["metrics"]["rmse"])
    return best_name, results[best_name]


def save_to_model_registry(name: str, result: dict, horizon_hours: int, is_real_forecast: bool) -> None:
    import joblib

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(result["model"], model_path)
    print(f"[ok] saved model locally to {model_path}")

    if not is_real_forecast:
        print("[warn] NOT uploading to Model Registry — this was a nowcast test run on "
              "too little data. Re-run once you have enough historical rows.")
        return

    import hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    mr = project.get_model_registry()

    hw_model = mr.python.create_model(
        name=MODEL_REGISTRY_NAME,
        metrics=result["metrics"],
        description=f"AQI {horizon_hours}h-ahead forecast — best model: {name}",
    )
    hw_model.save(model_path)
    print(f"[ok] saved '{name}' to Hopsworks Model Registry as '{MODEL_REGISTRY_NAME}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = fetch_training_data()

    if len(df) == 0:
        print("[error] no data in the feature store yet — let the feature pipeline run longer.")
        sys.exit(1)

    X, y, horizon_hours, is_real_forecast = build_training_frame(df)

    if len(X) == 0:
        print("[error] no usable rows after building the forecasting target — "
              "need more historical data.")
        sys.exit(1)

    results = train_and_evaluate_models(X, y)
    best_name, best_result = pick_best_model(results)
    print(f"[info] best model: {best_name} (lowest RMSE)")

    save_to_model_registry(best_name, best_result, horizon_hours, is_real_forecast)


if __name__ == "__main__":
    main()