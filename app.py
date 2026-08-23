"""
Streamlit dashboard for the Pearls AQI Predictor project.

Shows:
- Current AQI reading for Lahore
- 3-day-ahead AQI forecast from the trained model
- Historical AQI trend chart
- Feature importance / explainability (SHAP if available, model's own
  importances as a reliable fallback if SHAP isn't installed)
- A hazard alert banner when the forecast crosses unhealthy thresholds

Usage:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
LOCAL_CSV_PATH = BASE_DIR / "data" / "features.csv"
MODEL_DIR = BASE_DIR / "trained_models"

FEATURE_COLUMNS = [
    "hour", "day", "day_of_week", "month",
    "pm2_5", "pm10", "co", "no2", "o3", "so2", "nh3",
    "temperature", "humidity", "pressure", "wind_speed",
]

# US EPA AQI category thresholds, used for the color-coded gauge and alerts
AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

st.set_page_config(page_title="Pearls AQI Predictor — Lahore", page_icon="🌫️", layout="wide")


# ---------------------------------------------------------------------------
# Data + model loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    if not LOCAL_CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOCAL_CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource
def load_model():
    if not MODEL_DIR.exists():
        return None, None
    model_files = list(MODEL_DIR.glob("*.pkl"))
    if not model_files:
        return None, None
    model_path = model_files[0]
    return joblib.load(model_path), model_path.stem


def get_aqi_category(aqi: float):
    for low, high, label, color in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label, color
    return "Hazardous", AQI_CATEGORIES[-1][3]


# ---------------------------------------------------------------------------
# Explainability (SHAP if available, feature importance as a fallback)
# ---------------------------------------------------------------------------

def explain_prediction(model, X_row: pd.DataFrame, background: pd.DataFrame):
    try:
        import shap
        # Use a sample of historical rows as the background reference so SHAP
        # has something to compare the current prediction against — using the
        # row itself as its own background always yields all-zero contributions.
        background_sample = background.sample(min(30, len(background)), random_state=42).astype(float)
        X_row = X_row.astype(float)
        explainer = shap.Explainer(model.predict, background_sample)
        shap_values = explainer(X_row)
        contributions = pd.Series(shap_values.values[0], index=FEATURE_COLUMNS)
        return contributions.sort_values(key=abs, ascending=False), "SHAP"
    except Exception:
        # Fallback: use the model's own feature importance / coefficients.
        # Not as precise as SHAP for a single prediction, but still gives a
        # genuinely useful "what's driving this" view with zero extra setup risk.
        if hasattr(model, "feature_importances_"):
            contributions = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        elif hasattr(model, "coef_"):
            contributions = pd.Series(model.coef_, index=FEATURE_COLUMNS)
        else:
            return None, None
        return contributions.sort_values(key=abs, ascending=False), "Model importance (SHAP unavailable)"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.title("🌫️ Pearls AQI Predictor — Lahore")
    st.caption("Serverless AQI forecasting pipeline — feature store, automated collection, and a 3-day-ahead model")

    df = load_data()
    model, model_name = load_model()

    if df.empty:
        st.error("No data found yet. Run feature_pipeline.py at least once to collect data.")
        return
    if model is None:
        st.warning("No trained model found. Run training_pipeline.py first — showing live data only.")

    latest = df.iloc[-1]
    current_aqi = latest["aqi"]
    category, color = get_aqi_category(current_aqi)

    forecast_aqi = None
    forecast_category = None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current AQI (Lahore)", f"{current_aqi:.0f}", category)
        st.markdown(
            f'<div style="background-color:{color};padding:8px;border-radius:8px;'
            f'text-align:center;color:black;font-weight:bold;">{category}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        if model is not None:
            X_latest = latest[FEATURE_COLUMNS].to_frame().T
            forecast_aqi = float(model.predict(X_latest)[0])
            forecast_category, forecast_color = get_aqi_category(forecast_aqi)
            st.metric("Forecast (~72h ahead)", f"{forecast_aqi:.0f}", forecast_category)
            st.markdown(
                f'<div style="background-color:{forecast_color};padding:8px;border-radius:8px;'
                f'text-align:center;color:black;font-weight:bold;">{forecast_category}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.metric("Forecast (~72h ahead)", "—")

    with col3:
        st.metric("Model in use", model_name or "—")
        st.caption(f"Last updated: {latest['timestamp']}")

    if forecast_aqi is not None and forecast_aqi >= 151:
        st.error(
            f"⚠️ HAZARD ALERT: Forecasted AQI ({forecast_aqi:.0f}) is in the "
            f"'{forecast_category}' range. Sensitive groups should limit outdoor exposure."
        )

    st.divider()

    st.subheader("Historical AQI Trend")
    fig = px.line(df, x="timestamp", y="aqi", title=None)
    fig.update_traces(line_color="#1f77b4")
    for low, high, label, color in AQI_CATEGORIES:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.08, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    if "backfilled" in df.columns:
        st.caption(
            "Note: rows marked as backfilled use approximated weather values "
            "(real historical weather isn't available on the free API tier) — "
            "pollutant values and AQI itself are real, retrieved readings."
        )

    st.divider()

    st.subheader("What's Driving This Forecast?")
    if model is not None:
        contributions, method = explain_prediction(model, X_latest, df[FEATURE_COLUMNS])
        if contributions is not None:
            st.caption(f"Method: {method}")
            fig2 = go.Figure(go.Bar(
                x=contributions.values,
                y=contributions.index,
                orientation="h",
                marker_color=["#d62728" if v > 0 else "#2ca02c" for v in contributions.values],
            ))
            fig2.update_layout(xaxis_title="Contribution to forecast", yaxis_title="Feature", height=450)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Explainability not available for this model type.")
    else:
        st.info("Train a model first to see explainability.")

    st.divider()

    with st.expander("View raw feature data"):
        st.dataframe(df.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
