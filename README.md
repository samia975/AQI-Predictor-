# 🌫️ Pearls AQI Predictor — Lahore

An end-to-end serverless machine learning system that forecasts Lahore's Air Quality Index (AQI) up to **3 days ahead**. Built for the 10Pearls SHINE Internship Program.

## 🔗 Quick Links

- **📊 Live Dashboard:** [saima-aqi-predictor.streamlit.app](https://saima-aqi-predictor.streamlit.app/)
- **📄 Full Project Report:** [AQI_Predictor_Final_Report.docx](./AQI_Predictor_Final_Report.docx)

## 📋 Project Overview

This project implements the full ML lifecycle for AQI forecasting:

1. **Feature Pipeline** (`feature_pipeline.py`) — fetches live weather + pollution data for Lahore from OpenWeather, computes the US EPA AQI, and engineers time-based and derived features.
2. **Historical Backfill** (`backfill_historical.py`) — recovers real historical pollutant data using OpenWeather's Air Pollution History API.
3. **Feature Store** — Hopsworks (free tier), with a local CSV backup for redundancy.
4. **Automation** — GitHub Actions runs the feature pipeline hourly and commits fresh data automatically (`.github/workflows/feature-pipeline.yml`).
5. **Training Pipeline** (`training_pipeline.py`) — trains and evaluates Ridge Regression, Random Forest, and a Neural Network; registers the best model to the Hopsworks Model Registry.
6. **Dashboard** (`dashboard/app.py`) — Streamlit app showing current AQI, the 3-day forecast, historical trends, SHAP explainability, and hazard alerts.

## 🛠️ Tech Stack

Python 3.12 · OpenWeather API · Hopsworks Feature Store & Model Registry · GitHub Actions · scikit-learn · SHAP · Streamlit · Plotly

## 📈 Results

Best model: **Ridge Regression**, forecasting AQI 72 hours ahead.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 70.76 | 22.01 | 0.058 |
| Random Forest | 71.44 | 17.42 | 0.040 |
| Neural Network | 73.19 | 29.58 | -0.008 |

See the [full report](./AQI_Predictor_Final_Report.docx) for architecture details, engineering challenges (Python/Hopsworks compatibility, a feature-store materialization outage and its recovery, etc.), and an honest discussion of model limitations.

## ▶️ Running Locally

```bash
pip install -r requirements.txt
export OPENWEATHER_API_KEY="your_key"
export HOPSWORKS_API_KEY="your_key"
export HOPSWORKS_PROJECT_NAME="your_project"

python feature_pipeline.py       # collect one data point
python training_pipeline.py      # train and register a model

cd dashboard
pip install -r requirements.txt
streamlit run app.py             # launch the dashboard
```
