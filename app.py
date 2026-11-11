import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ---------------------------------------------------------
# G35 Demand Forecasting - Streamlit Frontend
# ---------------------------------------------------------

st.set_page_config(
    page_title="G35 | Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "G35_Demand_Model.pkl"
DATA_PATH = BASE_DIR / "artifacts" / "G35_Cleaned_Data.csv"
FORECAST_PATH = BASE_DIR / "artifacts" / "G35_Forecast.csv"
RESULTS_PATH = BASE_DIR / "artifacts" / "G35_Model_Results.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    return data.sort_values("Date")

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

st.title("📦 G35 — Demand Forecasting System")
st.caption("Daily sales demand analysis using a Random Forest machine learning pipeline")

# Check that essential files exist
missing_files = [str(p.name) for p in [MODEL_PATH, DATA_PATH] if not p.exists()]
if missing_files:
    st.error("Required project file(s) missing: " + ", ".join(missing_files))
    st.info("Keep the model and cleaned dataset inside the artifacts folder.")
    st.stop()

try:
    model = load_model()
    data = load_data()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.warning("Use the same scikit-learn version used to train the model (1.6.1).")
    st.stop()

forecast_saved = load_csv(FORECAST_PATH)
results = load_csv(RESULTS_PATH)

# Sidebar controls
st.sidebar.header("Project Information")
st.sidebar.write("**Team:** G35")
st.sidebar.write("**Algorithm:** Random Forest Regressor")
st.sidebar.write("**Forecast horizon:** 7 days")
st.sidebar.markdown("---")
st.sidebar.caption("The uploaded historical dataset covers April–June 2022. Forecasts are demonstrations based on this historical period.")

# KPI cards
latest_date = data["Date"].max()
total_units = data["Demand"].sum()
average_demand = data["Demand"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Historical Records", f"{len(data):,} days")
c2.metric("Total Units in Dataset", f"{total_units:,.0f}")
c3.metric("Average Daily Demand", f"{average_demand:,.1f}")

st.markdown(f"**Historical data range:** {data['Date'].min().date()} to {latest_date.date()}")

# Historical demand chart
st.subheader("1. Historical Daily Demand")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(data["Date"], data["Demand"], linewidth=1.8)
ax.set_xlabel("Date")
ax.set_ylabel("Units sold")
ax.grid(alpha=0.25)
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

# Forecast section
st.subheader("2. Generate Future Demand Forecast")
st.write("The app uses the saved Random Forest pipeline and the most recent historical demand to predict the next 7 days.")

if st.button("Generate 7-Day Forecast", type="primary"):
    history = data.set_index("Date")["Demand"].astype(float).copy()
    forecast_rows = []

    for _ in range(7):
        next_date = history.index[-1] + pd.Timedelta(days=1)

        # Build the same features used during model training
        input_row = pd.DataFrame([{
            "Day": next_date.day,
            "Weekday": next_date.dayofweek,
            "Month": next_date.month,
            "Lag1": history.iloc[-1],
            "Lag2": history.iloc[-2],
            "Lag7": history.iloc[-7],
            "Rolling_Mean": history.iloc[-7:].mean()
        }])

        prediction = float(model.predict(input_row)[0])
        prediction = max(0, prediction)

        forecast_rows.append({
            "Date": next_date,
            "Predicted Demand": round(prediction, 2)
        })
        history.loc[next_date] = prediction

    generated_forecast = pd.DataFrame(forecast_rows)
    st.session_state["generated_forecast"] = generated_forecast

if "generated_forecast" in st.session_state:
    generated_forecast = st.session_state["generated_forecast"]
    st.dataframe(generated_forecast, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    recent = data.tail(21)
    ax2.plot(recent["Date"], recent["Demand"], label="Historical Demand")
    ax2.plot(generated_forecast["Date"], generated_forecast["Predicted Demand"],
             marker="o", linestyle="--", label="Predicted Demand")
    ax2.set_title("Historical vs 7-Day Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Units")
    ax2.legend()
    ax2.grid(alpha=0.25)
    fig2.autofmt_xdate()
    st.pyplot(fig2)
    plt.close(fig2)

    csv_bytes = generated_forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download New Forecast CSV",
        data=csv_bytes,
        file_name="G35_New_7_Day_Forecast.csv",
        mime="text/csv"
    )

elif not forecast_saved.empty:
    st.info("A previously saved forecast is available below.")
    st.dataframe(forecast_saved, use_container_width=True)

# Model evaluation
st.subheader("3. Model Performance")
if not results.empty:
    st.dataframe(results, use_container_width=True)
    if {"Model", "RMSE"}.issubset(results.columns):
        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        ax3.bar(results["Model"], results["RMSE"])
        ax3.set_ylabel("RMSE (lower is better)")
        ax3.set_title("Model RMSE Comparison")
        ax3.grid(axis="y", alpha=0.25)
        st.pyplot(fig3)
        plt.close(fig3)
else:
    st.warning("Model results CSV was not found. Forecasting will still work.")

# Feature importance
st.subheader("4. Feature Importance")
try:
    estimator = model.named_steps["model"]
    feature_names = ["Day", "Weekday", "Month", "Lag1", "Lag2", "Lag7", "Rolling_Mean"]
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": estimator.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    ax4.set_xlabel("Importance")
    ax4.set_title("Random Forest Feature Importance")
    ax4.grid(axis="x", alpha=0.25)
    st.pyplot(fig4)
    plt.close(fig4)
except Exception as error:
    st.info(f"Feature importance is not available: {error}")

st.markdown("---")
st.caption("G35 Academic Machine Learning Project | Demand Forecasting using Random Forest")
