import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Setting page configuration
st.set_page_config(page_title="Demand Forecasting", page_icon="📦", layout="wide")

# Defining project file paths
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "artifacts" / "G35_Demand_Model.pkl"
DATA_PATH = BASE / "artifacts" / "G35_Cleaned_Data.csv"
FORECAST_PATH = BASE / "artifacts" / "G35_Forecast.csv"
RESULTS_PATH = BASE / "artifacts" / "G35_Model_Results.csv"

# Loading the trained ML model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Loading historical demand data
@st.cache_data
def load_data():
    d = pd.read_csv(DATA_PATH)
    d["Date"] = pd.to_datetime(d["Date"])
    return d.sort_values("Date")

# Loading saved CSV files
@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

# Displaying project heading
st.title("📦 Demand Forecasting System")
st.write("Daily demand prediction using Random Forest and Extra Trees.")
st.caption("Team Project: G35")

# Checking required project files
missing = [p.name for p in [MODEL_PATH, DATA_PATH] if not p.exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing))
    st.stop()

# Loading model and dataset
try:
    model = load_model()
    data = load_data()
except Exception as e:
    st.error(f"Could not load model or dataset: {e}")
    st.stop()

saved_forecast = load_csv(FORECAST_PATH)
results = load_csv(RESULTS_PATH)

# Showing project summary
st.sidebar.header("Project Details")
st.sidebar.write("Project: Demand Forecasting")
st.sidebar.write("Forecast period: 7 days")
st.sidebar.caption("Historical dataset covers a limited period in 2022.")

# Showing key performance indicators
c1, c2, c3 = st.columns(3)
c1.metric("Historical Days", f"{len(data):,}")
c2.metric("Total Units Sold", f"{data['Demand'].sum():,.0f}")
c3.metric("Average Daily Demand", f"{data['Demand'].mean():,.1f}")

st.write(f"Data period: {data['Date'].min().date()} to {data['Date'].max().date()}")

# Plotting historical demand with Streamlit's built-in chart
st.subheader("1. Historical Daily Demand")
st.line_chart(data.set_index("Date")[["Demand"]], use_container_width=True)

# Generating future demand predictions
st.subheader("2. Generate 7-Day Forecast")
if st.button("Generate 7-Day Forecast", type="primary"):
    history = data.set_index("Date")["Demand"].astype(float).copy()
    rows = []

    for _ in range(7):
        next_date = history.index[-1] + pd.Timedelta(days=1)

        # Creating input features using previous demand
        x_new = pd.DataFrame([{
            "Day": next_date.day,
            "Weekday": next_date.dayofweek,
            "Month": next_date.month,
            "Lag1": history.iloc[-1],
            "Lag2": history.iloc[-2],
            "Lag7": history.iloc[-7],
            "Rolling_Mean": history.iloc[-7:].mean()
        }])

        # Predicting demand for the next day
        pred = max(0, float(model.predict(x_new)[0]))
        rows.append({"Date": next_date, "Predicted Demand": round(pred, 2)})
        history.loc[next_date] = pred

    st.session_state["forecast_result"] = pd.DataFrame(rows)

# Displaying generated forecast and chart
if "forecast_result" in st.session_state:
    forecast = st.session_state["forecast_result"]
    st.dataframe(forecast, use_container_width=True)

    st.subheader("3. Historical vs Forecast Demand")
    recent = data.tail(21).set_index("Date")[["Demand"]]
    future = forecast.rename(columns={"Predicted Demand": "Demand"}).set_index("Date")[["Demand"]]
    st.line_chart(pd.concat([recent, future]), use_container_width=True)

    # Downloading forecast as CSV
    st.download_button(
        "Download Forecast CSV",
        forecast.to_csv(index=False).encode("utf-8"),
        "Demand_Forecast_Next_7_Days.csv",
        "text/csv"
    )
elif not saved_forecast.empty:
    st.info("Previously saved forecast:")
    st.dataframe(saved_forecast, use_container_width=True)

# Displaying model evaluation metrics
st.subheader("4. Model Performance")
if not results.empty:
    st.dataframe(results, use_container_width=True)
    if {"Model", "RMSE"}.issubset(results.columns):
        st.bar_chart(results.set_index("Model")[["RMSE"]])
else:
    st.info("Model results file was not found.")

# Displaying feature importance
st.subheader("5. Feature Importance")
try:
    model_step = model.named_steps["model"]
    names = ["Day", "Weekday", "Month", "Lag1", "Lag2", "Lag7", "Rolling_Mean"]
    importance = pd.DataFrame(
        {"Importance": model_step.feature_importances_}, index=names
    ).sort_values("Importance")
    st.bar_chart(importance)
except Exception as e:
    st.info(f"Feature importance unavailable: {e}")

st.markdown("---")
st.caption("Academic Machine Learning Project | Demand Forecasting")
