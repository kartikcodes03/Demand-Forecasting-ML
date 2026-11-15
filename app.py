import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page setup
st.set_page_config(page_title="Demand Forecasting", page_icon="📦", layout="wide")

# Project file locations
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "artifacts" / "Demand_Model.pkl"
SAMPLE_PATH = BASE / "artifacts" / "Cleaned_Data.csv"
RESULTS_PATH = BASE / "artifacts" / "Model_Results.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def prepare_data(df):
    # Standardize column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ["date", "order date", "order_date"]), None)
    qty_col = next((c for c in df.columns if c.lower() in ["demand", "qty", "quantity", "units sold", "units_sold"]), None)

    if date_col is None or qty_col is None:
        raise ValueError("CSV must contain a Date column and a Demand or Qty column.")

    # Convert dates and quantities
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
    df = df.dropna(subset=[date_col, qty_col])

    # Aggregate sales into daily demand
    daily = df.groupby(df[date_col].dt.normalize())[qty_col].sum().reset_index()
    daily.columns = ["Date", "Demand"]
    daily = daily.sort_values("Date").reset_index(drop=True)

    if len(daily) < 7:
        raise ValueError("At least 7 different dates are required.")
    return daily

def make_forecast(model, daily, days=7):
    # Generate future dates using the uploaded historical data
    history = daily.set_index("Date")["Demand"].astype(float).copy()
    output = []
    for _ in range(days):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        features = pd.DataFrame([{
            "Day": next_date.day,
            "Weekday": next_date.dayofweek,
            "Month": next_date.month,
            "Lag1": history.iloc[-1],
            "Lag2": history.iloc[-2],
            "Lag7": history.iloc[-7],
            "Rolling_Mean": history.iloc[-7:].mean()
        }])
        prediction = max(0, float(model.predict(features)[0]))
        output.append({"Date": next_date, "Predicted Demand": round(prediction, 2)})
        history.loc[next_date] = prediction
    return pd.DataFrame(output)

st.title("📦 Demand Forecasting System")
st.write("Upload your own historical sales CSV and predict demand for the next 7 days.")

# Load the trained Random Forest pipeline
if not MODEL_PATH.exists():
    st.error("Model file missing. Check artifacts/Demand_Model.pkl in GitHub.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load trained model: {e}")
    st.stop()

st.sidebar.header("Project Details")
st.sidebar.write("Model: Random Forest Pipeline")
st.sidebar.write("Forecast horizon: 7 days")

# Upload the user's CSV file
st.subheader("1. Upload Your Sales Data")
st.write("Required columns: Date and Demand or Qty.")
st.caption("Example: Date, Qty  |  01-06-2022, 25")
uploaded_file = st.file_uploader("Choose your CSV file", type=["csv"])

daily = None
if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file, on_bad_lines="skip")
        daily = prepare_data(raw_data)
        st.success("Your CSV was uploaded and processed successfully.")
        st.write("Daily demand preview:")
        st.dataframe(daily.tail(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not process CSV: {e}")
        st.info("Please check the column names and make sure there are at least 7 dates.")
        st.stop()
else:
    st.info("Upload a CSV file to create a forecast.")
    if SAMPLE_PATH.exists() and st.checkbox("Use sample project data"):
        daily = prepare_data(pd.read_csv(SAMPLE_PATH))
        st.dataframe(daily.tail(10), use_container_width=True)

if daily is not None:
    st.subheader("2. Uploaded Data Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Historical Days", f"{len(daily):,}")
    c2.metric("Total Demand", f"{daily['Demand'].sum():,.0f}")
    c3.metric("Average Daily Demand", f"{daily['Demand'].mean():,.1f}")
    st.write(f"Date range: {daily['Date'].min().date()} to {daily['Date'].max().date()}")

    st.subheader("3. Historical Demand Graph")
    st.line_chart(daily.set_index("Date")[["Demand"]])

    st.subheader("4. Generate 7-Day Forecast")
    if st.button("Predict Next 7 Days", type="primary"):
        st.session_state["user_forecast"] = make_forecast(model, daily, 7)

    if "user_forecast" in st.session_state:
        forecast = st.session_state["user_forecast"]
        st.success("Your 7-day forecast is ready.")
        st.dataframe(forecast, use_container_width=True)
        st.subheader("5. Historical vs Predicted Demand")
        actual = daily.tail(21).set_index("Date")[["Demand"]]
        future = forecast.rename(columns={"Predicted Demand": "Demand"}).set_index("Date")[["Demand"]]
        st.line_chart(pd.concat([actual, future]))
        st.download_button(
            "Download Forecast CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            "Your_Demand_Forecast.csv",
            "text/csv"
        )

    st.subheader("6. Model Evaluation")
    if RESULTS_PATH.exists():
        st.dataframe(pd.read_csv(RESULTS_PATH), use_container_width=True)

st.markdown("---")
st.caption("Academic Machine Learning Project | Demand Forecasting")
