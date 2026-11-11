# Demand-Forecasting-ML

A machine learning project that analyzes historical Amazon sales and forecasts daily product demand using a **Random Forest Regressor**. The project also compares Random Forest and Extra Trees models.

## Project features

- Data cleaning and daily demand aggregation
- Exploratory data analysis and visualizations
- Feature engineering using calendar, lag and rolling-average features
- Scikit-learn preprocessing and prediction pipeline
- Random Forest and Extra Trees model comparison
- Model evaluation using MAE, RMSE and R²
- Seven-day recursive demand forecast
- Interactive Streamlit frontend
- Downloadable forecast CSV

## Repository structure

```text
Demand-Forecasting-ML/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── artifacts/
    ├── Demand_Model.pkl
    ├── Cleaned_Data.csv
    ├── Forecast.csv
    └── Model_Results.csv
```

## Dataset

The project uses the Amazon Sale Report dataset. The original raw CSV is not included in this repository because it is large. The cleaned daily dataset required by the frontend is included in `artifacts/`.

The available cleaned data covers a limited period in 2022. Therefore, the forecast is an academic demonstration and should not be treated as a reliable long-term business forecast.

## Run the project locally

### 1. Install Python

Use Python 3.10 or 3.11 for a compatible environment.

### 2. Download or clone this repository

```bash
git clone https://github.com/YOUR-USERNAME/G35-Demand-Forecasting.git
cd G35-Demand-Forecasting
```

### 3. Create and activate a virtual environment (recommended)

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the frontend

```bash
streamlit run app.py
```

The app will open in your browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**.
4. Choose this repository and branch (`main`).
5. Set the main file path to `app.py`.
6. Select **Deploy**.

Keep `app.py`, `requirements.txt`, and the `artifacts` folder in the repository. Do not rename the artifact files unless you also update their paths in `app.py`.

## Model details

**Target:** Total quantity sold per day (`Qty` aggregated by date).

**Input features:**
- `Day`
- `Weekday`
- `Month`
- `Lag1`
- `Lag2`
- `Lag7`
- `Rolling_Mean`

**Algorithms compared:**
- Random Forest Regressor
- Extra Trees Regressor

**Evaluation metrics:**
- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — Coefficient of Determination

The saved model is a complete scikit-learn Pipeline containing `StandardScaler` and `RandomForestRegressor`.

## Important compatibility note

The supplied model was saved with **scikit-learn 1.6.1**. The `requirements.txt` pins that version so the saved model can be loaded consistently. Scikit-learn warns that loading pickle/joblib models across different versions is not guaranteed to be compatible.

## Team

 **Project Title:** Building a Demand Forecasting Model  
**Project Type:** Machine Learning / Regression / Time-Series Forecasting

---

Academic project. Forecast quality depends on the amount and quality of historical data.
