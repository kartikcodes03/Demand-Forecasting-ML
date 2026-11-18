Demand Forecasting ML — Render Deployment
A web application for daily demand forecasting. The application is built with Python and Streamlit, uses a saved machine-learning model, and is hosted on Render.

Live application: https://demand-forecasting-ml-fzew.onrender.com
GitHub repository: https://github.com/kartikcodes03/Demand-Forecasting-ML
Project Features
- Upload a historical sales CSV file
- Read date and demand/quantity columns
- Aggregate transactions into daily demand
- View historical demand and summary statistics
- Generate a seven-day demand forecast using the saved model
- View forecast results and charts
- Download forecast results as a CSV file
Technology Stack
Area	Technology
Programming language	Python
Web application framework	Streamlit
Data processing	Pandas, NumPy
Machine learning	Scikit-learn
Saved model loading	Joblib
Hosting and deployment	Render
Source code repository	GitHub


Hosting clarification: The application is hosted on Render. Streamlit is the Python framework used to build the web interface; this project is not deployed on Streamlit Community Cloud.

Repository Structure
Demand-Forecasting-ML/
├── app.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── Demand_Model.pkl
    ├── Cleaned_Data.csv
    ├── Forecast.csv
    └── Model_Results.csv
The filenames and paths in this structure must match the actual files in the GitHub repository and the paths used in app.py.
Uploading Your Own CSV
Upload a CSV containing one date column and one demand/quantity column.

Accepted date column names:
- Date
- Order Date
- Order_Date
Accepted demand column names:
- Demand
- Qty
- Quantity
- Units Sold
- Units_Sold
Example:
Date,Qty
01-06-2022,120
02-06-2022,135
03-06-2022,128
04-06-2022,150
05-06-2022,142
06-06-2022,160
07-06-2022,155
At least seven different dates are required by the current application. If multiple transactions occur on the same date, their quantities are added to calculate daily demand.
How Prediction Works
The application prepares daily demand history from the uploaded CSV and passes model features to the saved trained model. The current application code uses date and lag/rolling-demand features such as:
- Day
- Weekday
- Month
- Lag1
- Lag2
- Lag7
- Rolling_Mean
The model generates the forecast one day at a time for the next seven days.

Important: Uploading a CSV does not automatically retrain the model. The uploaded file supplies demand history to the existing trained model. Forecasts may be unreliable if the uploaded data is substantially different from the data used during model training.
Run Locally
1. Clone the repository:
   git clone https://github.com/kartikcodes03/Demand-Forecasting-ML.git
   cd Demand-Forecasting-ML
2. Install dependencies:
   pip install -r requirements.txt
3. Start the application locally:
   streamlit run app.py
4. Open the local address shown in the terminal (usually http://localhost:8501).
Deploy on Render
This project is deployed on Render as a Web Service.
1. Push the application code and required model/data files to the GitHub repository.
2. In Render, choose New → Web Service.
3. Connect the GitHub repository: kartikcodes03/Demand-Forecasting-ML.
4. Select the main branch.
5. If app.py is in the repository root, leave Root Directory blank.
6. Set the Build Command:
   pip install -r requirements.txt
7. Set the Start Command:
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
8. Create the Web Service and wait for the deployment to complete.
Render can automatically redeploy the service when new commits are pushed to the connected GitHub branch.
Model Compatibility
The saved model must be loaded with compatible versions of its Python and scikit-learn dependencies. The scikit-learn version in requirements.txt should match the version used when the model was trained and saved. For this project, the previously specified version was 1.6.1; verify it against the actual training environment before changing it.
Project Information
- Team: G35
- Project: Building a Demand Forecasting Model
- Application hosting: Render
This project is intended for academic demonstration. Forecasts are estimates and should be validated before being used for business decisions.
