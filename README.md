# EV_Charging_Demand_Intelligence
An interactive Streamlit dashboard for EV charging demand analysis, machine learning-based demand prediction and classification, and time-series forecasting. Built with Python, Pandas, Plotly, and Scikit-learn to transform charging data into actionable insights for understanding demand trends and supporting data-driven decisions.
# ⚡ EV Charging Demand Intelligence Dashboard

An interactive Streamlit dashboard for exploring electric-vehicle (EV) charging demand, understanding historical usage patterns, and generating machine-learning predictions and baseline time-series forecasts.

The project brings together **business analytics, regression, classification, and time-series forecasting** in one dashboard.

## Project objectives

- Explore charging-demand trends by month, hour, location, charger type, and charger company.
- Summarize key business indicators such as average demand, peak-demand periods, and demand patterns.
- Predict a demand value for a selected charging scenario using a trained regression model.
- Classify a scenario as **Low Demand** or **High Demand** using a trained classification model.
- Compare baseline time-series forecasting methods and evaluate their errors on held-out months.

## Dashboard sections

### 1. Executive Overview
A high-level view of the charging data, including summary KPIs and charts for demand by month, location, charger type/company, weekday versus weekend, charging duration, and time of day.

### 2. Regression Prediction
Uses a saved preprocessing pipeline and trained regression model to estimate demand for a user-defined scenario. The prediction is shown alongside historical context to make the result easier to interpret.

### 3. Demand Classification
Uses a saved classification model to estimate whether a selected scenario falls into the **Low Demand** or **High Demand** class. Where supported by the model, the dashboard also displays class probabilities.

### 4. Time-Series Forecasting
Aggregates demand by month and compares baseline forecasting approaches, including a 3-month moving average and Simple Exponential Smoothing. Forecast performance is evaluated on a chronological holdout period using:

- **MAE (Mean Absolute Error):** average absolute difference between actual and predicted demand.
- **RMSE (Root Mean Squared Error):** a metric that penalizes larger forecast errors more heavily.

The method with the lowest holdout MAE is selected as the baseline recommendation for the displayed outlook. Forecasts are estimates, not guarantees.

## Technology stack

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- scikit-learn
- LightGBM
- Joblib
- Statsmodels (used for exponential smoothing when available)

## Repository structure

The dashboard expects the data and saved model files to be arranged like this:

```text
your-project/
├── EV_Charging_Dashboard_Executive.py
├── ChargingRecords.csv
├── demand_preprocessor.pkl
├── lightgbm_demand_model.pkl
├── requirements.txt
├── README.md
└── saved_models/
    ├── ev_demand_classifier.pkl
    ├── ev_preprocessor.pkl
    └── demand_threshold.pkl
```

**Important:** The CSV and `.pkl` files are runtime dependencies. The dashboard cannot load predictions without the expected data and trained model artifacts. If you do not want to publish those files, keep them locally and document how to obtain them, or adjust the application to load data/models from another approved location.

## Setup and run locally

### 1. Clone your repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Replace the placeholders with your GitHub username and repository name.

### 2. Create and activate a virtual environment (recommended)

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
py -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the required data and model files

Place `ChargingRecords.csv`, `demand_preprocessor.pkl`, and `lightgbm_demand_model.pkl` beside the dashboard script. Place the three classification artifacts in the `saved_models/` folder, using the filenames shown in the repository structure above.

The CSV should contain the columns required by the dashboard and trained models, including fields such as `Demand`, `StartDatetime`, `Location`, `ChargerType`, `ChargerCompany`, and `Duration`. The exact feature names must match those expected by the saved preprocessing/model artifacts.

### 5. Start the dashboard

```bash
streamlit run EV_Charging_Dashboard_Executive.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Deploying with Streamlit Community Cloud

1. Push the dashboard script, `requirements.txt`, and `README.md` to GitHub.
2. In Streamlit Community Cloud, create an app and select your repository and branch.
3. Set the main file path to `EV_Charging_Dashboard_Executive.py`.
4. Make sure the app can access the CSV and all required model artifacts at the paths expected by the code.
5. Deploy the app.

If the dataset or model artifacts are too large or should not be public, do **not** commit them to a public repository. Use an approved private storage or deployment mechanism and update the loading paths in the application accordingly.

## Notes and limitations

- Model predictions depend on the quality, coverage, and feature definitions of the training data.
- Time-series forecasts are baseline estimates and should be validated with more historical data and repeated backtesting before operational decisions are made.
- A low holdout MAE means smaller average errors on the selected test period; it does not mean the model is a particular percentage accurate.
- Saved Python model files (`.pkl`) should only be loaded from trusted sources. Pickle-based files can execute code during loading.
- Ensure that the dataset and any included artifacts can legally and safely be shared before publishing them.

## Future improvements

- Add automated data validation and clearer handling of missing or unavailable files.
- Expand time-series backtesting and compare additional forecasting models.
- Add prediction intervals or uncertainty estimates where appropriate.
- Add model monitoring and retraining workflows as new charging data becomes available.

---

**Project:** EV Charging Demand Intelligence  
**Application:** Streamlit dashboard  
**Purpose:** Historical analytics, demand prediction, classification, and baseline forecasting
