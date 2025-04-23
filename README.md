# ⏳ Time Series Forecasting with Scikit-learn + Streamlit

This project demonstrates time series forecasting using Scikit-learn models like Random Forest and Gradient Boosting, combined with a Streamlit dashboard for interactive exploration.

## 📦 Features

- Lag features, rolling averages, and date-based feature engineering
- Train/test split on time-based index
- Interactive model selection (Random Forest / Gradient Boosting)
- Visual forecast vs actual plot
- Performance metrics (Mean Squared Error)

## 📁 Dataset Format

Ensure your `data/data.csv` has the following structure:

```csv
date,value
2023-01-01,100
2023-01-02,105
...
```

## 🚀 How to Run

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```

2. Place your time series CSV file in the `data/` folder as `data.csv`

3. Launch the dashboard:

```bash
streamlit run src/main.py
```

## 📊 Output

- Interactive line chart comparing actual vs predicted values
- Mean Squared Error display

---

🧠 Powered by Scikit-learn, Streamlit, and Python.
