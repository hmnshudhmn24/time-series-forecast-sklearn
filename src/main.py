import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/data.csv', parse_dates=['date'], index_col='date')
target = 'value'

# Feature engineering
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df['lag1'] = df[target].shift(1)
df['lag2'] = df[target].shift(2)
df['rolling_mean3'] = df[target].rolling(window=3).mean()

df.dropna(inplace=True)

# Split data
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Streamlit dashboard
st.title("📈 Time Series Forecasting Dashboard")

model_option = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])

if model_option == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("📊 Forecast vs Actual")
fig, ax = plt.subplots()
ax.plot(y_test.index, y_test.values, label='Actual')
ax.plot(y_test.index, y_pred, label='Forecast')
ax.legend()
st.pyplot(fig)

mse = mean_squared_error(y_test, y_pred)
st.write(f"**Mean Squared Error:** {mse:.2f}")
