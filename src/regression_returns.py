import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# ----------------------------
# Download data
# ----------------------------
stock = input('Stock ticker: ')
num_days = int(input('Number of days for analysis (integer): '))

start_date = datetime.datetime.now() - datetime.timedelta(days=num_days)
df = yf.download(stock, start=start_date)
df.dropna(inplace=True)

# ----------------------------
# Compute log returns + lag features
# ----------------------------
df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
df['Lag1'] = df['LogReturn'].shift(1)
df['Lag2'] = df['LogReturn'].shift(2)
df.dropna(inplace=True)

# ----------------------------
# Regression model
# ----------------------------
X = df[['Lag1', 'Lag2']]
y = df['LogReturn']
X = sm.add_constant(X)  # add intercept

model = sm.OLS(y, X).fit()
print(model.summary())

# ----------------------------
# Predict next day's return
# ----------------------------
latest = df[['Lag1', 'Lag2']].iloc[[-1]]   # double brackets = DataFrame
latest = sm.add_constant(latest, has_constant='add')

pred = model.predict(latest)

last_price = df['Close'].iloc[-1]
predicted_price = (last_price * np.exp(pred.iloc[0])).item()

print(f"\nPredicted next log-return: {pred.iloc[0]:.6f}")
print(f"Predicted next close price: ${predicted_price:.2f}")

