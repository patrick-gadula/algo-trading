import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import sys

# ----------------------------
# Download data
# ----------------------------
stock = input('Stock ticker: ')
num_days = int(input('Number of days for analysis (integer): '))

if num_days < 100:
    print("\nRegression model needs more than 100+ days")
    sys.exit()

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
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

# ----------------------------
# Predict next day's return
# ----------------------------
latest = df[['Lag1', 'Lag2']].iloc[[-1]]
latest = sm.add_constant(latest, has_constant='add')

pred = model.predict(latest)

last_price = float(df['Close'].iloc[-1])
predicted_price = (last_price * np.exp(pred.iloc[0])).item()

print(f"\nPredicted next log-return: {pred.iloc[0]:.6f}")
print(f"Predicted next close price: ${predicted_price:.2f}")

# ----------------------------
# Turn prediction into trade signal
# ----------------------------
entry_price = last_price
take_profit = predicted_price
# risk management: stop loss at +/-2% from entry
if pred.iloc[0] > 0:
    stop_loss = entry_price * 0.98
    print(f"\n📈 Signal: LONG {stock}")
    print(f"  Entry at ~${entry_price:.2f}")
    print(f"  Take profit target: ${take_profit:.2f}")
    print(f"  Stop loss: ${stop_loss:.2f}")
    print("  Holding period: ~1 day (next trading session)")
else:
    stop_loss = entry_price * 1.02
    print(f"\n📉 Signal: SHORT {stock}")
    print(f"  Entry at ~${entry_price:.2f}")
    print(f"  Take profit target: ${take_profit:.2f}")
    print(f"  Stop loss: ${stop_loss:.2f}")
    print("  Holding period: ~1 day (next trading session)")
