# Algorithmic Trading Starter Project

This project is a beginner-friendly introduction to algorithmic trading.  
I built two scripts to test and demonstrate simple strategies on stock data from Yahoo Finance.

---

## 📊 SMA Crossover Strategy (`main.py`)

My script tests moving average crossover strategies:

- **SMA (Simple Moving Average)** = average stock price over N days.
- If a short SMA (e.g. 10-day) crosses above a long SMA (e.g. 200-day) → **buy**.
- If it crosses below → **sell**.

What it does:
1. Downloads stock data for a ticker I choose.
2. Tries different SMA pairs (10/50, 20/100, etc.).
3. Finds which pair gives the best returns.
4. Runs the strategy with that pair and plots buy/sell signals on the price chart.
5. Tracks portfolio value starting at $100,000.

---

## 📈 Regression on Returns (`regression_returns.py`)

My script builds a simple predictive model:

- Computes **log-returns** (log of today’s price / yesterday’s price).
- Uses yesterday’s return (Lag1) and two days ago (Lag2) to predict tomorrow’s return.
- Fits a regression model:  
  `Today’s return = a + b*Lag1 + c*Lag2`.
- Prints the regression summary.
- Predicts tomorrow’s log-return and the expected next closing price.

---

## 🧠 In Plain English

- `main.py` = backtests a basic trading rule using moving averages.  
- `regression_returns.py` = tries to predict tomorrow’s price from the last two days’ returns.  

Both scripts follow the same process: **get data → apply rules/model → see results**.

---

## ⚠️ Disclaimer
This is for learning only, not financial advice.
