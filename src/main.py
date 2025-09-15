# Import libraries
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
plt.style.use('fivethirtyeight')

# ----------------------------
# Buy/Sell strategy function
# ----------------------------
def buy_sell_signal(data):
    buy_signal = []
    sell_signal = []
    open_position = []
    funds = [100000] * len(data)
    last_funds = 100000
    flag = 0  # 0 = no open position, 1 = long position
    last_pos = 0

    for i in range(len(data)):
        price = data['Price'].iloc[i]
        sma20 = data['SMA_short'].iloc[i]
        sma50 = data['SMA_long'].iloc[i]

        if sma20 > sma50:
            if flag == 0:
                flag = 1
                buy_signal.append(price)
                last_pos = last_funds / price
                funds[i] = last_funds
                open_position.append(last_pos)
                sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                last_funds = price * last_pos
                funds[i] = last_funds
                open_position.append(last_pos)
                sell_signal.append(np.nan)
        elif sma20 < sma50:
            if flag == 1:
                flag = 0
                buy_signal.append(np.nan)
                last_funds = last_pos * price
                funds[i] = last_funds
                open_position.append(0)
                sell_signal.append(price)
            else:
                buy_signal.append(np.nan)
                funds[i] = last_funds
                open_position.append(0)
                sell_signal.append(np.nan)
        else:
            buy_signal.append(np.nan)
            open_position.append(0)
            sell_signal.append(np.nan)
            funds[i] = last_funds

    return buy_signal, sell_signal, open_position, funds, flag

# ----------------------------
# SMA evaluation function
# ----------------------------
def evaluate_strategy(df, short_window, long_window, initial_capital=100000):
    df = df.copy()
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()

    position = 0.0
    funds = initial_capital

    for i in range(len(df)):
        price = df['Close'].iloc[i].item()   # safe scalar
        sma_short = df['SMA_short'].iloc[i]
        sma_long = df['SMA_long'].iloc[i]

        if pd.isna(sma_short) or pd.isna(sma_long):
            continue  # skip until SMAs are valid

        # Buy
        if sma_short > sma_long and position == 0:
            position = funds / price
            funds = 0
        # Sell
        elif sma_short < sma_long and position > 0:
            funds = position * price
            position = 0

    # Liquidate at end if holding
    if position > 0:
        funds = position * df['Close'].iloc[-1].item()

    return float(funds)


# ----------------------------
# Main script
# ----------------------------
stock = input('Stock ticker: ')
num_days = int(input('Number of days for analysis (integer): '))

start_date = datetime.datetime.now() - datetime.timedelta(days=num_days)
df = yf.download(stock, start=start_date)
df.dropna(how='any', inplace=True)

# Step 1: Find best SMA pair
best_result = 0
best_pair = None

for short in [10, 20, 30]:
    for long in [50, 100, 150, 200]:
        if short < long:
            result = evaluate_strategy(df, short, long)
            print(f"SMA {short}/{long}: Final value ${result:,.2f}")
            if result > best_result:
                best_result = result
                best_pair = (short, long)

print("\nBest SMA pair:", best_pair, "Final Portfolio Value:", f"${best_result:,.2f}")

# Step 2: Build DataFrame with best SMAs
df['SMA_short'] = df['Close'].rolling(window=best_pair[0]).mean()
df['SMA_long'] = df['Close'].rolling(window=best_pair[1]).mean()

Data = pd.DataFrame()
Data['Price'] = df['Close']
Data['SMA_short'] = df['SMA_short']
Data['SMA_long'] = df['SMA_long']
Data['funds'] = 100000

# Step 3: Apply buy/sell strategy
buy_sell = buy_sell_signal(Data)
Data['Buy_price'] = buy_sell[0]
Data['Sell_price'] = buy_sell[1]
Data['Open_pos'] = buy_sell[2]
Data['live_pos'] = Data['Open_pos'] * Data['Price']
Data['funds'] = buy_sell[3]

# Step 4: Plot results
plt.figure(figsize=(15, 8))
plt.plot(Data['Price'], label=stock, linewidth=1)
plt.plot(Data['SMA_short'], label=f'SMA{best_pair[0]}', linewidth=0.7)
plt.plot(Data['SMA_long'], label=f'SMA{best_pair[1]}', linewidth=0.7)
plt.scatter(Data.index, Data['Buy_price'], label='Buy', marker='^', color='g')
plt.scatter(Data.index, Data['Sell_price'], label='Sell', marker='v', color='r')
plt.title(f'{stock} SMA {best_pair[0]}/{best_pair[1]} Buy-Sell Strategy')
plt.xlabel(f'Last {num_days} days')
plt.ylabel('Close price ($)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("strategy_chart.png")
print("Chart saved as strategy_chart.png")
