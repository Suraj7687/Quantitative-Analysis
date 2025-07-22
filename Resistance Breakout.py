
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import copy
import time
import matplotlib.pyplot as plt 


def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2['ATR']

def CAGR(DF): #CAGR gives long term growth rate.
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df)/(252*78)
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(252*78)
    return vol

def sharpe(DF,rf): #gives risk-adjusted performance.
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF): #Worst historical loss.
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

# Download historical data (monthly) for selected stocks

tickers = ["MSFT","AAPL","AMZN","INTC", "CSCO","VZ","IBM","TSLA","AMD"]
key_path = "C:\\Users\\Lenovo\\OneDrive\\Documents\\APIKEY\\key.txt"         
with open(key_path, 'r') as file:
    api_key = file.read().strip()

# Define tickers (IMPORTANT: Alpha Vantage works best with US tickers)

# Initialize TimeSeries
ts = TimeSeries(key=api_key, output_format='pandas')

ohlc_intraday = {}
api_call_count = 1
start_time = time.time() 

for ticker in tickers:
    try:
        data, meta = ts.get_intraday(symbol=ticker, interval='5min', outputsize='full')
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data.iloc[::-1]  # Reverse for chronological order
        data = data.between_time('9:35','16:00')
        ohlc_intraday[ticker] = data
        print(f"Downloaded: {ticker}")
    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
    
    api_call_count += 1
    if api_call_count == 5:
        api_call_count = 1
        time.sleep(60 - (time.time() - start_time) % 60.0)

tickers = ohlc_intraday.keys() # redefine tickers variable after removing any tickers with corrupted data

################################Backtesting####################################

# calculating ATR and rolling max price for each stock and consolidating this info by stock in a separate dataframe
ohlc_dict = copy.deepcopy(ohlc_intraday)
tickers_signal = {}
tickers_ret = {}
for ticker in tickers:
    print("calculating ATR and rolling max price for ",ticker)
    ohlc_dict[ticker]["ATR"] = ATR(ohlc_dict[ticker],20)
    ohlc_dict[ticker]["roll_max_cp"] = ohlc_dict[ticker]["High"].rolling(20).max()
    ohlc_dict[ticker]["roll_min_cp"] = ohlc_dict[ticker]["Low"].rolling(20).min()
    ohlc_dict[ticker]["roll_max_vol"] = ohlc_dict[ticker]["Volume"].rolling(20).max()
    ohlc_dict[ticker].dropna(inplace=True)
    tickers_signal[ticker] = ""
    tickers_ret[ticker] = [0]


buy_signals = {}
sell_signals = {}
# identifying signals and calculating daily return (stop loss factored in)
for ticker in tickers:
    tickers_ret[ticker] = [0]
    tickers_signal[ticker] = ""
    buy_signals[ticker] = []
    sell_signals[ticker] = []

    print("calculating returns for", ticker)
    df = ohlc_dict[ticker]
    for i in range(1, len(df)):
        if tickers_signal[ticker] == "":
            tickers_ret[ticker].append(0)

            if df["High"].iloc[i] >= df["roll_max_cp"].iloc[i] and df["Volume"].iloc[i] > 1.5 * df["roll_max_vol"].iloc[i-1]:
                tickers_signal[ticker] = "Buy"
                buy_signals[ticker].append((df.index[i], df["Close"].iloc[i]))

            elif df["Low"].iloc[i] <= df["roll_min_cp"].iloc[i] and df["Volume"].iloc[i] > 1.5 * df["roll_max_vol"].iloc[i-1]:
                tickers_signal[ticker] = "Sell"
                sell_signals[ticker].append((df.index[i], df["Close"].iloc[i]))

        elif tickers_signal[ticker] == "Buy":
            if df["Low"].iloc[i] < df["Close"].iloc[i-1] - df["ATR"].iloc[i-1]:
                tickers_signal[ticker] = ""
                tickers_ret[ticker].append(((df["Close"].iloc[i-1] - df["ATR"].iloc[i-1]) / df["Close"].iloc[i-1]) - 1)

            elif df["Low"].iloc[i] <= df["roll_min_cp"].iloc[i] and df["Volume"].iloc[i] > 1.5 * df["roll_max_vol"].iloc[i-1]:
                tickers_signal[ticker] = "Sell"
                sell_signals[ticker].append((df.index[i], df["Close"].iloc[i]))
                tickers_ret[ticker].append((df["Close"].iloc[i] / df["Close"].iloc[i-1]) - 1)

            else:
                tickers_ret[ticker].append((df["Close"].iloc[i] / df["Close"].iloc[i-1]) - 1)

        elif tickers_signal[ticker] == "Sell":
            if df["High"].iloc[i] > df["Close"].iloc[i-1] + df["ATR"].iloc[i-1]:
                tickers_signal[ticker] = ""
                tickers_ret[ticker].append((df["Close"].iloc[i-1] / (df["Close"].iloc[i-1] + df["ATR"].iloc[i-1])) - 1)

            elif df["High"].iloc[i] >= df["roll_max_cp"].iloc[i] and df["Volume"].iloc[i] > 1.5 * df["roll_max_vol"].iloc[i-1]:
                tickers_signal[ticker] = "Buy"
                buy_signals[ticker].append((df.index[i], df["Close"].iloc[i]))
                tickers_ret[ticker].append((df["Close"].iloc[i-1] / df["Close"].iloc[i]) - 1)

            else:
                tickers_ret[ticker].append((df["Close"].iloc[i-1] / df["Close"].iloc[i]) - 1)

    ohlc_dict[ticker]["ret"] = np.array(tickers_ret[ticker])


#plotting signals on the chart.

ticker = "MSFT"
df = ohlc_dict[ticker]

buy_dates, buy_prices = zip(*buy_signals[ticker]) if buy_signals[ticker] else ([], [])
sell_dates, sell_prices = zip(*sell_signals[ticker]) if sell_signals[ticker] else ([], [])

plt.figure(figsize=(14, 6))
plt.plot(df.index, df["Close"], label="Close Price", color='blue')

plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy Signal', zorder=5)
plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal', zorder=5)

plt.title(f"{ticker} - Price Chart with Buy/Sell Signals")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# calculating overall strategy's KPIs
strategy_df = pd.DataFrame() #This creates an empty DataFrame named strategy_df.
#Think of it like a blank Excel sheet.
#It will be used to store return values of each stock and then the overall strategy return.
for ticker in tickers:
    strategy_df[ticker] = ohlc_dict[ticker]["ret"]
strategy_df["ret"] = strategy_df.mean(axis=1)
CAGR(strategy_df)
sharpe(strategy_df,0.025)
max_dd(strategy_df)  


# vizualization of strategy return
(1+strategy_df["ret"]).cumprod().plot()
plt.show()

#calculating individual stock's KPIs
cagr = {}
sharpe_ratio = {}
max_drawdown = {}
for ticker in tickers:
    print("calculating KPI's for",ticker)
    cagr[ticker] = CAGR(ohlc_dict[ticker])
    sharpe_ratio[ticker] = sharpe(ohlc_dict[ticker], 0.025)
    max_drawdown[ticker] = max_dd(ohlc_dict[ticker])


KPI_df = pd.DataFrame([cagr, sharpe_ratio, max_drawdown], index = ["Return(CAGR)", "Sharpe", "Max Drawdown"])
KPI_df.T