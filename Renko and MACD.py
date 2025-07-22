import numpy as np 
import pandas as pd 
from stocktrends import Renko
import statsmodels.api as sm 
from alpha_vantage.timeseries import TimeSeries
import copy 
import time 
import datetime
import matplotlib.pyplot as plt 


def MACD(DF, a,b,c):
    df = DF.copy()
    df["MA_Fast"] = df["Adj Close"].ewm(span = a, min_periods = a).mean()
    df["MA_Slow"] = df["Adj Close"].ewm(span = b, min_periods = b).mean()
    df["MACD"] = df["MA_Fast"] - df["MA_Slow"]
    df["Signal"] = df["MACD"].ewm(span=c, min_periods = c).mean()
    df.dropna(inplace = True)
    return (df["MACD"],df["Signal"])

def ATR(DF,n):
    df = DF.copy()
    df["H-L"] = abs(df["High"] - df["Low"])
    df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1, skipna = False)
    df["ATR"] = df["TR"].rolling(n).mean()
    df2 = df.drop(["H-L","H-PC","L-PC"],axis = 1)
    return df2

#calculates the angle (in degrees) of the line of best fit (using linear regression)
# over a rolling window of size n for a given time series (ser
def slope(ser, n):
    slopes = [0.0] * (n - 1)

    for i in range(n, len(ser) + 1):
        y = ser.iloc[i - n:i].values  # ✅ Get a slice of n values
        x = np.arange(n)
        # Handle constant y values
        if np.min(y) == np.max(y):
            slopes.append(0.0)
            continue
        # Normalize
        y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        # Add constant and fit OLS
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slope_value = results.params[-1]  # Slope coefficient
        slopes.append(slope_value)
    # Convert slope to degrees
    return pd.Series(np.rad2deg(np.arctan(slopes)), index=ser.index)

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    df2.brick_size = max(0.5,round(ATR(DF,120)["ATR"][-1],0))
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"].iloc[i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"].iloc[i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"].iloc[i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"].iloc[i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df


def CAGR(DF):
    df = DF.copy()
    df["cum_return"] = (1+df["ret"]).cumprod()
    n = len(df)/(252*78)
    CAGR = (df["cum_return"].tolist()[-1]**(1/n)) - 1
    return CAGR

def volatility(DF):
    df = DF.copy()
    vol = df["ret"].std()*np.sqrt(252*78)
    return vol

def sharpe(DF, rf):
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr

def max_dd(DF):
    df = DF.copy()
    df["cum_return"] = (1+df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


#Downloading the tickers data.

tickers = ["MSFT","AAPL","AMZN","INTC", "CSCO","VZ","IBM","QCOM","LYFT"]

key_path = "C:\\Users\\Lenovo\\OneDrive\\Documents\\APIKEY\\key.txt"
with open(key_path, 'r') as file:
    api_key = file.read().strip()
    
ts = TimeSeries(key=open(key_path,'r').read(), output_format = 'pandas')

ohlc_intraday = {}
api_call_count = 1
start_time = time.time()

for ticker in tickers:
    try:
        data, meta = ts.get_intraday(symbol=ticker, interval = "5min", outputsize = 'full')
        data.columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        data = data.iloc[::-1]
        data = data.between_time('9:35', '16:00')
        ohlc_intraday[ticker] = data
        print(f"Downloaded: {ticker}")
    except Exception as e:
        print(f"failed to fetch {ticker} : {e}")
        
    api_call_count += 1
    if api_call_count == 5:
        api_call_count = 1
        time.sleep(60 - (time.time() - start_time)%60.0)
        
tickers = ohlc_intraday.keys()


#Merging renko df with original ohlc df
ohlc_renko = {}
df = copy.deepcopy(ohlc_intraday)
tickers_signal = {}
tickers_ret = {}
for ticker in tickers:
    print("merging for ",ticker)
    renko = renko_DF(df[ticker])
    renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
    df[ticker] = df[ticker].copy()
    df[ticker]["Date"] = pd.to_datetime(df[ticker].index)
    renko["Date"] = pd.to_datetime(renko["Date"])
    ohlc_renko[ticker] = df[ticker].merge(renko.loc[:,["Date","bar_num"]],how="outer",on="Date")
    ohlc_renko[ticker]["bar_num"].fillna(method='ffill',inplace=True)
    ohlc_renko[ticker]["macd"]= MACD(ohlc_renko[ticker],12,26,9)[0]
    ohlc_renko[ticker]["macd_sig"]= MACD(ohlc_renko[ticker],12,26,9)[1]
    ohlc_renko[ticker]["macd_slope"] = slope(ohlc_renko[ticker]["macd"],5)
    ohlc_renko[ticker]["macd_sig_slope"] = slope(ohlc_renko[ticker]["macd_sig"],5)
    tickers_signal[ticker] = ""
    tickers_ret[ticker] = []


#Identifying signals and calculating daily returns. 
buy_signals = {}
sell_signals = {}
for ticker in tickers:
    print("Calculating returns and storing signals for",ticker)
    tickers_ret[ticker] = []
    tickers_signal[ticker] = ""
    buy_signals[ticker] = []
    sell_signals[ticker] = []
    df = ohlc_renko[ticker]
    
    for i in range(len(df)):
        if tickers_signal[ticker] == "":
            tickers_ret[ticker].append(0)
            if i > 0:
                if df["bar_num"].iloc[i]>=2 and df["macd"].iloc[i]>df["macd_sig"].iloc[i] and df["macd_slope"].iloc[i] > df["macd_sig_slope"].iloc[i]:
                    tickers_signal[ticker] = "Buy"
                    buy_signals[ticker].append((df.index[i], df["Adj Close"].iloc[i]))
                elif df["bar_num"].iloc[i] <=-2 and df["macd"].iloc[i] < df["macd_sig"].iloc[i] and df["macd_slope"].iloc[i] < df["macd_sig_slope"].iloc[i]:
                    tickers_signal[ticker] = "Sell"
                    sell_signals[ticker].append((df.index[i], df["Adj Close"].iloc[i]))
                
        elif tickers_signal[ticker] == "Buy":
            #Calculating returnduring buying/uptrend/bullish
            if i > 0:
                ret = (df["Adj Close"].iloc[i]/df["Adj Close"][i-1]) - 1 
                tickers_ret[ticker].append(ret)
            else:
                tickers_ret[ticker].append(0)
                
            if i > 0:
                if df["bar_num"].iloc[i] <= -2 and df["macd"].iloc[i] < df["macd_sig"].iloc[i] and df["macd_slope"].iloc[i] < df["macd_sig_slope"].iloc[i]:
                    tickers_signal[ticker] = "Sell"
                    sell_signals[ticker].append((df.index[i], df["Adj Close"].iloc[i]))
                elif df["macd"].iloc[i] < df["macd_sig"].iloc[i] and df["macd_slope"].iloc[i] > df["macd_sig_slope"].iloc[i]:
                    tickers_signal[ticker] = ""
                    
        elif tickers_signal[ticker] == "Sell":
            if i > 0:
                ret = (df["Adj Close"][i-1]/df["Adj Close"].iloc[i]) - 1
                tickers_ret[ticker].append(ret)
            else:
                tickers_ret[ticker].append(0)
                
            if i > 0:
                if df["bar_num"].iloc[i] >= 2 and df["macd"].iloc[i] > df["macd_sig"].iloc[i] and df["macd_slope"].iloc[i] > df["macd_sig_slope"].iloc[i]:
                    tickers_signal[ticker] = "Buy"
                    buy_signals[ticker].append((df.index[i], df["Adj Close"].iloc[i]))
                elif df["macd"].iloc[i] > df["macd_sig"].iloc[i] and df["macd_slope"].iloc[i] > df["macd_sig_slope"].iloc[i]:
                    tickers_signal[ticker] = ""
    df["ret"] = np.array(tickers_ret[ticker])
    
#Plotting Buying and sell signals
ohlc_renko = {}
for ticker in tickers:
    df = ohlc_renko[ticker]
#df = ohlc_renko[ticker]


# Extract dates and prices of signals
buy_dates, buy_prices = zip(*buy_signals[ticker]) if buy_signals[ticker] else ([], [])
sell_dates, sell_prices = zip(*sell_signals[ticker]) if sell_signals[ticker] else ([], [])

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["Adj Close"], label="Close Price", color='blue')

plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy Signal', zorder=5)
plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal', zorder=5)

plt.title(f"{ticker} - Price Chart with Buy/Sell Signals")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

    
#Calculating overall KPI's 
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc_renko[ticker]["ret"]
strategy_df["ret"] = strategy_df.mean(axis=1)
CAGR(strategy_df)
sharpe(strategy_df, 0.025)
max_dd(strategy_df)


#Visualizing strategy returns 
(1+strategy_df["ret"]).cumprod().plot()
plt.show()
    
#Calculating Individual stock's KPI's
cagr = {}
sharpe_ratio = {}
max_drawdown = {}
for ticker in tickers:
    print("Calculating KPI's for", ticker)
    cagr[ticker] = CAGR(ohlc_renko[ticker])
    sharpe_ratio[ticker] = sharpe(ohlc_renko[ticker],0.025)
    max_drawdown[ticker] = max_dd(ohlc_renko[ticker])
    
    
KPI_df = pd.DataFrame([cagr, sharpe_ratio, max_drawdown], index = ["Return(CAGR)","Sharpe Ratio", "Max Drawdown"])
KPI_df.T


# Example tickers list
tickers = ["MSFT","AAPL","AMZN","INTC", "CSCO","VZ","IBM","QCOM","LYFT"]

# Assuming these dictionaries exist
# ohlc_renko = {ticker1: df1, ticker2: df2, ...}
# buy_signals = {ticker1: [(date1, price1), ...], ticker2: [...], ...}
# sell_signals = {ticker1: [(date1, price1), ...], ticker2: [...], ...}

for ticker in tickers:
    df = ohlc_renko[ticker]

    # Extract buy/sell dates and prices safely
    buy_data = buy_signals.get(ticker, [])
    sell_data = sell_signals.get(ticker, [])

    buy_dates, buy_prices = zip(*buy_data) if buy_data else ([], [])
    sell_dates, sell_prices = zip(*sell_data) if sell_data else ([], [])

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["Adj Close"], label="Close Price", color='blue')

    if buy_dates:
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    if sell_dates:
        plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal', zorder=5)

    plt.title(f"{ticker} - Price Chart with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

