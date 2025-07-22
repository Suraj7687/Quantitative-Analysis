import yfinance as yf 
import numpy as np 
import statsmodels.api as sm 
from alpha_vantage.timeseries import TimeSeries
import copy 
import time 
import datetime

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
    
   
def StoscOsc(DF, k, d):

    df = DF.copy()
    df["L14"] = df["Low"].rolling(window=k).min()
    df["H14"] = df["High"].rolling(window=k).max()
    df["%K"] = (df["Adj Close"] - df["L14"]) / (df["H14"] - df["L14"]) * 100
    df["%D"] = df["%K"].rolling(window=d).mean()
    df.dropna(inplace=True)
    return df[["%K", "%D"]]

for ticker in ohlc_intraday:
    ohlc_intraday[ticker][["%K", "%D"]] = StoscOsc(ohlc_intraday[ticker], 8, 3)
    ohlc_intraday[ticker] = ohlc_intraday[ticker].iloc[::-1]
    