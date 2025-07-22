import yfinance as yf 
import pandas as pd

stocks = ["AMZN", "GOOG", "MSFT"]
ohlcv_data = {}

#Looping over tickers and storing OHLCV dataframe in dictionary.
for ticker in stocks:
    temp = yf.download(ticker, period = "1mo", interval = "5m", multi_level_index = False)
    temp["Adj Close"] = temp["Close"]
    temp.dropna(how = 'any', inplace = True)
    ohlcv_data[ticker] = temp
    
def Boll_Band(DF, n=14):
    df = DF.copy()
    df["MB"] = df["Adj Close"].rolling(n).mean()
    df["UB"] = df["MB"] + 2*df["Adj Close"].rolling(n).std()
    df["LB"] = df["MB"] - 2*df["Adj Close"].rolling(n).std()
    df["BB_Width"] = df["UB"] - df["LB"]
    return df[["MB", "UB", "LB", "BB_Width"]]

for ticker in ohlcv_data:
    ohlcv_data[ticker][["MB", "UB", "LB", "BB_Width"]] = Boll_Band(ohlcv_data[ticker])