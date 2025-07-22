import yfinance as yf

stocks = ["AAPL","AMZN", "MSFT", "GOOG"]
ohlcv_data = {}

for ticker in stocks:
    temp = yf.download(ticker, period = "1mo", interval = "5m", multi_level_index = False)
    temp["Adj Close"] = temp["Close"]
    temp.dropna(how="any", inplace = True)
    ohlcv_data[ticker] = temp
   
    
    
def ATR(DF, n=14):
    df = DF.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = df["High"] - df["Adj Close"].shift(1)
    df["L-PC"] = df["Low"] - df["Adj Close"].shift(1)
    df["TR"] = df[["H-L","H-PC", "L-PC"]].max(axis=1, skipna = False)
    df["ATR"] = df["TR"].ewm(span = n, min_periods = n).mean()
    return df["ATR"]
    
for ticker in ohlcv_data:
    ohlcv_data[ticker]["ATR"] = ATR(ohlcv_data[ticker])