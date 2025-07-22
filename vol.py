import yfinance as yf
import numpy as np
import copy

stocks = ["AMZN", "GOOG", "MSFT"]
ohlcv_data = {}

for ticker in stocks:
    temp = yf.download(ticker, period = "7mo", interval = "1d", multi_level_index = False)
    temp["Adj Close"] = temp["Close"]
    temp.dropna(how="any", inplace = True)
    ohlcv_data[ticker] = temp[["Open","High","Low","Close","Adj Close", "Close"]]
    
    
def volatility(DF):
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    vol = df["return"].std()*np.sqrt(252)
    return vol

ohlc_dict = copy.deepcopy(ohlcv_data)


for ticker in ohlcv_data:
    f = ohlc_dict[ticker]
    f["return"] = f["Adj Close"].pct_change()
    ohlc_dict[ticker] = f
    
for ticker in ohlcv_data:
    print("Volatility of {} = {}".format(ticker, volatility(ohlcv_data[ticker])))
    