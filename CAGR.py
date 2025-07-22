import yfinance as yf

stocks = ["AMZN", "GOOG", "MSFT"]
ohlcv_data = {}

for ticker in stocks:
    temp = yf.download(ticker, period = "7mo", interval = "1d", multi_level_index = False)
    temp["Adj Close"] = temp["Close"]
    temp.dropna(how = "any", inplace = True)
    ohlcv_data[ticker] = temp[["Open","High", "Low", "Close", "Adj Close", "Volume"]]
    
    
def CAGR(DF):
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    df["cum_return"] = (1+df["return"]).cumprod()
    n = len(df)/252 #252 no. of trading days. and if working for intraday divide again 252/8.
    CAGR = (df["cum_return"].iloc[-1])**(1/n) - 1
    return CAGR

for ticker in ohlcv_data:
    print("CAGR for {} = {}".format(ticker, CAGR(ohlcv_data[ticker])))
 
