import yfinance as yf
import matplotlib.pyplot as plt

stocks = ["AMZN", "GOOG", "MSFT"]
ohlcv_data = {}

for ticker in stocks:
    temp = yf.download(ticker, period= "7mo", interval = "1d", multi_level_index = False)
    temp["Adj Close"] = temp["Close"]
    temp.dropna(how="any", inplace = True)
    ohlcv_data[ticker] = temp
    
def CAGR(DF):
    df = DF.copy()
    df["return"] = DF["Adj Close"].pct_change()
    df["cum_return"] = (1+df["return"]).cumprod()
    n = len(df)/252 #252 no. of trading days. and if working for intraday divide again 252/8.
    CAGR = (df["cum_return"].iloc[-1])**(1/n) - 1
    return CAGR  
    
def max_dd(DF):
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["return"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    return (df["drawdown"] / df["cum_roll_max"]).max()

def calmar(DF):
    df = DF.copy()
    return CAGR(df)/max_dd(df)

for ticker in ohlcv_data:
    print("max drawdown of {} = {}".format(ticker, max_dd(ohlcv_data[ticker])))
    print("calmar ratio of {} = {}".format(ticker, calmar(ohlcv_data[ticker])))
