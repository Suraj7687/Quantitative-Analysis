import yfinance as yf 
import numpy as np 
import copy 
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt 

def CAGR(DF):
    df = DF.copy()
    df["cum_return"] = (1+df["mon_return"]).cumprod()
    n = len(df)/12
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    df = DF.copy()
    vol = df["mon_return"].std()*np.sqrt(12)
    return vol

def sharpe(DF, rf):
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr

def max_dd(DF):
    df = DF.copy()
    df["cum_return"] = (1+df["mon_return"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd



stocks = ['AMZN', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 
          'CVX', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'KO', 'JPM',
              'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'SHW', 
          'TRV', 'UNH', 'CRM', 'NVDA', 'VZ', 'V', 'WMT', 'DIS']


ohlc_mon = {}
start = dt.datetime.today() - dt.timedelta(3650)
end = dt.datetime.today()

for ticker in stocks:
    temp = yf.download(ticker, start, end, interval = '1mo', multi_level_index = False)
    temp["Adj Close"] = temp["Close"]
    temp.dropna(inplace = True, how = 'all')
    ohlc_mon[ticker] = temp
    
stocks = ohlc_mon.keys()#redefine ticker variable after removing any ticker with corrupted data.

#Calculating monthly return for each stock and consolidating return info by stock in a seperate dataframe 
ohlc_dict = copy.deepcopy(ohlc_mon)
return_df = pd.DataFrame()
for ticker in stocks:
    ohlc_dict[ticker]["mon_return"] = ohlc_dict[ticker]["Adj Close"].pct_change()
    return_df[ticker] = ohlc_dict[ticker]["mon_return"]
return_df.dropna(inplace = True)

m = 6 #keeping good performing stocks in my portfolio
x = 3 #stocks with bad performance which needs to be removed. 
def pflio(DF, m, x):
    df = DF.copy()
    portfolio = []
    monthly_ret = [0] #initial return to be kept 0. 
    for i in range(1,len(df)):
        if len(portfolio) > 0:
            monthly_ret.append(df[portfolio].iloc[i,:].mean())
            bad_stocks =  df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
            portfolio = [t for t in portfolio if t not in bad_stocks]
        fill = m - len(portfolio)
        new_picks = df[[t for t in stocks if t not in portfolio]].iloc[i,:].sort_values(ascending = False)[:fill].index.values.tolist()
        portfolio = portfolio + new_picks
        print(portfolio)        
    monthly_ret_df = pd.DataFrame(np.array(monthly_ret),columns = ['mon_return'])
    return monthly_ret_df

CAGR(pflio(return_df, 6, 3))
sharpe(pflio(return_df, 6, 3), 0.025)
max_dd(pflio(return_df, 6, 3))

DJI = yf.download("^DJI",dt.date.today()-dt.timedelta(3650),dt.date.today(),interval='1mo', multi_level_index = False)
DJI['Adj Close'] = DJI['Close']
DJI["mon_return"] = DJI["Adj Close"].pct_change().fillna(0)
CAGR(DJI)
sharpe(DJI, 0.025)


fig, ax = plt.subplots()
plt.plot((1+pflio(return_df, 6, 3)).cumprod())
plt.plot((1+DJI["mon_return"][2:].reset_index(drop = True)).cumprod())
plt.title("Index Return vs Strategy Return")
plt.ylabel("cumulative return")
plt.xlabel("months")
plt.legend(["Strategy Return", "Index Return"])
plt.show()
            


    
    