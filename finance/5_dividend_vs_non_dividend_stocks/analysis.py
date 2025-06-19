import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


non_dividend_stock_tickers = ['GOOGL', 'MRNA', 'TSLA', 'META', 'UBER']

dividend_stock_tickers = ['CSCO', 'JNJ', 'MCD', 'VZ', 'MMM']


start = '2020-01-01'
end = '2025-01-01'



non_div_data = yf.download(non_dividend_stock_tickers, start=start, end=end)
if non_div_data is not None and 'Close' in non_div_data:
    non_div_df = non_div_data['Close']


div_data = yf.download(dividend_stock_tickers, start=start, end=end)
if div_data is not None and 'Close' in div_data:
    div_df = div_data['Close']


div_returns_df = div_df.pct_change().dropna()
non_div_returns_df = non_div_df.pct_change().dropna()


dividends = pd.DataFrame(index=pd.date_range(start=start, end=end, freq='D'))
dividends.index = pd.to_datetime(dividends.index).date



for ticker in dividend_stock_tickers:
    dividend_data = yf.Ticker(ticker).get_dividends()
    dividend_data.index = pd.to_datetime(dividend_data.index).date

    

    dividends[ticker] = dividend_data.reindex(dividends.index)


dividends.dropna(how='all', inplace=True)




div_total_return_df = div_df.copy()
non_div_total_return_df = non_div_df.copy()


div_cumulative_price_return = (1 + div_returns_df).cumprod()
non_div_cumulative_price_return = (1 + non_div_returns_df).cumprod()


print(dividends)
