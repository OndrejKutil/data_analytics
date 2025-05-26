import yfinance as yf

tickers = ["AAPL", "JPM", "XOM", "TSLA", "SPY"]
data = yf.download(tickers, start="2021-01-01", end="2024-01-01")["Close"]
data.to_csv("./portfolio_risk_and_performance_analysis/data/stock_data.csv")