# LSTM trading

- Script: main.py
- Task: predict next-day close using LSTM on BTC-USD with engineered features (SMA/EMA/MACD/RSI).
- Pipeline: scale -> create sequences -> train LSTM with callbacks -> evaluate -> plot -> 5‑day forecast.
- Output: metrics in console and plots (PNG).
- Run: `python main.py` (adjust ticker/period if needed).
