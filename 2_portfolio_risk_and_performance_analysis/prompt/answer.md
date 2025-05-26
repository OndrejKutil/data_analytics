# **Case Study: Portfolio Risk and Performance Analysis**

**Scenario:**\
You are a junior analyst at an investment firm. Your manager asks you to evaluate the risk-return profile of a portfolio consisting of multiple assets. The goal is to assess diversification, compute key risk metrics, and visualize the portfolio’s performance.

---

## **Tasks:**

### **1. Data Collection**

- Get historical stock data for **5 assets** from different sectors:

  - **AAPL** (Technology)
  - **JPM** (Financials)
  - **XOM** (Energy)
  - **TSLA** (Growth stock)
  - **SPY** (Market benchmark – S&P 500 ETF)

- Retrieve **daily adjusted closing prices** for the last **3 years** using `yfinance`:

  ```python
  import yfinance as yf
  import pandas as pd

  tickers = ["AAPL", "JPM", "XOM", "TSLA", "SPY"]
  data = yf.download(tickers, start="2021-01-01", end="2024-01-01")["Adj Close"]
  data.to_csv("portfolio_data.csv")
  ```

---

### **2. Data Cleaning & Returns Calculation**

- Compute **daily log returns**:

  ```python
  returns = data.pct_change().dropna()
  log_returns = np.log(1 + returns)
  ```

- Handle missing values if any.

---

### **3. Portfolio Performance Metrics**

- Assume an **equal-weighted portfolio** (each stock = 20% weight).

- Compute **expected return** using historical mean:

  $$
  E(R_p) = \sum (w_i \cdot E(R_i))
  $$

- Compute **portfolio variance & standard deviation**:

  $$
  \sigma_p^2 = w^T \Sigma w
  $$

  where **Σ** is the covariance matrix.

- Compute **Sharpe Ratio** (risk-adjusted return):

  $$
  SR = \frac{E(R_p) - R_f}{\sigma_p}
  $$

  Assume **risk-free rate ********\(R_f\)******** = 2%**.

---

### **4. Risk & Diversification Analysis**

- Compute **correlation matrix** between assets.
- Identify assets that contribute most to **portfolio risk**.
- Compare portfolio volatility with **S&P 500 (SPY).**

---

### **5. Visualization**

- **Portfolio cumulative returns** (vs. S&P 500).
- **Rolling volatility** over time.
- **Correlation heatmap** to show diversification.
- **Efficient frontier (optional, more advanced later)**.

---

## **Learning Outcomes**

✅ Hands-on **portfolio analysis & risk management**\
✅ **Statistical analysis** with real-world data\
✅ **Data visualization & storytelling**

Would you like me to generate the data for you, or do you prefer to run the script yourself? 🚀
