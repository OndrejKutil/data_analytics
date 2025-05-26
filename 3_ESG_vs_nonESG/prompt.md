📊 Sample Task: ESG vs Non-ESG Investment Performance Analysis
🧩 Objective
Compare the historical returns, volatility, and risk-adjusted performance (e.g., Sharpe ratio) of ESG-focused ETFs vs. non-ESG ETFs over the past 5–10 years.

🔧 1. Dataset
You can use Yahoo Finance to fetch ETF data via yfinance.

Example ETFs:

ESG ETF: ESGU (iShares ESG Aware MSCI USA ETF)

Non-ESG ETF: SPY (S&P 500 ETF)

Fetch adjusted closing prices weekly or monthly from 2015 to today.

📌 2. Key Tasks
A. Data Collection
Use yfinance to download historical data.

Calculate monthly returns from adjusted closing prices.

B. Exploratory Data Analysis
Plot return time series.

Compute descriptive statistics (mean, median, std dev).

Check for missing data and handle it.

C. Visualization
Line plot of cumulative returns.

Histogram or KDE plot of monthly returns.

Boxplot comparison of return distributions.

D. Statistical Testing
T-test or Mann-Whitney U test on returns: Is ESG significantly different from non-ESG?

Correlation and beta vs. market (optional).

E. Risk & Performance Metrics
Annualized return, volatility.

Sharpe Ratio (assume risk-free rate of 0–1%).

Max drawdown (optional).

💬 Sample Guiding Questions
Do ESG ETFs perform better or worse than traditional ETFs over the long run?

How does the volatility compare between the two?

Is there a statistically significant difference in return distributions?

🛠️ Stretch Task (Optional)
Add more ETFs:

ESG: SUSA, ESGV

Non-ESG: DIA, IVV

Try sector analysis: Is ESG under-/over-performing in specific sectors (e.g., tech, energy)?

🧠 Skills Practiced
Time series analysis

Data visualization (matplotlib, seaborn)

Data cleaning and transformation (pandas)

Statistical inference

Performance evaluation in finance