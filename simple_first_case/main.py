import pandas as pd

df = pd.read_csv('./simple_first_case/sample_stock_data.csv')

# print(df.iloc[0]['AAPL'])

columns = list(df.columns)
columns.remove('Date')

monthly_returns = {}


for stock in columns:
    returns = []
    for i in range(1, 12, 1):
        returns.append(float((df.iloc[i][stock] - df.iloc[i-1][stock]) / df.iloc[i-1][stock]))
    monthly_returns[stock] = returns

# print(monthly_returns)

monthly_prices = {}

for stock in columns:
    prices = []
    for i in range(1, 12, 1):
        prices.append(float(df.iloc[i][stock]))
    monthly_prices[stock] = prices

# print(monthly_prices)


import matplotlib.pyplot as plt

for stock in columns:
    plt.plot(monthly_prices[stock], label=stock)

plt.xlabel('Month')
plt.ylabel('Price')
plt.title('Monthly Prices')
plt.legend()
plt.show()


for stock in columns:
    plt.plot(monthly_returns[stock], label=stock)

plt.xlabel('Month')
plt.ylabel('Return')
plt.title('Monthly Returns')
plt.legend()
plt.show()
