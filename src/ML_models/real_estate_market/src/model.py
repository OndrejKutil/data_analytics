import matplotlib.pyplot as plt
import pandas as pd
import os
import pprint

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(FILE_DIR, '..', 'listings.csv')

df = pd.read_csv(CSV_PATH)

# find any rows with null values
null_rows = df[df.isnull().any(axis=1)]
# print(f"Rows with null values:\n{null_rows}")

df = df.dropna(subset=['price_czk', 'area_sqm'])

# scattwer plot of area vs price
plt.figure(figsize=(10, 6))
plt.scatter(df['area_sqm'], df['price_czk'], alpha=0.5)
plt.title('Area vs Price')
plt.xlabel('Area (sqm)')
plt.ylabel('Price (CZK)')
plt.grid()
plt.savefig(os.path.join(FILE_DIR, '..', 'area_vs_price.png'))

# filter out extreme values
df_original = df.copy()
df = df[df['area_sqm'] < 500]

# calculate mean price per square meter, ignoring rows with missing data
df['price_per_sqm'] = df.apply(lambda row: row['price_czk'] / row['area_sqm'] if row['area_sqm'] > 0 else None, axis=1)
mean_price_per_sqm = df['price_per_sqm'].mean()

# calculate mean price per square meter for different prague locations
mean_price_by_location = df.groupby('prague_location')['price_per_sqm'].mean().sort_values(ascending=False)
pprint.pprint(mean_price_by_location)


# linear regression of area vs price
from sklearn.linear_model import LinearRegression

x = df[['area_sqm']]
y = df['price_czk']
model = LinearRegression()
model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_

print(f"Linear Regression Model: price_czk = {slope:.2f} * area_sqm + {intercept:.2f}")

# plot regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['area_sqm'], df['price_czk'], alpha=0.5)
plt.plot(df['area_sqm'], model.predict(x), color='red', linewidth=2)
plt.title('Area vs Price with Regression Line')
plt.xlabel('Area (sqm)')
plt.ylabel('Price (CZK)')
plt.grid()
plt.savefig(os.path.join(FILE_DIR, '..', 'area_vs_price_regression.png'))
