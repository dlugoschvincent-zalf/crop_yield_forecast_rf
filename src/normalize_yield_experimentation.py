import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv("../yield_data/Final_data.csv")

# Filter and process the data
ww_yield = (
    df.query("var == 'ww' and measure == 'yield'")
    .rename(columns={"value": "ww_yield"})
    .set_index(["nuts_id", "year"])[["ww_yield"]]
)

# Select the region of interest, for example, 'region_1'
region = "DE80K"
ww_yield_region = ww_yield.loc[region]

X = ww_yield_region.index
X = np.reshape(X, (len(X), 1))
y = ww_yield_region.values


pf = PolynomialFeatures(degree=10)

Xp = pf.fit_transform(X)

md2 = LinearRegression()
md2.fit(Xp, y)
trendp = md2.predict(Xp)
plt.plot(X, y)
plt.plot(X, trendp)
plt.legend(["data", "polynomial trend"])
plt.show()

detrpoly = [y[i] - trendp[i] for i in range(0, len(y))]
plt.plot(X, detrpoly)
plt.title("polynomially detrended data")
plt.show()

r2 = r2_score(y, trendp)
rmse = np.sqrt(mean_squared_error(y, trendp))
print("r2:", r2)
print("rmse", rmse)


scaler = MinMaxScaler((5, 10))
detrpoly_scaled = scaler.fit_transform(detrpoly)

plt.plot(X, detrpoly_scaled)
plt.title("polynomially detrended data scaled")
plt.show()
