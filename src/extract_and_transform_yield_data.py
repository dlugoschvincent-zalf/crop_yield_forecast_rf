import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xarray as xr
import random


# Load the data
df = pd.read_csv("../yield_data/Final_data.csv")

# Filter and process the data for winter wheat yield
ww_yield = (
    df.query("var == 'ww' and measure == 'yield'")
    .rename(columns={"value": "ww_yield"})
    .set_index(["nuts_id", "year"])[["ww_yield"]]
)

# ww_yield.dropna(inplace=True)

ww_yield_ds = xr.Dataset.from_dataframe(ww_yield)

# --------------------------------------------------------------------------------
# 1. AVERAGE DETRENDING
#    Goal: Remove the global average trend from each region's yield.
# --------------------------------------------------------------------------------

# Calculate detrended yield by subtracting the average yield of that year across all regions
ww_yield_ds["ww_yield_average_detrended"] = ww_yield_ds["ww_yield"] - ww_yield_ds[
    "ww_yield"
].mean(dim="nuts_id")


# --------------------------------------------------------------------------------
# 2. YIELD ANOMALY CALCULATION
#    Goal: Calculate how much the yield in a specific year deviates from the recent trend.
# --------------------------------------------------------------------------------

ww_yield_ds["ww_yield_anomaly_5yr"] = ww_yield_ds["ww_yield"] - ww_yield_ds[
    "ww_yield"
].rolling(year=5, center=False).mean().shift(year=2).mean(dim="nuts_id")


# --------------------------------------------------------------------------------
# 3. MIN MAX SCALING
#    Goal: Remove regional differenced
# --------------------------------------------------------------------------------
# Calculate min and max values for each nutsid
min_vals = ww_yield_ds["ww_yield_average_detrended"].groupby("nuts_id").min(dim="year")
max_vals = ww_yield_ds["ww_yield_average_detrended"].groupby("nuts_id").max(dim="year")
ww_yield_ds["ww_yield_average_detrended_min_max"] = (
    ww_yield_ds["ww_yield_average_detrended"] - min_vals
) / (max_vals - min_vals)


# Calculate min and max values for each nutsid
min_vals = ww_yield_ds["ww_yield_anomaly_5yr"].groupby("nuts_id").min(dim="year")
max_vals = ww_yield_ds["ww_yield_anomaly_5yr"].groupby("nuts_id").max(dim="year")
ww_yield_ds["ww_yield_anomaly_5yr_min_max"] = (
    ww_yield_ds["ww_yield_anomaly_5yr"] - min_vals
) / (max_vals - min_vals)


print(ww_yield_ds["ww_yield_average_detrended_min_max"].sel(nuts_id="DE80J").values)
print(ww_yield_ds["ww_yield_average_detrended"].sel(nuts_id="DE80J").values)
print(ww_yield_ds["ww_yield_anomaly_5yr"].sel(nuts_id="DE80J").values)

ww_yield = ww_yield.dropna()

ww_yield_ds.to_netcdf("../output/targets/processed_yield_simpler.nc")

# Create a new DataFrame to store the results
processed_data = []

all_regions = ww_yield.index.get_level_values("nuts_id").unique()

# Calculate average yearly yield upfront. This is used for one of the detrending methods.
average_yearly_yield = ww_yield.groupby("year")["ww_yield"].mean()

average_yearly_yield_rolling_5_years = (
    ww_yield.groupby("year")["ww_yield"].rolling(window=5, center=False).mean().shift(2)
)

# Loop through each region (NUTS ID)
for region in all_regions:
    ww_yield_region = ww_yield.loc[region]
    X = ww_yield_region.index.values.astype(int)
    X = np.reshape(X, (len(X), 1))
    y = ww_yield_region.values

    # --------------------------------------------------------------------------------
    # 1. STANDARDIZATION (SCALING)
    #    Goal: Transform data to have zero mean and unit variance.
    #    This helps in comparing and combining data from different distributions.
    # --------------------------------------------------------------------------------

    scaler = StandardScaler()
    # Fit to the data and then transform it. We're scaling the original yield data here.
    ww_yield_standardized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # --------------------------------------------------------------------------------
    # 2. DETRENDING USING POLYNOMIAL REGRESSION (Applied before and after scaling)
    #    Goal: Remove the long-term trend from the time series.
    #    This helps isolate the year-to-year variability from the overall trend.
    # --------------------------------------------------------------------------------

    # A. DETRENDING AFTER STANDARDIZATION
    pf = PolynomialFeatures(degree=3)  # Create polynomial features (up to degree 3)
    Xp = pf.fit_transform(X)  # Transform the year values into polynomial features
    md2 = LinearRegression()
    md2.fit(
        Xp, ww_yield_standardized
    )  # Fit linear regression to the standardized yield data
    trendp_scaled = md2.predict(Xp)  # Predict the trend based on the fitted model
    # Calculate the detrended data by subtracting the predicted trend from the standardized data
    ww_yield_standardized_detrended = ww_yield_standardized - trendp_scaled

    # B. DETRENDING BEFORE STANDARDIZATION
    pf = PolynomialFeatures(degree=3)
    Xp = pf.fit_transform(X)
    md2 = LinearRegression()
    md2.fit(Xp, y)  # Fit linear regression to the original yield data
    trendp_region = md2.predict(Xp)
    # Calculate the detrended data by subtracting the predicted trend from the original data
    ww_yield_detrended = y - trendp_region

    # Standardize the already detrended yield data
    ww_yield_detrended_standardized = scaler.fit_transform(
        ww_yield_detrended.reshape(-1, 1)
    ).flatten()

    # --------------------------------------------------------------------------------
    # 1. AVERAGE DETRENDING
    #    Goal: Remove the global average trend from each region's yield.
    # --------------------------------------------------------------------------------

    # Calculate detrended yield by subtracting the average yield of that year across all regions
    ww_yield_average_detrended = ww_yield_region.copy()
    for year in ww_yield_average_detrended.index:
        ww_yield_average_detrended.loc[year, "ww_yield"] -= average_yearly_yield.loc[
            year
        ]

    # --------------------------------------------------------------------------------
    # 4. YIELD ANOMALY CALCULATION
    #    Goal: Calculate how much the yield in a specific year deviates from the recent trend.
    # --------------------------------------------------------------------------------

    # Calculate 5-year moving average anomaly. This shows how much a particular year's yield is
    # above or below the average yield of the surrounding 5 years.
    ww_yield_anomaly_5yr = ww_yield_region["ww_yield"] - ww_yield_region[
        "ww_yield"
    ].rolling(window=5, center=False).mean().shift(2)

    # Calculate 5-year moving average anomaly based on average detrended data. This combines
    # the average detrending with the anomaly calculation to show deviations from the detrended average.
    ww_yield_anomaly_5yr_average_detrended = (
        ww_yield_average_detrended
        - ww_yield_average_detrended.rolling(window=5, center=False).mean().shift(2)
    )

    # --------------------------------------------------------------------------------
    # 5. STORE THE PROCESSED DATA
    # --------------------------------------------------------------------------------

    for i, year in enumerate(ww_yield_region.index):
        processed_data.append(
            [
                region,
                year,
                ww_yield_region.iloc[i].values[0],
                ww_yield_detrended[i],
                ww_yield_standardized[i],
                ww_yield_detrended_standardized[i],
                ww_yield_standardized_detrended[i],
                ww_yield_average_detrended.iloc[i],
                ww_yield_anomaly_5yr.iloc[i],  # Add yield anomaly
                ww_yield_anomaly_5yr_average_detrended.iloc[i],
            ]
        )

# Create the new DataFrame
processed_df = pd.DataFrame(
    processed_data,
    columns=[
        "nuts_id",
        "year",
        "ww_yield",
        "ww_yield_detrended_poly",
        "ww_yield_standardized",
        "ww_yield_detrended_then_standardized",
        "ww_yield_standardized_then_detrended",
        "ww_yield_average_detrended",
        "ww_yield_anomaly_5yr",
        "ww_yield_anomaly_5yr_average_detrended",
    ],
)
processed_df.set_index(["nuts_id", "year"], inplace=True)

yield_ds = xr.Dataset.from_dataframe(processed_df)

# Add descriptions to the data variables in the netCDF file
yield_ds["ww_yield"].attrs["long_name"] = "Winter Wheat Yield"
yield_ds["ww_yield"].attrs["units"] = "Tons per Hectare"
yield_ds["ww_yield_detrended_poly"].attrs[
    "long_name"
] = "Winter Wheat Yield (Detrended using Polynomial Regression)"
yield_ds["ww_yield_detrended_poly"].attrs[
    "description"
] = "Original yield with the long-term trend removed using polynomial regression."
yield_ds["ww_yield_standardized"].attrs[
    "long_name"
] = "Winter Wheat Yield (Standardized)"
yield_ds["ww_yield_standardized"].attrs[
    "description"
] = "Yield data standardized to have zero mean and unit variance."
yield_ds["ww_yield_detrended_then_standardized"].attrs[
    "long_name"
] = "Winter Wheat Yield (Detrended, then Standardized)"
yield_ds["ww_yield_detrended_then_standardized"].attrs[
    "description"
] = "Yield data detrended using polynomial regression, then standardized to have zero mean and unit variance."
yield_ds["ww_yield_standardized_then_detrended"].attrs[
    "long_name"
] = "Winter Wheat Yield (Standardized, then Detrended)"
yield_ds["ww_yield_standardized_then_detrended"].attrs[
    "description"
] = "Yield data standardized to have zero mean and unit variance, then detrended using polynomial regression"
yield_ds["ww_yield_average_detrended"].attrs[
    "long_name"
] = "Winter Wheat Yield (Average Detrended)"
yield_ds["ww_yield_average_detrended"].attrs[
    "description"
] = "Yield data detrended by subtracting the average yield of that year across all regions."
yield_ds["ww_yield_anomaly_5yr"].attrs[
    "long_name"
] = "Winter Wheat Yield Anomaly (5-year)"
yield_ds["ww_yield_anomaly_5yr"].attrs[
    "description"
] = "Difference between the yield and the 5-year rolling mean, highlighting deviations from the recent trend."
yield_ds["ww_yield_anomaly_5yr_average_detrended"].attrs[
    "long_name"
] = "Winter Wheat Yield Anomaly (5-year) on Average Detrended Data"
yield_ds["ww_yield_anomaly_5yr_average_detrended"].attrs[
    "description"
] = "5-year anomaly calculated on the average detrended yield data."

yield_ds.to_netcdf("../output/targets/processed_yield.nc")

print(yield_ds.data_vars)
print(yield_ds["ww_yield_average_detrended"].sel(nuts_id="DE80J", year=2020).values)


# Example plotting for a single region with new transformations
example_region = "DE80J"

plt.figure(figsize=(12, 8))
plt.plot(
    yield_ds["year"].values,
    yield_ds["ww_yield"].sel(nuts_id=example_region).values,
    label="Original Yield",
)
plt.plot(
    yield_ds["year"].values,
    yield_ds["ww_yield_detrended_poly"].sel(nuts_id=example_region).values,
    label="Detrended Yield (Poly)",
)
plt.plot(
    yield_ds["year"].values,
    yield_ds["ww_yield_standardized"].sel(nuts_id=example_region).values,
    label="Standardized Yield",
)
plt.plot(
    yield_ds["year"].values,
    yield_ds["ww_yield_detrended_then_standardized"].sel(nuts_id=example_region).values,
    label="Detrended then Standardized Yield",
)
plt.plot(
    yield_ds["year"].values,
    yield_ds["ww_yield_standardized_then_detrended"].sel(nuts_id=example_region).values,
    label="Standardized then Detrended Yield",
)
plt.plot(
    yield_ds["year"].values,
    yield_ds["ww_yield_average_detrended"].sel(nuts_id=example_region).values,
    label="Average Detrended Yield",
)
plt.plot(
    yield_ds["year"].values,
    yield_ds["ww_yield_anomaly_5yr"].sel(nuts_id=example_region).values,
    label="Yield Anomaly (5-year)",
)

plt.plot(
    yield_ds["year"].values,
    yield_ds["ww_yield_anomaly_5yr_average_detrended"]
    .sel(nuts_id=example_region)
    .values,
    label="Yield Anomaly (5-year) on Average Detrended",
)

plt.legend()
plt.title(f"Yield and Processed Data for Region {example_region}")
plt.show()

num_random_regions = 5
random_regions = random.sample(list(all_regions), num_random_regions)

# Create the plot
plt.figure(figsize=(15, 6))
for i, region in enumerate(random_regions):
    plt.subplot(1, num_random_regions, i + 1)
    plt.plot(
        yield_ds["year"].values,
        yield_ds["ww_yield"].sel(nuts_id=region).values,
        label="Original Yield",
    )
    plt.plot(
        yield_ds["year"].values,
        yield_ds["ww_yield_detrended_then_standardized"].sel(nuts_id=region).values + 6,
        label="Detrended and then scaled yield",
    )
    plt.plot(
        yield_ds["year"].values,
        yield_ds["ww_yield_anomaly_5yr_average_detrended"].sel(nuts_id=region).values
        + 6,
        label="Yield anomaly based on average detrending",
    )
    plt.title(f"Region {region}")
    plt.legend(fontsize="small")

    plt.xlim(1979, 2021)  # Set x-axis limits
    plt.ylim(3, 10)  # Set y-axis limits

plt.tight_layout()
plt.show()
