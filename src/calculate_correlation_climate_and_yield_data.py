import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.preprocessing import QuantileTransformer

yield_ds = xr.open_dataset("./../output/processed_yield.nc")

aggregated_climate_data_ds = xr.open_dataset("./agg-test.nc")

merged_data = xr.merge([yield_ds, aggregated_climate_data_ds]).sel(
    year=slice(1986, 2021)
)
print(merged_data.data_vars)


correlation_results = []

# List of yield transformations to compare
yield_transformations = [
    "ww_yield",
    "ww_yield_average_detrended",
]

for nuts_id in merged_data["nuts_id"].values:
    if (merged_data["ww_yield"].sel(nuts_id=nuts_id).notnull().sum().values >= 36) and (
        merged_data["pr_monthly"].sel(nuts_id=nuts_id).notnull().all().values
    ):
        for climate_var in [
            "tas_monthly",
            "rsds_monthly",
            "pr_monthly",
            "days_in_99_percentile_sfcWind_monthly",
            "days_in_99_percentile_pr_monthly",
            "frost_days_monthly",
            "sfcWind_monthly",
        ]:

            for yield_var in yield_transformations:
                # Calculate correlation
                yearly_mean = (
                    merged_data[climate_var].sel(nuts_id=nuts_id).mean(dim=["month"])
                )
                yield_data = merged_data[yield_var].sel(nuts_id=nuts_id)

                # Quantile transform yield data
                qt = QuantileTransformer()  # Create a transformer object
                yield_data_transformed = qt.fit_transform(
                    yield_data.values.reshape(-1, 1)
                )  # Reshape for the transformer and apply

                yield_data_transformed = np.log(yield_data.values)
                correlation = stats.pearsonr(yearly_mean.values, yield_data_transformed)

                print(correlation)

                correlation_results.append(
                    [nuts_id, climate_var, yield_var, correlation.statistic]
                )

# Create DataFrame from results
correlation_df = pd.DataFrame(
    correlation_results,
    columns=["nuts_id", "climate_var", "yield_transformation", "correlation"],
)

correlation_df.to_csv("../output/correllations_test.csv")

# Add a column for correlation sign
correlation_df["correlation_sign"] = correlation_df["correlation"].apply(
    lambda x: "Positive" if x >= 0 else "Negative"
)

average_correlations = correlation_df.groupby("yield_transformation")[
    "correlation"
].mean()
print(average_correlations)

avg_corr_by_var = correlation_df.groupby(["yield_transformation", "climate_var"])[
    "correlation"
].mean()
print(avg_corr_by_var)

top_correlations = correlation_df.groupby(["yield_transformation"])[
    "correlation"
].nlargest(
    10
)  # Top 10
print(top_correlations)


# Plotting with Separation of Positive and Negative Correlations
plt.figure(figsize=(12, 6))
sns.boxplot(
    x="yield_transformation",
    y="correlation",
    hue="correlation_sign",
    data=correlation_df,
    palette={"Positive": "blue", "Negative": "red"},
)
plt.xticks(rotation=45, ha="right")
plt.title("Correlation Between Monthly Climate and Yield Transformations")
plt.show()

# Specify the nuts_id and climate variable you want to plot
target_nuts_id = "DE80J"  # Replace with the actual nuts_id
target_climate_var = "tas_monthly"  # Replace with the actual climate variable

# Filter the DataFrame for the specific region and climate variable
filtered_correlation_df = correlation_df[
    (correlation_df["nuts_id"] == target_nuts_id)
    & (correlation_df["climate_var"] == target_climate_var)
]

# Plotting
plt.figure(figsize=(12, 6))
sns.boxplot(
    x="yield_transformation",
    y="correlation",
    hue="correlation_sign",
    data=filtered_correlation_df,
    palette={"Positive": "blue", "Negative": "red"},
)
plt.xticks(rotation=45, ha="right")
plt.title(
    f"Correlation Between {target_climate_var} and Yield Transformations (Region: {target_nuts_id})"
)
plt.show()

# Create a separate plot for each yield transformation
for yield_var in correlation_df["yield_transformation"].unique():
    plt.figure(figsize=(10, 6))

    # Filter data for the current yield transformation
    subset_df = correlation_df[correlation_df["yield_transformation"] == yield_var]

    # Create the boxplot
    sns.boxplot(
        x="climate_var",
        y="correlation",
        hue="correlation_sign",
        data=subset_df,
        palette={"Positive": "blue", "Negative": "red"},
    )

    plt.xticks(rotation=45, ha="right")
    plt.title(f"Correlations for {yield_var} (Region: {target_nuts_id})")
    plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
    plt.show()
