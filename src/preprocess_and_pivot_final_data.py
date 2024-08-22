import xarray as xr
import pandas as pd


def pivot_dataframe(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Pivots the given DataFrame for the specified variables,
    handling both monthly and weekly data in a single output.

    Args:
        df (pd.DataFrame): The DataFrame to pivot.
        variables (list): A list of variables to pivot.
                          Variable names should follow the pattern
                          'variable_timescale' (e.g., 'gdd_monthly').

    Returns:
        pd.DataFrame: The pivoted DataFrame containing both monthly
                       and weekly pivoted data.
    """

    dfs = []
    for var in variables:
        # Extract time scale from variable name
        time_scale = var.split("_")[-1]  # Get 'monthly' or 'weekly'

        if time_scale not in ["quarterly", "monthly", "weekly"]:
            raise ValueError(
                f"Invalid time_scale in variable name: {var}. "
                "Use 'monthly' or 'weekly'."
            )

        if time_scale == "quarterly":
            pivot_columns = [f"{var}_quarter_{i}" for i in range(1, 4)]  # Qurter 1-3
            pivot_on = "quarter"
        elif time_scale == "monthly":
            pivot_columns = [f"{var}_month_{i}" for i in range(1, 10)]  # Months 1-9
            pivot_on = "month"
        else:  # time_scale =="weekly"
            pivot_columns = [f"{var}_week_{i}" for i in range(1, 40)]  # Weeks 1-39
            pivot_on = "week"

        df_temp = df.pivot_table(
            index=[
                "nuts_id",
                "year",
                "mean_regional_ww_yield",
                "ww_yield_average_detrended",
            ],
            columns=pivot_on,
            values=var,
            aggfunc="first",
        ).reset_index()

        df_temp.columns = [
            "nuts_id",
            "year",
            "mean_regional_ww_yield",
            "ww_yield_average_detrended",
        ] + pivot_columns

        dfs.append(df_temp)

    # Merge all pivoted data into a single DataFrame
    df_final = dfs[0]
    for i in range(1, len(dfs)):
        df_final = pd.merge(
            df_final,
            dfs[i],
            on=[
                "nuts_id",
                "year",
                "mean_regional_ww_yield",
                "ww_yield_average_detrended",
            ],
        )

    return df_final


# Load and preprocess data
ds = xr.open_dataset(
    "../output/features/final_data_features_aggregated_on_nuts_inc_spi.nc"
)
yield_ds = xr.open_dataset("../output/targets/processed_yield_simpler.nc")

ds = ds.merge(yield_ds[["ww_yield_average_detrended", "ww_yield"]])

print(ds.data_vars)
df_relevant_data = (
    ds.sel(
        year=slice(1979, 2021),
        month=slice(1, 9),
        week=slice(1, 39),
        quarter=slice(1, 3),
    )
    .to_dataframe()
    .dropna()
    .reset_index()
)

# Define years to exclude for testing
years_to_exclude = range(2015, 2022)

# Split into training and test sets based on years
df_train = pd.DataFrame(
    df_relevant_data[~df_relevant_data["year"].isin(years_to_exclude)].copy()
)

df_test = pd.DataFrame(
    df_relevant_data[df_relevant_data["year"].isin(years_to_exclude)].copy()
)

# Calculate training regional dataset means for yield data (to be used for both train and test)
regional_yield_means = pd.DataFrame(
    df_train.groupby("nuts_id").agg(
        mean_regional_ww_yield=("ww_yield", "mean"),
    )
)

# Apply regional means to training data
df_train = df_train.merge(regional_yield_means, on="nuts_id", how="left")


# Apply regional means to test data
df_test = df_test.merge(regional_yield_means, on="nuts_id", how="left")

# --- Pivoting the Data ---
variables_to_pivot = [
    "spi_quarterly",
    "days_in_99_percentile_sfcWind_monthly",
    "days_in_99_percentile_pr_monthly",
    "frost_days_monthly",
    "spi_monthly",
    # "frost_days_weekly",
    "days_avg_temp_above_28_weekly",
    "pr_monthly",
    "rsds_monthly",
    "tasmax_monthly",
    "hurs_monthly",
    "tas_monthly",
    "sfcWind_monthly",
    "tasmin_monthly",
    # "pr_weekly",
    # "rsds_weekly",
    # "tasmax_weekly",
    # "hurs_weekly",
    # "tas_weekly",
    # "sfcWind_weekly",
    # "tasmin_weekly",
]

df_train = pivot_dataframe(df_train, variables_to_pivot)
df_test = pivot_dataframe(df_test, variables_to_pivot)

df_test_split_list = [
    df_test[df_test["year"] == year_to_exclude] for year_to_exclude in years_to_exclude
]

df_train = df_train.set_index(["year", "nuts_id"]).sort_index()
df_test = df_test.set_index(["year", "nuts_id"]).sort_index()

df_train.to_csv("../output/final_runs/ready_to_extract_data/training_data_pivoted.csv")
df_test.to_csv("../output/final_runs/ready_to_extract_data/test_data_pivoted.csv")
