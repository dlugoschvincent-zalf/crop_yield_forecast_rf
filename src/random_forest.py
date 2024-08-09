import xarray as xr
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    TimeSeriesSplit,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
import numpy as np
import matplotlib.pyplot as plt

# Supress scientific notation
np.set_printoptions(suppress=True)


# Function to evaluate and print model results
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{model_name}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    return y_pred


def plot_results(y_test, y_pred, filename_suffix=""):
    # 1. Scatter Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted Values {filename_suffix}")

    # Draw a 45-degree line (perfect prediction line)
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        linestyle="--",
        color="red",
    )

    plt.savefig(f"../plots/scatter_{filename_suffix}.png")
    plt.close()

    # 2. Line Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Values")
    plt.plot(y_pred, label="Predicted Values")
    plt.xlabel("Data Point Index")
    plt.ylabel("Values")
    plt.title(f"Actual vs. Predicted Values over Index {filename_suffix}")
    plt.legend()
    plt.savefig(f"../plots/line_plot_{filename_suffix}.png")
    plt.close()

    # 3. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residual Plot {filename_suffix}")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.savefig(f"../plots/residual_{filename_suffix}.png")
    plt.close()

    # 4. Absolute Error Plot
    absolute_errors = np.abs(np.abs(y_pred) - np.abs(y_test))
    x_values = range(len(absolute_errors))
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, absolute_errors, label="Absolute Error", marker="o")
    plt.xlabel("Data Point")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error {filename_suffix}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../plots/absolute_errors_{filename_suffix}.png")
    plt.close()


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
                "mean_regional_pr",
                "mean_regional_frost_days",
                "mean_regional_ww_yield",
                "mean_yearly_pr",
                "mean_yearly_rsds",
                "mean_yearly_hurs",
                "mean_yearly_tasmin",
                "mean_yearly_tasmax",
                "mean_yearly_tas",
                "mean_yearly_sfcWind",
                "ww_yield",
                "ww_yield_average_detrended",
            ],
            columns=pivot_on,
            values=var,
            aggfunc="first",
        ).reset_index()

        df_temp.columns = [
            "nuts_id",
            "year",
            "mean_regional_pr",
            "mean_regional_frost_days",
            "mean_regional_ww_yield",
            "mean_yearly_pr",
            "mean_yearly_rsds",
            "mean_yearly_hurs",
            "mean_yearly_tasmin",
            "mean_yearly_tasmax",
            "mean_yearly_tas",
            "mean_yearly_sfcWind",
            "ww_yield",
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
                "mean_regional_pr",
                "mean_regional_frost_days",
                "mean_regional_ww_yield",
                "mean_yearly_pr",
                "mean_yearly_rsds",
                "mean_yearly_hurs",
                "mean_yearly_tasmin",
                "mean_yearly_tasmax",
                "mean_yearly_tas",
                "mean_yearly_sfcWind",
                "ww_yield",
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

# Calculate training data yearly regional means for climate data
train_yearly_climate_means = pd.DataFrame(
    df_train.groupby(["year", "nuts_id"]).agg(
        mean_yearly_pr=("pr_monthly", "mean"),
        mean_yearly_rsds=("rsds_monthly", "mean"),
        mean_yearly_tasmax=("tasmax_monthly", "mean"),
        mean_yearly_tasmin=("tasmin_monthly", "mean"),
        mean_yearly_tas=("tas_monthly", "mean"),
        mean_yearly_hurs=("hurs_monthly", "mean"),
        mean_yearly_sfcWind=("sfcWind_monthly", "mean"),
        mean_yearly_frost_days=("frost_days_monthly", "mean"),
    )
)

# Calculate full dataset yearly regional means for climate data (for test set)
full_yearly_climate_means = pd.DataFrame(
    df_relevant_data.groupby(["year", "nuts_id"]).agg(
        mean_yearly_pr=("pr_monthly", "mean"),
        mean_yearly_rsds=("rsds_monthly", "mean"),
        mean_yearly_tasmax=("tasmax_monthly", "mean"),
        mean_yearly_tasmin=("tasmin_monthly", "mean"),
        mean_yearly_tas=("tas_monthly", "mean"),
        mean_yearly_hurs=("hurs_monthly", "mean"),
        mean_yearly_sfcWind=("sfcWind_monthly", "mean"),
        mean_yearly_frost_days=("frost_days_monthly", "mean"),
    )
)

# Calculate training data regional means for climate data
train_regional_climate_means = pd.DataFrame(
    df_train.groupby("nuts_id").agg(
        mean_regional_pr=("pr_monthly", "mean"),
        mean_regional_frost_days=("frost_days_monthly", "mean"),
    )
)

# Calculate full dataset regional means for climate data (for test set)
full_regional_climate_means = pd.DataFrame(
    df_relevant_data.groupby("nuts_id").agg(
        mean_regional_pr=("pr_monthly", "mean"),
        mean_regional_frost_days=("frost_days_monthly", "mean"),
    )
)

# Calculate training regional dataset means for yield data (to be used for both train and test)
regional_yield_means = pd.DataFrame(
    df_train.groupby("nuts_id").agg(
        mean_regional_ww_yield=("ww_yield", "mean"),
    )
)

# Apply yearly means to training data
df_train = df_train.merge(
    train_yearly_climate_means, on=["year", "nuts_id"], how="left"
)

# Apply regional means to training data
df_train = df_train.merge(train_regional_climate_means, on="nuts_id", how="left")
df_train = df_train.merge(regional_yield_means, on="nuts_id", how="left")

# Apply yearly means to test data
df_test = df_test.merge(full_yearly_climate_means, on=["year", "nuts_id"], how="left")

# Apply regional means to test data
df_test = df_test.merge(full_regional_climate_means, on="nuts_id", how="left")
df_test = df_test.merge(regional_yield_means, on="nuts_id", how="left")

# --- Pivoting the Data ---
variables_to_pivot = [
    "spi_quarterly",
    "days_in_99_percentile_sfcWind_monthly",
    "days_in_99_percentile_pr_monthly",
    "frost_days_monthly",
    "spi_monthly",
    "frost_days_weekly",
    "days_avg_temp_above_28_weekly",
    "pr_monthly",
    "rsds_monthly",
    "tasmax_monthly",
    "hurs_monthly",
    "tas_monthly",
    "sfcWind_monthly",
    "tasmin_monthly",
    "pr_weekly",
    "rsds_weekly",
    "tasmax_weekly",
    "hurs_weekly",
    "tas_weekly",
    "sfcWind_weekly",
    "tasmin_weekly",
]

df_train = pivot_dataframe(df_train, variables_to_pivot)
df_test = pivot_dataframe(df_test, variables_to_pivot)

df_test_split_list = [
    df_test[df_test["year"] == year_to_exclude] for year_to_exclude in years_to_exclude
]

df_train = df_train.set_index(["year", "nuts_id"]).sort_index()
df_test = df_test.set_index(["year", "nuts_id"]).sort_index()

df_train.to_csv("../output/training_data_pivoted.csv")
df_test.to_csv("../output/test_data_pivoted.csv")
# --- Feature Selection ---

features_to_consider = (
    [
        # "mean_yearly_pr",
        # "mean_yearly_rsds",
        # "mean_yearly_hurs",
        # "mean_yearly_tasmin",
        # "mean_yearly_tasmax",
        # "mean_yearly_tas",
        # "mean_yearly_sfcWind",
        # "elevation",
        # "mean_regional_pr",
        # "mean_regional_frost_days",
        # "mean_regional_ww_yield_anomaly_percent_weighted",
        "mean_regional_ww_yield",
        # "spi",
        # "gdd_monthly",
        # "frost_days_monthly",
        # "days_max_temp_above_28_monthly",
        # "days_avg_temp_above_28_monthly",
        # "days_in_97_5_percentile_tas_monthly",
        # "days_in_95_percentile_pr_monthly",
        # "days_in_95_percentile_rsds_monthly",
        # "days_in_90_percentile_sfcWind_monthly",
        # "days_in_95_percentile_sfcWind_monthly",
        # "gdd_weekly",
        # "days_max_temp_above_28_weekly",
        # "days_avg_temp_above_28_weekly",
        # "days_in_97_5_percentile_tas_weekly",
        # "days_in_95_percentile_pr_weekly",
        # "days_in_95_percentile_rsds_weekly",
        # "days_in_90_percentile_sfcWind_weekly",
        # "days_in_95_percentile_sfcWind_weekly",
        # "frost_days_weekly",
        # "tas_monthly",
        # "tasmax_monthly",
        # "tasmin_monthly",
        # "rsds_monthly",
        # "pr_monthly",
        # "hurs_monthly",
        # "sfcWind_monthly",
        # "tas_weekly",
        # "tasmax_weekly",
        # "tasmin_weekly",
        # "rsds_weekly",
        # "pr_weekly",
        # "hurs_weekly",
        # "sfcWind_weekly",
    ]
    # + [f"spi_quarterly_quarter_{i}" for i in range(1, 4)]
    + [f"spi_monthly_month_{i}" for i in range(1, 10)]
    + [f"days_in_99_percentile_sfcWind_monthly_month_{i}" for i in range(1, 10)]
    + [f"days_in_99_percentile_pr_monthly_month_{i}" for i in range(1, 10)]
    + [f"frost_days_monthly_month_{i}" for i in range(1, 10)]
    # + [f"pr_monthly_month_{i}" for i in range(1, 10)]
    # + [f"tasmax_monthly_month_{i}" for i in range(1, 10)]
    # + [f"tasmin_monthly_month_{i}" for i in range(1, 10)]
    # + [f"tas_monthly_month_{i}" for i in range(1, 10)]
    # + [f"hurs_monthly_month_{i}" for i in range(1, 10)]
    # + [f"sfcWind_monthly_month_{i}" for i in range(1, 10)]
    # + [f"rsds_monthly_month_{i}" for i in range(1, 10)]
    # + [f"frost_days_weekly_week_{i}" for i in range(1, 40)]
    # + [f"days_avg_temp_above_28_weekly_week_{i}" for i in range(1, 40)]
    # + [f"rsds_weekly_week_{i}" for i in range(1, 40)]
    # + [f"pr_weekly_week_{i}" for i in range(1, 40)]
    # + [f"tasmax_weekly_week_{i}" for i in range(1, 40)]
    # + [f"tasmin_weekly_week_{i}" for i in range(1, 40)]
    # + [f"tas_weekly_week_{i}" for i in range(1, 40)]
    # + [f"hurs_weekly_week_{i}" for i in range(1, 40)]
    # + [f"sfcWind_weekly_week_{i}" for i in range(1, 40)]
)
target = "ww_yield_average_detrended"
# target = "ww_yield_anomaly_5yr"

# --- Features and target ---

X_train_years_excluded = df_train[features_to_consider]
y_train_years_excluded = df_train[target]

df_train[features_to_consider + [target]].to_csv(
    "../output/training_data_years_excluded.csv"
)

X_test_excluded_years = df_test[features_to_consider]
X_test_excluded_years = X_test_excluded_years.sort_index()
y_test_excluded_years = df_test[target]
y_test_excluded_years = y_test_excluded_years.sort_index()
df_test[features_to_consider + [target]].to_csv(
    "../output/testing_data_excluded_years.csv"
)

X_test_excluded_years_split = [
    df_year[features_to_consider] for df_year in df_test_split_list
]

y_test_excluded_years_split = [df_year[target] for df_year in df_test_split_list]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_years_excluded, y_train_years_excluded, test_size=0.10, shuffle=True
)

X_train = X_train.sort_index()
y_train = y_train.sort_index()

X_val = X_val.sort_index()
y_val = y_val.sort_index()

# Hyperparameter grid for RandomizedSearchCV
param_grid = {"n_estimators": [500]}


# Create the Random Forest model
rf = RandomForestRegressor()

tscv = TimeSeriesSplit(n_splits=5)
# Use RandomizedSearchCV for hyperparameter optimization
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
    n_jobs=-1,
    verbose=2,
)

grid_search.fit(X_train, y_train)

pd.DataFrame(grid_search.cv_results_).to_csv("../output/grid_search_results.csv")
# Get the best model from hyperparameter tuning
best_rf = grid_search.best_estimator_

# Evaluate the best model
y_pred_train = evaluate_model(best_rf, X_train, y_train, "Random Forest Training Data")
plot_results(y_train.values, y_pred_train, "Training_Data")

y_pred_val = evaluate_model(best_rf, X_val, y_val, "Random Forest Validation Data")
plot_results(y_val.values, y_pred_val, "Validation_Data")

y_pred_test_excluded = evaluate_model(
    best_rf,
    X_test_excluded_years,
    y_test_excluded_years,
    f"Random Forest Test Excluded Years: {', '.join(map(str,years_to_exclude))}",
)

plot_results(
    y_test_excluded_years.values,
    y_pred_test_excluded,
    f"Test_Excluded_Years_{'_'.join(map(str,years_to_exclude))}",
)


for i, year in enumerate(years_to_exclude):
    y_pred_test_excluded_single_year = evaluate_model(
        best_rf,
        X_test_excluded_years_split[i].values,
        y_test_excluded_years_split[i].values,
        f"Random Forest Test Excluded Year: {year}",
    )

    plot_results(
        y_test_excluded_years_split[i].values,
        y_pred_test_excluded_single_year,
        f"Test_Excluded_Year_{year}",
    )

feature_importance = pd.DataFrame(
    {"feature": X_train.columns, "importance": best_rf.feature_importances_}
).sort_values("importance", ascending=False)
print("\nRandom Forest Feature Importance:")
print(feature_importance)
feature_importance.to_csv("../output/feature_importance.csv")

print(grid_search.best_params_)

# Get the best estimator from grid search
best_estimator = grid_search.best_estimator_

# Get the 'n_estimators' values used in the grid search
n_estimators_values = list(set(grid_search.cv_results_["param_n_estimators"].data))

# Initialize lists to store RMSE values for training, validation, and test sets
train_rmse = []
validation_rmse = []
test_excluded_years_rmse = []
n_estimators_values.sort()
# Calculate RMSE for each value of n_estimators
for n_estimators in n_estimators_values:
    best_estimator.set_params(n_estimators=n_estimators)
    best_estimator.fit(X_train, y_train)

    # Predict on training, validation (using CV results), and test sets
    y_train_pred = best_estimator.predict(X_train)
    y_val_pred = grid_search.predict(X_val)  # Use CV predictions for validation set
    y_test_excluded_years_pred = best_estimator.predict(X_test_excluded_years)

    # Calculate RMSE and append to respective lists
    train_rmse.append(root_mean_squared_error(y_train, y_train_pred))
    validation_rmse.append(root_mean_squared_error(y_val, y_val_pred))
    test_excluded_years_rmse.append(
        root_mean_squared_error(y_test_excluded_years, y_test_excluded_years_pred)
    )

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, train_rmse, label="Training RMSE", marker="o")
plt.plot(n_estimators_values, validation_rmse, label="Validation RMSE", marker="x")
plt.plot(
    n_estimators_values,
    test_excluded_years_rmse,
    label="Test RMSE (Excluded years)",
    marker="s",
)
plt.xlabel("n_estimators")
plt.ylabel("RMSE")
plt.title("RMSE on Training, Validation, and Test Sets")
plt.legend()
plt.grid(True)
plt.xscale("log")
plt.savefig("../plots/n_trees_vs_error.png")
