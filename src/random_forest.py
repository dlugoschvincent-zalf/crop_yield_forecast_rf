import xarray as xr
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm

# Load and preprocess data (as in your original script)
ds = xr.open_dataset("../output/all_inputs_aggregated_on_nuts.nc")

df_relevant_data = ds.sel(year=slice(2007, 2021)).to_dataframe().reset_index().dropna()

# Drop oct, nov, dec
# df_relevant_data = df_relevant_data[df_relevant_data.month < 10]

# Create a list to store DataFrames for each variable
dfs = []

# Loop through the variables to pivot
for var in ["spi", "gdd", "frost_days"]:
    df_temp = df_relevant_data.pivot_table(
        index=[
            "nuts_id",
            "year",
            "elevation",
            "ww_yield_anomaly_percent_weighted",
            "ww_yield",
            "precip",
        ],
        columns="month",
        values=var,
        aggfunc="first",
    ).reset_index()

    df_temp.columns = [
        "nuts_id",
        "year",
        "elevation",
        "ww_yield_anomaly_percent_weighted",
        "ww_yield",
        "precip",
    ] + [f"{var}_month_{i}" for i in range(1, 13)]

    dfs.append(df_temp)

# Merge all DataFrames together
df_final = dfs[0]
for i in range(1, len(dfs)):
    df_final = pd.merge(
        df_final,
        dfs[i],
        on=[
            "nuts_id",
            "year",
            "elevation",
            "ww_yield_anomaly_percent_weighted",
            "ww_yield",
            "precip",
        ],
    )


df_relevant_data = df_final

years_to_exclude = [2020]  # List of years to exclude

df_years_excluded = df_relevant_data[
    ~df_relevant_data["year"].isin(years_to_exclude)
].copy()

df_excluded_years = df_relevant_data[
    df_relevant_data["year"].isin(years_to_exclude)
].copy()


# Regional precipitation average excluding test years
df_years_excluded["mean_regional_precip"] = df_years_excluded.groupby("nuts_id")[
    "precip"
].transform("mean")


# Regional precipitation average including all years
df_excluded_years["mean_regional_precip"] = df_relevant_data.groupby("nuts_id")[
    "precip"
].transform("mean")


df_relevant_data["ww_yield_anomaly_percent_weighted_mean_all_years"] = (
    df_relevant_data.groupby("nuts_id")["ww_yield_anomaly_percent_weighted"].transform(
        "mean"
    )
)

df_years_excluded.to_csv("../output/data_years_excluded_pivoted.csv")
df_excluded_years.to_csv("../output/data_excluded_years_pivoted.csv")


# Features and target
X_years_excluded = df_years_excluded[
    ["elevation", "mean_regional_precip"]
    + [f"spi_month_{i}" for i in range(1, 13)]
    + [f"gdd_month_{i}" for i in range(1, 13)]
    + [f"frost_days_month_{i}" for i in range(1, 13)]
]

y_years_excluded = df_years_excluded["ww_yield"]

X_test_excluded_years = df_excluded_years[
    ["elevation", "mean_regional_precip"]
    + [f"spi_month_{i}" for i in range(1, 13)]
    + [f"gdd_month_{i}" for i in range(1, 13)]
    + [f"frost_days_month_{i}" for i in range(1, 13)]
]

y_test_excluded_years = df_excluded_years["ww_yield"]

X_train, X_val, y_train, y_val = train_test_split(
    X_years_excluded, y_years_excluded, test_size=0.2
)


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


# Hyperparameter grid for RandomizedSearchCV
param_grid = {
    "n_estimators": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        14,
        16,
        18,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
    ],
}

# Create the Random Forest model
rf = RandomForestRegressor()

# Use RandomizedSearchCV for hyperparameter optimization
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(X_train, y_train)

# Get the best model from hyperparameter tuning
best_rf = grid_search.best_estimator_

# Evaluate the best model
evaluate_model(
    best_rf,
    X_train,
    y_train,
    "Random Forest (Randomized Search) Training Data",
)

evaluate_model(
    best_rf,
    X_val,
    y_val,
    "Random Forest (Randomized Search) Validation included years",
)

evaluate_model(
    best_rf,
    X_test_excluded_years,
    y_test_excluded_years,
    "Random Forest (Randomized Search) Test excluded years",
)

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
plt.savefig("../plots/n_trees_vs_error.png")

# Evaluate tho model with yearly averages
y_test_all_regions_avg = df_relevant_data[
    df_relevant_data["year"].isin(years_to_exclude)
]["ww_yield_anomaly_percent_weighted_mean_all_years"]

feature_importance = pd.DataFrame(
    {"feature": X_train.columns, "importance": best_rf.feature_importances_}
).sort_values("importance", ascending=False)
print("\nRandom Forest Feature Importance:")
print(feature_importance)

print(grid_search.best_params_)

y_pred_to_plot = best_rf.predict(X_test_excluded_years)
# y_pred_to_plot = best_rf.predict(X_years_excluded)

y_test_to_plot = y_test_excluded_years.values
# y_test_to_plot = y_years_excluded.values

# 1. Scatter Plot (for visualizing individual data points)
plt.figure(figsize=(12, 6))
plt.scatter(
    y_test_to_plot, y_pred_to_plot, alpha=0.5
)  # Adjust alpha for point transparency
plt.xlabel("Actual Values (Test Data)")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.savefig("../plots/scatter.png")

# 2. Line Plot (useful for time series or ordered data)
plt.figure(figsize=(12, 6))
plt.plot(y_test_to_plot, label="Actual Values")
plt.plot(y_pred_to_plot, label="Predicted Values")
plt.xlabel("Data Point Index")
plt.ylabel("Values")
plt.title("Actual vs. Predicted Values over Time/Index")
plt.legend()
plt.savefig("../plots/line_plot.png")

# 3. Line Plot (Average over year and region) (useful for time series or ordered data)
plt.figure(figsize=(12, 6))
plt.plot(y_test_all_regions_avg.values, label="Actual Values")
plt.plot(y_pred_to_plot, label="Predicted Values")
plt.xlabel("Data Point Index")
plt.ylabel("Values")
plt.title("Actual vs. Predicted Values over Time/Index")
plt.legend()
plt.savefig("../plots/line_plot_all_regions.png")

# 4. Residual Plot (for analyzing prediction errors)
residuals = y_test_to_plot - y_pred_to_plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_to_plot, residuals, alpha=0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.axhline(y=0, color="r", linestyle="--")  # Add a horizontal line at zero residual
plt.savefig("../plots/residual.png")

absolute_errors = np.abs(np.abs(y_pred_to_plot) - np.abs(y_test_to_plot))

# Create x-axis values for plotting (assuming your data has some inherent order)
x_values = range(len(absolute_errors))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x_values, absolute_errors, label="Absolute Error", marker="o")
plt.xlabel("Data Point")  # Adjust label if your x-axis represents something else
plt.ylabel("Absolute Error")
plt.title("Absolute Error Between Predicted and Actual Values")
plt.legend()
plt.grid(True)
plt.savefig("../plots/absolute_erros.png")

y_pred_to_plot.sort()
y_test_to_plot.sort()

# Calculate mean and standard deviation for y_pred
mean_pred = statistics.mean(y_pred_to_plot)
sd_pred = statistics.stdev(y_pred_to_plot)

# Calculate mean and standard deviation for y_test
mean_test = statistics.mean(y_test_to_plot)
sd_test = statistics.stdev(y_test_to_plot)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot the normal distribution of y_pred (blue)
plt.plot(
    y_pred_to_plot,
    norm.pdf(y_pred_to_plot, mean_pred, sd_pred),
    label="Predicted",
    color="blue",
)

# Plot the normal distribution of y_test (orange)
plt.plot(
    y_test_to_plot,
    norm.pdf(y_test_to_plot, mean_test, sd_test),
    label="Actual",
    color="orange",
)

plt.xlabel("Yield Anomaly (%)")
plt.ylabel("Probability Density")
plt.title("Normal Distribution of Predicted and Actual Yield Anomalies")
plt.legend()
plt.grid(True)

plt.savefig("../plots/normal_distributions_combined.png")
