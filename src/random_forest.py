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

    plt.savefig(f"../output/final_runs/plots/scatter_{filename_suffix}.png")
    plt.close()

    # 2. Line Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Values")
    plt.plot(y_pred, label="Predicted Values")
    plt.xlabel("Data Point Index")
    plt.ylabel("Values")
    plt.title(f"Actual vs. Predicted Values over Index {filename_suffix}")
    plt.legend()
    plt.savefig(f"../output/final_runs/plots/line_plot_{filename_suffix}.png")
    plt.close()

    # 3. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residual Plot {filename_suffix}")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.savefig(f"../output/final_runs/plots/residual_{filename_suffix}.png")
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
    plt.savefig(f"../output/final_runs/plots/absolute_errors_{filename_suffix}.png")
    plt.close()


if __name__ == "__main__":
    df_train = pd.read_csv(
        "../output/final_runs/ready_to_extract_data/training_data_pivoted.csv"
    )
    df_test = pd.read_csv(
        "../output/final_runs/ready_to_extract_data/test_data_pivoted.csv"
    )

    years_to_exclude = sorted(df_test["year"].unique())

    df_test_split_list = [
        df_test[df_test["year"] == year_to_exclude]
        .set_index(["year", "nuts_id"])
        .sort_index()
        for year_to_exclude in years_to_exclude
    ]

    df_train = df_train.set_index(["year", "nuts_id"]).sort_index()
    df_test = df_test.set_index(["year", "nuts_id"]).sort_index()

    # --- Feature Selection ---

    # Variant 1: Only mean regional yield
    features_to_consider_variant_1 = [
        "mean_regional_ww_yield",
    ]

    # Variant 2: derived features only and mean regional yield
    features_to_consider_variant_2 = (
        [
            "mean_regional_ww_yield",
        ]
        + [f"spi_monthly_month_{i}" for i in range(1, 10)]
        + [f"days_in_99_percentile_sfcWind_monthly_month_{i}" for i in range(1, 10)]
        + [f"days_in_99_percentile_pr_monthly_month_{i}" for i in range(1, 10)]
        + [f"frost_days_monthly_month_{i}" for i in range(1, 10)]
        + [f"days_avg_temp_above_28_weekly_week_{i}" for i in range(1, 40)]
    )

    # Variant 3: Only derived features
    features_to_consider_variant_3 = (
        [f"spi_monthly_month_{i}" for i in range(1, 10)]
        + [f"days_in_99_percentile_sfcWind_monthly_month_{i}" for i in range(1, 10)]
        + [f"days_in_99_percentile_pr_monthly_month_{i}" for i in range(1, 10)]
        + [f"frost_days_monthly_month_{i}" for i in range(1, 10)]
        + [f"days_avg_temp_above_28_weekly_week_{i}" for i in range(1, 40)]
    )

    # Variant 4: raw monthly data and mean regional yield
    features_to_consider_variant_4 = (
        [
            "mean_regional_ww_yield",
        ]
        + [f"pr_monthly_month_{i}" for i in range(1, 10)]
        + [f"tasmax_monthly_month_{i}" for i in range(1, 10)]
        + [f"tasmin_monthly_month_{i}" for i in range(1, 10)]
        + [f"tas_monthly_month_{i}" for i in range(1, 10)]
        + [f"hurs_monthly_month_{i}" for i in range(1, 10)]
        + [f"sfcWind_monthly_month_{i}" for i in range(1, 10)]
        + [f"rsds_monthly_month_{i}" for i in range(1, 10)]
    )

    # Variant 5: only raw monthly data
    features_to_consider_variant_5 = (
        [f"pr_monthly_month_{i}" for i in range(1, 10)]
        + [f"tasmax_monthly_month_{i}" for i in range(1, 10)]
        + [f"tasmin_monthly_month_{i}" for i in range(1, 10)]
        + [f"tas_monthly_month_{i}" for i in range(1, 10)]
        + [f"hurs_monthly_month_{i}" for i in range(1, 10)]
        + [f"sfcWind_monthly_month_{i}" for i in range(1, 10)]
        + [f"rsds_monthly_month_{i}" for i in range(1, 10)]
    )

    target = "ww_yield_average_detrended"

    features_to_consider_variantions = [
        features_to_consider_variant_1,
        features_to_consider_variant_2,
        features_to_consider_variant_3,
        features_to_consider_variant_4,
        features_to_consider_variant_5,
    ]

    # --- Features and target ---
    for index, features_to_consider in enumerate(features_to_consider_variantions):
        index = index + 1
        print(f"Running Variant {index}")
        X_train_years_excluded = df_train[features_to_consider]
        y_train_years_excluded = df_train[target]

        X_test_excluded_years = df_test[features_to_consider]
        y_test_excluded_years = df_test[target]

        X_test_excluded_years_split = [
            df_year[features_to_consider] for df_year in df_test_split_list
        ]

        y_test_excluded_years_split = [
            df_year[target] for df_year in df_test_split_list
        ]

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_years_excluded,
            y_train_years_excluded,
            test_size=0.10,
            shuffle=True,
            random_state=42,
        )

        X_train = X_train.sort_index()
        y_train = y_train.sort_index()

        X_val = X_val.sort_index()
        y_val = y_val.sort_index()

        # Hyperparameter grid for RandomizedSearchCV in this case we only test for different number of trees
        param_grid = {
            "n_estimators": [1, 10, 50, 100, 300, 500],
            "max_features": [0.66],
            "random_state": [42],
        }

        # Create the Random Forest model
        rf = RandomForestRegressor(random_state=42)

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

        # Get the best model from hyperparameter tuning
        best_rf = grid_search.best_estimator_

        # Evaluate the best model
        y_pred_train = evaluate_model(
            best_rf, X_train, y_train, "Random Forest Training Data"
        )
        plot_results(y_train.values, y_pred_train, f"Training_Data_variant_{index}")

        y_pred_val = evaluate_model(
            best_rf, X_val, y_val, "Random Forest Validation Data"
        )
        plot_results(y_val.values, y_pred_val, f"Validation_Data_variant_{index}")

        y_pred_test_excluded = evaluate_model(
            best_rf,
            X_test_excluded_years,
            y_test_excluded_years,
            f"Random Forest Test Excluded Years: {', '.join(map(str,years_to_exclude))}",
        )

        plot_results(
            y_test_excluded_years.values,
            y_pred_test_excluded,
            f"Test_{'_'.join(map(str,years_to_exclude))}_variant_{index}",
        )

        for i, year in enumerate(years_to_exclude):
            y_pred_test_excluded_single_year = evaluate_model(
                best_rf,
                X_test_excluded_years_split[i],
                y_test_excluded_years_split[i],
                f"Random Forest Test Excluded Year: {year}",
            )

            plot_results(
                y_test_excluded_years_split[i].values,
                y_pred_test_excluded_single_year,
                f"Test_Excluded_Year_{year}_variant_{index}",
            )

        feature_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": best_rf.feature_importances_}
        ).sort_values("importance", ascending=False)
        print(f"\nRandom Forest Feature Importance variant {index}:")
        print(feature_importance)
        feature_importance.to_csv(
            f"../output/final_runs/feature_importance/feature_importance_variant_{index}.csv"
        )

        print(grid_search.best_params_)

        # Get the best estimator from grid search
        best_estimator = grid_search.best_estimator_

        # Get the 'n_estimators' values used in the grid search
        n_estimators_values = list(
            set(grid_search.cv_results_["param_n_estimators"].data)
        )

        # Initialize lists to store RMSE values for training, validation, and test sets
        train_rmse = []
        validation_rmse = []
        test_excluded_years_rmse = []
        n_estimators_values.sort()
        # Calculate RMSE for each value of n_estimators
        # only important when utilizing the grid search and wanting to visualize the effect of the n_estimators hyperparameter on model performance
        for n_estimators in n_estimators_values:
            best_estimator.set_params(n_estimators=n_estimators, random_state=42)
            best_estimator.fit(X_train, y_train)

            # Predict on training, validation (using CV results), and test sets
            y_train_pred = best_estimator.predict(X_train)
            y_val_pred = grid_search.predict(
                X_val
            )  # Use CV predictions for validation set
            y_test_excluded_years_pred = best_estimator.predict(X_test_excluded_years)

            # Calculate RMSE and append to respective lists
            train_rmse.append(root_mean_squared_error(y_train, y_train_pred))
            validation_rmse.append(root_mean_squared_error(y_val, y_val_pred))
            test_excluded_years_rmse.append(
                root_mean_squared_error(
                    y_test_excluded_years, y_test_excluded_years_pred
                )
            )

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(n_estimators_values, train_rmse, label="Training RMSE", marker="o")
        plt.plot(
            n_estimators_values, validation_rmse, label="Validation RMSE", marker="x"
        )
        plt.plot(
            n_estimators_values,
            test_excluded_years_rmse,
            label=f"Test RMSE (Excluded years)",
            marker="s",
        )
        plt.xlabel("n_estimators")
        plt.ylabel("RMSE")
        plt.title("RMSE on Training, Validation, and Test Sets variant {index}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"../output/final_runs/plots/n_trees_vs_error_variant_{index}.png")
