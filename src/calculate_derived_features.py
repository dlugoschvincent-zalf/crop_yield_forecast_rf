import xarray as xr
from dask.distributed import Client
import os


# ----- Helper Functions -----
def calculate_gdd(tasmin: xr.DataArray, tasmax: xr.DataArray, freq: str, base_temp=10):
    tasmax = xr.where(tasmax > 30, 30, tasmax)
    tavg = (tasmin + tasmax) / 2
    daily_gdd = xr.where(tavg > base_temp, tavg - base_temp, 0)
    return daily_gdd.resample(time=freq).sum(dim="time")


def calculate_frost_days(data: xr.DataArray, freq: str):
    return (data < 0).resample(time=freq).sum(dim="time")


def calculate_days_above_threshold(tas_array: xr.DataArray, threshold_temp, freq):
    return (tas_array > threshold_temp).resample(time=freq).sum(dim="time")


def calculate_days_in_percentile(data: xr.DataArray, freq, q):
    threshold = data.quantile(q, dim="time")
    threshold = threshold.drop_vars("quantile")  # Drop the quantile coordinate
    return (data > threshold).resample(time=freq).sum(dim="time")


# ----- Main Calculation Function -----
def calculate_climate_indices(ds: xr.Dataset, output_dir: str, client: Client):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List of calculations for both weekly and monthly
    calculations = [
        ("gdd_week", calculate_gdd(ds["tasmin"], ds["tasmax"], "W")),
        ("gdd_month", calculate_gdd(ds["tasmin"], ds["tasmax"], "ME")),
        ("frost_days_week", calculate_frost_days(ds["tas"], "W")),
        ("frost_days_month", calculate_frost_days(ds["tas"], "ME")),
        (
            "days_max_temp_above_28_week",
            calculate_days_above_threshold(ds["tasmax"], 28, "W"),
        ),
        (
            "days_max_temp_above_28_month",
            calculate_days_above_threshold(ds["tasmax"], 28, "ME"),
        ),
        (
            "days_avg_temp_above_28_week",
            calculate_days_above_threshold(ds["tas"], 28, "W"),
        ),
        (
            "days_avg_temp_above_28_month",
            calculate_days_above_threshold(ds["tas"], 28, "ME"),
        ),
        (
            "days_in_97_5_percentile_tas_week",
            calculate_days_in_percentile(ds["tas"], "W", 0.975),
        ),
        (
            "days_in_97_5_percentile_tas_month",
            calculate_days_in_percentile(ds["tas"], "ME", 0.975),
        ),
        (
            "days_in_95_percentile_pr_week",
            calculate_days_in_percentile(ds["pr"], "W", 0.95),
        ),
        (
            "days_in_95_percentile_pr_month",
            calculate_days_in_percentile(ds["pr"], "ME", 0.95),
        ),
        (
            "days_in_95_percentile_rsds_week",
            calculate_days_in_percentile(ds["rsds"], "W", 0.95),
        ),
        (
            "days_in_95_percentile_rsds_month",
            calculate_days_in_percentile(ds["rsds"], "ME", 0.95),
        ),
        (
            "days_in_90_percentile_sfcWind_week",
            calculate_days_in_percentile(ds["sfcWind"], "W", 0.90),
        ),
        (
            "days_in_90_percentile_sfcWind_month",
            calculate_days_in_percentile(ds["sfcWind"], "ME", 0.90),
        ),
        (
            "days_in_95_percentile_sfcWind_week",
            calculate_days_in_percentile(ds["sfcWind"], "W", 0.95),
        ),
        (
            "days_in_95_percentile_sfcWind_month",
            calculate_days_in_percentile(ds["sfcWind"], "ME", 0.95),
        ),
    ]

    # Perform calculations and save each result separately
    for name, calc_func in calculations:
        print(f"Calculating {name}...")
        ds_result = xr.Dataset()
        ds_result[name] = calc_func.compute(client=client)
        output_file = os.path.join(output_dir, f"{name}.nc")

        print(f"Saving {output_file}...")
        saver = ds_result.to_netcdf(output_file, compute=False)
        saver.compute(client=client)

        print(f"Saved {output_file}")
        del ds_result

    print("All calculations completed and saved.")


def merge_and_save_results(output_dir: str, client: Client):
    print("Merging results...")
    # List all .nc files in the output directory
    nc_files_weekly = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith("week.nc")
    ]

    # Open all weekly datasets using open_mfdataset
    merged_ds_weekly = xr.open_mfdataset(
        nc_files_weekly, chunks={"time": "auto", "lat": "auto", "lon": "auto"}
    )

    nc_files_monthly = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith("month.nc")
    ]

    # Open all monthly datasets using open_mfdataset
    merged_ds_monthly = xr.open_mfdataset(
        nc_files_monthly, chunks={"time": "auto", "lat": "auto", "lon": "auto"}
    )

    # Save the merged dataset with progress bar
    print(f"Saving final output to ../output/combined_climate_indices_monthly.nc...")

    encoding_monthly = {
        "gdd_month": {"compression": "zstd"},
        "frost_days_month": {"compression": "zstd"},
        "days_max_temp_above_28_month": {"compression": "zstd"},
        "days_avg_temp_above_28_month": {"compression": "zstd"},
        "days_in_97_5_percentile_tas_month": {"compression": "zstd"},
        "days_in_95_percentile_pr_month": {"compression": "zstd"},
        "days_in_95_percentile_rsds_month": {"compression": "zstd"},
        "days_in_90_percentile_sfcWind_month": {"compression": "zstd"},
        "days_in_95_percentile_sfcWind_month": {"compression": "zstd"},
    }
    saver = merged_ds_monthly.to_netcdf(
        "../output/combined_climate_indices_monthly.nc",
        compute=False,
        encoding=encoding_monthly,
    )

    saver.compute(client=client)

    print(f"Saving final output to ../output/combined_climate_indices_weekly.nc...")

    encoding_weekly = {
        "gdd_week": {"compression": "zstd"},
        "frost_days_week": {"compression": "zstd"},
        "days_max_temp_above_28_week": {"compression": "zstd"},
        "days_avg_temp_above_28_week": {"compression": "zstd"},
        "days_in_97_5_percentile_tas_week": {"compression": "zstd"},
        "days_in_95_percentile_pr_week": {"compression": "zstd"},
        "days_in_95_percentile_rsds_week": {"compression": "zstd"},
        "days_in_90_percentile_sfcWind_week": {"compression": "zstd"},
        "days_in_95_percentile_sfcWind_week": {"compression": "zstd"},
    }
    saver = merged_ds_weekly.to_netcdf(
        "../output/combined_climate_indices_weekly.nc",
        compute=False,
        encoding=encoding_weekly,
    )

    saver.compute(client=client)

    print("Final output saved.")


if __name__ == "__main__":
    # Set up Dask client
    client = Client()

    # Define chunking strategy
    chunks = {"time": -1, "lat": "auto", "lon": "auto"}

    # Open dataset with chunking
    ds = xr.open_dataset("../output/allyears_compressed_merged.nc", chunks=chunks)

    # Define output directories
    intermediate_dir = "../output/intermediate"

    # Calculate all indices and save intermediate results
    print("Calculating climate indices...")
    calculate_climate_indices(ds, intermediate_dir, client)

    # merge_and_save_results(intermediate_dir, client)
