import geopandas as gpd
import xarray as xr
import rasterio
from rasterio import transform
from rasterio.mask import mask
import numpy as np
import pandas as pd
from dask.distributed import Client
import rioxarray
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm

np.set_printoptions(suppress=True)


def plot_yield_and_climate_variable(
    nuts_id: str, clipped_dataset: xr.Dataset, ww_yield_ds: xr.Dataset, variable: str
):

    # Calculate regional mean precipitation
    regional_mean_pr = (
        clipped_dataset["pr"].resample(time="YE").mean(dim=["time", "lat", "lon"])
    )

    # Get winter wheat yield for the region
    regional_yield = ww_yield_ds["ww_yield_detrended"].sel(nuts_id=nuts_code)

    # --- Align time ranges ---
    # Find the common time range
    common_years = np.intersect1d(
        regional_mean_pr.time.dt.year.values, regional_yield.year.values
    )

    # Select data only within the common time range
    regional_mean_pr = regional_mean_pr.sel(
        time=regional_mean_pr.time.dt.year.isin(common_years)
    )
    regional_yield = regional_yield.sel(year=common_years)

    # --- Plotting ---
    fig, ax1 = plt.subplots()

    ax1.plot(
        common_years,
        regional_mean_pr,
        color="blue",
        label=variable,
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Precipitation", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()

    ax2.plot(common_years, regional_yield, color="red", label="Yield")
    ax2.set_ylabel("Yield", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.legend()
    plt.title(f"{variable} and Yield for {nuts_code}")
    plt.savefig(f"../plots/climate_vs_yield/{variable}_vs_yield_{nuts_id}.png")


def plot_region_cropped_ww_pr(nuts_id: str, clipped: xr.Dataset, gdf: gpd.GeoDataFrame):
    ax = gdf.plot(facecolor="none", edgecolor="black")
    clipped["pr"].isel(time=0).plot(ax=ax, zorder=-1)
    plt.title(f"{nuts_code} cropped by ww")
    plt.savefig(f"../plots/cropped_regions/{nuts_id}_cropped_by_ww.png", dpi=300)


def calculate_days_in_percentile(data: xr.DataArray, freq, q):
    threshold = data.quantile(q, dim="time")
    threshold = threshold.drop_vars("quantile")  # Drop the quantile coordinate
    return (data > threshold).resample(time=freq).sum(dim="time")


def calculate_frost_days(data: xr.DataArray, freq: str):
    return (data < 0).resample(time=freq).sum(dim="time")


def calculate_days_above_threshold(tas_array: xr.DataArray, threshold_temp, freq):
    return (tas_array > threshold_temp).resample(time=freq).sum(dim="time")


def calculate_spi_monthly(precip_data: xr.DataArray):
    resampled_precip_data = precip_data.resample(time="ME").mean()
    spi_ds = xr.full_like(resampled_precip_data, np.nan)

    print("Processing Monthly SPI")
    for lat in resampled_precip_data.lat:
        for lon in resampled_precip_data.lon:
            location_data = resampled_precip_data.sel(lat=lat, lon=lon).dropna("time")
            if location_data.size == 0:
                continue
            for month in range(1, 13):
                monthly_data = location_data.sel(
                    time=location_data.time.dt.month == month
                )
                if monthly_data.size == 0:
                    continue
                # Handle zero values
                monthly_data = monthly_data.where(monthly_data != 0, 0.000001)
                params = gamma.fit(monthly_data.values, floc=0)
                cdf = gamma.cdf(monthly_data.values, *params)
                spi_values = norm.ppf(cdf, loc=0, scale=1)
                spi_ds.loc[dict(lat=lat, lon=lon, time=monthly_data.time)] = spi_values
    return spi_ds


def calculate_spi_quarterly(precip_data: xr.DataArray):
    resampled_precip_data = precip_data.resample(time="QE").mean()
    spi_ds = xr.full_like(resampled_precip_data, np.nan)
    print("Processing Quarterly SPI")
    for lat in resampled_precip_data.lat:
        for lon in resampled_precip_data.lon:
            location_data = resampled_precip_data.sel(lat=lat, lon=lon).dropna("time")
            if location_data.size == 0:
                continue
            for quarter in range(1, 5):
                quarterly_data = location_data.sel(
                    time=location_data.time.dt.quarter == quarter
                )
                if quarterly_data.size == 0:
                    continue
                # Handle zero values
                quarterly_data = quarterly_data.where(quarterly_data != 0, 0.000001)
                params = gamma.fit(quarterly_data.values, floc=0)
                cdf = gamma.cdf(quarterly_data.values, *params)
                spi_values = norm.ppf(cdf, loc=0, scale=1)
                spi_ds.loc[dict(lat=lat, lon=lon, time=quarterly_data.time)] = (
                    spi_values
                )
    return spi_ds


if __name__ == "__main__":
    client = Client()

    # Define file paths
    file_paths = {
        "shapefile": "../masks/NUTS3/NUTS250_N3.shp",
        "crop_mask": "../masks/CropMasks/CTM_2018_WiWh_EPSG4326_654_866_1000m.tif",
        "elevation": "../elevation_data/elevation_amber_conform_med_EPSG4326_654_866_1000m.tif",
        "yield": "../output/targets/processed_yield.nc",
        "output": "../output/features/final_data_features_aggregated_on_nuts_inc_spi.nc",
        "raw_climate_data": "../climate_data/1979_2023_allvariables_compressed.nc",
    }

    # Create dictionaries to store results
    result_dicts = {
        "quarterly_features": {},
        "monthly_features": {},
        "weekly_features": {},
        "monthly_raw_data": {},
        "weekly_raw_data": {},
    }

    quarterly_features = ["spi_quarterly"]

    monthly_features = [
        "days_in_99_percentile_sfcWind_monthly",
        "days_in_99_percentile_pr_monthly",
        "frost_days_monthly",
        "spi_monthly",
    ]

    weekly_features = ["frost_days_weekly", "days_avg_temp_above_28_weekly"]

    raw_data = [
        "rsds",
        "pr",
        "hurs",
        "tasmax",
        "tasmin",
        "tas",
        "sfcWind",
    ]

    # Initialize result dictionaries
    for feature in quarterly_features:
        result_dicts["quarterly_features"][feature] = {}
    for feature in monthly_features:
        result_dicts["monthly_features"][feature] = {}
    for feature in weekly_features:
        result_dicts["weekly_features"][feature] = {}
    for variable in raw_data:
        result_dicts["monthly_raw_data"][f"{variable}_monthly"] = {}
    for variable in raw_data:
        result_dicts["weekly_raw_data"][f"{variable}_weekly"] = {}

    # Open climate data ds
    raw_climate_data_ds = xr.open_dataset(file_paths["raw_climate_data"])
    raw_climate_data_ds = raw_climate_data_ds.chunk(
        {"time": -1, "lat": "auto", "lon": "auto"}
    )

    # Set rio spatial dims and crs
    raw_climate_data_ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    raw_climate_data_ds.rio.write_crs("epsg:4326", inplace=True)

    # Load shapefile and ensure CRS matches the NetCDF
    gdf = gpd.GeoDataFrame(gpd.read_file(file_paths["shapefile"]).to_crs("EPSG:4326"))

    # Group the GeoDataFrame by NUTS code
    nuts_groups = gdf.groupby("NUTS_CODE")

    # open yield dataset
    ww_yield_ds = xr.open_dataset(file_paths["yield"])

    # Open winter wheat crop mask
    ww_crop_mask = rasterio.open(file_paths["crop_mask"])

    masked_data = raw_climate_data_ds.where(ww_crop_mask.read(1) == 1)

    output_file = "../output/derived_features.nc"

    nuts_ids = []
    for nuts_code, nuts_group in nuts_groups:
        try:
            number_of_available_years = (
                ww_yield_ds["ww_yield"].sel(nuts_id=nuts_code).notnull().sum().values
            )
        except KeyError:
            number_of_available_years = 0

        if number_of_available_years >= 40:
            clipped: xr.Dataset = masked_data.rio.clip(
                nuts_group["geometry"].apply(mapping),
                "epsg:4326",
                drop=True,
            )

            clipped.load()
            number_of_points_ww_growth = (
                clipped["pr"].isel(time=0).notnull().sum().values
            )

            if number_of_points_ww_growth >= 15:

                result_dicts["quarterly_features"]["spi_quarterly"][nuts_code] = (
                    calculate_spi_quarterly(clipped["pr"])
                    .mean(dim=["lat", "lon"])
                    .compute(client=client)
                )

                result_dicts["monthly_features"]["spi_monthly"][nuts_code] = (
                    calculate_spi_monthly(clipped["pr"])
                    .mean(dim=["lat", "lon"])
                    .compute(client=client)
                )

                result_dicts["monthly_features"][
                    "days_in_99_percentile_sfcWind_monthly"
                ][nuts_code] = (
                    calculate_days_in_percentile(clipped["sfcWind"], "ME", 0.99)
                    .mean(dim=["lat", "lon"])
                    .compute(client=client)
                )

                result_dicts["monthly_features"]["days_in_99_percentile_pr_monthly"][
                    nuts_code
                ] = (
                    calculate_days_in_percentile(clipped["pr"], "ME", 0.99)
                    .mean(dim=["lat", "lon"])
                    .compute(client=client)
                )

                result_dicts["monthly_features"]["frost_days_monthly"][nuts_code] = (
                    calculate_frost_days(clipped["tas"], "ME")
                    .mean(dim=["lat", "lon"])
                    .compute(client=client)
                )

                result_dicts["weekly_features"]["frost_days_weekly"][nuts_code] = (
                    calculate_frost_days(clipped["tas"], "W")
                    .mean(dim=["lat", "lon"])
                    .compute(client=client)
                )

                result_dicts["weekly_features"]["days_avg_temp_above_28_weekly"][
                    nuts_code
                ] = (
                    calculate_days_above_threshold(clipped["tas"], 28, "W")
                    .mean(dim=["lat", "lon"])
                    .compute(client=client)
                )

                for variable in raw_data:
                    result_dicts["monthly_raw_data"][f"{variable}_monthly"][
                        nuts_code
                    ] = (
                        clipped[variable]
                        .resample(time="ME")
                        .mean(dim=["time", "lat", "lon"])
                        .compute(client=client)
                    )
                print("Processed monthly raw data")

                for variable in raw_data:
                    result_dicts["weekly_raw_data"][f"{variable}_weekly"][nuts_code] = (
                        clipped[variable]
                        .resample(time="W")
                        .mean(dim=["time", "lat", "lon"])
                        .compute(client=client)
                    )
                print("Processed weekly raw data")

                print(f"Processed nuts_id:{nuts_code}")
                nuts_ids.append(nuts_code)

    # clear out all 53rd weeks
    for feature_key, features in result_dicts["weekly_features"].items():
        for nuts_id, weekly_data in features.items():
            result_dicts["weekly_features"][feature_key][nuts_id] = weekly_data.where(
                weekly_data.time.dt.isocalendar().week != 53, drop=True
            )

    for variable_key, variables in result_dicts["weekly_raw_data"].items():
        for nuts_id, weekly_raw_data in variables.items():
            result_dicts["weekly_raw_data"][variable_key][nuts_id] = (
                weekly_raw_data.where(
                    weekly_raw_data.time.dt.isocalendar().week != 53, drop=True
                )
            )

    years = result_dicts["quarterly_features"]["spi_quarterly"][
        nuts_ids[0]
    ].time.dt.year.values

    quarters = result_dicts["quarterly_features"]["spi_quarterly"][
        nuts_ids[0]
    ].time.dt.quarter.values

    quarterly_multi_index = pd.MultiIndex.from_arrays(
        [years, quarters], names=("year", "quarter")
    )

    years = result_dicts["monthly_features"]["frost_days_monthly"][
        nuts_ids[0]
    ].time.dt.year.values

    months = result_dicts["monthly_features"]["frost_days_monthly"][
        nuts_ids[0]
    ].time.dt.month.values

    monthly_multi_index = pd.MultiIndex.from_arrays(
        [years, months], names=("year", "month")
    )

    years = result_dicts["weekly_features"]["frost_days_weekly"][
        nuts_ids[0]
    ].time.dt.year.values

    weeks = (
        result_dicts["weekly_features"]["frost_days_weekly"][nuts_ids[0]]
        .time.dt.isocalendar()
        .week.values
    )

    weekly_multi_index = pd.MultiIndex.from_arrays(
        [years, weeks], names=("year", "week")
    )

    def create_value_array(feature_dict, multi_index):
        value_store = np.zeros((len(nuts_ids), len(multi_index)))
        for i, nuts_id in enumerate(nuts_ids):
            value_store[i, :] = feature_dict[nuts_id].values
        return value_store

    quarterly_values = {
        feature: create_value_array(
            result_dicts["quarterly_features"][feature], quarterly_multi_index
        )
        for feature in quarterly_features
    }

    monthly_values = {
        feature: create_value_array(
            result_dicts["monthly_features"][feature], monthly_multi_index
        )
        for feature in monthly_features
    }

    weekly_values = {
        feature: create_value_array(
            result_dicts["weekly_features"][feature], weekly_multi_index
        )
        for feature in weekly_features
    }

    monthly_raw_data_values = {
        variable: create_value_array(
            result_dicts["monthly_raw_data"][f"{variable}_monthly"], monthly_multi_index
        )
        for variable in raw_data
    }

    weekly_raw_data_values = {
        variable: create_value_array(
            result_dicts["weekly_raw_data"][f"{variable}_weekly"], weekly_multi_index
        )
        for variable in raw_data
    }

    # Create xarray Dataset
    data_vars = {
        **{
            feature: (
                ["nuts_id", "year", "quarter"],
                quarterly_values[feature].reshape(len(nuts_ids), -1, 4),
            )
            for feature in quarterly_features
        },
        **{
            feature: (
                ["nuts_id", "year", "month"],
                monthly_values[feature].reshape(len(nuts_ids), -1, 12),
            )
            for feature in monthly_features
        },
        **{
            feature: (
                ["nuts_id", "year", "week"],
                weekly_values[feature].reshape(len(nuts_ids), -1, 52),
            )
            for feature in weekly_features
        },
        **{
            f"{variable}_monthly": (
                ["nuts_id", "year", "month"],
                monthly_raw_data_values[variable].reshape(len(nuts_ids), -1, 12),
            )
            for variable in raw_data
        },
        **{
            f"{variable}_weekly": (
                ["nuts_id", "year", "week"],
                weekly_raw_data_values[variable].reshape(len(nuts_ids), -1, 52),
            )
            for variable in raw_data
        },
    }

    aggregated_ds = xr.Dataset(
        data_vars,
        coords={
            "nuts_id": nuts_ids,
            "year": np.unique(years),
            "quarter": np.arange(1, 5),
            "month": np.arange(1, 13),
            "week": np.arange(1, 53),
        },
    )

    aggregated_ds.to_netcdf(file_paths["output"])
