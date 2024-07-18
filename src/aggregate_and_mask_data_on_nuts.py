import geopandas as gpd
import xarray as xr
import rasterio
from rasterio import transform
from rasterio.mask import mask
import numpy as np
import pandas as pd
from dask.distributed import Client


def calculate_yield_anomaly(ds):
    """Calculates various yield anomalies:
    1. Compared to the
    2. Compared to the total average over all regions for the 5 years prior to the previous year
    3. Compared to the total weighted average for the same 5-year period, weighted by total arable land

    Args:
        ds: xarray Dataset with dimensions 'nuts_id' and 'year',
            and variables 'ww_yield' and 'ArabLand'.
    Returns:
        xarray Dataset with the yield anomalies and percentage yield anomalies.
    """

    # 1. Anomaly based on total average over all regions for the same 5-year period
    # Calculate 5-year rolling mean shifted to align with the desired comparison window
    total_mean = (
        ds["ww_yield"]
        .rolling(year=5, center=False)
        .mean()
        .shift(year=2)
        .mean(dim="nuts_id")
    )
    ds["ww_yield_anomaly"] = ds["ww_yield"] - total_mean
    ds["ww_yield_anomaly_percent"] = (ds["ww_yield_anomaly"] / total_mean) * 100

    # 2. Anomaly based on weighted average for the same 5-year period
    weighted_sum = (
        (ds["ww_yield"] * ds["ArabLand"])
        .rolling(year=5, center=False)
        .mean()
        .shift(year=2)
        .sum(dim="nuts_id")
    )
    weights_sum = (
        ds["ArabLand"]
        .rolling(year=5, center=False)
        .mean()
        .shift(year=2)
        .sum(dim="nuts_id")
    )
    weighted_mean = weighted_sum / weights_sum
    ds["ww_yield_anomaly_weighted"] = ds["ww_yield"] - weighted_mean
    ds["ww_yield_anomaly_percent_weighted"] = (
        ds["ww_yield_anomaly_weighted"] / weighted_mean
    ) * 100

    return ds


if __name__ == "__main__":
    client = Client()

    # Define file paths
    file_paths = {
        "spi": "../output/spi_data_monthly.nc",
        "derived_features_monthly": "../output/combined_climate_indices_monthly.nc",
        "derived_features_weekly": "../output/combined_climate_indices_weekly.nc",
        "shapefile": "../masks/NUTS3/NUTS250_N3.shp",
        "crop_mask": "../masks/CropMasks/CTM_2018_WiWh_EPSG4326_654_866_1000m.tif",
        "elevation": "../elevation_data/elevation_amber_conform_med_EPSG4326_654_866_1000m.tif",
        "yield": "../yield_data/Final_data.csv",
        "output": "../output/all_inputs_aggregated_on_nuts_big.nc",
        "allyears_monthly_merged": "../output/allyears_uncompressed_monthly_merged.nc",
    }

    # Load datasets
    datasets = {
        "spi": xr.load_dataset(
            file_paths["spi"],
            chunks={"times": "auto", "lat": "auto", "lon": "auto"},
        ),
        "derived_features_monthly": xr.open_dataset(
            file_paths["derived_features_monthly"],
            chunks={"times": "auto", "lat": "auto", "lon": "auto"},
        ),
        "derived_features_weekly": xr.open_dataset(
            file_paths["derived_features_weekly"],
            chunks={"times": "auto", "lat": "auto", "lon": "auto"},
        ),
        "allyears_monthly_merged": xr.load_dataset(
            file_paths["allyears_monthly_merged"],
            chunks={"times": "auto", "lat": "auto", "lon": "auto"},
        ),
    }

    # remove the 53rd week now. Before finding a better solution
    datasets["derived_features_weekly"] = datasets["derived_features_weekly"].where(
        datasets["derived_features_weekly"].time.dt.isocalendar().week != 53, drop=True
    )

    # Load shapefile and ensure CRS matches the NetCDF
    gdf = gpd.read_file(file_paths["shapefile"]).to_crs("EPSG:4326")

    # Load yield data
    df = pd.read_csv(file_paths["yield"])

    # Create dictionaries to store results
    result_dicts = {
        "valid_coords_by_nuts": {},
        "avg_elevation_by_nuts": {},
        "monthly_features": {},
        "weekly_features": {},
        "monthly_raw_data": {},
    }

    monthly_features = [
        "gdd",
        "frost_days",
        "days_max_temp_above_28",
        "days_avg_temp_above_28",
        "days_in_97_5_percentile_tas",
        "days_in_95_percentile_pr",
        "days_in_95_percentile_rsds",
        "days_in_90_percentile_sfcWind",
        "days_in_95_percentile_sfcWind",
    ]

    weekly_features = [
        "gdd",
        "frost_days",
        "days_max_temp_above_28",
        "days_avg_temp_above_28",
        "days_in_97_5_percentile_tas",
        "days_in_95_percentile_pr",
        "days_in_95_percentile_rsds",
        "days_in_90_percentile_sfcWind",
        "days_in_95_percentile_sfcWind",
    ]

    monthly_raw_data = ["rsds", "pr", "hurs", "tasmax", "tasmin", "tas", "sfcWind"]

    # Initialize result dictionaries
    for feature in monthly_features:
        result_dicts["monthly_features"][feature] = {}
    result_dicts["monthly_features"]["spi"] = {}
    for feature in weekly_features:
        result_dicts["weekly_features"][feature] = {}

    for variable in monthly_raw_data:
        result_dicts["monthly_raw_data"][variable] = {}

    # Group the GeoDataFrame by NUTS code
    nuts_groups = gdf.groupby("NUTS_CODE")

    delayed_computations = []
    # Open raster files for masking
    with rasterio.open(file_paths["crop_mask"]) as crop_mask, rasterio.open(
        file_paths["elevation"]
    ) as elev_raster:
        # Iterate over each NUTS region
        for nuts_code, nuts_group in nuts_groups:
            print(nuts_code)
            # Combine geometries for the current NUTS code
            combined_geometry = nuts_group["geometry"].union_all()

            # Mask and crop raster data
            crop_mask_data, crop_transform = mask(
                crop_mask, [combined_geometry], crop=True
            )
            elev_data, _ = mask(elev_raster, [combined_geometry], crop=True)

            # Find valid pixel coordinates within the crop mask
            valid_pixels = np.where(crop_mask_data[0] == 1)
            lons, lats = transform.xy(crop_transform, valid_pixels[0], valid_pixels[1])
            result_dicts["valid_coords_by_nuts"][nuts_code] = list(zip(lats, lons))

            # Calculate average elevation for valid pixels
            masked_elev = np.ma.masked_where(crop_mask_data[0] == 0, elev_data[0])
            result_dicts["avg_elevation_by_nuts"][nuts_code] = np.ma.mean(masked_elev)

            # Process time series data if enough valid points are available
            if len(result_dicts["valid_coords_by_nuts"][nuts_code]) >= 10:

                def extract_and_average(dataset: xr.Dataset, variable_name: str):
                    time_series_list = [
                        dataset[variable_name].sel(lat=lat, lon=lon, method="nearest")
                        for lat, lon in result_dicts["valid_coords_by_nuts"][nuts_code]
                    ]
                    return xr.concat(time_series_list, dim="point").mean(
                        dim="point", skipna=True
                    )

                result_dicts["monthly_features"]["spi"][nuts_code] = (
                    extract_and_average(datasets["spi"], "spi").compute(client=client)
                )
                # Extract and store time series data
                for feature in monthly_features:
                    result_dicts["monthly_features"][feature][nuts_code] = (
                        extract_and_average(
                            datasets["derived_features_monthly"], f"{feature}_month"
                        ).compute(client=client)
                    )

                for feature in weekly_features:
                    result_dicts["weekly_features"][feature][nuts_code] = (
                        extract_and_average(
                            datasets["derived_features_weekly"], f"{feature}_week"
                        ).compute(client=client)
                    )

                for variable in monthly_raw_data:
                    result_dicts["monthly_raw_data"][variable][nuts_code] = (
                        extract_and_average(
                            datasets["allyears_monthly_merged"], variable
                        )
                    ).compute(client=client)

    # Prepare data for xarray Dataset
    nuts_ids = list(result_dicts["monthly_features"]["gdd"].keys())
    time = result_dicts["monthly_features"]["gdd"][nuts_ids[0]].time
    years = time.dt.year.values
    months = time.dt.month.values

    multi_index = pd.MultiIndex.from_arrays([years, months], names=("year", "month"))
    print(multi_index)
    time = result_dicts["weekly_features"]["gdd"][nuts_ids[0]].time
    years = time.dt.year.values
    weeks = time.dt.isocalendar().week.values

    multi_index_weekly = pd.MultiIndex.from_arrays(
        [years, weeks], names=("year", "week")
    )

    print(multi_index_weekly)

    # Create arrays for values
    def create_value_array(feature_dict, multi_index):
        value_store = np.zeros((len(nuts_ids), len(multi_index)))
        for i, nuts_id in enumerate(nuts_ids):
            value_store[i, :] = feature_dict[nuts_id].values
        return value_store

    # Create multi-index for year and month/week

    monthly_values = {
        feature: create_value_array(
            result_dicts["monthly_features"][feature], multi_index
        )
        for feature in monthly_features
    }
    weekly_values = {
        feature: create_value_array(
            result_dicts["weekly_features"][feature], multi_index_weekly
        )
        for feature in weekly_features
    }
    raw_data_values = {
        variable: create_value_array(
            result_dicts["monthly_raw_data"][variable], multi_index
        )
        for variable in monthly_raw_data
    }

    elevation_values = np.array(
        [result_dicts["avg_elevation_by_nuts"][nuts_id] for nuts_id in nuts_ids]
    )

    # Create xarray Dataset
    data_vars = {
        "elevation": (["nuts_id"], elevation_values),
        **{
            f"{feature}_monthly": (
                ["nuts_id", "year", "month"],
                monthly_values[feature].reshape(len(nuts_ids), -1, 12),
            )
            for feature in monthly_features
        },
        **{
            f"{feature}_weekly": (
                ["nuts_id", "year", "week"],
                weekly_values[feature].reshape(len(nuts_ids), -1, 52),
            )
            for feature in weekly_features
        },
        **{
            variable: (
                ["nuts_id", "year", "month"],
                raw_data_values[variable].reshape(len(nuts_ids), -1, 12),
            )
            for variable in monthly_raw_data
        },
    }

    aggregated_ds = xr.Dataset(
        data_vars,
        coords={
            "nuts_id": nuts_ids,
            "year": np.unique(years),
            "month": np.arange(1, 13),
            "week": np.arange(1, 53),
        },
    )

    # Prepare wheat yield data
    ww_yield = (
        df.query("var == 'ww' and measure == 'yield'")
        .rename(columns={"value": "ww_yield"})
        .set_index(["nuts_id", "year"])[["ww_yield"]]
    )
    arab_land = (
        df.query("var == 'ArabLand' and measure == 'area'")
        .rename(columns={"value": "ArabLand"})
        .set_index(["nuts_id", "year"])[["ArabLand"]]
    )
    yield_df = pd.merge(ww_yield, arab_land, left_index=True, right_index=True)

    # Convert to xarray Dataset and calculate yield anomalies
    yield_ds = calculate_yield_anomaly(xr.Dataset.from_dataframe(yield_df))

    # Merge datasets and save to NetCDF
    ds_final = xr.merge([aggregated_ds, yield_ds])
    ds_final.to_netcdf(file_paths["output"])

    # Verify the structure of the saved NetCDF file
    test_ds = xr.open_dataset(file_paths["output"])
    print(test_ds)
