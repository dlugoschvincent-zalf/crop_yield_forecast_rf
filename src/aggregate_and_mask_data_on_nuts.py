import geopandas as gpd
import xarray as xr
import rasterio
from rasterio import transform
from rasterio.mask import mask
import numpy as np
import pandas as pd


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


# Read and prepare the data

# Define file paths
precip_file = "../output/yearly_avg_precipitation.nc"
gdd_file = "../output/growing_degree_days.nc"
frost_days_file = "../output/frost_days.nc"
spi_file = "../output/spi_data_monthly.nc"
shapefile_path = "../masks/NUTS3/NUTS250_N3.shp"
crop_mask_file = "../masks/CropMasks/CTM_2018_WiWh_EPSG4326_654_866_1000m.tif"
elevation_file = (
    "../elevation_data/elevation_amber_conform_med_EPSG4326_654_866_1000m.tif"
)
yield_file = "../yield_data/Final_data.csv"
output_file = "../output/all_inputs_aggregated_on_nuts.nc"

# Load datasets
precip_ds = xr.load_dataset(precip_file)
gdd_ds = xr.load_dataset(gdd_file)
frost_days_ds = xr.load_dataset(frost_days_file)
spi_ds = xr.load_dataset(spi_file)

# Load shapefile and ensure CRS matches the NetCDF
gdf = gpd.GeoDataFrame(gpd.read_file(filename=shapefile_path))
netcdf_crs = "EPSG:4326"
gdf = gdf.to_crs(netcdf_crs)

# Load yield data
df = pd.read_csv(yield_file)

# Create dictionaries to store results
valid_coords_by_nuts = {}  # Store valid lat/lon pairs for each NUTS region
avg_elevation_by_nuts = {}
mean_spi_by_nuts = {}
mean_precip_by_nuts = {}
mean_gdd_by_nuts = {}
mean_frost_days_by_nuts = {}


# Group the GeoDataFrame by NUTS code
if gdf is None:
    raise SystemExit("GDF not defined")

nuts_groups = gdf.groupby("NUTS_CODE")

# Open raster files for masking
with rasterio.open(crop_mask_file) as crop_mask, rasterio.open(
    elevation_file
) as elev_raster:
    # Iterate over each NUTS region
    for nuts_code, nuts_group in nuts_groups:
        # Combine geometries for the current NUTS code
        combined_geometry = nuts_group["geometry"].union_all()

        # Mask and crop raster data
        crop_mask_data, crop_transform = mask(crop_mask, [combined_geometry], crop=True)
        elev_data, _ = mask(elev_raster, [combined_geometry], crop=True)

        # Find valid pixel coordinates within the crop mask
        valid_pixels = np.where(crop_mask_data[0] == 1)
        lons, lats = transform.xy(crop_transform, valid_pixels[0], valid_pixels[1])
        valid_coords_by_nuts[nuts_code] = list(zip(lats, lons))

        # Calculate average elevation for valid pixels
        masked_elev = np.ma.masked_where(crop_mask_data[0] == 0, elev_data[0])
        avg_elevation_by_nuts[nuts_code] = np.ma.mean(masked_elev)

        # Process time series data if enough valid points are available
        if len(valid_coords_by_nuts[nuts_code]) >= 10:
            # Function to extract and average time series data
            def extract_and_average(dataset, variable_name):
                time_series_list = []
                for lat, lon in valid_coords_by_nuts[nuts_code]:
                    time_series = dataset[variable_name].sel(
                        lat=lat, lon=lon, method="nearest"
                    )
                    time_series_list.append(time_series)
                time_series_array = xr.concat(time_series_list, dim="point")
                return time_series_array.mean(dim="point", skipna=True)

            # Extract and store time series data
            mean_spi_by_nuts[nuts_code] = extract_and_average(spi_ds, "spi")

            mean_precip_by_nuts[nuts_code] = extract_and_average(
                precip_ds, "yearly_avg_precipitation"
            )

            mean_gdd_by_nuts[nuts_code] = extract_and_average(gdd_ds, "gdd")

            mean_frost_days_by_nuts[nuts_code] = extract_and_average(
                frost_days_ds, "frost_days"
            )


# Prepare data for xarray Dataset
nuts_ids = list(mean_spi_by_nuts.keys())
time = mean_spi_by_nuts[nuts_ids[0]].time
years = time.dt.year.values
months = time.dt.month.values

# Create multi-index for year and month
multi_index = pd.MultiIndex.from_arrays([years, months], names=("year", "month"))

# Initialize arrays to store data
spi_values = np.zeros((len(nuts_ids), len(multi_index)))
precip_values = np.zeros((len(nuts_ids), len(np.unique(years))))
gdd_values = np.zeros((len(nuts_ids), len(multi_index)))
frost_days_values = np.zeros((len(nuts_ids), len(multi_index)))
elevation_values = np.array([avg_elevation_by_nuts[nuts_id] for nuts_id in nuts_ids])


# Populate data arrays
for i, nuts_id in enumerate(nuts_ids):
    spi_values[i, :] = mean_spi_by_nuts[nuts_id].values
    precip_values[i, :] = mean_precip_by_nuts[nuts_id].values
    gdd_values[i, :] = mean_gdd_by_nuts[nuts_id].values
    frost_days_values[i, :] = mean_frost_days_by_nuts[nuts_id].values

# Create xarray Dataset
aggregated_ds = xr.Dataset(
    {
        "spi": (
            ["nuts_id", "year", "month"],
            spi_values.reshape(len(nuts_ids), -1, 12),
        ),
        "elevation": (["nuts_id"], elevation_values),
        "precip": (["nuts_id", "year"], precip_values),
        "gdd": (
            ["nuts_id", "year", "month"],
            gdd_values.reshape(len(nuts_ids), -1, 12),
        ),
        "frost_days": (
            ["nuts_id", "year", "month"],
            frost_days_values.reshape(len(nuts_ids), -1, 12),
        ),
    },
    coords={"nuts_id": nuts_ids, "year": np.unique(years), "month": np.arange(1, 13)},
)

# Prepare wheat yield data
ww_yield = (
    df.query("var == 'ww' and measure == 'yield'")
    .rename(columns={"value": "ww_yield"})
    .set_index(["nuts_id", "year"])[["ww_yield"]]
)

# Prepare arable land data
arab_land = (
    df.query("var == 'ArabLand' and measure == 'area'")
    .rename(columns={"value": "ArabLand"})
    .set_index(["nuts_id", "year"])[["ArabLand"]]
)

# Merge the yield dataframes
yield_df = pd.merge(ww_yield, arab_land, left_index=True, right_index=True)

# Convert to xarray Dataset
yield_ds = xr.Dataset.from_dataframe(yield_df)

# Calculate and add the yield anomalies
yield_ds = calculate_yield_anomaly(yield_ds)

ds_final = xr.merge([aggregated_ds, yield_ds])

# Save the dataset to a NetCDF file
ds_final.to_netcdf(output_file)

# Verify the structure of the saved NetCDF file
test_ds = xr.open_dataset(output_file)
print(test_ds)
