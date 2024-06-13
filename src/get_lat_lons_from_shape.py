import geopandas as gpd
import xarray as xr
import numpy as np
from shapely.geometry import Point

# Load the shapefile using geopandas
shapefile_path = "../masks/NUTS3/NUTS250_N3.shp"  # Replace with your shapefile path
gdf = gpd.read_file(shapefile_path)
netcdf_crs = "EPSG:4326"
print(gdf.crs)
gdf = gdf.to_crs(netcdf_crs)
print(gdf.crs)

# Load the NetCDF dataset using xarray
netcdf_path = "../climate_data/amber/2020/zalf_pr_amber_2020_v1-0.nc"
# Replace with your NetCDF path
ds = xr.open_dataset(netcdf_path)

# Select the region you want to extract data for (e.g., by index or name)
region_index = 0  # Replace with the index or name of your target region

selected_region = gdf.iloc[region_index]

print(selected_region)

# Create a boolean mask indicating points within the selected region
lon, lat = np.meshgrid(ds["lon"], ds["lat"])
mask = np.array(
    [
        selected_region.geometry.contains(Point(lon_val, lat_val))
        for lon_val, lat_val in zip(lon.flatten(), lat.flatten())
    ]
)
mask = mask.reshape(lon.shape)

# Extract the latitude and longitude values using the mask
lat_within_region = lat[mask]  # Index the lat grid directly
lon_within_region = lon[mask]  # Index the lon grid directly

print("Latitudes within region:\n", lat_within_region)
print("Longitudes within region:\n", lon_within_region)

# You can now use these lat/lon combinations to extract data from your NetCDF file
# data_within_region = ds["your_variable_name"].where(mask)
