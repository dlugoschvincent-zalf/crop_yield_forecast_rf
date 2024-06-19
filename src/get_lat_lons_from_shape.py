import geopandas as gpd
import xarray as xr
from shapely.geometry import Point
import rasterio
from rasterio import transform
from rasterio.mask import mask
import numpy as np

# Load shapefile and ensure CRS matches the NetCDF
shapefile_path = "../masks/NUTS3/NUTS250_N3.shp"
gdf = gpd.read_file(shapefile_path)
netcdf_crs = "EPSG:4326"
gdf = gdf.to_crs(netcdf_crs)

# Load NetCDF dataset (if you need data from it later)
netcdf_path = "../climate_data/amber/2020/zalf_pr_amber_2020_v1-0.nc"
ds = xr.open_dataset(netcdf_path)

# Load GeoTIFF
crop_mask_file = "../masks/CropMasks/CTM_2018_WiWh_EPSG4326_654_866_1000m.tif"

# Create the dictionary to store results
valid_lat_lon_by_nuts = {}

# Use spatial indexing to speed up intersection checks
gdf.sindex

with rasterio.open(crop_mask_file) as src:
    for index, row in gdf.iterrows():
        nuts_code = row["NUTS_CODE"]
        geometry = row["geometry"]

        # Crop the raster data to the bounding box of the current NUTS region
        crop_mask, crop_transform = mask(src, [geometry], crop=True)

        # Find the valid pixels within the cropped region
        valid_pixels = np.where(crop_mask[0] == 1)

        # Convert pixel coordinates to lat/lon
        lons, lats = transform.xy(crop_transform, valid_pixels[0], valid_pixels[1])

        valid_lat_lon_by_nuts.setdefault(nuts_code, []).extend(zip(lats, lons))

# Print or use the dictionary
# print(valid_lat_lon_by_nuts)

print(valid_lat_lon_by_nuts["DEG0N"])
