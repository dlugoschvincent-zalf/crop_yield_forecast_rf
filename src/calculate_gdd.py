import xarray as xr
import numpy as np

# Open the NetCDF files containing minimum and maximum temperatures
ds_min = xr.open_dataset(
    "./allyears_uncompressed_tasmin.nc",
    chunks={"time": "auto", "lat": "auto", "lon": "auto"},
)
ds_max = xr.open_dataset(
    "./allyears_uncompressed_tasmax.nc",
    chunks={"time": "auto", "lat": "auto", "lon": "auto"},
)

# Access the temperature variables (replace with actual variable names)
tasmin = ds_min["tasmin"]
tasmax = ds_max["tasmax"]


# Define a function to calculate growing degree days (GDD) per month
def calculate_gdd(tasmin, tasmax, base_temp=10):
    """
    Calculates growing degree days (GDD) per month.

    Args:
    tasmin: Daily minimum temperature DataArray.
    tasmax: Daily maximum temperature DataArray.
    base_temp: Base temperature for GDD calculation (default: 10°C).

    Returns:
    DataArray containing monthly GDD.
    """

    # Cap tasmax at 30 C
    tasmax = tasmax.where(tasmax <= 30, 30)

    # Calculate daily average temperature
    tavg = (tasmin + tasmax) / 2

    # Calculate GDD for each day
    daily_gdd = tavg - base_temp
    daily_gdd = daily_gdd.where(daily_gdd > 0, 0)  # Set negative values to 0

    # Resample to monthly sums
    monthly_gdd = daily_gdd.resample(time="1ME").sum(dim="time")
    return monthly_gdd


# Calculate growing degree days
gdd = calculate_gdd(tasmin, tasmax, base_temp=10)  # Set your desired base temperature

gdd = gdd.compute()

# Extract year and month as separate coordinates

ds_gdd = xr.Dataset({"gdd": gdd})

ds_gdd["gdd"].attrs["long_name"] = "Growing Degree Days"

print(ds_gdd)
# Save to NetCDF
ds_gdd.to_netcdf("growing_degree_days.nc")
# Print the results
np.set_printoptions(suppress=True)
print(ds_gdd["gdd"].isel(lat=200, lon=300).values)
