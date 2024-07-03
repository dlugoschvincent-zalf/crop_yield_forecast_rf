import xarray as xr
import numpy as np

# Open the NetCDF file using xarray
ds = xr.open_dataset("../output/allyears_uncompressed_tasmin.nc")

# Access the temperature variable (replace 'temperature' with the actual variable name)
temperature = ds["tasmin"]


# Define a function to calculate frost days (days with temperature below 0°C)
def count_frost_days(data):
    return (data < 0).resample(time="1ME").sum(dim="time")


# Apply the function to the temperature data, grouping by month
frost_days_per_month = count_frost_days(temperature)

ds_frost_days = xr.Dataset({"frost_days": frost_days_per_month})

ds_frost_days["frost_days"].attrs["long_name"] = "Frost Days"
# Print the results
print(frost_days_per_month.isel(lat=200, lon=300, time=0).values)

print(ds_frost_days)
# Save to NetCDF
ds_frost_days.to_netcdf("frost_days.nc")
# Print the results
np.set_printoptions(suppress=True)
print(ds_frost_days["frost_days"].isel(lat=200, lon=300).values)
