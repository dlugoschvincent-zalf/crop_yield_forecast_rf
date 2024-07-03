import xarray as xr
import numpy as np


def calculate_yearly_average_precipitation(input_file, output_file):
    # Open the NetCDF file
    ds = xr.open_dataset(input_file)

    # Identify the precipitation variable
    # You may need to change 'precipitation' to the actual variable name in your file
    precip_var = "pr"

    # Calculate yearly average
    yearly_avg = ds[precip_var].groupby("time.year").mean("time")

    # Create a new dataset with the yearly averages
    ds_yearly = xr.Dataset({"yearly_avg_precipitation": yearly_avg})

    # Add attributes
    ds_yearly["yearly_avg_precipitation"].attrs["units"] = ds[precip_var].attrs.get(
        "units", "unknown"
    )
    ds_yearly["yearly_avg_precipitation"].attrs[
        "long_name"
    ] = "Yearly average precipitation"

    # Save to a new NetCDF file
    ds_yearly.to_netcdf(output_file)

    print(f"Yearly average precipitation saved to {output_file}")


# Usage
input_file = "../output/allyears_uncompressed_pr.nc"
output_file = "../output/yearly_avg_pr.nc"

calculate_yearly_average_precipitation(input_file, output_file)
