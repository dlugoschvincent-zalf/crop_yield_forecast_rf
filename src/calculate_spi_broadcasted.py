import xarray as xr
import numpy as np
from scipy.stats import gamma, norm
from dask.distributed import Client


def calculate_spi(data: xr.DataArray):
    """Calculates SPI for the entire dataset using broadcasting."""

    # Calculate monthly data for all locations at once
    monthly_data = data.groupby("time.month")

    # Calculate SPI for each month using broadcasting
    spi = xr.apply_ufunc(
        _calculate_spi_for_month,
        monthly_data,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    return spi


def _calculate_spi_for_month(month_data: np.ndarray):
    """Calculates SPI for a single month across all locations.

    This function is intended to be used with xr.apply_ufunc.
    """
    # Handle zero values
    zero_mask = month_data == 0
    if np.all(zero_mask):
        return np.full_like(month_data, np.nan)

    if np.any(zero_mask):
        non_zero_data = month_data[~zero_mask]
        params = gamma.fit(non_zero_data, floc=0)
        cdf = gamma.cdf(month_data, *params)
        cdf[zero_mask] = gamma.cdf(np.min(non_zero_data) / 2, *params)
    else:
        params = gamma.fit(month_data, floc=0)
        cdf = gamma.cdf(month_data, *params)

    spi_values = norm.ppf(cdf, loc=0, scale=1)
    return spi_values


if __name__ == "__main__":
    client = Client()

    ds = xr.open_dataset("../output/allyears_compressed_merged.nc")

    processed_precip_data = ds["pr"].resample(time="ME").mean()

    # Calculate SPI using broadcasting
    spi = calculate_spi(processed_precip_data)

    # Create a new dataset for SPI
    spi_ds = xr.Dataset()
    spi_ds["spi"] = spi
    spi_ds = spi_ds.assign_coords(
        time=processed_precip_data.time,
        lat=processed_precip_data.lat,
        lon=processed_precip_data.lon,
    )
    spi_ds["spi"].attrs["standard_name"] = "standardized_precipitation_index"
    spi_ds["spi"].attrs["units"] = "unitless"
    spi_ds["spi"].attrs["long_name"] = "Standardized Precipitation Index (90-day)"

    # Save to a new NetCDF file
    output_file = "../output/spi_data_monthly_br.nc"
    spi_ds.to_netcdf(output_file)
