import xarray as xr
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import subprocess
from scipy.stats import gamma, norm
from dask.distributed import Client


def calculate_monthly_spi_chunk(chunk: xr.DataArray, month: int) -> xr.DataArray:
    """
    Calculates the Standardized Precipitation Index (SPI) for a specific month
    across all years in a chunk of precipitation data.

    Args:
        chunk (xr.DataArray): A chunk of precipitation data with dimensions
                              (time, lat, lon).
        month (int): The month for which to calculate SPI (1=January, 12=December).

    Returns:
        xr.DataArray: A chunk of SPI values for the specified month, with
                      dimensions (time, lat, lon).
    """

    spi_chunk = xr.full_like(chunk, np.nan)

    for lat in chunk.lat:
        for lon in chunk.lon:
            # Select data for the specified month at this grid point
            precip_monthly = chunk.sel(lat=lat, lon=lon).sel(
                time=chunk.time.dt.month == month
            )

            # Remove missing values
            valid_data = precip_monthly.dropna("time")

            # Skip if no valid data for this month at this grid point
            if not valid_data.size:
                continue

            # Fit a Gamma distribution to the valid monthly precipitation data
            params = gamma.fit(valid_data)

            # Calculate the CDF of the monthly precipitation values
            cdf = gamma.cdf(precip_monthly, *params)

            # Transform CDF into SPI values
            spi_chunk.loc[dict(lat=lat, lon=lon, time=precip_monthly.time)] = norm.ppf(
                cdf, loc=0, scale=1
            )

    return spi_chunk


if __name__ == "__main__":
    client = Client()

    # Open the NetCDF file with chunking
    ds = xr.open_dataset("./allyears_uncompressed.nc", chunks={"lat": 25, "lon": 25})

    # Select spatial subset
    lat_min, lat_max = 100, 200
    lon_min, lon_max = 100, 200
    ds = ds.isel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Calculate monthly SPI for each month
    spi_monthly = []
    for month in range(1, 13):  # Iterate over months 1 to 12
        spi_month = xr.map_blocks(calculate_monthly_spi_chunk, ds["pr"], args=[month])
        spi_monthly.append(spi_month.compute(client=client))

    # Combine monthly SPI results into a single Dataset
    spi_ds = xr.Dataset(
        {f"spi_{month:02d}": spi for month, spi in enumerate(spi_monthly, start=1)},
        coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
    )

    # --- Saving the SPI Data ---
    # Add metadata attributes to the SPI variables
    for month in range(1, 13):
        spi_ds[f"spi_{month:02d}"].attrs[
            "standart_name"
        ] = "standardized_precipitation_index"
        spi_ds[f"spi_{month:02d}"].attrs["units"] = "unitless"
        spi_ds[f"spi_{month:02d}"].attrs[
            "long_name"
        ] = f"Standardized Precipitation Index (monthly, month={month:02d})"

    # Save the SPI data to a new NetCDF file
    spi_ds.to_netcdf("spi_data_monthly.nc")

    spi_entry = spi_ds["spi"].isel(lat=80, lon=80)

    spi_entry.to_dataframe().to_csv("120_120_rolling_90days.csv")

    # Create the time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(spi_entry.time, spi_entry.values)

    plt.title(
        f"3-Month SPI at Latitude: {spi_entry.lat.data}, Longitude: {spi_entry.lon.data}"
    )
    plt.xlabel("Time")
    plt.ylabel("SPI")
    plt.grid(True)
    plt.axhline(
        y=0, color="black", linestyle="--", linewidth=0.8
    )  # Add a horizontal line at SPI=0

    plt.savefig("tmp.png")
    subprocess.call(["kitty", "+kitten", "icat", "--align", "left", "tmp.png"])
