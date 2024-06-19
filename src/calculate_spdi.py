import xarray as xr
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import subprocess
from scipy.stats import gamma, norm
from dask.distributed import Client


def calculate_spi_chunk(chunk: xr.DataArray):
    """Calculates SPI for a single chunk of data."""
    spi_chunk = xr.full_like(chunk, np.nan)
    for lat in chunk.lat:
        for lon in chunk.lon:
            for i in range(1, 13):
                precip_point = chunk.sel(
                    lat=lat, lon=lon, time=chunk.time.dt.month == i
                )
                valid_data = precip_point.dropna("time")
                if not valid_data.values.size:
                    continue  # Skip if no valid data
                params = gamma.fit(valid_data)
                cdf = gamma.cdf(precip_point, *params)
                spi_chunk.loc[dict(lat=lat, lon=lon, time=chunk.time.dt.month == i)] = (
                    norm.ppf(cdf, loc=0, scale=1)
                )
    return spi_chunk


if __name__ == "__main__":
    client = Client()

    ds = xr.open_dataset("./allyears_uncompressed.nc")

    lat_min, lat_max = 150, 200
    lon_min, lon_max = 150, 200

    ds = ds.isel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    processed_precip_data = (
        ds["pr"].resample(time="ME").mean().chunk({"lat": 10, "lon": 10, "time": -1})
    )

    # processed_precip_data = (
    #     ds["pr"]
    #     .resample(time="QM-DEC")
    #     .mean()
    #     .chunk({"lat": 10, "lon": 10, "time": -1})
    # )

    spi = xr.map_blocks(
        calculate_spi_chunk,
        processed_precip_data,
    )

    spi = spi.compute(client=client)

    # Create a new dataset for SPI
    spi_ds = xr.Dataset()
    spi_ds["spi"] = spi  # Assign the computed SPI

    # Add essential coordinates from the original dataset
    spi_ds = spi_ds.assign_coords(
        time=processed_precip_data.time,
        lat=processed_precip_data.lat,
        lon=processed_precip_data.lon,
    )

    # Optionally add attributes for SPI (e.g., units, description)
    spi_ds["spi"].attrs["standart_name"] = "standardized_precipitation_index"
    spi_ds["spi"].attrs["units"] = "unitless"
    spi_ds["spi"].attrs["long_name"] = "Standardized Precipitation Index (90-day)"

    # Save to a new NetCDF file
    spi_ds.to_netcdf("spi_data.nc")

    spi_entry = spi_ds["spi"].isel(lat=40, lon=40)

    spi_entry.to_dataframe().to_csv("test_output_monthly_spi.csv")

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
