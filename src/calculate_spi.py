import xarray as xr
import numpy as np
from scipy.stats import gamma, norm
from dask.distributed import Client


def calculate_spi_monthly(dataset: xr.DataArray):
    spi_ds = xr.full_like(dataset, np.nan)
    print("Processing Monthly SPI")

    count = 0
    for lat in dataset.lat:
        for lon in dataset.lon:
            location_data = dataset.sel(lat=lat, lon=lon).dropna("time")
            count += 1
            if location_data.size == 0:
                continue
            for month in range(1, 13):
                monthly_data = location_data.sel(
                    time=location_data.time.dt.month == month
                )
                if monthly_data.size == 0:
                    continue

                # Handle zero values
                # monthly_data = monthly_data.where(monthly_data != 0, 0.000001)

                params = gamma.fit(monthly_data.values, floc=0)
                cdf = gamma.cdf(monthly_data.values, *params)

                spi_values = norm.ppf(cdf, loc=0, scale=1)
                if count % 100 == 0:
                    print(f"Position: {count}")
                    print(spi_values)
                spi_ds.loc[dict(lat=lat, lon=lon, time=monthly_data.time)] = spi_values
    return spi_ds


def calculate_spi_quarterly(dataset: xr.DataArray):
    spi_ds = xr.full_like(dataset, np.nan)
    print("Processing Quarterly SPI")

    count = 0
    for lat in dataset.lat:
        for lon in dataset.lon:
            location_data = dataset.sel(lat=lat, lon=lon).dropna("time")

            count += 1
            if location_data.size == 0:
                continue

            for quarter in range(1, 5):
                quarterly_data = location_data.sel(
                    time=location_data.time.dt.quarter == quarter
                )
                if quarterly_data.size == 0:
                    continue

                # Handle zero values
                # quarterly_data = quarterly_data.where(quarterly_data != 0, 0.000001)

                params = gamma.fit(quarterly_data.values, floc=0)
                cdf = gamma.cdf(quarterly_data.values, *params)

                spi_values = norm.ppf(cdf, loc=0, scale=1)
                if count % 100 == 0:
                    print(f"Position: {count}")
                    print(spi_values)
                spi_ds.loc[dict(lat=lat, lon=lon, time=quarterly_data.time)] = (
                    spi_values
                )
    return spi_ds


if __name__ == "__main__":
    client = Client()

    ds = xr.open_dataset(
        "../../netcdf_testing/output/1979_2023_allvariables_compressed.nc",
        chunks={"lat": "auto", "lon": "auto", "time": "auto"},
    )

    print("Opened Dataset")
    # processed_precip_data_monthly = ds["pr"].resample(time="ME").mean()
    # processed_precip_data_monthly.load()
    processed_precip_data_quarterly = ds["pr"].resample(time="QE").mean()
    processed_precip_data_quarterly.load()

    print("Resampled Monthly")

    # spi_monthly = calculate_spi_monthly(processed_precip_data_monthly)
    spi_quarterly = calculate_spi_quarterly(processed_precip_data_quarterly)

    # Create a new dataset for SPI
    spi_ds = xr.Dataset()
    # spi_ds["spi_monthly"] = spi_monthly
    spi_ds["spi_quarterly"] = spi_quarterly

    spi_ds = spi_ds.assign_coords(
        time=processed_precip_data_quarterly.time,
        lat=processed_precip_data_quarterly.lat,
        lon=processed_precip_data_quarterly.lon,
    )

    # spi_ds["spi_monthly"].attrs["standard_name"] = "standardized_precipitation_index"
    # spi_ds["spi_monthly"].attrs["units"] = "unitless"
    # spi_ds["spi_monthly"].attrs[
    #     "long_name"
    # ] = "Standardized Precipitation Index (Monthly)"

    spi_ds["spi_quarterly"].attrs["standard_name"] = "standardized_precipitation_index"
    spi_ds["spi_quarterly"].attrs["units"] = "unitless"
    spi_ds["spi_quarterly"].attrs[
        "long_name"
    ] = "Standardized Precipitation Index (Quarterly)"

    # Save to a new NetCDF file
    output_file = "../output/spi_data_quarterly_new.nc"
    spi_ds.to_netcdf(output_file)
