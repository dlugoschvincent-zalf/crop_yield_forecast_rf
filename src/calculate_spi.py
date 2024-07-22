import xarray as xr
import numpy as np
from scipy.stats import gamma, norm
from dask.distributed import Client
import logging
from datetime import datetime

# Set up logging (as in the previous example)
log_filename = (
    f"../logs/spi_calculation_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=log_filename,
    filemode="w",
)

# Add console logging if desired
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


def calculate_spi_chunk(chunk: xr.DataArray):
    """Calculates SPI for a single chunk of data,
    aware that some locations might not have data by design.
    """
    spi_chunk = xr.full_like(chunk, np.nan)
    total_locations = chunk.sizes["lat"] * chunk.sizes["lon"]
    processed_locations = 0
    no_data_locations = 0

    for lat in chunk.lat:
        for lon in chunk.lon:
            location_data = chunk.sel(lat=lat, lon=lon).dropna("time")

            if location_data.size == 0:
                no_data_locations += 1
                continue

            processed_locations += 1
            monthly_data = [
                location_data.sel(time=location_data.time.dt.month == m)
                for m in range(1, 13)
            ]

            for month, month_data in enumerate(monthly_data, start=1):
                if month_data.size == 0:
                    continue

                try:
                    # Handle zero values
                    # If monthly aggregates are zero a zero mask is created to prevent these month to go into the calculation as is. A small epsilon is added to the zeros before fitting the gamma distribution parameters preventing causing a crash in the calculation.
                    zero_mask = month_data.values == 0
                    month_data.values[zero_mask] = 0.000001

                    params = gamma.fit(month_data.values, floc=0)
                    cdf = gamma.cdf(month_data.values, *params)

                    spi_values = norm.ppf(cdf, loc=0, scale=1)
                    spi_chunk.loc[dict(lat=lat, lon=lon, time=month_data.time)] = (
                        spi_values
                    )

                except Exception as e:
                    logging.warning(
                        f"Error processing lat {lat}, lon {lon}, month {month}: {str(e)}"
                    )

    logging.info(
        f"Chunk processing complete. "
        f"Processed locations: {processed_locations}, "
        f"No-data locations: {no_data_locations}, "
        f"Total locations: {total_locations}"
    )

    return spi_chunk


if __name__ == "__main__":
    logging.info("Starting SPI calculation process")
    client = Client()
    logging.info(f"Dask client initialized: {client}")

    ds = xr.open_dataset("../output/allyears_compressed_merged.nc")
    logging.info("Dataset opened successfully")

    # Log original data shape
    logging.info(f"Original data shape: {ds['pr'].shape}")

    # Check for locations with no data
    no_data_mask = ds["pr"].isnull().all(dim="time")
    no_data_count = no_data_mask.sum().values
    logging.info(f"Locations with no data: {no_data_count}")

    processed_precip_data = (
        ds["pr"].resample(time="ME").mean().chunk({"lat": 50, "lon": 50, "time": -1})
    )

    logging.info("Precipitation data processed and chunked")

    # Log processed data shape
    logging.info(f"Processed data shape: {processed_precip_data.shape}")

    # Check if no-data locations are preserved after processing
    processed_no_data_mask = processed_precip_data.isnull().all(dim="time")
    processed_no_data_count = processed_no_data_mask.sum().values
    logging.info(f"Locations with no data after processing: {processed_no_data_count}")

    # Log specific location data (choose a location known to have data)
    lat, lon = 48.25466202036988, 7.711184504946912
    logging.info(f"Sample data at ({lat}, {lon}):")
    logging.info(f"Original: {ds['pr'].sel(lat=lat, lon=lon).values}")
    logging.info(f"Processed: {processed_precip_data.sel(lat=lat, lon=lon).values}")

    logging.info("Starting SPI calculation")
    spi = xr.map_blocks(
        calculate_spi_chunk,
        processed_precip_data,
    )
    spi = spi.compute(client=client)
    logging.info("SPI calculation completed")

    # Log SPI data shape
    logging.info(f"SPI data shape: {spi.shape}")

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
    spi_ds["spi"].attrs["long_name"] = "Standardized Precipitation Index (Quarterly)"
    # spi_ds["spi"].attrs["long_name"] = "Standardized Precipitation Index (Monthly)"

    # Log SPI data for the sample location
    logging.info(f"SPI data at ({lat}, {lon}):")
    logging.info(spi_ds["spi"].sel(lat=lat, lon=lon).values)

    # Check for locations with no SPI data
    spi_no_data_mask = spi_ds["spi"].isnull().all(dim="time")
    spi_no_data_count = spi_no_data_mask.sum().values
    logging.info(f"Locations with no SPI data: {spi_no_data_count}")

    # Compare non-NaN counts
    original_valid_count = (~no_data_mask).sum().values
    processed_valid_count = (~processed_no_data_mask).sum().values
    spi_valid_count = (~spi_no_data_mask).sum().values
    logging.info(
        f"Locations with data: Original: {original_valid_count}, "
        f"Processed: {processed_valid_count}, SPI: {spi_valid_count}"
    )

    # Save to a new NetCDF file
    output_file = "../output/spi_data_monthly.nc"
    spi_ds.to_netcdf(output_file)
    logging.info(f"SPI data saved to {output_file}")

    logging.info("SPI calculation process completed")
