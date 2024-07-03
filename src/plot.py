import xarray as xr
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import subprocess
from scipy.stats import gamma, norm
from dask.distributed import Client
import pandas as pd
from datetime import datetime

spi_ds = xr.load_dataset("./spi_and_elevation_data_by_nuts_masked_ww.nc")
print(spi_ds)
spi_entry = spi_ds["spi"].sel(nuts_id="DE80J")

spi_entry.to_dataframe().to_csv("test_output_seasonal_spi.csv")
print(spi_entry.values.flatten())

# Create the time series plot
plt.figure(figsize=(12, 6))

start_date = pd.to_datetime(f"{spi_entry.year.values.min()}-01-01")
end_date = pd.to_datetime(f"{spi_entry.year.values.max()}-12-01")
monthly_dates = pd.date_range(start_date, end_date, freq="MS")

plt.plot(monthly_dates, spi_entry.values.flatten())

plt.title(f"3-Month SPI at: {spi_entry.nuts_id}")
plt.xlabel("Time")
plt.ylabel("SPI")
plt.grid(True)
plt.axhline(
    y=0, color="black", linestyle="--", linewidth=0.8
)  # Add a horizontal line at SPI=0

plt.savefig("tmp.png")
subprocess.call(["kitty", "+kitten", "icat", "--align", "left", "tmp.png"])
