import xarray as xr
import numpy as np
from scipy.stats import gamma
import scipy.stats as stats
import matplotlib.pyplot as plt
import subprocess


def calculate_spi_dataset(ds: xr.Dataset, scale, variable_name="pr", max_points=100):
    """
    Calculates the Standardized Precipitation Index (SPI) and adds it as a new variable to the input dataset.

    Args:
        ds (xr.Dataset): The input dataset containing precipitation data.
        scale (int): Time scale in months for SPI calculation.
        min_valid_points (int): Minimum number of valid data points for SPI calculation.
        variable_name (str): The name of the precipitation variable in the dataset.
        max_points (int, optional): Maximum number of grid points to process. If None, processes all points.

    Returns:
        xr.Dataset: The input dataset with a new variable 'spi' containing SPI values.
    """

    precip = ds[variable_name]
    precip_rolling = precip.rolling(time=scale, center=False).sum()
    spi = xr.full_like(precip_rolling, np.nan)

    count = 0  # Initialize point counter
    for lat in precip.lat:
        for lon in precip.lon:
            if max_points is not None and count >= max_points:
                print("Reached maximum point limit. Exiting loop.")
                ds["spi"] = spi  # Assign to the 'spi' variable in the original dataset
                return ds

            precip_point = precip_rolling.sel(lat=lat, lon=lon)
            valid_data = precip_point.dropna("time")

            if not valid_data.values.size:
                spi.loc[dict(lat=lat, lon=lon)] = np.nan
                continue

            # print(f"Lat:{lat.data}, Lon: {lon.data}")

            params = gamma.fit(valid_data)
            cdf = gamma.cdf(precip_point, *params)
            spi.loc[dict(lat=lat, lon=lon)] = stats.norm.ppf(cdf, loc=0, scale=1)

            count += 1

    ds["spi"] = spi  # Assign SPI data to the new 'spi' variable

    return ds


# Example usage:
ds = xr.open_mfdataset(
    [
        "../climate_data/amber/2015/zalf_pr_amber_2015_v1-0.nc",
        "../climate_data/amber/2016/zalf_pr_amber_2016_v1-0.nc",
        "../climate_data/amber/2017/zalf_pr_amber_2017_v1-0.nc",
        "../climate_data/amber/2018/zalf_pr_amber_2018_v1-0.nc",
        "../climate_data/amber/2019/zalf_pr_amber_2019_v1-0.nc",
        "../climate_data/amber/2020/zalf_pr_amber_2020_v1-0.nc",
        "../climate_data/amber/2021/zalf_pr_amber_2021_v1-0.nc",
        "../climate_data/amber/2022/zalf_pr_amber_2022_v1-0.nc",
        "../climate_data/amber/2023/zalf_pr_amber_2023_v1-0.nc",
    ]
)
ds.load()
ds = calculate_spi_dataset(ds, scale=90, variable_name="pr", max_points=100)

# Now the original 'ds' dataset has a new variable called 'spi'

np.set_printoptions(suppress=True)
print(len(ds["spi"].sel(lat="54.908170918201535", lon="8.70730856035551").values))
ds["spi"].sel(lat="54.908170918201535", lon="8.70730856035551").to_dataframe().to_csv(
    "54.908170918201535_8.70730856035551_2018_rolling_90days.csv"
)

# Select the specific point (latitude, longitude)
selected_lat = "54.908170918201535"
selected_lon = "8.70730856035551"
spi_point = ds["spi"].sel(lat=selected_lat, lon=selected_lon)

# Create the time series plot
plt.figure(figsize=(12, 6))
plt.plot(spi_point.time, spi_point.values)

plt.title(f"3-Month SPI at Latitude: {selected_lat}, Longitude: {selected_lon}")
plt.xlabel("Time")
plt.ylabel("SPI")
plt.grid(True)
plt.axhline(
    y=0, color="black", linestyle="--", linewidth=0.8
)  # Add a horizontal line at SPI=0

# plt.show()
plt.savefig("tmp.png")
subprocess.call(["kitty", "+kitten", "icat", "--align", "left", "tmp.png"])
