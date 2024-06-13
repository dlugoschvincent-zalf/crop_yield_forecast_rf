import xarray as xr
import numpy as np
import pprint

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

selected_lat = "54.908170918201535"
selected_lon = "8.70730856035551"
pr_point = ds["pr"].sel(lat=selected_lat, lon=selected_lon)
# Calculate the mean precipitation for each year
mean_precip_yearly = pr_point.groupby("time.year").mean("time")

# Print the result

# print(mean_precip_yearly.values)
np.set_printoptions(threshold=1000000)

pprint.pp(ds.variables["latitude_longitude"].attrs)
