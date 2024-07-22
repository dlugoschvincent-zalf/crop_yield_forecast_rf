import xarray as xr
from dask.diagnostics.progress import ProgressBar

ds_path_list = [
    "../climate_data/amber/2007/zalf_merged_amber_2007_v1-0.nc",
    "../climate_data/amber/2008/zalf_merged_amber_2008_v1-0.nc",
    "../climate_data/amber/2009/zalf_merged_amber_2009_v1-0.nc",
    "../climate_data/amber/2010/zalf_merged_amber_2010_v1-0.nc",
    "../climate_data/amber/2011/zalf_merged_amber_2011_v1-0.nc",
    "../climate_data/amber/2012/zalf_merged_amber_2012_v1-0.nc",
    "../climate_data/amber/2013/zalf_merged_amber_2013_v1-0.nc",
    "../climate_data/amber/2014/zalf_merged_amber_2014_v1-0.nc",
    "../climate_data/amber/2015/zalf_merged_amber_2015_v1-0.nc",
    "../climate_data/amber/2016/zalf_merged_amber_2016_v1-0.nc",
    "../climate_data/amber/2017/zalf_merged_amber_2017_v1-0.nc",
    "../climate_data/amber/2018/zalf_merged_amber_2018_v1-0.nc",
    "../climate_data/amber/2019/zalf_merged_amber_2019_v1-0.nc",
    "../climate_data/amber/2020/zalf_merged_amber_2020_v1-0.nc",
    "../climate_data/amber/2021/zalf_merged_amber_2021_v1-0.nc",
    "../climate_data/amber/2022/zalf_merged_amber_2022_v1-0.nc",
    "../climate_data/amber/2023/zalf_merged_amber_2023_v1-0.nc",
]

ds = xr.open_mfdataset(ds_path_list)
encoding = {
    "pr": {"compression": "zstd"},
    "hurs": {"compression": "zstd"},
    "sfcWind": {"compression": "zstd"},
    "tasmin": {"compression": "zstd"},
    "tasmax": {"compression": "zstd"},
    "tas": {"compression": "zstd"},
    "rsds": {"compression": "zstd"},
}
saver = ds.to_netcdf(
    "../output/allyears_compressed_merged.nc", compute=False, encoding=encoding
)
with ProgressBar():
    results = saver.compute()
print("saved")
