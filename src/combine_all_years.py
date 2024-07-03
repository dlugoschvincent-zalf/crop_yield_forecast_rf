import xarray as xr

ds_path_list = [
    "../climate_data/amber/2007/zalf_tasmax_amber_2007_v1-0.nc",
    "../climate_data/amber/2008/zalf_tasmax_amber_2008_v1-0.nc",
    "../climate_data/amber/2009/zalf_tasmax_amber_2009_v1-0.nc",
    "../climate_data/amber/2010/zalf_tasmax_amber_2010_v1-0.nc",
    "../climate_data/amber/2011/zalf_tasmax_amber_2011_v1-0.nc",
    "../climate_data/amber/2012/zalf_tasmax_amber_2012_v1-0.nc",
    "../climate_data/amber/2013/zalf_tasmax_amber_2013_v1-0.nc",
    "../climate_data/amber/2014/zalf_tasmax_amber_2014_v1-0.nc",
    "../climate_data/amber/2015/zalf_tasmax_amber_2015_v1-0.nc",
    "../climate_data/amber/2016/zalf_tasmax_amber_2016_v1-0.nc",
    "../climate_data/amber/2017/zalf_tasmax_amber_2017_v1-0.nc",
    "../climate_data/amber/2018/zalf_tasmax_amber_2018_v1-0.nc",
    "../climate_data/amber/2019/zalf_tasmax_amber_2019_v1-0.nc",
    "../climate_data/amber/2020/zalf_tasmax_amber_2020_v1-0.nc",
    "../climate_data/amber/2021/zalf_tasmax_amber_2021_v1-0.nc",
    "../climate_data/amber/2022/zalf_tasmax_amber_2022_v1-0.nc",
    "../climate_data/amber/2023/zalf_tasmax_amber_2023_v1-0.nc",
]
ds_list = []
for ds_path in ds_path_list:
    ds_list.append(xr.open_dataset(ds_path))
ds = xr.concat(ds_list, dim="time")

ds.to_netcdf(
    "../output/allyears_uncompressed_tasmax.nc", encoding={"tasmax": {"zlib": False}}
)
print("saved")
