import xarray as xr
from dask.distributed import Client

if __name__ == "__main__":
    # Set up Dask client
    client = Client()

    ds = xr.open_dataset(
        "../output/allyears_compressed_merged.nc",
        chunks={"time": "auto", "lat": "auto", "lon": "auto"},
    )

    ds_monthly = (
        ds.resample(time="1ME").mean(dim="time", skipna=True).compute(client=client)
    )

    # Rename variables by adding "_monthly"
    ds_monthly = ds_monthly.rename_vars(
        {var: f"{var}_monthly" for var in ds_monthly.data_vars}
    )

    saver = ds_monthly.to_netcdf(
        "../output/allyears_uncompressed_monthly_merged.nc", compute=False
    )
    saver.compute(client=client)
