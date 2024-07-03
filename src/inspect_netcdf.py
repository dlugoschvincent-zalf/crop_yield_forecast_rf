import xarray as xr
import numpy as np


ds = xr.open_dataset("../output/all_inputs_aggregated_on_nuts.nc")
np.set_printoptions(suppress=True)

print(ds["gdd"].sel(nuts_id="DE80J", year=2022).values)

print(ds["frost_days"].sel(nuts_id="DE80J", year=2022).values)

print(ds["spi"].sel(nuts_id="DE80J", year=2022).values)

print(ds["precip"].sel(nuts_id="DE80J", year=2022).values)
