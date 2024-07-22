import xarray as xr
import dask.array as da
from scipy.stats import gamma
import numpy as np
import warnings


def calculate_gamma_parameters(ds, pr_var="pr"):
    """
    Calculates the shape (alpha) and scale (beta) parameters of a gamma distribution
    for each latitude-longitude point in a dataset containing precipitation data.

    Args:
        ds (xarray.Dataset): Input dataset containing precipitation data.
        pr_var (str): Name of the precipitation variable in the dataset (default: 'pr').

    Returns:
        xarray.Dataset: A dataset with the same coordinates as the input dataset,
                         containing the calculated gamma distribution parameters:
                         - alpha: Shape parameter.
                         - beta: Scale parameter.
    """

    def _gamma_params_chunk(data, min_data_points=5, eps=1e-8):
        """Calculate gamma parameters, handling potential errors."""

        valid_data = data[~np.isnan(data)]
        if len(valid_data) >= min_data_points:
            # 1. Add a small epsilon to handle zeros:
            valid_data = valid_data + eps

            # 2. Try/Except to catch fitting errors:
            try:
                alpha, _, beta = gamma.fit(valid_data, floc=0)
            except (ValueError, RuntimeWarning) as e:
                warnings.warn(f"Gamma fitting failed: {e}")
                alpha, beta = np.nan, np.nan

            return np.array([alpha, beta])
        else:
            return np.array([np.nan, np.nan])

    # Calculate gamma parameters using dask.array.apply_along_axis for parallelization
    data_dims = ds[pr_var].dims[1:]  # Dimensions excluding time
    gamma_params = da.apply_along_axis(
        _gamma_params_chunk,
        axis=0,
        arr=ds[pr_var].data,
    )

    # Create a new dataset to store the parameters
    ds_params = xr.Dataset(
        {
            "alpha": (data_dims, gamma_params[:, 0]),
            "beta": (data_dims, gamma_params[:, 1]),
        },
        coords=ds.coords,
    )

    return ds_params


# Example usage:
# Assuming your netCDF file is named 'precip_data.nc'
ds = xr.open_dataset(
    "../output/allyears_compressed_merged.nc", chunks={"time": "auto"}
)  # Adjust chunk size as needed
ds_gamma = calculate_gamma_parameters(ds, pr_var="pr")
ds_gamma.to_netcdf("gamma_parameters.nc")  # Save results to a new netCDF file
