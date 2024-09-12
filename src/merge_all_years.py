import xarray as xr
import os
import glob
from dask.distributed import Client

BASE_DIR = os.path.join("netcdf_files", "amber")
OUTPUT_DIR = "output"
start_year = 1979
end_year = 2023
YEARS = range(start_year, end_year + 1)  # Updated to include up to 2023
VARIABLES = ["pr", "tasmin", "tasmax", "tas", "rsds", "sfcwind", "hurs"]
VARIABLES_IN_FILE = ["pr", "tasmin", "tasmax", "tas", "rsds", "sfcWind", "hurs"]


if __name__ == "__main__":
    client = Client()
    # Generate list of all files
    all_files = []
    for year in YEARS:
        year_dir = os.path.join(BASE_DIR, str(year))
        if os.path.exists(year_dir):
            for variable in VARIABLES:
                file_pattern = os.path.join(
                    year_dir, f"zalf_{variable}_amber_{year}_v1-0.nc"
                )
                matching_files = glob.glob(file_pattern)
                all_files.extend(matching_files)

    print(f"Found {len(all_files)} files. Merging all files...")
    ds = xr.open_mfdataset(
        all_files, chunks={"time": "auto", "lat": "auto", "lon": "auto"}
    )

    # Set up encoding for compression
    encoding = {var: {"compression": "zstd"} for var in VARIABLES_IN_FILE}

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(
        OUTPUT_DIR, f"{start_year}_{end_year}_allvariables_compressed_merged.nc"
    )

    print("Saving merged and compressed dataset...")
    saver = ds.to_netcdf(output_file, compute=False, encoding=encoding)

    saver.compute(client=client)
    print(f"Saved merged and compressed file to: {output_file}")

    print("Merging and compression completed.")
