import xarray as xr

from scipy.stats import gamma, norm
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# ds = xr.open_dataset("./spi_data.nc")


# print((ds.time.dt.month == 1)["spi"].values)


# precip_month = ds.isel(lat=80, lon=80, time=ds.time.dt.month == 4)
#
# print(precip_month["spi"].data)
#
ds = xr.open_dataset("./allyears_uncompressed.nc")

point_data = ds["pr"].isel(lat=200, lon=200)
print(point_data.values)
shape_fit, loc_fit, scale_fit = gamma.fit(point_data.values)

# Generate x values for plotting
x = np.linspace(-2, 2, 100)  # Adjust range based on data

# Calculate the PDF of the fitted gamma
y_fit = gamma.pdf(x, a=shape_fit, loc=loc_fit, scale=scale_fit)

# Plot the fitted gamma distribution
plt.plot(x, y_fit, label="Fitted Gamma")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Fitted Gamma Distribution")
plt.grid(True)
plt.legend()

plt.savefig("tmp_gamma.png")
subprocess.call(["kitty", "+kitten", "icat", "--align", "left", "tmp_gamma.png"])
