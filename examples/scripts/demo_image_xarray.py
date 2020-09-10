""" Dmonstration for XImage operations


"""

import astropy.units as u
import numpy
import xarray
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame
from rascil.processing_components.simulation.testing_support import create_test_image

frequency = numpy.linspace(1.0e8, 1.1e8, 8)
phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')

ximg = create_test_image(cellsize=0.001, phasecentre=phasecentre, frequency=frequency,
                         polarisation_frame=PolarisationFrame("stokesIQUV"))

assert numpy.max(ximg.data) > 0.0

print(ximg)

# Show channel 0, stokes I image
import matplotlib.pyplot as plt
ximg2d = ximg.data.sel({"frequency": frequency[0], "polarisation": "I"})
ximg2d.plot.imshow()
plt.show()

# Select I, V polarisations
print(ximg.data.sel({"polarisation": ["I", "V"]}))

# Apply a mask to the data
ximg.data = ximg.data.where(numpy.hypot(ximg.data["lon"] - numpy.mean(ximg.data["lon"]),
                                        ximg.data["lat"] - numpy.mean(ximg.data["lat"])) < 0.03)

print(ximg)
print(numpy.sum(ximg.data))

# Apply a function across the image in chunks using dask transparently
ximg.data = ximg.data.chunk({"lon": 32, "lat": 32})
print(ximg)
ximg.data = xarray.apply_ufunc(numpy.sqrt, ximg.data,
                               dask="parallelized",
                               output_dtypes=[float])

# Show the resulting image
ximg.data = ximg.data.sel({"frequency": frequency[0], "polarisation": "I"})
ximg.data.plot.imshow()
plt.show()
