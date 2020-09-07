""" Unit tests for XImage operations


"""

import unittest
import dask

import xarray
import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame
from rascil.data_models.memory_data_models_xarray import XImage
from rascil.processing_components.simulation.testing_support import create_test_image


class TestXimage(unittest.TestCase):
    
    def test_create_ximage(self):
        frequency = numpy.linspace(1.0e8, 1.1e8, 8)
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        
        im = create_test_image(cellsize=0.001, phasecentre=phasecentre, frequency=frequency,
                               polarisation_frame=PolarisationFrame("stokesIQUV"))
        ximg = XImage(data=im.data, wcs=im.wcs, frequency=frequency, phasecentre=phasecentre,
                      polarisation_frame=PolarisationFrame("stokesIQUV"))
        
        print("\nInitial ximg")
        print(ximg)
        
        import matplotlib.pyplot as plt
        ximg2d = ximg.data.sel({"frequency": frequency[0], "polarisation": "I"})
        ximg2d.plot.imshow()
        plt.show()

        print("\nSelection by polarisation")
        print(ximg.data.sel({"polarisation": ["I", "V"]}))
        print("\nMasking")
        ximg.data = ximg.data.where(numpy.hypot(ximg.data["lon"] - numpy.mean(ximg.data["lon"]),
                                      ximg.data["lat"] - numpy.mean(ximg.data["lat"])) < 6.0)
        
        print(ximg)
        print(numpy.sum(ximg.data))

        ximg.data = ximg.data.chunk({"lon": 32, "lat": 32})
        print(ximg)
        
        ximg.data = xarray.apply_ufunc(numpy.sqrt, ximg.data,
                                 dask="parallelized",
                                 output_dtypes=[float])

        ximg.data = ximg.data.sel({"frequency": frequency[0], "polarisation": "I"})
        ximg.data.plot.imshow()
        plt.show()


if __name__ == '__main__':
    unittest.main()
