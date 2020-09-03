""" Unit tests for XImage operations


"""

import unittest
import dask

import xarray
import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame
from rascil.processing_components.image.image_x import create_ximage, convert_image_to_ximage
from rascil.processing_components.simulation.testing_support import create_test_image


class TestXimage(unittest.TestCase):
    
    def test_create_ximage(self):
        frequency = numpy.linspace(1.0e8, 1.1e8, 8)
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        
        im = create_test_image(cellsize=0.001, phasecentre=phasecentre, frequency=frequency,
                               polarisation_frame=PolarisationFrame("stokesIQUV"))
        ximg = create_ximage(data=im.data, wcs=im.wcs, axes=(256, 256),
                             cellsize=0.001,
                             frequency=frequency,
                             phasecentre=phasecentre,
                             polarisation_frame=PolarisationFrame("stokesIQUV"))
        
        print("\nInitial ximg")
        print(ximg)
        
        import matplotlib.pyplot as plt
        ximg2d = ximg.sel({"frequency": frequency[0], "polarisation": "I"})
        ximg2d.plot.imshow()
        plt.show()

        print("\nSelection by polarisation")
        print(ximg.sel({"polarisation": ["I", "V"]}))
        print("\nMasking")
        ximg = ximg.where(numpy.hypot(ximg["lon"] - numpy.mean(ximg["lon"]),
                                      ximg["lat"] - numpy.mean(ximg["lat"])) < 6.0)
        
        print(ximg)
        print(numpy.sum(ximg))

        ximg = ximg.chunk({"lon": 32, "lat": 32})
        print(ximg)
        
        ximg = xarray.apply_ufunc(numpy.sqrt, ximg,
                                 dask="parallelized",
                                 output_dtypes=[float])

        ximg2d = ximg.sel({"frequency": frequency[0], "polarisation": "I"})
        ximg2d.plot.imshow()
        plt.show()

    def test_convert_ximage(self):
        frequency = numpy.linspace(1.0e8, 1.1e8, 8)
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    
        im = create_test_image(cellsize=0.001, phasecentre=phasecentre, frequency=frequency,
                               polarisation_frame=PolarisationFrame("stokesIQUV"))
        ximg = convert_image_to_ximage(im)
        print("\nInitial ximg")
        print(ximg)
    
        import matplotlib.pyplot as plt
        ximg2d = ximg.sel({"frequency": frequency[0], "polarisation": "I"})
        ximg2d.plot.imshow()
        plt.show()
    
        print("\nSelection by polarisation")
        print(ximg.sel({"polarisation": ["I", "V"]}))
        print("\nMasking")
        ximg = ximg.where(numpy.hypot(ximg["lon"] - numpy.mean(ximg["lon"]),
                                      ximg["lat"] - numpy.mean(ximg["lat"])) < 6.0)
    
        print(ximg)
        print(numpy.sum(ximg))
    
        ximg = ximg.chunk({"lon": 32, "lat": 32})
        print(ximg)
    
        ximg = xarray.apply_ufunc(numpy.sqrt, ximg,
                                  dask="parallelized",
                                  output_dtypes=[float])
    
        ximg2d = ximg.sel({"frequency": frequency[0], "polarisation": "I"})
        ximg2d.plot.imshow()
        plt.show()

    def test_convert_ximage_terse(self):
        frequency = numpy.linspace(1.0e8, 1.1e8, 8)
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        im = create_test_image(cellsize=0.001, phasecentre=phasecentre, frequency=frequency,
                               polarisation_frame=PolarisationFrame("stokesIQUV"))
        ximg = convert_image_to_ximage(im)
        
        import matplotlib.pyplot as plt
        ximg2d = ximg.sel({"frequency": frequency[0], "polarisation": "I"})
        ximg2d.plot.imshow()
        plt.show()
        # Masking
        ximg = ximg.where(numpy.hypot(ximg["lon"] - numpy.mean(ximg["lon"]),
                                      ximg["lat"] - numpy.mean(ximg["lat"])) < 6.0)
        # Chunk - this returns a dask array
        ximg = ximg.chunk({"lon": 32, "lat": 32})
        # Apply a ufunc, using dask
        ximg = xarray.apply_ufunc(numpy.sqrt, ximg,
                                  dask="parallelized",
                                  output_dtypes=[float])
        # Plot the result
        ximg2d = ximg.sel({"frequency": frequency[0], "polarisation": "I"})
        ximg2d.plot.imshow()
        plt.show()


if __name__ == '__main__':
    unittest.main()
