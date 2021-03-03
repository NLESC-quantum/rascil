""" Unit tests for image operations

"""
import logging
import os
import unittest

import matplotlib.pyplot as plt

import astropy.units as u
import numpy
import xarray
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_test_image, show_image

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestImageSelection(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")

        self.image = create_test_image(
            cellsize=0.0001,
            phasecentre=SkyCoord(
                ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
            ),
            frequency=numpy.linspace(0.8e9, 1.2e9, 5),
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            channel_bandwidth=1e7 * numpy.ones([5]),
        )

        # assert numpy.max(self.image["pixels"]) > 0.0, "Test image is empty"
        self.cellsize = 180.0 * 0.0001 / numpy.pi
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_image_iselect_channel(self):
        subim = self.image.isel({"frequency": slice(0, 2)}, drop=False)
        assert subim["pixels"].shape == (2, 4, 256, 256), subim["pixels"].shape

    def test_image_iselect_channel_pol(self):
        subim = self.image.isel({"frequency": 0, "polarisation": [0, 3]}, drop=False)
        assert subim["pixels"].shape == (2, 256, 256)
        assert subim.dims == {"polarisation": 2, "y": 256, "x": 256}
        numpy.testing.assert_array_equal(subim.coords["polarisation"], ["I", "V"])

    def test_image_where_frequency(self):
        subim = self.image.where(self.image["frequency"] > 8.3e8, drop=False)
        subim = subim.dropna(dim="frequency", how="all")
        assert subim["pixels"].shape == (1, 4, 256, 256)
        assert subim.dims == {
            "frequency": 1,
            "polarisation": 4,
            "y": 256,
            "x": 256,
        }, subim.dims
        numpy.testing.assert_array_equal(subim.coords["frequency"], [8.4e8])

    def test_image_where_polarisation(self):
        subim = self.image.where(self.image["polarisation"] == "I", drop=False)
        subim = subim.dropna(dim="polarisation", how="all")
        assert subim["pixels"].shape == (5, 1, 256, 256)
        assert subim.dims == {"frequency": 5, "polarisation": 1, "y": 256, "x": 256}
        numpy.testing.assert_array_equal(subim.coords["polarisation"], ["I"])

    def test_image_where_radius_radec(self):
        nchan, npol, ny, nx = self.image["pixels"].shape
        secd = 1.0 / numpy.cos(numpy.deg2rad(self.image.dec))
        r = numpy.hypot(
            (self.image.ra - self.image.ra[ny // 2, nx // 2]) * secd,
            self.image.dec - self.image.dec[ny // 2, nx // 2],
        )
        show_image(self.image.where(r < 0.3, 0.0))
        plt.show()

    def test_image_where_radius_xy(self):
        nchan, npol, ny, nx = self.image["pixels"].shape
        r = numpy.hypot(
            self.image.x - self.image.x[nx // 2], self.image.y - self.image.y[ny // 2]
        )
        show_image(self.image.where(r < 0.3, 0.0))
        plt.show()


if __name__ == "__main__":
    unittest.main()
