""" Unit tests for skycomponents

"""
import os
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components import create_image, create_griddata_from_image
from rascil.processing_components.skycomponent.operations import create_skycomponent
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.data_models.xarray_coordinate_support import image_wcs, griddata_wcs

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestXarrayCoordinateSupport(unittest.TestCase):
    def setUp(self):

        self.persist = os.getenv("RASCIL_PERSIST", False)

        from rascil.data_models.parameters import rascil_path

        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.results_dir = rascil_path("test_results")

    def actualSetup(self, dopol=False):

        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")
            self.pol_flux = numpy.array([1.0, -0.8, 0.2, 0.01])
        else:
            self.vis_pol = PolarisationFrame("stokesI")
            self.image_pol = PolarisationFrame("stokesI")
            self.pol_flux = numpy.array([1.0])

        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 6)
        self.image_frequency = numpy.linspace(0.9e8, 1.1e8, 5)
        self.image_channel_bandwidth = numpy.array(5 * [5e6])
        self.component_frequency = numpy.linspace(0.8e8, 1.2e8, 7)
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.image_frequency,
            channel_bandwidth=self.image_channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
            polarisation_frame=self.vis_pol,
            zerow=True,
        )
        self.vis["vis"].data *= 0.0

        # Create model
        self.model = create_image(
            npixel=256,
            cellsize=0.0015,
            phasecentre=self.vis.phasecentre,
            frequency=self.image_frequency,
            channel_bandwidth=self.image_channel_bandwidth,
            polarisation_frame=self.image_pol,
        )

        dphasecentre = SkyCoord(
            ra=+181.0 * u.deg, dec=-58.0 * u.deg, frame="icrs", equinox="J2000"
        )
        flux_scale = numpy.power(self.component_frequency / 1e8, -0.7)
        self.flux = numpy.outer(flux_scale, self.pol_flux)
        self.sc = create_skycomponent(
            direction=dphasecentre,
            flux=self.flux,
            frequency=self.component_frequency,
            polarisation_frame=self.image_pol,
        )

    def test_image_conversion(self):
        self.actualSetup()
        print(self.model.image_acc.wcs)
        print(image_wcs(self.model))

    def test_image_conversion_pol(self):
        self.actualSetup(dopol=True)
        print(self.model.image_acc.wcs)
        print(image_wcs(self.model))

    def test_griddata_conversion(self):
        self.actualSetup()
        gd = create_griddata_from_image(self.model)
        assert gd.griddata_acc.shape == self.model.image_acc.shape
        assert (
            gd.griddata_acc.polarisation_frame.type
            == self.model.image_acc.polarisation_frame.type
        )

    def test_griddata_conversion_pol(self):
        self.actualSetup(dopol=True)
        gd = create_griddata_from_image(self.model)
        assert gd.griddata_acc.shape == self.model.image_acc.shape
        assert (
            gd.griddata_acc.polarisation_frame.type
            == self.model.image_acc.polarisation_frame.type
        )


if __name__ == "__main__":
    unittest.main()
