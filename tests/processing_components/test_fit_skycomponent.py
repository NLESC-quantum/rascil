""" Unit tests for skycomponents

"""
import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.data_models.data_model_helpers import export_skycomponent_to_hdf5
from rascil.processing_components import create_image, export_image_to_fits
from rascil.processing_components import (
    insert_skycomponent,
    create_skycomponent,
    fit_skycomponent,
    fit_skycomponent_spectral_index,
)
from rascil.processing_components.simulation import create_named_configuration

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestFitSkycomponent(unittest.TestCase):
    def setUp(self):

        self.persist = os.getenv("RASCIL_PERSIST", False)

        from rascil.data_models.parameters import rascil_path

        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.dir = rascil_path("test_results")

    def actualSetup(self, dopol=False):

        if dopol:
            self.image_pol = PolarisationFrame("stokesIQUV")
            self.pol_flux = numpy.array([1.0, -0.8, 0.2, 0.01])
        else:
            self.image_pol = PolarisationFrame("stokesI")
            self.pol_flux = numpy.array([1.0])

        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 6)
        self.image_frequency = numpy.linspace(0.9e8, 1.1e8, 5)
        self.image_channel_bandwidth = numpy.array(5 * [5e6])
        self.component_frequency = numpy.linspace(0.8e8, 1.2e8, 7)
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )

        # Create model
        self.model = create_image(
            npixel=256,
            cellsize=0.0015,
            phasecentre=self.phasecentre,
            frequency=self.image_frequency,
            channel_bandwidth=self.image_channel_bandwidth,
            polarisation_frame=self.image_pol,
        )

        self.dphasecentre = SkyCoord(
            ra=+181.0 * u.deg, dec=-58.0 * u.deg, frame="icrs", equinox="J2000"
        )
        flux_scale = numpy.power(self.component_frequency / 1e8, -0.7)
        self.flux = numpy.outer(flux_scale, self.pol_flux)
        self.sc = create_skycomponent(
            direction=self.dphasecentre,
            flux=self.flux,
            frequency=self.component_frequency,
            polarisation_frame=self.image_pol,
        )

    def test_fit_skycomponent_pswf(self):
        """Put a point source on a location and then try to recover by fitting"""
        self.actualSetup()

        self.model = insert_skycomponent(self.model, self.sc, insert_method="PSWF")

        newsc = fit_skycomponent(self.model, self.sc)

        assert newsc.shape == "Point"
        separation = newsc.direction.separation(self.sc.direction).rad
        assert separation < 1e-7, separation

        if self.persist:
            export_image_to_fits(
                self.model, "%s/test_fit_skycomponent.fits" % (self.dir)
            )
            export_skycomponent_to_hdf5(
                newsc, "%s/test_fit_skycomponent.hdf" % (self.dir)
            )

    def test_fit_skycomponent_spectral_index(self):

        self.actualSetup()
        spec_indx = fit_skycomponent_spectral_index(self.sc)

        assert abs(spec_indx - (-0.7)) < 1e-7


if __name__ == "__main__":
    unittest.main()
