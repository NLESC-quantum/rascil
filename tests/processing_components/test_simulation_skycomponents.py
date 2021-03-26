"""Unit tests for functions in simulations/skycomponents.py

"""

import logging
import sys
import unittest

import astropy.units as u
import numpy
from numpy.testing import assert_almost_equal
from astropy.coordinates import SkyCoord

from rascil.data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components.simulation.skycomponents import addnoise_skycomponent
from rascil.processing_components.simulation import (
    create_low_test_skycomponents_from_gleam,
)
from rascil.processing_components.skycomponent.base import copy_skycomponent

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestSimSkycomponents(unittest.TestCase):
    def setUp(self):

        from rascil.data_models.parameters import rascil_path, rascil_data_path

        self.dir = rascil_path("test_results")

        self.frequency = numpy.array([1e8])
        self.phasecentre = SkyCoord(
            ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )

        self.noise = 1.0e-5

    def test_addnoise_skycomponents(self):

        self.components = create_low_test_skycomponents_from_gleam(
            flux_limit=0.1,
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
            radius=0.5,
        )

        original = copy_skycomponent(self.components)
        result = addnoise_skycomponent(self.components, noise=self.noise, mode="both")
        ra_orig = numpy.array([comp.direction.ra.radian for comp in original])
        dec_orig = numpy.array([comp.direction.dec.radian for comp in original])
        flux_orig = numpy.array([comp.flux[0, 0] for comp in original])

        ra_result = numpy.array([comp.direction.ra.radian for comp in result])
        dec_result = numpy.array([comp.direction.dec.radian for comp in result])
        flux_result = numpy.array([comp.flux[0, 0] for comp in result])

        assert len(result) == len(original)

        assert numpy.any(numpy.not_equal(ra_orig,ra_result))
        assert numpy.any(numpy.not_equal(dec_orig,dec_result))
        assert numpy.any(numpy.not_equal(flux_orig,flux_result))

        assert_almost_equal(numpy.mean(ra_orig), numpy.mean(ra_result), decimal=3)
        assert_almost_equal(numpy.mean(dec_orig), numpy.mean(dec_result), decimal=3)
        assert_almost_equal(numpy.mean(flux_orig), numpy.mean(flux_result), decimal=3)


if __name__ == "__main__":
    unittest.main()
