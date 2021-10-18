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

        self.results_dir = rascil_path("test_results")

        self.central_frequency = numpy.array([1e8])
        self.component_frequency = numpy.linspace(0.8e8, 1.2e8, 7)
        self.phasecentre = SkyCoord(
            ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )

        self.noise = 1.0e-5

    def test_addnoise_skycomponents_direction(self):

        self.components = create_low_test_skycomponents_from_gleam(
            flux_limit=0.1,
            phasecentre=self.phasecentre,
            frequency=self.central_frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
            radius=0.5,
        )

        original = copy_skycomponent(self.components)
        add_direc = addnoise_skycomponent(
            self.components, noise=self.noise, mode="direction"
        )

        assert len(add_direc) == len(original)

        ra_orig = numpy.array([comp.direction.ra.radian for comp in original])
        dec_orig = numpy.array([comp.direction.dec.radian for comp in original])

        ra_result = numpy.array([comp.direction.ra.radian for comp in add_direc])
        dec_result = numpy.array([comp.direction.dec.radian for comp in add_direc])

        assert numpy.any(numpy.not_equal(ra_orig, ra_result))
        assert numpy.any(numpy.not_equal(dec_orig, dec_result))

        assert_almost_equal(numpy.mean(ra_orig), numpy.mean(ra_result), decimal=3)
        assert_almost_equal(numpy.mean(dec_orig), numpy.mean(dec_result), decimal=3)

    def test_addnoise_skycomponents_singleflux(self):

        self.components = create_low_test_skycomponents_from_gleam(
            flux_limit=0.1,
            phasecentre=self.phasecentre,
            frequency=self.central_frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
            radius=0.5,
        )

        original = copy_skycomponent(self.components)

        result = addnoise_skycomponent(
            self.components, noise=self.noise, mode="flux_central"
        )
        flux_orig = numpy.array([comp.flux[0, 0] for comp in original])
        flux_result = numpy.array([comp.flux[0, 0] for comp in result])

        assert len(result) == len(original)
        assert numpy.any(numpy.not_equal(flux_orig, flux_result))
        assert_almost_equal(numpy.mean(flux_orig), numpy.mean(flux_result), decimal=3)

    def test_addnoise_skycomponents_multiflux(self):

        self.components = create_low_test_skycomponents_from_gleam(
            flux_limit=0.1,
            phasecentre=self.phasecentre,
            frequency=self.component_frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
            radius=0.5,
        )

        original = copy_skycomponent(self.components)

        add_central = addnoise_skycomponent(
            self.components, noise=self.noise, mode="flux_central"
        )

        nchan = len(self.component_frequency)
        centre = nchan // 2
        flux_orig = numpy.array([comp.flux[centre, 0] for comp in original])
        flux_result_ctr = numpy.array([comp.flux[centre, 0] for comp in add_central])

        assert numpy.any(numpy.not_equal(flux_orig, flux_result_ctr))
        assert_almost_equal(
            numpy.mean(flux_orig), numpy.mean(flux_result_ctr), decimal=3
        )

        add_all = addnoise_skycomponent(
            self.components, noise=self.noise, mode="flux_all"
        )
        flux_orig = numpy.array([comp.flux for comp in original])
        flux_result_all = numpy.array([comp.flux for comp in add_all])

        assert flux_result_all.shape == (len(self.components), nchan, 1)

        assert numpy.any(numpy.not_equal(flux_orig, flux_result_all))
        assert_almost_equal(
            numpy.mean(flux_orig), numpy.mean(flux_result_all), decimal=3
        )


if __name__ == "__main__":
    unittest.main()
