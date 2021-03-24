"""
Unit tests for plot_skycomponents

"""
import logging
import unittest
import sys

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.testing import assert_array_almost_equal
from rascil.data_models import PolarisationFrame, rascil_path, rascil_data_path
from rascil.processing_components.simulation import (
    create_low_test_skycomponents_from_gleam,
)
from rascil.processing_components.skycomponent.operations import (
    filter_skycomponents_by_flux,
)
from rascil.processing_components.skycomponent.plot_skycomponent import (
    plot_skycomponents_positions,
    plot_skycomponents_position_distance,
    plot_skycomponents_flux,
    plot_skycomponents_flux_ratio,
    plot_skycomponents_flux_histogram,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestPlotSkycomponent(unittest.TestCase):
    def setUp(self):

        self.dir = rascil_path("test_results")

        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e6])
        self.phasecentre = SkyCoord(
            ra=+0.0 * u.deg, dec=-55.0 * u.deg, frame="icrs", equinox="J2000"
        )

        self.components = create_low_test_skycomponents_from_gleam(
            flux_limit=0.1,
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
            radius=0.5,
        )
        self.plot_file = self.dir+"/test_plot_skycomponents"

    def test_plot_positions(self):

        ra_test, dec_test = plot_skycomponents_positions(
            self.components[: len(self.components) // 2],
            self.components,
            plot_file=self.plot_file,
        )

        assert len(ra_test) == len(dec_test)
        assert len(ra_test) == len(self.components) // 2

    def test_plot_position_distance(self):

        ra_error, dec_error = plot_skycomponents_position_distance(
            self.components[: len(self.components) // 2],
            self.components,
            self.phasecentre,
            plot_file=self.plot_file,
        )
       
        assert_array_almost_equal(ra_error, 0.0)
        assert_array_almost_equal(dec_error, 0.0)
        assert len(ra_error) == len(self.components) // 2

    def test_plot_flux(self):

        filtered = filter_skycomponents_by_flux(self.components, flux_min=0.1)
        flux_in, flux_out = plot_skycomponents_flux(
            filtered, self.components, plot_file=self.plot_file
        )
      
        assert_array_almost_equal(flux_in, flux_out)
        assert len(flux_in) == len(filtered)

    def test_plot_flux_ratio(self):

        filtered = filter_skycomponents_by_flux(self.components, flux_min=0.1)
        dist, flux_ratio = plot_skycomponents_flux_ratio(
            filtered, self.components, self.phasecentre, plot_file=self.plot_file
        )
        
        assert_array_almost_equal(flux_ratio, 1.0)
        assert len(flux_ratio) == len(filtered)

    def test_plot_flux_histogram(self):

        filtered = filter_skycomponents_by_flux(self.components, flux_min=0.1)
        [flux_in, flux_out] = plot_skycomponents_flux_histogram(
            filtered, self.components, plot_file=self.plot_file
        )
    
        assert len(flux_out) <= len(self.components)
        assert len(flux_in) <= len(filtered)


if __name__ == "__main__":
    unittest.main()
