"""
Unit tests for plot_skycomponents

Output files:

    test_plot_skycomponents_flux_histogram.png
    test_plot_skycomponents_flux_ratio.png
    test_plot_skycomponents_flux_value.png
    test_plot_skycomponents_position_distance.png
    test_plot_skycomponents_position_error.png
    test_plot_skycomponents_position_value.png
    test_plot_skycomponents_position_quiver.png
    test_plot_skycomponents_gaussian_beam_position.png
"""
import logging
import unittest
import sys
import os

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
    addnoise_skycomponent,
)
from rascil.processing_components.image import create_image, export_image_to_fits
from rascil.processing_components.skycomponent.operations import restore_skycomponent
from rascil.processing_components.skycomponent.plot_skycomponent import (
    plot_skycomponents_positions,
    plot_skycomponents_position_distance,
    plot_skycomponents_flux,
    plot_skycomponents_flux_ratio,
    plot_skycomponents_flux_histogram,
    plot_skycomponents_position_quiver,
    plot_gaussian_beam_position,
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
            flux_limit=3.0,
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
            radius=0.25,
        )

        self.cellsize = 0.0015
        self.model = create_image(
            npixel=512,
            cellsize=self.cellsize,
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        self.clean_beam = {
            "bmaj": 5.0 * numpy.rad2deg(self.cellsize),
            "bmin": 3.0 * numpy.rad2deg(self.cellsize),
            "bpa": -30.0,
        }

        self.model = restore_skycomponent(
            self.model, self.components, clean_beam=self.clean_beam
        )
        export_image_to_fits(
            self.model, self.dir + "/test_plot_skycomponents_model.fits"
        )

        self.noise = 1.0e-6
        self.plot_file = self.dir + "/test_plot_skycomponents"

        # clean up existing png files
        filelist = [f for f in os.listdir(self.dir) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(self.dir, f))

    def test_plot_positions(self):

        comp_test = addnoise_skycomponent(
            self.components, noise=self.noise, mode="direction"
        )
        ra_test, dec_test = plot_skycomponents_positions(
            comp_test,
            self.components,
            img_size=self.cellsize,
            plot_file=self.plot_file,
        )

        assert len(ra_test) == len(dec_test)
        assert len(ra_test) == len(comp_test)

        assert os.path.exists(self.plot_file + "_position_value.png")
        assert os.path.exists(self.plot_file + "_position_error.png")

    def test_plot_position_distance(self):

        comp_test = addnoise_skycomponent(
            self.components, noise=self.noise, mode="direction"
        )
        ra_error, dec_error = plot_skycomponents_position_distance(
            comp_test,
            self.components,
            self.phasecentre,
            self.cellsize,
            plot_file=self.plot_file,
        )

        assert len(ra_error) == len(comp_test)
        assert os.path.exists(self.plot_file + "_position_distance.png")

    def test_plot_flux(self):

        comp_test = addnoise_skycomponent(
            self.components, noise=self.noise, mode="flux"
        )

        flux_in, flux_out = plot_skycomponents_flux(
            comp_test, self.components, plot_file=self.plot_file
        )

        assert_array_almost_equal(flux_in, flux_out, decimal=3)
        assert len(flux_in) == len(self.components)
        assert len(flux_out) == len(comp_test)
        assert os.path.exists(self.plot_file + "_flux_value.png")

    def test_plot_flux_ratio(self):

        comp_test = addnoise_skycomponent(
            self.components, noise=self.noise, mode="flux"
        )
        dist, flux_ratio = plot_skycomponents_flux_ratio(
            comp_test, self.components, self.phasecentre, plot_file=self.plot_file
        )

        assert_array_almost_equal(flux_ratio, 1.0, decimal=3)
        assert len(flux_ratio) <= len(comp_test)
        assert os.path.exists(self.plot_file + "_flux_ratio.png")

    def test_plot_flux_histogram(self):

        comp_test = addnoise_skycomponent(
            self.components, noise=self.noise, mode="flux"
        )
        [flux_in, flux_out] = plot_skycomponents_flux_histogram(
            comp_test,
            self.components,
            plot_file=self.plot_file,
            nbins=100,
        )

        assert len(flux_out) <= len(self.components)
        assert len(flux_in) <= len(comp_test)
        assert os.path.exists(self.plot_file + "_flux_histogram.png")

    def test_plot_position_quiver(self):

        comp_test = addnoise_skycomponent(
            self.components, noise=self.noise, mode="direction"
        )
        ra_error, dec_error = plot_skycomponents_position_quiver(
            comp_test,
            self.components,
            self.phasecentre,
            num=100,
            plot_file=self.plot_file,
        )

        assert_array_almost_equal(ra_error, 0.0, decimal=3)
        assert_array_almost_equal(dec_error, 0.0, decimal=3)
        assert len(ra_error) == 27
        assert os.path.exists(self.plot_file + "_position_quiver.png")

    def test_plot_gaussian_beam_position(self):

        comp_test = addnoise_skycomponent(
            self.components, noise=self.noise, mode="direction"
        )
        bmaj, bmin = plot_gaussian_beam_position(
            comp_test,
            self.components,
            self.phasecentre,
            self.model,
            plot_file=self.plot_file,
        )

        assert len(bmaj) == len(bmin)
        assert os.path.exists(self.plot_file + "_gaussian_beam_position.png")


if __name__ == "__main__":
    unittest.main()
