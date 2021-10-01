""" Unit tests for image Taylor terms

"""
import logging
import os
import unittest

import numpy

from rascil.processing_components.image.taylor_terms import (
    calculate_image_list_frequency_moments,
    calculate_image_list_from_frequency_taylor_terms,
    calculate_image_frequency_moments,
    calculate_image_from_frequency_taylor_terms,
    calculate_frequency_taylor_terms_from_image_list,
)
from rascil.processing_components import (
    create_empty_image_like,
    image_scatter_channels,
)
from rascil.processing_components.image.operations import (
    export_image_to_fits,
)
from rascil.processing_components.simulation import (
    create_test_image,
    create_low_test_image_from_gleam,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestImage(unittest.TestCase):
    def setUp(self):

        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")

        self.m31image = create_test_image()

        # assert numpy.max(self.m31image["pixels"]) > 0.0, "Test image is empty"
        self.cellsize = 180.0 * 0.0001 / numpy.pi
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_calculate_image_frequency_moments(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512, cellsize=0.0001, frequency=frequency, flux_limit=1.0
        )
        if self.persist:
            export_image_to_fits(
                original_cube, fitsfile="%s/test_moments_cube.fits" % (self.results_dir)
            )
        cube = create_empty_image_like(original_cube)
        moment_cube = calculate_image_frequency_moments(cube, nmoment=3)
        if self.persist:
            export_image_to_fits(
                moment_cube,
                fitsfile="%s/test_moments_moment_cube.fits" % (self.results_dir),
            )
        reconstructed_cube = calculate_image_from_frequency_taylor_terms(
            cube, moment_cube
        )
        print(reconstructed_cube.image_acc.wcs)
        if self.persist:
            export_image_to_fits(
                reconstructed_cube,
                fitsfile="%s/test_moments_reconstructed_cube.fits" % (self.results_dir),
            )
        error = numpy.std(
            reconstructed_cube["pixels"].data - original_cube["pixels"].data
        )
        assert error < 0.2, error

    def test_calculate_image_frequency_moments_1(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512, cellsize=0.0001, frequency=frequency, flux_limit=1.0
        )
        if self.persist:
            export_image_to_fits(
                original_cube, fitsfile="%s/test_moments_1_cube.fits" % (self.dir)
            )
        cube = create_empty_image_like(original_cube)
        moment_cube = calculate_image_frequency_moments(cube, nmoment=1)
        if self.persist:
            export_image_to_fits(
                moment_cube, fitsfile="%s/test_moments_1_moment_cube.fits" % (self.dir)
            )
        reconstructed_cube = calculate_image_from_frequency_taylor_terms(
            cube, moment_cube
        )
        if self.persist:
            export_image_to_fits(
                reconstructed_cube,
                fitsfile="%s/test_moments_1_reconstructed_cube.fits" % (self.dir),
            )
        error = numpy.std(
            reconstructed_cube["pixels"].data - original_cube["pixels"].data
        )
        assert error < 0.2

    def test_calculate_taylor_terms(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512, cellsize=0.0001, frequency=frequency, flux_limit=1.0
        )
        original_list = image_scatter_channels(original_cube)
        taylor_term_list = calculate_frequency_taylor_terms_from_image_list(
            original_list, nmoment=3
        )
        assert len(taylor_term_list) == 3


if __name__ == "__main__":
    unittest.main()
