""" Unit tests for xarray operations

"""
import logging
import os
import unittest
import numpy

from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.processing_components.xarray.operations import (
    import_xarray_from_fits,
    export_xarray_to_fits,
)
from rascil.processing_components.simulation.testing_support import create_test_image

from rascil.processing_components import fft_image_to_griddata

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestXarrayOperations(unittest.TestCase):
    def setUp(self):
        self.persist = os.getenv("RASCIL_PERSIST", False)
        self.results_dir = rascil_path("test_results")

    def test_read_write_screen(self):
        screen = import_xarray_from_fits(
            rascil_data_path("models/test_mpc_screen.fits")
        )
        assert screen["pixels"].data.shape == (1, 3, 2000, 2000), screen[
            "pixels"
        ].data.shape
        assert numpy.unravel_index(
            numpy.argmax(screen["pixels"].data), screen["pixels"].data.shape
        ) == (0, 2, 79, 1814)
        fitsfile = rascil_path("test_results/test_export_xarray.fits")
        export_xarray_to_fits(
            screen,
            fitsfile,
        )
        assert os.path.isfile(fitsfile)

    def test_read_write_screen_complex_fails(self):
        screen = import_xarray_from_fits(
            rascil_data_path("models/test_mpc_screen.fits")
        )
        screen = screen.astype("complex")
        with self.assertRaises(AssertionError):
            export_xarray_to_fits(
                screen, rascil_path("test_results/test_export_xarray.fits")
            )

    def test_write_griddata(self):
        model = create_test_image()
        gd = fft_image_to_griddata(model)
        fitsfiles = [
            rascil_path("test_results/test_export_xarray_griddata_real.fits"),
            rascil_path("test_results/test_export_xarray_griddata_imag.fits"),
        ]

        export_xarray_to_fits(gd, fitsfiles)
        for fitsfile in fitsfiles:
            assert os.path.isfile(fitsfile)
