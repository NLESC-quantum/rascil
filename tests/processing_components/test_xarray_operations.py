""" Unit tests for mpc

"""
import logging
import os
import unittest
import numpy

from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.processing_components.xarray.operations import import_xarray_from_fits, export_xarray_to_fits

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)


class TestXarrayOperations(unittest.TestCase):
    def setUp(self):
        self.persist = os.getenv("RASCIL_PERSIST", False)
        self.dir = rascil_path('test_results')
    
    def test_read_write_screen(self):
        screen = import_xarray_from_fits(rascil_data_path('models/test_mpc_screen.fits'))
        assert screen.shape == (1, 3, 2000, 2000), screen.shape
        assert numpy.unravel_index(numpy.argmax(screen.values), screen.shape) == (0, 2, 79, 1814)
        export_xarray_to_fits(screen, rascil_path('test_results/test_export_xarray.fits'))
    
    def test_read_write_screen_complex_fails(self):
        screen = import_xarray_from_fits(rascil_data_path('models/test_mpc_screen.fits'))
        screen = screen.astype('complex')
        with self.assertRaises(AssertionError):
            export_xarray_to_fits(screen, rascil_path('test_results/test_export_xarray.fits'))
    
    def test_read_write_screen_complex(self):
        screen = import_xarray_from_fits(rascil_data_path('models/test_mpc_screen.fits'))
        screen = screen.astype('complex')
        fitsfiles = [rascil_path('test_results/test_export_xarray_real.fits'),
                     rascil_path('test_results/test_export_xarray_imag.fits')]
        export_xarray_to_fits(screen, fitsfiles)
