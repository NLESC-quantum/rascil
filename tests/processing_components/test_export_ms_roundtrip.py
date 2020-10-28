# -*- coding: utf-8 -*-

"""Unit test for the measurementset module."""

import os
import sys
import time
import shutil
import tempfile
import unittest

from rascil.data_models.memory_data_models import Configuration
from rascil.data_models.polarisation import ReceptorFrame

from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.time import Time

import logging
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame, rascil_path
from rascil.processing_components import create_blockvisibility, export_image_to_fits, \
    create_named_configuration, create_test_image, \
    create_image_from_visibility, advise_wide_field, invert_2d, predict_2d, \
    export_blockvisibility_to_ms, create_blockvisibility_from_ms


try:
    import casacore
    from casacore.tables import table  # pylint: disable=import-error
    from rascil.processing_components.visibility import msv2
    from rascil.processing_components.visibility.msv2fund import Stand, Antenna
    run_ms_tests = True
#            except ModuleNotFoundError:
except:
    run_ms_tests = False


class measurementset_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the lsl.writer.measurementset.Ms
    class."""

    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        numpy.seterr(all='ignore')

    def test_roundtrip(self):
        if run_ms_tests==False:
            return

        results_dir = rascil_path("test_results")

        # Construct LOW core configuration
        lowr3 = create_named_configuration('LOWBD2', rmax=750.0)

        # We create the visibility. This just makes the uvw, time, antenna1, antenna2,
        # weight columns in a table. We subsequently fill the visibility value in by
        # a predict step.

        times = numpy.zeros([1])
        frequency = numpy.array([1e8])
        channel_bandwidth = numpy.array([1e6])
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame='icrs',
                               equinox='J2000')
        vt = create_blockvisibility(lowr3, times, frequency, channel_bandwidth=channel_bandwidth,
                                    weight=1.0, phasecentre=phasecentre,
                                    polarisation_frame=PolarisationFrame('stokesI'))

        # Find the recommended imaging parameters
        advice = advise_wide_field(vt, guard_band_image=3.0, delA=0.1,
                                   oversampling_synthesised_beam=4.0,
                                   verbose=False)
        cellsize = advice['cellsize']

        # Read the venerable test image, constructing a RASCIL Image
        m31image = create_test_image(frequency=frequency, cellsize=cellsize,
                                     phasecentre=vt.phasecentre)

        # Predict the visibility for the Image
        vt = predict_2d(vt, m31image, context='2d')

        model = create_image_from_visibility(vt, cellsize=cellsize, npixel=512)
        dirty_before, sumwt = invert_2d(vt, model, context='2d')
        #export_image_to_fits(dirty_before, '%s/imaging_dirty_before.fits' % (results_dir))

        # print("Before: Max, min in dirty image = %.6f, %.6f, sumwt = %f" %
        #       (dirty_before.data.max(), dirty_before.data.min(), sumwt))

        msname = "{dir}/test_roundtrip.ms".format(dir=results_dir)
        ms = export_blockvisibility_to_ms(msname, [vt])
        vt_after = create_blockvisibility_from_ms(msname)[0]

        # Make the dirty image and point spread function
        model = create_image_from_visibility(vt_after, cellsize=cellsize, npixel=512)
        dirty_after, sumwt = invert_2d(vt_after, model, context='2d')

        # print("After: Max, min in dirty image = %.6f, %.6f, sumwt = %f" %
        #       (dirty_after.data.max(), dirty_after.data.min(), sumwt))

        #export_image_to_fits(dirty_after, '%s/imaging_dirty_after.fits' % (results_dir))

        error = numpy.max(numpy.abs(dirty_after.data - dirty_before.data)) / numpy.max(numpy.abs(dirty_after.data))
        # print("Maximum fractional difference in dirty image before, after writing to MS = {}".format(error))

        assert error < 1e08, "Maximum fractional difference in dirty image before,after writing to MS execeeds tolerance"

        

