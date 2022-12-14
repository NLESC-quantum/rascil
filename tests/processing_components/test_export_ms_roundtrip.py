# -*- coding: utf-8 -*-

"""Unit test for the measurementset module."""

import unittest

import numpy
import xarray
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame, rascil_path
from rascil.processing_components import (
    create_blockvisibility,
    export_image_to_fits,
    create_named_configuration,
    create_test_image,
    create_image_from_visibility,
    advise_wide_field,
    export_blockvisibility_to_ms,
    create_blockvisibility_from_ms,
)
from rascil.processing_components.imaging.imaging import (
    predict_blockvisibility,
    invert_blockvisibility,
)


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

        numpy.seterr(all="ignore")

    def test_roundtrip(self):
        if run_ms_tests is False:
            return

        results_dir = rascil_path("test_results")

        # Construct LOW core configuration
        lowr3 = create_named_configuration("LOWBD2", rmax=750.0)

        # We create the visibility. This just makes the uvw, time, antenna1, antenna2,
        # weight columns in a table. We subsequently fill the visibility value in by
        # a predict step.

        times = numpy.zeros([1])
        frequency = numpy.array([1e8, 1.1e8, 1.2e8])
        channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
        )
        vt = create_blockvisibility(
            lowr3,
            times,
            frequency,
            channel_bandwidth=channel_bandwidth,
            weight=1.0,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
        )

        # Find the recommended imaging parameters
        advice = advise_wide_field(
            vt,
            guard_band_image=3.0,
            delA=0.1,
            oversampling_synthesised_beam=4.0,
            verbose=False,
        )
        cellsize = advice["cellsize"]

        # Read the venerable test image, constructing a RASCIL Image
        m31image = create_test_image(
            cellsize=cellsize, frequency=frequency, phasecentre=vt.phasecentre
        )

        # Predict the visibility for the Image
        vt = predict_blockvisibility(vt, m31image, context="2d")

        model = create_image_from_visibility(vt, cellsize=cellsize, npixel=512)
        dirty_before, sumwt = invert_blockvisibility(vt, model, context="2d")
        export_image_to_fits(
            dirty_before,
            "{dir}/test_roundtrip_dirty_before.fits".format(dir=results_dir),
        )

        # print("Before: Max, min in dirty image = %.6f, %.6f, sumwt = %f" %
        #      (dirty_before.data.max(), dirty_before.data.min(), sumwt))

        msname = "{dir}/test_roundtrip.ms".format(dir=results_dir)
        export_blockvisibility_to_ms(msname, [vt])
        vt_after = create_blockvisibility_from_ms(msname)[0]

        # Temporarily flag autocorrelations until MS writer is fixed
        vt_after["flags"] = xarray.where(
            vt_after["uvdist_lambda"] > 0.0, vt_after["flags"], 1.0
        )

        # Make the dirty image and point spread function
        model = create_image_from_visibility(vt_after, cellsize=cellsize, npixel=512)
        dirty_after, sumwt = invert_blockvisibility(vt_after, model, context="2d")
        export_image_to_fits(
            dirty_after, "{dir}/test_roundtrip_dirty_after.fits".format(dir=results_dir)
        )

        # print("After: Max, min in dirty image = %.6f, %.6f, sumwt = %f" %
        #      (dirty_after.data.max(), dirty_after.data.min(), sumwt))

        # export_image_to_fits(dirty_after, '%s/imaging_dirty_after.fits' % (results_dir))

        error = numpy.max(
            numpy.abs(dirty_after["pixels"].data - dirty_before["pixels"].data)
        ) / numpy.max(numpy.abs(dirty_before["pixels"].data))
        # print("Maximum fractional difference in peak of dirty image before, after writing to MS = {}".format(error))

        assert (
            error < 1e-8
        ), "Maximum fractional difference in peak of dirty image before, after writing to MS execeeds tolerance"
