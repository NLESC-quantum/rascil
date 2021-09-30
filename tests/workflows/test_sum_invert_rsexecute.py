"""Unit tests for sum_invert_results and sum_invert_results_rsexecute

The invert results are a list of tuples where each tuple hos an image and a weight
matrix.

"""
import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.parameters import rascil_path
from rascil.processing_components import (
    create_image,
    qa_image,
)
from rascil.workflows import sum_invert_results, sum_invert_results_rsexecute
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestSumInvert(unittest.TestCase):
    def setUp(self):
        rsexecute.set_client(use_dask=True)

        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.testimage = create_image(
            npixel=256, frequency=self.frequency, phasecentre=self.phasecentre
        )

    def test_sum_invert(self):
        """Test that sum_invert_results gives the correct results for a >1 element list and
        that sum_invert_results_rsexecute gives the same results as sum_invert_results
        """
        results = list()
        nchan, npol, _, _ = self.testimage["pixels"].shape
        for i in range(10):
            im = self.testimage.copy(deep=True)
            im["pixels"].data[...] = i + 1
            wt = (i + 1) * numpy.ones([nchan, npol])
            results.append((im, wt))
        sum_py = sum_invert_results(results)
        sum_dask = rsexecute.compute(sum_invert_results_rsexecute(results), sync=True)
        qa = qa_image(sum_py[0])

        expected = {
            "max": 7.0,
            "min": 7.0,
            "rms": 0.0,
            "sum": 1376256.0,
        }
        for field in ["max", "min", "rms", "sum"]:
            numpy.testing.assert_almost_equal(
                qa.data[field], expected[field], err_msg=str(qa)
            )
        qa_sum = qa_image(sum_dask[0])
        for field in ["max", "min", "rms", "sum"]:
            numpy.testing.assert_approx_equal(qa.data[field], qa_sum.data[field])
        numpy.testing.assert_array_almost_equal_nulp(sum_py[1], sum_dask[1])

    def test_sum_invert_single(self):
        """Test that sum_invert_results gives the correct results for a single
        element list and that sum_invert_results_rsexecute gives the same
        results as sum_invert_results
        """
        nchan, npol, _, _ = self.testimage["pixels"].shape
        im = self.testimage.copy(deep=True)
        im["pixels"].data[...] = 1
        wt = numpy.ones([nchan, npol])
        result = [(im, wt)]
        sum_py = sum_invert_results(result)
        sum_dask = rsexecute.compute(sum_invert_results_rsexecute(result), sync=True)
        qa = qa_image(sum_py[0])

        expected = {
            "max": 1.0,
            "min": 1.0,
            "rms": 0.0,
            "sum": 196608.0,
        }
        for field in ["max", "min", "rms", "sum"]:
            numpy.testing.assert_almost_equal(
                qa.data[field], expected[field], err_msg=str(qa)
            )
        qa_sum = qa_image(sum_dask[0])
        for field in ["max", "min", "rms", "sum"]:
            numpy.testing.assert_approx_equal(qa.data[field], qa_sum.data[field])
        numpy.testing.assert_array_almost_equal_nulp(sum_py[1], sum_dask[1])

    def tearDown(self):
        rsexecute.close()
