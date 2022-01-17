""" Unit processing_components for Array functions


"""
import unittest
import numpy
import logging

from rascil.processing_components.util.array_functions import average_chunks

from rascil.processing_components.util.array_functions import average_chunks2
from rascil.processing_components.util.array_functions import average_chunks, average_chunks2
log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestArray_functions(unittest.TestCase):
    def test_average_chunks(self):
        arr = numpy.linspace(0.0, 100.0, 11)
        wts = numpy.ones_like(arr)
        carr, cwts = average_chunks(arr, wts, 2)
        assert len(carr) == len(cwts)
        answerarr = numpy.array([5.0, 25.0, 45.0, 65.0, 85.0, 100.0])
        answerwts = numpy.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
        numpy.testing.assert_array_equal(carr, answerarr)
        numpy.testing.assert_array_equal(cwts, answerwts)

    def test_average_chunks_exact(self):
        arr = numpy.linspace(0.0, 90.0, 10)
        wts = numpy.ones_like(arr)
        carr, cwts = average_chunks(arr, wts, 2)
        assert len(carr) == len(cwts)
        answerarr = numpy.array([5.0, 25.0, 45.0, 65.0, 85.0])
        answerwts = numpy.array([2.0, 2.0, 2.0, 2.0, 2.0])
        numpy.testing.assert_array_equal(carr, answerarr)
        numpy.testing.assert_array_equal(cwts, answerwts)

    def test_average_chunks_zero(self):
        arr = numpy.linspace(0.0, 90.0, 10)
        wts = numpy.ones_like(arr)
        carr, cwts = average_chunks(arr, wts, 0)
        assert len(carr) == len(cwts)
        numpy.testing.assert_array_equal(carr, arr)
        numpy.testing.assert_array_equal(cwts, wts)

    def test_average_chunks_single(self):
        arr = numpy.linspace(0.0, 100.0, 11)
        wts = numpy.ones_like(arr)
        carr, cwts = average_chunks(arr, wts, 12)
        assert len(carr) == len(cwts)
        answerarr = numpy.array([50.0])
        answerwts = numpy.array([11.0])
        numpy.testing.assert_array_equal(carr, answerarr)
        numpy.testing.assert_array_equal(cwts, answerwts)

    def test_average_chunks2_1d(self):
        arr = numpy.linspace(0.0, 100.0, 11).reshape(
            [1, 11]
        )  # pylint: disable=no-member
        wts = numpy.ones_like(arr)
        carr, cwts = average_chunks2(arr, wts, (1, 2))
        assert len(carr) == len(cwts)
        answerarr = numpy.array([[5.0, 25.0, 45.0, 65.0, 85.0, 100.0]])
        answerwts = numpy.array([[2.0, 2.0, 2.0, 2.0, 2.0, 1.0]])
        numpy.testing.assert_array_equal(carr, answerarr)
        numpy.testing.assert_array_equal(cwts, answerwts)

    def test_average_chunks2_1d_trans(self):
        arr = numpy.linspace(0.0, 100.0, 11).reshape(
            [11, 1]
        )  # pylint: disable=no-member
        wts = numpy.ones_like(arr)
        carr, cwts = average_chunks2(arr, wts, (2, 1))
        assert len(carr) == len(cwts)
        answerarr = numpy.array([[5.0], [25.0], [45.0], [65.0], [85.0], [100.0]])
        answerwts = numpy.array([[2.0], [2.0], [2.0], [2.0], [2.0], [1.0]])
        numpy.testing.assert_array_equal(carr, answerarr)
        numpy.testing.assert_array_equal(cwts, answerwts)

    def test_average_chunks2_2d(self):
        arr = numpy.linspace(0.0, 120.0, 121).reshape(
            11, 11
        )  # pylint: disable=no-member
        wts = numpy.ones_like(arr)
        carr, cwts = average_chunks2(arr, wts, (5, 2))
        assert len(carr) == len(cwts)
        answerarr = numpy.array([32.0, 87.0, 120.0])
        answerwts = numpy.array([5.0, 5.0, 1.0])
        numpy.testing.assert_array_equal(carr[:, 5], answerarr)
        numpy.testing.assert_array_equal(cwts[:, 5], answerwts)


if __name__ == "__main__":
    unittest.main()
