""" Unit tests for testing support


"""
import logging
import unittest

import numpy

# Import the base and then make a global version
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class Testrsexecute(unittest.TestCase):
    def setUp(self):
        rsexecute.set_client(use_dask=True)

    def tearDown(self):
        rsexecute.close()

    def test_useFunction(self):
        def square(x):
            return x**2

        graph = rsexecute.execute(square)(numpy.arange(10))
        result = rsexecute.compute(graph, sync=True)
        assert (
            result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
        ).all(), result

    def test_restart(self):
        def square(x):
            return x**2

        for trial in range(10):
            rsexecute.close()
            rsexecute.set_client(use_dask=True)
            graph = rsexecute.execute(square)(numpy.arange(10))
            result = rsexecute.compute(graph, sync=True)
            assert (
                result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
            ).all(), result

        rsexecute.close()

    def test_useDaskAsync(self):
        def square(x):
            return x**2

        graph = rsexecute.execute(square)(numpy.arange(10))
        result = rsexecute.compute(graph).result()
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()

    def test_useDaskSync(self):
        def square(x):
            return x**2

        graph = rsexecute.execute(square)(numpy.arange(10))
        result = rsexecute.compute(graph, sync=True)
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
