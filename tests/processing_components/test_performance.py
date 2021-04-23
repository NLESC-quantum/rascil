""" Unit tests for performance monitoring

"""
import logging
import os
import unittest

import numpy

from rascil.processing_components.simulation import (
    create_test_image,
)
from rascil.processing_components.util.performance import (
    performance_store_dict,
    performance_qa_image,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestPerformance(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")
        self.json_file = rascil_path("test_results/test_performance.json")

        self.m31image = create_test_image()

        # assert numpy.max(self.m31image["pixels"]) > 0.0, "Test image is empty"
        self.cellsize = 180.0 * 0.0001 / numpy.pi
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_qa_image(self):
        performance_qa_image(self.json_file, "restored", self.m31image, mode="w")


if __name__ == "__main__":
    unittest.main()
