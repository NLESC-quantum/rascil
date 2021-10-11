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
    performance_read,
    performance_environment,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestPerformance(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")
        self.json_file = rascil_path("test_results/test_performance.json")

        self.m31image = create_test_image()
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_qa_image(self):
        """Test that the QA for an image is written correctly"""
        performance_qa_image(self.json_file, "restored", self.m31image, mode="w")
        performance = performance_read(self.json_file)
        assert "restored" in performance
        assert "max" in performance["restored"]

    def test_qa_file_exception(self):
        """Check for non-existant file

        :return:
        """
        with self.assertRaises(FileNotFoundError):
            performance = performance_read("Doesnotexist.json")

    def test_environment(self):
        """Test that the environment information is written correctly"""
        performance_environment(self.json_file, mode="w")
        performance = performance_read(self.json_file)
        assert "environment" in performance
        assert "git" in performance["environment"]


if __name__ == "__main__":
    unittest.main()
