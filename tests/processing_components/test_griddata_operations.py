""" Unit tests for image operations


"""
import logging
import unittest

import numpy

from rascil.processing_components.griddata.operations import create_griddata_from_image
from rascil.processing_components.simulation import create_test_image

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestGridData(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")

        self.m31image = create_test_image(cellsize=0.0001)
        self.cellsize = 180.0 * 0.0001 / numpy.pi

    def test_create_griddata_from_image(self):
        m31model_by_image = create_griddata_from_image(self.m31image)
        assert (
            m31model_by_image.griddata_acc.shape[0] == self.m31image.image_acc.shape[0]
        )
        assert (
            m31model_by_image.griddata_acc.shape[1] == self.m31image.image_acc.shape[1]
        )
        assert (
            m31model_by_image.griddata_acc.shape[2] == self.m31image.image_acc.shape[2]
        )
        assert (
            m31model_by_image.griddata_acc.shape[3] == self.m31image.image_acc.shape[3]
        )


if __name__ == "__main__":
    unittest.main()
