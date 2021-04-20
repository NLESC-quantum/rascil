"""Unit tests for image iteration


"""
import logging
import os
import unittest

import numpy

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.gather_scatter import (
    image_gather_channels,
    image_scatter_channels,
)
from rascil.processing_components.simulation import create_test_image
from rascil.workflows import image_gather_channels_rsexecute
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestImageGatherScattersGraph(unittest.TestCase):
    def setUp(self):

        rsexecute.set_client(use_dask=True)

        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def tearDown(self):
        rsexecute.close()

    def test_scatter_gather_channel(self):
        for nchan in [128, 16]:
            m31cube = create_test_image(
                frequency=numpy.linspace(1e8, 1.1e8, nchan),
                polarisation_frame=PolarisationFrame("stokesI"),
            )

            for subimages in [16, 8, 2, 1]:
                image_list = rsexecute.execute(image_scatter_channels)(
                    m31cube, subimages=subimages
                )
                m31cuberec = rsexecute.execute(image_gather_channels)(
                    image_list, m31cube, subimages=subimages
                )
                m31cuberec = rsexecute.compute(m31cuberec, sync=True)
                diff = m31cube["pixels"].data - m31cuberec["pixels"].data
                assert numpy.max(numpy.abs(diff)) == 0.0, (
                    "Scatter gather failed for %d" % subimages
                )

    def test_gather_channel(self):
        for nchan in [128, 16]:
            m31cube = create_test_image(
                frequency=numpy.linspace(1e8, 1.1e8, nchan),
                polarisation_frame=PolarisationFrame("stokesI"),
            )
            image_list = rsexecute.execute(image_scatter_channels)(
                m31cube, subimages=nchan
            )
            m31cuberec = rsexecute.execute(image_gather_channels)(
                image_list, None, subimages=nchan
            )
            m31cuberec = rsexecute.compute(m31cuberec, sync=True)
            assert m31cube["pixels"].shape == m31cuberec["pixels"].shape
            diff = m31cube["pixels"].data - m31cuberec["pixels"].data
            assert numpy.max(numpy.abs(diff)) == 0.0, (
                "Scatter gather failed for %d" % nchan
            )

    def test_gather_channel_workflow(self):
        for nchan in [128, 16]:
            m31cube = create_test_image(
                frequency=numpy.linspace(1e8, 1.1e8, nchan),
                polarisation_frame=PolarisationFrame("stokesI"),
            )
            image_list = rsexecute.execute(image_scatter_channels, nout=nchan)(
                m31cube, subimages=nchan
            )
            image_list = rsexecute.compute(image_list, sync=True)
            m31cuberec = image_gather_channels_rsexecute(image_list)
            m31cuberec = rsexecute.compute(m31cuberec, sync=True)
            assert m31cube["pixels"].shape == m31cuberec["pixels"].shape
            diff = m31cube["pixels"].data - m31cuberec["pixels"].data
            assert numpy.max(numpy.abs(diff)) == 0.0, (
                "Scatter gather failed for %d" % nchan
            )


if __name__ == "__main__":
    unittest.main()
