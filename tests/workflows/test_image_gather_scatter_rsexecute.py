"""Unit tests for image iteration


"""
import os
import logging
import unittest

import numpy

from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components.image.operations import export_image_to_fits
from rascil.processing_components.image.operations import create_empty_image_like
from rascil.processing_components.image.gather_scatter import image_gather_facets, image_scatter_facets, image_gather_channels, \
    image_scatter_channels
from rascil.processing_components.simulation import create_test_image
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)

class TestImageGatherScattersGraph(unittest.TestCase):
    
    def setUp(self):
        from distributed import Client
        client = Client(threads_per_worker=4, n_workers=4, processes=True)
        print(client)
        rsexecute.set_client(use_dask=True, client=client)
    
        from rascil.data_models.parameters import rascil_path, rascil_data_path
        self.dir = rascil_path('test_results')
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def tearDown(self):
        rsexecute.close()

    def test_scatter_gather_channel(self):
        for nchan in [128, 16]:
            m31cube = create_test_image(frequency=numpy.linspace(1e8, 1.1e8, nchan),
                                        polarisation_frame=PolarisationFrame('stokesI'))
            
            for subimages in [16, 8, 2, 1]:
                image_list = rsexecute.execute(image_scatter_channels)(m31cube, subimages=subimages)
                m31cuberec = rsexecute.execute( image_gather_channels)(image_list, m31cube, subimages=subimages)
                m31cuberec = rsexecute.compute(m31cuberec, sync=True)
                diff = m31cube["pixels"].data - m31cuberec["pixels"].data
                assert numpy.max(numpy.abs(diff)) == 0.0, "Scatter gather failed for %d" % subimages
    
    def test_gather_channel(self):
        for nchan in [128, 16]:
            m31cube = create_test_image(frequency=numpy.linspace(1e8, 1.1e8, nchan),
                                        polarisation_frame=PolarisationFrame('stokesI'))
            image_list = rsexecute.execute(image_scatter_channels)(m31cube, subimages=nchan)
            m31cuberec = rsexecute.execute(image_gather_channels)(image_list, None, subimages=nchan)
            m31cuberec = rsexecute.compute(m31cuberec, sync=True)
            assert m31cube.shape == m31cuberec.shape
            diff = m31cube["pixels"].data - m31cuberec["pixels"].data
            assert numpy.max(numpy.abs(diff)) == 0.0, "Scatter gather failed for %d" % nchan


if __name__ == '__main__':
    unittest.main()
