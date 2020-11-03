""" Unit tests for pipelines expressed via rsexecute
"""

import argparse
import logging
import os
import sys
import unittest

import numpy

from distributed import Client

# These are the RASCIL functions we need
from rascil.data_models import rascil_data_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_blockvisibility_from_ms, \
    concatenate_visibility,deconvolve_cube, restore_cube, image_gather_channels, \
    create_image_from_visibility
from rascil.processing_components.image.operations import export_image_to_fits, qa_image
from rascil.processing_components.visibility import blockvisibility_where

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute, \
    get_dask_client
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import invert_list_rsexecute_workflow

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))

class TestDPrepB(unittest.TestCase):
    def setUp(self):
        
        rsexecute.set_client(use_dask=True, verbose=True)
        print(rsexecute.client)
        print(rsexecute.client.scheduler)

        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def tearDown(self):
        if rsexecute.client is not None:
            print(rsexecute.client)
            print(rsexecute.client.scheduler)
        rsexecute.close()
    
    def test_pipeline(self):
        """
        This executes a DPREPB pipeline: deconvolution of calibrated spectral line data.

        """
        
        
        nchan = 40
        uvmax = 450.0
        cellsize = 0.00015
        npixel = 256
        
        context = '2d'
        
        input_vis = [rascil_data_path('vis/sim-1.ms'), rascil_data_path('vis/sim-2.ms')]
        
        import time
        
        start = time.time()
        
        # Define a function to be executed by Dask to load the data, combine it, and select
        # only the short baselines. We load each channel separately.
        def load_ms(c):
            v1 = create_blockvisibility_from_ms(input_vis[0], start_chan=c, end_chan=c)[0]
            v2 = create_blockvisibility_from_ms(input_vis[1], start_chan=c, end_chan=c)[0]
            vf = concatenate_visibility([v1, v2])
            vf.configuration.diameter[...] = 35.0
            vf = blockvisibility_where(vf, vf.uvdist_lambda < uvmax)
            return vf
        
        # Construct the graph to load the data and persist the graph on the Dask cluster.
        vis_list = [rsexecute.execute(load_ms)(c) for c in range(nchan)]
        vis_list = rsexecute.persist(vis_list)
        
        # Construct the graph to define the model images and persist the graph to the cluster
        model_list = [rsexecute.execute(create_image_from_visibility)
                      (v, npixel=npixel, cellsize=cellsize,
                       polarisation_frame=PolarisationFrame("stokesIQUV"),
                       nchan=1) for v in vis_list]
        model_list = rsexecute.compute(model_list, sync=True)
        model_list = rsexecute.scatter(model_list)
        
        # Construct the graphs to make the dirty image and psf, and persist these to the cluster
        dirty_list = invert_list_rsexecute_workflow(vis_list, template_model_imagelist=model_list, normalize=False,
                                                    context="2d")
        psf_list = invert_list_rsexecute_workflow(vis_list, template_model_imagelist=model_list, normalize=False,
                                                  context="2d", dopsf=True)
        
        # Construct the graphs to do the clean and restoration, and gather the channel images
        # into one image. Persist the graph on the cluster
        def deconvolve(d, p, m):
            c, resid = deconvolve_cube(d[0], p[0], m, threshold=0.01, fracthresh=0.01,
                                       window_shape='quarter', niter=100, gain=0.1,
                                       algorithm='hogbom-complex')
            r = restore_cube(c, p[0], resid)
            return r
        
        restored_list = [rsexecute.execute(deconvolve)(dirty_list[c], psf_list[c],
                                                       model_list[c])
                         for c in range(nchan)]
        restored_cube = rsexecute.execute(image_gather_channels, nout=1)(restored_list)
        # Up to this point all we have is a graph. Now we compute it and get the
        # final restored cleaned cube. During the compute, Dask shows diagnostic pages
        # at http://127.0.0.1:8787
        restored_cube = rsexecute.compute(restored_cube, sync=True)
        
        # Save the cube
        print("Processing took %.3f s" % (time.time() - start))
        export_image_to_fits(restored_cube,
                             '%s/test_dprepb_rsexecute_%s_clean_restored_cube.fits'
                             % (self.dir, context))
        
        qa = qa_image(restored_cube, context='CLEAN restored cube')
        print(qa)

        assert restored_cube["pixels"].data.shape == (40, 4, 256, 256)
        assert numpy.abs(qa.data['max'] - 4.01307082) < 1.0e-6, str(qa)
        assert numpy.abs(qa.data['min'] + 0.52919064) < 1.0e-6, str(qa)


if __name__ == '__main__':
    unittest.main()
