"""
This executes a DPREPB pipeline: deconvolution of calibrated spectral line data.

"""

import argparse
import logging

import numpy

from dask.distributed import Client

# These are the RASCIL functions we need
from rascil.data_models import PolarisationFrame, rascil_data_path
from rascil.processing_components import create_blockvisibility_from_ms, \
    concatenate_visibility, \
    deconvolve_cube, restore_cube, export_image_to_fits, qa_image, \
    image_gather_channels, create_image_from_visibility
from rascil.processing_components.visibility import blockvisibility_where

from rascil.workflows import invert_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute, get_dask_client

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark pipelines in numpy and dask')
    parser.add_argument('--use_dask', type=str, default='False', help='Use Dask?')
    parser.add_argument('--npixel', type=int, default=256,
                        help='Number of pixels per axis')
    parser.add_argument('--context', dest='context', default='ng',
                        help='Context: 2d|awprojection|ng')
    parser.add_argument('--nchan', type=int, default=40,
                        help='Number of channels to process')
    args = parser.parse_args()
    print(args)

    # Put the results in current directory
    results_dir = './'
    dask_dir = './dask-work-space'
    # Since the processing is distributed over multiple processes we have to tell each Dask worker
    # where to send the log messages
    def init_logging():
        logging.basicConfig(filename='%s/dprepb-pipeline.log' % results_dir,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


    log = logging.getLogger()
    logging.info("Starting Imaging pipeline")

    if args.use_dask:
        rsexecute.set_client(use_dask=True, verbose=True)
        print(rsexecute.client)
    else:
        rsexecute.set_client(use_dask=False)

    rsexecute.run(init_logging)

    nchan = args.nchan
    uvmax = 450.0
    cellsize = 0.00015
    npixel = args.npixel

    context = args.context

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
    model_list = rsexecute.persist(model_list)

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
    print(qa_image(restored_cube, context='CLEAN restored cube'))
    export_image_to_fits(restored_cube,
                         '%s/dprepb_rsexecute_%s_clean_restored_cube.fits'
                         % (results_dir, context))

    rsexecute.close()
