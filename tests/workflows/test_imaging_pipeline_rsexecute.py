""" Unit tests for pipelines expressed via rsexecute
"""

import os
import pprint
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt

# These are the RASCIL functions we need
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import show_image, export_image_to_fits, qa_image, \
    create_low_test_image_from_gleam, create_image_from_visibility, advise_wide_field
from rascil.workflows import invert_list_rsexecute_workflow, predict_list_rsexecute_workflow, \
    continuum_imaging_list_rsexecute_workflow, simulate_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute, get_dask_client

pp = pprint.PrettyPrinter()

import logging

log = logging.getLogger("rascil-logger")
logging.info("Starting imaging-pipeline")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

class TestImagingPipeline(unittest.TestCase):
    def setUp(self):

        rsexecute.set_client(use_dask=True, verbose=True)

        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def tearDown(self):
        rsexecute.close()
    
    def test_pipeline(self):
        nfreqwin = 5
        ntimes = 5
        rmax = 300.0
        frequency = numpy.linspace(1e8, 1.2e8, nfreqwin)
        if nfreqwin > 1:
            channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
        else:
            channel_bandwidth = numpy.array([2e7])
        times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
        phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        
        bvis_list = simulate_list_rsexecute_workflow('LOWBD2',
                                                     frequency=frequency,
                                                     channel_bandwidth=channel_bandwidth,
                                                     times=times,
                                                     phasecentre=phasecentre,
                                                     order='frequency',
                                                     rmax=rmax, format='blockvis',
                                                     zerow=True)
        
        log.info('%d elements in vis_list' % len(bvis_list))
        log.info('About to make visibility')
        bvis_list = rsexecute.compute(bvis_list, sync=True)
        
        advice_low = advise_wide_field(bvis_list[0], guard_band_image=8.0, delA=0.02)
        advice_high = advise_wide_field(bvis_list[-1], guard_band_image=8.0, delA=0.02)
        
        npixel = advice_high['npixels2']
        cellsize = min(advice_low['cellsize'], advice_high['cellsize'])
        
        gleam_model = [rsexecute.execute(create_low_test_image_from_gleam, nout=1)
                       (npixel=npixel,
                        frequency=[frequency[f]],
                        channel_bandwidth=[channel_bandwidth[f]],
                        cellsize=cellsize,
                        phasecentre=phasecentre,
                        polarisation_frame=PolarisationFrame("stokesI"),
                        flux_limit=1.0,
                        applybeam=True)
                       for f, freq in enumerate(frequency)]
        log.info('About to make GLEAM model')
        gleam_model = rsexecute.persist(gleam_model)
        
        log.info('About to run predict to get predicted visibility')
        future_vis_graph = rsexecute.scatter(bvis_list)
        predicted_vislist = predict_list_rsexecute_workflow(future_vis_graph, gleam_model,
                                                            context='2d')
        predicted_vislist = rsexecute.persist(predicted_vislist)
        
        model_list = [rsexecute.execute(create_image_from_visibility, nout=1)(bvis_list[f],
                                                                      npixel=npixel,
                                                                      frequency=[frequency[f]],
                                                                      channel_bandwidth=[channel_bandwidth[f]],
                                                                      cellsize=cellsize,
                                                                      phasecentre=phasecentre,
                                                                      polarisation_frame=PolarisationFrame("stokesI"),
                                                                      chunksize=None)
                      for f, freq in enumerate(frequency)]
        # Works ok if the model_list is precalculated
        model_list = rsexecute.compute(model_list, sync=True)

        dirty_list = invert_list_rsexecute_workflow(predicted_vislist, model_list, context='2d')
        psf_list = invert_list_rsexecute_workflow(predicted_vislist, model_list, context='2d', dopsf=True)
        
        dirty_list = rsexecute.compute(dirty_list, sync=True)
        dirty = dirty_list[0][0]
        show_image(dirty, cm='Greys')
        plt.show()
        
        psf_list = rsexecute.compute(psf_list, sync=True)
        psf = psf_list[0][0]
        show_image(psf, cm='Greys')
        plt.show()

        continuum_imaging_list = \
            continuum_imaging_list_rsexecute_workflow(predicted_vislist,
                                                      model_imagelist=model_list,
                                                      context='2d',
                                                      algorithm='mmclean',
                                                      scales=[0],
                                                      niter=100, fractional_threshold=0.1,
                                                      threshold=0.01,
                                                      nmoment=1,
                                                      nmajor=5, gain=0.7,
                                                      deconvolve_facets=4,
                                                      deconvolve_overlap=32,
                                                      deconvolve_taper='tukey',
                                                      psf_support=64)
                
        centre = nfreqwin // 2
        continuum_imaging_list = rsexecute.compute(continuum_imaging_list, sync=True)
        deconvolved = continuum_imaging_list[0][centre]
        residual = continuum_imaging_list[1][centre]
        restored = continuum_imaging_list[2][centre]
        
        f = show_image(deconvolved, title='Clean image - no selfcal', cm='Greys',
                       vmax=0.1, vmin=-0.01)
        log.info(qa_image(deconvolved, context='Clean image - no selfcal'))
        
        plt.show()
        
        f = show_image(restored, title='Restored clean image - no selfcal',
                       cm='Greys', vmax=1.0, vmin=-0.1)
        log.info(qa_image(restored, context='Restored clean image - no selfcal'))
        plt.show()
        export_image_to_fits(restored, '%s/imaging-dask_continuum_imaging_restored.fits'
                             % (self.dir))
        
        f = show_image(residual[0], title='Residual clean image - no selfcal', cm='Greys')
        log.info(qa_image(residual[0], context='Residual clean image - no selfcal'))
        plt.show()
        export_image_to_fits(residual[0], '%s/imaging-dask_continuum_imaging_residual.fits'
                             % (self.dir))

if __name__ == '__main__':
    unittest.main()
