"""Unit tests for pipelines expressed via dask.delayed


"""
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.data_model_helpers import export_gaintable_to_hdf5
from rascil.data_models.memory_data_models import SkyModel
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.calibration.chain_calibration import create_calibration_controls
from rascil.processing_components.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from rascil.processing_components.image.operations import export_image_to_fits, qa_image, smooth_image
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from rascil.processing_components.simulation import simulate_gaintable
from rascil.processing_components.skycomponent.operations import insert_skycomponent
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.pipelines.pipeline_rsexecute import ical_list_rsexecute_workflow, \
    continuum_imaging_list_rsexecute_workflow
from rascil.workflows.rsexecute.pipelines.pipeline_skymodel_rsexecute import ical_skymodel_list_rsexecute_workflow, \
    continuum_imaging_skymodel_list_rsexecute_workflow

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestPipelineGraphs(unittest.TestCase):
    
    def setUp(self):
    
        # We always want the same numbers
        from numpy.random import default_rng
        self.rng = default_rng(1805550721)

        rsexecute.set_client(use_dask=True)
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        self.persist = os.getenv("RASCIL_PERSIST", True)
    
    def tearDown(self):
        rsexecute.close()
    
    def actualSetUp(self, add_errors=False, nfreqwin=5, dospectral=True, dopol=False, zerow=False):
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = nfreqwin
        self.ntimes = 3
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        if self.freqwin > 1:
            self.channelwidth = numpy.array(self.freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.channelwidth = numpy.array([1e6])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, 0.0, 0.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.bvis_list = \
            [rsexecute.execute(ingest_unittest_visibility, nout=1)(self.low,
                                                                   [self.frequency[i]],
                                                                   [self.channelwidth[i]],
                                                                   self.times,
                                                                   self.vis_pol,
                                                                   self.phasecentre,
                                                                   zerow=zerow)
             for i in range(nfreqwin)]
        self.bvis_list = rsexecute.persist(self.bvis_list)
        
        self.model_imagelist = [rsexecute.execute(create_unittest_model, nout=1)
                                (self.bvis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.0005)
                                for i in range(nfreqwin)]
        self.model_imagelist = rsexecute.compute(self.model_imagelist, sync=True)
        
        self.components_list = [rsexecute.execute(create_unittest_components)
                                (self.model_imagelist[freqwin], flux[freqwin, :][numpy.newaxis, :])
                                for freqwin, m in enumerate(self.model_imagelist)]
        self.components_list = rsexecute.compute(self.components_list, sync=True)
        
        self.bvis_list = [rsexecute.execute(dft_skycomponent_visibility)
                              (self.bvis_list[freqwin], self.components_list[freqwin])
                              for freqwin, _ in enumerate(self.bvis_list)]
        self.bvis_list = rsexecute.compute(self.bvis_list, sync=True)
        self.bvis_list = rsexecute.scatter(self.bvis_list)
        
        self.model_imagelist = [rsexecute.execute(insert_skycomponent, nout=1)
                                (self.model_imagelist[freqwin], self.components_list[freqwin])
                                for freqwin in range(nfreqwin)]
        self.model_imagelist = rsexecute.compute(self.model_imagelist, sync=True)
        
        model = self.model_imagelist[0]
        self.cmodel = smooth_image(model)
        if self.persist:
            export_image_to_fits(model, '%s/test_pipelines_rsexecute_model.fits' % self.dir)
            export_image_to_fits(self.cmodel, '%s/test_pipelines_rsexecute_cmodel.fits' % self.dir)
        
        if add_errors:
            seeds = [self.rng.integers(low=1, high=2 ** 32 - 1) for i in range(nfreqwin)]
            if nfreqwin == 5:
                assert seeds == [3822708302, 2154889844, 3073218956, 3754981936, 3778183766], seeds
            
            def sim_and_apply(vis, seed):
                gt = create_gaintable_from_blockvisibility(vis)
                gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.0, smooth_channels=1,
                                        leakage=0.0, seed=seed)
                return apply_gaintable(vis, gt)
            
            # Do this without Dask since the random number generation seems to go wrong
            self.bvis_list = [rsexecute.execute(sim_and_apply)(self.bvis_list[i], seeds[i])
                              for i in range(self.freqwin)]
            self.bvis_list = rsexecute.compute(self.bvis_list, sync=True)
            self.bvis_list = rsexecute.scatter(self.bvis_list)
        
        self.model_imagelist = [rsexecute.execute(create_unittest_model, nout=1)
                                (self.bvis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.0005)
                                for i in range(nfreqwin)]
        self.model_imagelist = rsexecute.persist(self.model_imagelist, sync=True)
        
    def test_continuum_imaging_pipeline(self):
        self.actualSetUp()
        
        continuum_imaging_list = \
            continuum_imaging_list_rsexecute_workflow(self.bvis_list,
                                                      model_imagelist=self.model_imagelist,
                                                      context='ng',
                                                      algorithm='mmclean',
                                                      scales=[0],
                                                      niter=100, fractional_threshold=0.1, threshold=0.01,
                                                      nmoment=2,
                                                      nmajor=5, gain=0.7,
                                                      deconvolve_facets=4, deconvolve_overlap=32,
                                                      deconvolve_taper='tukey', psf_support=64,
                                                      restore_facets=4,
                                                      do_wstacking=True)
        
        clean, residual, restored = rsexecute.compute(continuum_imaging_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre], '%s/test_pipelines_continuum_imaging_pipeline_rsexecute_clean.fits' %
                                 self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_continuum_imaging_pipeline_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_continuum_imaging_pipeline_rsexecute_restored.fits' % self.dir)
        
        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 100.01718910094274) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.06593874966691299) < 1.0e-7, str(qa)
    
    def test_continuum_imaging_pipeline_pol(self):
        self.actualSetUp(add_errors=False, dopol=True)
        
        self.bvis_list = rsexecute.compute(self.bvis_list, sync=True)
        
        continuum_imaging_list = \
            continuum_imaging_list_rsexecute_workflow(self.bvis_list,
                                                      model_imagelist=self.model_imagelist,
                                                      context='ng',
                                                      algorithm='mmclean',
                                                      scales=[0],
                                                      niter=100, fractional_threshold=0.1, threshold=0.01,
                                                      nmoment=2,
                                                      nmajor=5, gain=0.7,
                                                      deconvolve_facets=4, deconvolve_overlap=32,
                                                      deconvolve_taper='tukey', psf_support=64,
                                                      restore_facets=1,
                                                      do_wstacking=True)
        
        clean, residual, restored = rsexecute.compute(continuum_imaging_list, sync=True)
        
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_continuum_imaging_pol_pipeline_rsexecute_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_continuum_imaging_pipeline_pol_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_continuum_imaging_pipeline_pol_rsexecute_restored.fits' % self.dir)

        # shape: '(1, 4, 512, 512)'
        # max: '100.0176908638641'
        # min: '-0.0658984665865698'
        # maxabs: '100.0176908638641'
        # rms: '1.956630666919213'
        # sum: '94254.37064262747'
        # medianabs: '0.00013371252335181178'
        # medianabsdevmedian: '0.00015014922996476563'
        # median: '4.191860637553575e-05'

        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 100.01718910094273) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.06593874966691313) < 1.0e-7, str(qa)
    
    def test_ical_pipeline(self):
        self.actualSetUp(add_errors=True)
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timeslice'] = 'auto'
        
        ical_list = \
            ical_list_rsexecute_workflow(self.bvis_list,
                                         model_imagelist=self.model_imagelist,
                                         context='ng',
                                         algorithm='mmclean', facets=1,
                                         scales=[0],
                                         niter=100, fractional_threshold=0.1, threshold=0.01,
                                         nmoment=2,
                                         nmajor=5, gain=0.7,
                                         deconvolve_facets=4, deconvolve_overlap=32,
                                         deconvolve_taper='tukey', psf_support=64,
                                         restore_facets=1,
                                         calibration_context='T', controls=controls, do_selfcal=True,
                                         global_solution=False)
        clean, residual, restored, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre], '%s/test_pipelines_ical_pipeline_rsexecute_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_ical_pipeline_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre], '%s/test_pipelines_ical_pipeline_rsexecute_restored.fits' % self.dir)
            export_gaintable_to_hdf5(gt_list[centre]['T'], '%s/test_pipelines_ical_pipeline_rsexecute_gaintable.hdf5' %
                                     self.dir)
        
        qa = qa_image(restored[centre])

        # shape: '(1, 1, 512, 512)'
        # max: '100.01806496946458'
        # min: '-100.01806496946458'
        # maxabs: '100.01806496946458'
        # rms: '3.8296135273845415'
        # sum: '78539.5285735257'
        # medianabs: '0.008886031696449033'
        # medianabsdevmedian: '0.008008004747634967'
        # median: '0.004751479717392764'

        assert numpy.abs(qa.data['max'] - 100.01806496946458) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.06555462924332975) < 1.0e-7, str(qa)
    
    def test_ical_pipeline_pol(self):
        self.actualSetUp(add_errors=True, dopol=True)
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timeslice'] = 'auto'
        
        ical_list = \
            ical_list_rsexecute_workflow(self.bvis_list,
                                         model_imagelist=self.model_imagelist,
                                         context='ng',
                                         algorithm='mmclean', facets=1,
                                         scales=[0],
                                         niter=100, fractional_threshold=0.1, threshold=0.01,
                                         nmoment=2,
                                         nmajor=5, gain=0.7,
                                         deconvolve_facets=4, deconvolve_overlap=32,
                                         deconvolve_taper='tukey', psf_support=64,
                                         restore_facets=1,
                                         calibration_context='T', controls=controls, do_selfcal=True,
                                         global_solution=False)
        clean, residual, restored, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre], '%s/test_pipelines_ical_pipeline_pol_rsexecute_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_ical_pipeline__polrsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_ical_pipeline_pol_rsexecute_restored.fits' % self.dir)
            export_gaintable_to_hdf5(gt_list[centre]['T'],
                                     '%s/test_pipelines_ical_pipeline_pol_rsexecute_gaintable.hdf5' %
                                     self.dir)

        # shape: '(1, 4, 512, 512)'
        # max: '100.01794300704981'
        # min: '-0.06614721555038076'
        # maxabs: '100.01794300704981'
        # rms: '1.9566271382605045'
        # sum: '94247.42565540374'
        # medianabs: '0.00012119260895500113'
        # medianabsdevmedian: '0.00013779748982964998'
        # median: '3.7803846884311977e-05'

        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 100.01614515505372) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.06575395680077878) < 1.0e-7, str(qa)

    def test_ical_pipeline_global(self):
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timeslice'] = 'auto'
        
        ical_list = \
            ical_list_rsexecute_workflow(self.bvis_list,
                                         model_imagelist=self.model_imagelist,
                                         context='ng',
                                         algorithm='mmclean', facets=1,
                                         scales=[0],
                                         niter=100, fractional_threshold=0.1, threshold=0.01,
                                         nmoment=2,
                                         nmajor=5, gain=0.7,
                                         deconvolve_facets=4, deconvolve_overlap=32,
                                         deconvolve_taper='tukey', psf_support=64,
                                         restore_facets=1,
                                         calibration_context='T', controls=controls, do_selfcal=True,
                                         global_solution=True)
        clean, residual, restored, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_ical_global_pipeline_rsexecute_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_ical_global_pipeline_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_ical_global_pipeline_rsexecute_restored.fits' % self.dir)
            export_gaintable_to_hdf5(gt_list[0]['T'],
                                     '%s/test_pipelines_ical_global_pipeline_rsexecute_gaintable.hdf5' %
                                     self.dir)
        
        qa = qa_image(restored[centre])

        # shape: '(1, 1, 512, 512)'
        # max: '99.22485878707184'
        # min: '-0.6086389127999817'
        # maxabs: '99.22485878707184'
        # rms: '3.8003164925182054'
        # sum: '77899.8922405312'
        # medianabs: '0.07818787845745906'
        # medianabsdevmedian: '0.07802338649921994'
        # median: '0.00814423022916257'

        assert numpy.abs(qa.data['max'] - 99.2248587870718) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.6086389127999817) < 1.0e-7, str(qa)

    @unittest.skip("Non-deterministic")
    def test_ical_skymodel_pipeline_empty(self):
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timeslice'] = 'auto'

        skymodel_list = [rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = \
            ical_skymodel_list_rsexecute_workflow(self.bvis_list,
                                                  model_imagelist=self.model_imagelist,
                                                  skymodel_list=skymodel_list,
                                                  context='ng',
                                                  algorithm='mmclean', facets=1,
                                                  scales=[0],
                                                  niter=100, fractional_threshold=0.1, threshold=0.01,
                                                  nmoment=2,
                                                  nmajor=5, gain=0.7,
                                                  deconvolve_facets=4, deconvolve_overlap=32,
                                                  deconvolve_taper='tukey', psf_support=64,
                                                  restore_facets=1,
                                                  calibration_context='T', controls=controls, do_selfcal=True,
                                                  global_solution=False)
        clean, residual, restored, skymodel_list, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_restored.fits' % self.dir)
            export_gaintable_to_hdf5(gt_list[centre]['T'],
                                     '%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_gaintable.hdf5' %
                                     self.dir)
        
        qa = qa_image(restored[centre], context='test_ical_skymodel_pipeline_empty')

        # shape: '(1, 1, 512, 512)'
        # max: '100.07184138596327'
        # min: '-0.1291884422737903'
        # maxabs: '100.07184138596327'
        # rms: '3.7196897293378886'
        # sum: '72967.68130171177'
        # medianabs: '0.021580552027293375'
        # medianabsdevmedian: '0.02158925516978749'
        # median: '0.00039522135548293363'

        assert numpy.abs(qa.data['max'] - 100.07184138596327) < 1e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.1291884422737903) < 1e-7, str(qa)
    
    @unittest.skip("Non-deterministic")
    def test_ical_skymodel_pipeline_exact(self):
        
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timeslice'] = 'auto'
        
        skymodel_list = [rsexecute.execute(SkyModel)(components=comp_list)
                         for comp_list in self.components_list]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = \
            ical_skymodel_list_rsexecute_workflow(self.bvis_list,
                                                  model_imagelist=self.model_imagelist,
                                                  skymodel_list=skymodel_list,
                                                  context='ng',
                                                  algorithm='mmclean', facets=1,
                                                  scales=[0],
                                                  niter=100, fractional_threshold=0.1, threshold=0.01,
                                                  nmoment=2,
                                                  nmajor=5, gain=0.7,
                                                  deconvolve_facets=4, deconvolve_overlap=32,
                                                  deconvolve_taper='tukey', psf_support=64,
                                                  restore_facets=1,
                                                  calibration_context='T', controls=controls, do_selfcal=True,
                                                  global_solution=False)
        clean, residual, restored, sky_model_list, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_restored.fits' % self.dir)
            export_gaintable_to_hdf5(gt_list[centre]['T'],
                                     '%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_gaintable.hdf5' %
                                     self.dir)
        
        qa = qa_image(restored[centre], context='test_ical_skymodel_pipeline_exact')

        # shape: '(1, 1, 512, 512)'
        # max: '100.00008262713018'
        # min: '-0.0002528759240868758'
        # maxabs: '100.00008262713018'
        # rms: '3.7286679080216034'
        # sum: '73303.14947494591'
        # medianabs: '3.7566942647849636e-05'
        # medianabsdevmedian: '3.455461115723066e-05'
        # median: '1.885354764046226e-05'

        assert numpy.abs(qa.data['max'] - 100.02345654897059) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.0044566100882072355) < 1.0e-7, str(qa)
    
    @unittest.skip("Non-deterministic")
    def test_ical_skymodel_pipeline_partial(self):
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timeslice'] = 'auto'
        
        def downscale(comp):
            comp.flux *= 0.5
            return comp

        def downscale_list(cl):
            return [downscale(comp) for comp in cl]

        scaled_components_list = [rsexecute.execute(downscale_list)(comp_list)
                                  for comp_list in self.components_list]
        skymodel_list = [rsexecute.execute(SkyModel)(components=comp_list)
                         for comp_list in scaled_components_list]
        skymodel_list = rsexecute.compute(skymodel_list, sync=True)
        skymodel_list = rsexecute.scatter(skymodel_list)
        
        ical_list = \
            ical_skymodel_list_rsexecute_workflow(self.bvis_list,
                                                  model_imagelist=self.model_imagelist,
                                                  skymodel_list=skymodel_list,
                                                  context='ng',
                                                  algorithm='mmclean', facets=1,
                                                  scales=[0],
                                                  niter=100, fractional_threshold=0.1, threshold=0.01,
                                                  nmoment=2,
                                                  nmajor=5, gain=0.7,
                                                  deconvolve_facets=4, deconvolve_overlap=32,
                                                  deconvolve_taper='tukey', psf_support=64,
                                                  restore_facets=1,
                                                  calibration_context='T', controls=controls, do_selfcal=True,
                                                  global_solution=False)
        clean, residual, restored, skymodel_list, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_restored.fits' % self.dir)
            export_gaintable_to_hdf5(gt_list[centre]['T'],
                                     '%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_gaintable.hdf5' %
                                     self.dir)
        
        qa = qa_image(restored[centre], context='test_ical_skymodel_pipeline_partial')
        print(qa)
        
        assert numpy.abs(qa.data['max'] - 100.0172263142844) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.05661951481124512) < 1.0e-7, str(qa)
    
    def test_continuum_imaging_skymodel_pipeline_empty(self):
        self.actualSetUp()
        
        skymodel_list = [rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist]
        skymodel_list = rsexecute.persist(skymodel_list)
        
        continuum_imaging_list = \
            continuum_imaging_skymodel_list_rsexecute_workflow(self.bvis_list,
                                                               model_imagelist=self.model_imagelist,
                                                               skymodel_list=skymodel_list,
                                                               context='ng',
                                                               algorithm='mmclean', facets=1,
                                                               scales=[0],
                                                               niter=100, fractional_threshold=0.1, threshold=0.01,
                                                               nmoment=2,
                                                               nmajor=5, gain=0.7, deconvolve_facets=4,
                                                               deconvolve_overlap=32, deconvolve_taper='tukey',
                                                               psf_support=64, restore_facets=1)
        clean, residual, restored, skymodel = rsexecute.compute(continuum_imaging_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_continuum_imaging_skymodel_empty_rsexecute_clean.fits' %
                                 self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_continuum_imaging_skymodel_empty_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_continuum_imaging_skymodel_empty_rsexecute_restored.fits' % self.dir)
        
        qa = qa_image(restored[centre], context='restored')
        assert numpy.abs(qa.data['max'] - 100.01686183120408) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.06593874966691259) < 1.0e-7, str(qa)
    
    def test_continuum_imaging_skymodel_pipeline_partial(self):
        self.actualSetUp()
        
        def downscale(comp):
            comp.flux *= 0.5
            return comp

        def downscale_list(cl):
            return [downscale(comp) for comp in cl]

        scaled_components_list = [rsexecute.execute(downscale_list)(comp_list)
                                  for comp_list in self.components_list]
        skymodel_list = [rsexecute.execute(SkyModel)(components=comp_list)
                         for comp_list in scaled_components_list]
        skymodel_list = rsexecute.compute(skymodel_list, sync=True)
        skymodel_list = rsexecute.scatter(skymodel_list)

        continuum_imaging_list = \
            continuum_imaging_skymodel_list_rsexecute_workflow(self.bvis_list,
                                                               model_imagelist=self.model_imagelist,
                                                               skymodel_list=skymodel_list,
                                                               context='ng',
                                                               algorithm='mmclean', facets=1,
                                                               scales=[0],
                                                               niter=100, fractional_threshold=0.1, threshold=0.01,
                                                               nmoment=2,
                                                               nmajor=5, gain=0.7, deconvolve_facets=4,
                                                               deconvolve_overlap=32, deconvolve_taper='tukey',
                                                               psf_support=64, restore_facets=1)
        clean, residual, restored, skymodel = rsexecute.compute(continuum_imaging_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_continuum_imaging_skymodel_partial_rsexecute_clean.fits' %
                                 self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_continuum_imaging_skymodel_rsexecute_partial_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_continuum_imaging_skymodel_rsexecute_partial_restored.fits' % self.dir)
        
        qa = qa_image(restored[centre], context='restored')
        
        assert numpy.abs(qa.data['max'] - 100.00770215156139) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.03303737521096155) < 1.0e-7, str(qa)
    
    def test_continuum_imaging_skymodel_pipeline_exact(self):
        self.actualSetUp()
        
        skymodel_list = [rsexecute.execute(SkyModel)(components=comp_list) for comp_list in self.components_list]
        skymodel_list = rsexecute.persist(skymodel_list)
        
        continuum_imaging_list = \
            continuum_imaging_skymodel_list_rsexecute_workflow(self.bvis_list, model_imagelist=self.model_imagelist,
                                                               skymodel_list=skymodel_list,
                                                               context='ng',
                                                               algorithm='mmclean', facets=1,
                                                               scales=[0],
                                                               niter=100, fractional_threshold=0.1, threshold=0.01,
                                                               nmoment=2,
                                                               nmajor=5, gain=0.7,
                                                               deconvolve_facets=4, deconvolve_overlap=32,
                                                               deconvolve_taper='tukey', psf_support=64,
                                                               restore_facets=1)
        clean, residual, restored, skymodel = rsexecute.compute(continuum_imaging_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_continuum_imaging_skymodel_exact_rsexecute_clean.fits' %
                                 self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_continuum_imaging_skymodel_rsexecute_exact_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_continuum_imaging_skymodel_rsexecute_exact_restored.fits' % self.dir)
        
        qa = qa_image(restored[centre], context='restored')
        assert numpy.abs(qa.data['max'] - 100.0) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min']) < 1.0e-12, str(qa)
        
        qa = qa_image(residual[centre][0], context='residual')
        assert numpy.abs(qa.data['max']) < 1.0e-12, str(qa)
        assert numpy.abs(qa.data['min']) < 1.0e-12, str(qa)


if __name__ == '__main__':
    unittest.main()
