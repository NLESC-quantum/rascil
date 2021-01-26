""" Unit processing_components for rascil-imager

"""
import logging
import unittest
import numpy

from rascil.apps.rascil_imager import cli_parser, imager

from rascil.data_models.parameters import rascil_path

from rascil.processing_components import import_image_from_fits

from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from rascil.processing_components.image.operations import export_image_to_fits, qa_image, smooth_image
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from rascil.processing_components.simulation import simulate_gaintable
from rascil.processing_components.skycomponent.operations import insert_skycomponent
from rascil.processing_components import export_blockvisibility_to_ms, concatenate_blockvisibility_frequency
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.pipelines.pipeline_rsexecute import ical_list_rsexecute_workflow, \
    continuum_imaging_list_rsexecute_workflow
from rascil.workflows.rsexecute.pipelines.pipeline_skymodel_rsexecute import ical_skymodel_list_rsexecute_workflow, \
    continuum_imaging_skymodel_list_rsexecute_workflow

log = logging.getLogger('rascil-logger')
log.setLevel(logging.WARNING)

class TestRASCILimager(unittest.TestCase):
    
    def make_MS(self, add_errors=False, nfreqwin=5, dospectral=True, dopol=False, zerow=False):
        
        # We always want the same numbers
        from numpy.random import default_rng
        self.rng = default_rng(1805550721)


        rsexecute.set_client(use_dask=True)

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
        self.model_imagelist = rsexecute.persist(self.model_imagelist)
        
        self.components_list = [rsexecute.execute(create_unittest_components)
                                (self.model_imagelist[freqwin], flux[freqwin, :][numpy.newaxis, :])
                                for freqwin, m in enumerate(self.model_imagelist)]
        self.components_list = rsexecute.persist(self.components_list)
        
        self.bvis_list = [rsexecute.execute(dft_skycomponent_visibility)
                          (self.bvis_list[freqwin], self.components_list[freqwin])
                          for freqwin, _ in enumerate(self.bvis_list)]
        self.bvis_list = rsexecute.persist(self.bvis_list)
        
        self.model_imagelist = [rsexecute.execute(insert_skycomponent, nout=1)
                                (self.model_imagelist[freqwin], self.components_list[freqwin])
                                for freqwin in range(nfreqwin)]

        if self.persist:
            self.model_imagelist = rsexecute.compute(self.model_imagelist, sync=True)
        
            model = self.model_imagelist[0]
            self.cmodel = smooth_image(model)
            export_image_to_fits(model, '%s/test_rascil_imager_model.fits' % self.dir)
            export_image_to_fits(self.cmodel, '%s/test_rascil_imager_cmodel.fits' % self.dir)
        
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

        import shutil
        shutil.rmtree(rascil_path("test_results/test_rascil_imager.ms"), ignore_errors=True)
        self.bvis_list = rsexecute.compute(self.bvis_list, sync=True)
        self.bvis_list = [concatenate_blockvisibility_frequency(self.bvis_list)]
        export_blockvisibility_to_ms(rascil_path("test_results/test_rascil_imager.ms"),
                                     self.bvis_list)

        rsexecute.close()
    
    def setUp(self) -> None:
        
        self.persist = True
        self.dir = rascil_path('test_results')

        parser = cli_parser()
        self.args = parser.parse_args([])
        self.args.ingest_msname = rascil_path("test_results/test_rascil_imager.ms")
        self.args.ingest_vis_nchan = 1
        self.args.ingest_dds = range(7)
        self.args.ingest_chan_per_blockvis = 1
        self.args.imaging_npixel = 512
        self.args.imaging_cellsize = 0.0005
        
    
    def test_invert(self):
        
        self.make_MS(nfreqwin=7)
        
        self.args.mode = "invert"
        dirtyname = imager(self.args)
        dirty = import_image_from_fits(dirtyname)
        qa = qa_image(dirty)
        print(qa)

        assert dirty["pixels"].shape == (1, 1, self.args.imaging_npixel, self.args.imaging_npixel)
        numpy.testing.assert_allclose(qa.data['max'], 120.00887877621307, atol=1e-7)
        numpy.testing.assert_allclose(qa.data['min'], -15.736477105371158, atol=1e-7)

    def test_ical(self):
    
        self.make_MS(nfreqwin=7, add_errors=True)
    
        self.args.mode = "ical"
        self.args.ingest_vis_nchan = 7
        self.args.ingest_average_blockvis = False

        self.args.clean_nmajor = 5
        self.args.clean_niter = 1000
        self.args.clean_algorithm = "mmclean"
        self.args.clean_nmoment = 2
        self.args.clean_gain = 0.1
        self.args.clean_scales = [0]
        self.args.clean_threshold = 0.003
        self.args.clean_fractional_threshold = 0.3
        self.args.clean_facets = 1
        self.args.calibration_T_first_selfcal = 2
        self.args.calibration_T_phase_only = True
        self.args.calibration_T_timeslice = None
        self.args.calibration_G_first_selfcal = 5
        self.args.calibration_G_phase_only = False
        self.args.calibration_G_timeslice = 1200.0
        self.args.calibration_B_first_selfcal = 8
        self.args.calibration_B_phase_only = False
        self.args.calibration_B_timeslice = 1.0e5
        self.args.calibration_global_solution = True
        self.args.calibration_calibration_context = "TG"

        deconvolvedname, residualname, restoredname = imager(self.args)
        restored = import_image_from_fits(restoredname)
        qa = qa_image(restored)
        print(qa)
    
        assert restored["pixels"].shape == (1, 1, 512, 512)
        numpy.testing.assert_allclose(qa.data['max'], 100.21408262286327, atol=1e-7)
        numpy.testing.assert_allclose(qa.data['min'], -0.4232514054580332, atol=1e-7)

    def test_cip(self):
        
        self.make_MS(nfreqwin=7, add_errors=False)
    
        self.args.mode = "cip"
        self.args.ingest_vis_nchan = 7
        self.args.ingest_average_blockvis = False

        self.args.clean_nmajor = 5
        self.args.clean_niter = 1000
        self.args.clean_algorithm = "mmclean"
        self.args.clean_nmoment = 2
        self.args.clean_gain = 0.1
        self.args.clean_scales = [0]
        self.args.clean_threshold = 0.003
        self.args.clean_fractional_threshold = 0.3
        self.args.clean_facets = 1
    
        deconvolvedname, residualname, restoredname = imager(self.args)
        restored = import_image_from_fits(restoredname)
        qa = qa_image(restored)
        print(qa)
    
        assert restored["pixels"].shape == (1, 1, 512, 512)
        numpy.testing.assert_allclose(qa.data['max'], 101.17459680479055, atol=1e-7)
        numpy.testing.assert_allclose(qa.data['min'], -0.03995828592915829, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
