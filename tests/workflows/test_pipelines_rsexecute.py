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
from rascil.processing_components.calibration.chain_calibration import (
    create_calibration_controls,
)
from rascil.processing_components.calibration.operations import (
    create_gaintable_from_blockvisibility,
    apply_gaintable,
)
from rascil.processing_components.image.operations import (
    export_image_to_fits,
    qa_image,
    smooth_image,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import (
    ingest_unittest_visibility,
    create_unittest_model,
    create_unittest_components,
)
from rascil.processing_components.simulation import simulate_gaintable
from rascil.processing_components.skycomponent import (
    insert_skycomponent,
    copy_skycomponent,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.pipelines.pipeline_rsexecute import (
    ical_list_rsexecute_workflow,
    continuum_imaging_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.pipelines.pipeline_skymodel_rsexecute import (
    ical_skymodel_list_rsexecute_workflow,
    continuum_imaging_skymodel_list_rsexecute_workflow,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestPipelineGraphs(unittest.TestCase):
    def setUp(self):

        # We always want the same numbers
        from numpy.random import default_rng

        self.rng = default_rng(1805550721)

        rsexecute.set_client(use_dask=True)
        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def tearDown(self):
        rsexecute.close()

    def actualSetUp(
        self, add_errors=False, nfreqwin=5, dospectral=True, dopol=False, zerow=False
    ):

        self.npixel = 512
        self.low = create_named_configuration("LOWBD2", rmax=750.0)
        self.freqwin = nfreqwin
        self.ntimes = 3
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)

        if self.freqwin > 1:
            self.channelwidth = numpy.array(
                self.freqwin * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.channelwidth = numpy.array([1e6])

        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")
            f = numpy.array([100.0, 20.0, 0.0, 0.0])
        else:
            self.vis_pol = PolarisationFrame("stokesI")
            self.image_pol = PolarisationFrame("stokesI")
            f = numpy.array([100.0])

        if dospectral:
            flux = numpy.array(
                [f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency]
            )
        else:
            flux = numpy.array([f])

        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.bvis_list = [
            rsexecute.execute(ingest_unittest_visibility, nout=1)(
                self.low,
                [self.frequency[i]],
                [self.channelwidth[i]],
                self.times,
                self.vis_pol,
                self.phasecentre,
                zerow=zerow,
            )
            for i in range(nfreqwin)
        ]
        self.bvis_list = rsexecute.persist(self.bvis_list)

        self.model_imagelist = [
            rsexecute.execute(create_unittest_model, nout=1)(
                self.bvis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.0005
            )
            for i in range(nfreqwin)
        ]
        self.model_imagelist = rsexecute.persist(self.model_imagelist)

        self.components_list = [
            rsexecute.execute(create_unittest_components)(
                self.model_imagelist[freqwin], flux[freqwin, :][numpy.newaxis, :]
            )
            for freqwin, m in enumerate(self.model_imagelist)
        ]
        self.components_list = rsexecute.persist(self.components_list)

        self.bvis_list = [
            rsexecute.execute(dft_skycomponent_visibility)(
                self.bvis_list[freqwin], self.components_list[freqwin]
            )
            for freqwin, _ in enumerate(self.bvis_list)
        ]
        self.bvis_list = rsexecute.persist(self.bvis_list)

        self.model_imagelist = [
            rsexecute.execute(insert_skycomponent, nout=1)(
                self.model_imagelist[freqwin], self.components_list[freqwin]
            )
            for freqwin in range(nfreqwin)
        ]
        self.model_imagelist = rsexecute.persist(self.model_imagelist)

        if self.persist:
            model = rsexecute.compute(self.model_imagelist[0], sync=True)
            self.cmodel = smooth_image(model)

            export_image_to_fits(
                model, "%s/test_pipelines_rsexecute_model.fits" % self.dir
            )
            export_image_to_fits(
                self.cmodel, "%s/test_pipelines_rsexecute_cmodel.fits" % self.dir
            )

        if add_errors:
            seeds = [
                self.rng.integers(low=1, high=2 ** 32 - 1) for i in range(nfreqwin)
            ]
            if nfreqwin == 5:
                assert seeds == [
                    3822708302,
                    2154889844,
                    3073218956,
                    3754981936,
                    3778183766,
                ], seeds

            def sim_and_apply(vis, seed):
                gt = create_gaintable_from_blockvisibility(vis)
                gt = simulate_gaintable(
                    gt,
                    phase_error=0.1,
                    amplitude_error=0.0,
                    smooth_channels=1,
                    leakage=0.0,
                    seed=seed,
                )
                return apply_gaintable(vis, gt)

            # Do this without Dask since the random number generation seems to go wrong
            self.bvis_list = [
                rsexecute.execute(sim_and_apply)(self.bvis_list[i], seeds[i])
                for i in range(self.freqwin)
            ]
            self.bvis_list = rsexecute.compute(self.bvis_list, sync=True)
            self.bvis_list = rsexecute.scatter(self.bvis_list)

        self.model_imagelist = [
            rsexecute.execute(create_unittest_model, nout=1)(
                self.bvis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.0005
            )
            for i in range(nfreqwin)
        ]
        self.model_imagelist = rsexecute.persist(self.model_imagelist, sync=True)

    def test_continuum_imaging_pipeline(self):
        self.actualSetUp()

        continuum_imaging_list = continuum_imaging_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            context="ng",
            algorithm="mmclean",
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            do_wstacking=True,
        )

        clean, residual, restored = rsexecute.compute(continuum_imaging_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_continuum_imaging_pipeline_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_continuum_imaging_pipeline_rsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_continuum_imaging_pipeline_rsexecute_restored.fits"
                % self.dir,
            )

        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data["max"] - 100.03317941816398) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.14322579060089866) < 1.0e-7, str(qa)

    @unittest.skip("Not needed")
    def test_continuum_imaging_pipeline_pol(self):
        self.actualSetUp(add_errors=False, dopol=True)

        continuum_imaging_list = continuum_imaging_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            context="ng",
            algorithm="mmclean",
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            do_wstacking=True,
        )

        clean, residual, restored = rsexecute.compute(continuum_imaging_list, sync=True)

        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_continuum_imaging_pol_pipeline_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_continuum_imaging_pipeline_pol_rsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_continuum_imaging_pipeline_pol_rsexecute_restored.fits"
                % self.dir,
            )

        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data["max"] - 100.03317941816398) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.14322579060089866) < 1.0e-7, str(qa)

    def test_ical_pipeline(self):
        self.actualSetUp(add_errors=True)

        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        ical_list = ical_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        clean, residual, restored, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_ical_pipeline_rsexecute_clean.fits" % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_ical_pipeline_rsexecute_residual.fits" % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_ical_pipeline_rsexecute_restored.fits" % self.dir,
            )
            export_gaintable_to_hdf5(
                gt_list[centre]["T"],
                "%s/test_pipelines_ical_pipeline_rsexecute_gaintable.hdf5" % self.dir,
            )

        qa = qa_image(restored[centre])

        assert numpy.abs(qa.data["max"] - 100.04110915447107) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.13802297573216515) < 1.0e-7, str(qa)

    @unittest.skip("Not needed")
    def test_ical_pipeline_pol(self):
        self.actualSetUp(add_errors=True, dopol=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        ical_list = ical_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        clean, residual, restored, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_ical_pipeline_pol_rsexecute_clean.fits" % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_ical_pipeline__polrsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_ical_pipeline_pol_rsexecute_restored.fits"
                % self.dir,
            )
            export_gaintable_to_hdf5(
                gt_list[centre]["T"],
                "%s/test_pipelines_ical_pipeline_pol_rsexecute_gaintable.hdf5"
                % self.dir,
            )

        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data["max"] - 100.04262292227956) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.14023674377903816) < 1.0e-7, str(qa)

    @unittest.skip("Not needed")
    def test_ical_pipeline_global(self):
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        ical_list = ical_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=True,
        )
        clean, residual, restored, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_ical_global_pipeline_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_ical_global_pipeline_rsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_ical_global_pipeline_rsexecute_restored.fits"
                % self.dir,
            )
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_global_pipeline_rsexecute_gaintable.hdf5"
                % self.dir,
            )

        qa = qa_image(restored[centre])

        assert numpy.abs(qa.data["max"] - 99.2291010715883) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.566365494871639) < 1.0e-7, str(qa)

    @unittest.skip("Not needed")
    def test_ical_skymodel_pipeline_empty(self):
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_restored.fits"
                % self.dir,
            )
            export_gaintable_to_hdf5(
                gt_list[centre]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_gaintable.hdf5"
                % self.dir,
            )

        qa = qa_image(restored[centre], context="test_ical_skymodel_pipeline_empty")

        assert numpy.abs(qa.data["max"] - 100.03636094114569) < 1e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.1368896248916611) < 1e-7, str(qa)

    @unittest.skip("Not needed")
    def test_ical_skymodel_pipeline_empty_threshold(self):
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
            component_threshold=10.0,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_restored.fits"
                % self.dir,
            )
            export_gaintable_to_hdf5(
                gt_list[centre]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_rsexecute_gaintable.hdf5"
                % self.dir,
            )

        qa = qa_image(restored[centre], context="test_ical_skymodel_pipeline_empty")

        assert numpy.abs(qa.data["max"] - 100.03636094114567) < 1e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.13688962489166145) < 1e-7, str(qa)

    def test_ical_skymodel_pipeline_exact(self):

        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(
                components=comp_list, image=self.model_imagelist[icomp]
            )
            for icomp, comp_list in enumerate(self.components_list)
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_restored.fits"
                % self.dir,
            )
            export_gaintable_to_hdf5(
                gt_list[centre]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_gaintable.hdf5"
                % self.dir,
            )

        qa = qa_image(restored[centre], context="test_ical_skymodel_pipeline_exact")

        assert numpy.abs(qa.data["max"] - 100.01691390925012) < 1e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.015435903203822881) < 1e-7, str(qa)

    @unittest.skip("Not needed")
    def test_ical_skymodel_pipeline_partial(self):
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        def downscale(comp):
            comp.flux *= 0.5
            return comp

        def downscale_list(cl):
            return [downscale(comp) for comp in cl]

        scaled_components_list = [
            rsexecute.execute(downscale_list)(comp_list)
            for comp_list in self.components_list
        ]
        skymodel_list = [
            rsexecute.execute(SkyModel)(
                components=comp_list, image=self.model_imagelist[icomp]
            )
            for icomp, comp_list in enumerate(scaled_components_list)
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_restored.fits"
                % self.dir,
            )
            export_gaintable_to_hdf5(
                gt_list[centre]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_gaintable.hdf5"
                % self.dir,
            )

        qa = qa_image(restored[centre], context="test_ical_skymodel_pipeline_partial")

        assert numpy.abs(qa.data["max"] - 100.01749785037397) < 1e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.0683933958944093) < 1e-7, str(qa)

    @unittest.skip("Not needed")
    def test_continuum_imaging_skymodel_pipeline_empty(self):
        self.actualSetUp()

        continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=None,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_continuum_imaging_skymodel_empty_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_continuum_imaging_skymodel_empty_rsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_continuum_imaging_skymodel_empty_rsexecute_restored.fits"
                % self.dir,
            )

        qa = qa_image(restored[centre], context="restored")
        assert numpy.abs(qa.data["max"] - 100.03317941816397) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.14322579060089646) < 1.0e-7, str(qa)

    @unittest.skip("Not needed")
    def test_continuum_imaging_skymodel_pipeline_empty_threshold(self):
        self.actualSetUp()

        continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=None,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=1,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            component_threshold=10.0,
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_continuum_imaging_skymodel_empty_threshold_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_continuum_imaging_skymodel_empty_threshold_rsexecute_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_continuum_imaging_skymodel_empty_threshold_rsexecute_restored.fits"
                % self.dir,
            )

        qa = qa_image(restored[centre], context="restored")
        assert numpy.abs(qa.data["max"] - 100.01689049423403) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.06610616076695902) < 1.0e-7, str(qa)

    @unittest.skip("Not needed")
    def test_continuum_imaging_skymodel_pipeline_partial(self):
        self.actualSetUp()

        def downscale(comp):
            newcomp = copy_skycomponent(comp)
            newcomp.flux *= 0.5
            return newcomp

        def downscale_list(cl):
            return [downscale(comp) for comp in cl]

        scaled_components_list = [
            rsexecute.execute(downscale_list)(comp_list)
            for comp_list in self.components_list
        ]
        skymodel_list = [
            rsexecute.execute(SkyModel)(
                components=comp_list, image=self.model_imagelist[i]
            )
            for i, comp_list in enumerate(scaled_components_list)
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=1,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_continuum_imaging_skymodel_partial_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_continuum_imaging_skymodel_rsexecute_partial_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_continuum_imaging_skymodel_rsexecute_partial_restored.fits"
                % self.dir,
            )

        qa = qa_image(restored[centre], context="restored")

        assert numpy.abs(qa.data["max"] - 100.00784388600081) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.03304451079009792) < 1.0e-7, str(qa)

    @unittest.skip("Not needed")
    def test_continuum_imaging_skymodel_pipeline_exact(self):
        self.actualSetUp()

        skymodel_list = [
            rsexecute.execute(SkyModel)(
                components=comp_list, image=self.model_imagelist[i]
            )
            for i, comp_list in enumerate(self.components_list)
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="ng",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=1,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(
                clean[centre],
                "%s/test_pipelines_continuum_imaging_skymodel_exact_rsexecute_clean.fits"
                % self.dir,
            )
            export_image_to_fits(
                residual[centre][0],
                "%s/test_pipelines_continuum_imaging_skymodel_rsexecute_exact_residual.fits"
                % self.dir,
            )
            export_image_to_fits(
                restored[centre],
                "%s/test_pipelines_continuum_imaging_skymodel_rsexecute_exact_restored.fits"
                % self.dir,
            )

        qa = qa_image(restored[centre], context="restored")
        assert numpy.abs(qa.data["max"] - 100.0) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data["min"]) < 1.0e-12, str(qa)

        qa = qa_image(residual[centre][0], context="residual")
        assert numpy.abs(qa.data["max"]) < 1.0e-12, str(qa)
        assert numpy.abs(qa.data["min"]) < 1.0e-12, str(qa)


if __name__ == "__main__":
    unittest.main()
