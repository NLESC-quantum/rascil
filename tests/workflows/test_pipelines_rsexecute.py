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
from rascil.processing_components.image.gather_scatter import image_gather_channels
from rascil.processing_components.image.operations import (
    export_image_to_fits,
    qa_image,
    smooth_image,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import (
    create_named_configuration,
    decimate_configuration,
)
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

        self.results_dir = rascil_path("test_results")
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def tearDown(self):
        rsexecute.close()

    def actualSetUp(
        self, add_errors=False, nfreqwin=5, dospectral=True, dopol=False, zerow=False
    ):

        self.npixel = 512
        self.low = create_named_configuration("LOW", rmax=750.0)
        self.low = decimate_configuration(self.low, skip=3)
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
                self.bvis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.001
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
                model, "%s/test_pipelines_rsexecute_model.fits" % self.results_dir
            )
            export_image_to_fits(
                self.cmodel,
                "%s/test_pipelines_rsexecute_cmodel.fits" % self.results_dir,
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
                self.bvis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.001
            )
            for i in range(nfreqwin)
        ]
        self.model_imagelist = rsexecute.persist(self.model_imagelist, sync=True)

    def save_and_check(
        self, tag, clean, residual, restored, flux_max, flux_min, taylor=False
    ):

        if not taylor:
            clean = image_gather_channels(clean)
            residual = image_gather_channels([r[0] for r in residual])
            restored = image_gather_channels(restored)
            if self.persist:
                export_image_to_fits(
                    clean,
                    f"{self.results_dir }/test_pipelines_{tag}_rsexecute_deconvolved.fits",
                )
                export_image_to_fits(
                    residual,
                    f"{self.results_dir }/test_pipelines_{tag}_rsexecute_residual.fits",
                )
                export_image_to_fits(
                    restored,
                    f"{self.results_dir }/test_pipelines_{tag}_rsexecute_restored.fits",
                )
            qa = qa_image(restored)
            assert numpy.abs(qa.data["max"] - flux_max) < 1.0e-7, str(qa)
            assert numpy.abs(qa.data["min"] - flux_min) < 1.0e-7, str(qa)
        else:
            if self.persist:
                for moment, _ in enumerate(clean):
                    export_image_to_fits(
                        clean[moment],
                        f"{self.results_dir }/test_pipelines_{tag}_rsexecute_deconvolved_taylor{moment}.fits",
                    )
                for moment, _ in enumerate(clean):
                    export_image_to_fits(
                        residual[moment][0],
                        f"{self.results_dir }/test_pipelines_{tag}_rsexecute_residual_taylor{moment}.fits",
                    )
                for moment, _ in enumerate(clean):
                    export_image_to_fits(
                        restored[moment],
                        f"{self.results_dir }/test_pipelines_{tag}_rsexecute_restored_taylor{moment}.fits",
                    )
            qa = qa_image(restored[0])
            assert numpy.abs(qa.data["max"] - flux_max) < 1.0e-7, str(qa)
            assert numpy.abs(qa.data["min"] - flux_min) < 1.0e-7, str(qa)

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
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
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

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_rsexecute_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_empty",
            clean,
            residual,
            restored,
            116.76942026395899,
            -0.1883149303697396,
        )

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
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
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

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_threshold_rsexecute_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_empty_threshold",
            clean,
            residual,
            restored,
            116.76942026395895,
            -0.18831493036973768,
        )

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
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
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

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_exact",
            clean,
            residual,
            restored,
            116.9220347300889,
            -0.03710122171894675,
        )

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
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
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

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_partial",
            clean,
            residual,
            restored,
            116.8877123742255,
            -0.10058889464014169,
        )

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
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        self.save_and_check(
            "cip_skymodel_pipeline_empty",
            clean,
            residual,
            restored,
            116.80897017399957,
            -0.18430286786954833,
        )

    def test_continuum_imaging_skymodel_pipeline_empty_taylor_terms(self):
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
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            restored_output="taylor",
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        self.save_and_check(
            "cip_skymodel_pipeline_empty_taylor_terms",
            clean,
            residual,
            restored,
            101.20612449807776,
            -0.05374166540972006,
            taylor=True,
        )

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
            nmajor=3,
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

        self.save_and_check(
            "cip_skymodel_pipeline_empty_threshold_taylor",
            clean,
            residual,
            restored,
            116.84424764687517,
            -0.18926129194462596,
        )

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
            nmajor=3,
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

        self.save_and_check(
            "cip_skymodel_pipeline_partial",
            clean,
            residual,
            restored,
            116.87515181235142,
            -0.09463064597231315,
        )

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
            nmajor=3,
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

        self.save_and_check(
            "cip_skymodel_pipeline_exact",
            clean,
            residual,
            restored,
            116.90605597782766,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
