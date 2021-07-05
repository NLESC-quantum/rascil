""" Unit tests for pipelines expressed via dask.delayed


"""
import os
import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.parameters import rascil_path

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import (
    invert_list_rsexecute_workflow,
    deconvolve_list_rsexecute_workflow,
    residual_list_rsexecute_workflow,
    restore_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.processing_components import (
    export_image_to_fits,
    smooth_image,
    create_pb,
    qa_image,
    image_gather_channels,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import (
    ingest_unittest_visibility,
    create_unittest_model,
    create_unittest_components,
    insert_unittest_errors,
)
from rascil.workflows import remove_sumwt
from rascil.processing_components.skycomponent.operations import insert_skycomponent

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestImagingDeconvolveGraph(unittest.TestCase):
    def setUp(self):
        rsexecute.set_client(use_dask=True)
        self.dir = rascil_path("test_results")
        self.persist = os.getenv("RASCIL_PERSIST", True)

    def tearDown(self):
        rsexecute.close()

    def actualSetUp(
        self, add_errors=False, freqwin=5, dospectral=True, dopol=False, zerow=False
    ):

        self.npixel = 512
        self.low = create_named_configuration("LOWBD2", rmax=750.0)
        self.freqwin = freqwin
        self.vis_list = list()
        self.ntimes = 5
        cellsize = 0.0005
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)

        if freqwin > 1:
            self.channelwidth = numpy.array(
                freqwin * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.channelwidth = numpy.array([1e6])

        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
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
        self.vis_list = [
            rsexecute.execute(ingest_unittest_visibility)(
                self.low,
                [self.frequency[freqwin]],
                [self.channelwidth[freqwin]],
                self.times,
                self.vis_pol,
                self.phasecentre,
                zerow=zerow,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.model_imagelist = [
            rsexecute.execute(create_unittest_model, nout=freqwin)(
                self.vis_list[freqwin],
                self.image_pol,
                cellsize=cellsize,
                npixel=self.npixel,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]
        self.componentlist = [
            rsexecute.execute(create_unittest_components)(
                self.model_imagelist[freqwin], flux[freqwin, :][numpy.newaxis, :]
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.vis_list = [
            rsexecute.execute(dft_skycomponent_visibility)(
                self.vis_list[freqwin], self.componentlist[freqwin]
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        # Calculate the model convolved with a Gaussian.

        if self.persist:

            self.model_imagelist = [
                rsexecute.execute(insert_skycomponent, nout=1)(
                    self.model_imagelist[freqwin], self.componentlist[freqwin]
                )
                for freqwin, _ in enumerate(self.frequency)
            ]

            self.model_imagelist = rsexecute.compute(self.model_imagelist, sync=True)
            model = self.model_imagelist[0]

            self.cmodel = smooth_image(model)
            export_image_to_fits(
                model, "%s/test_imaging_deconvolve_rsexecute_model.fits" % self.dir
            )
            export_image_to_fits(
                self.cmodel,
                "%s/test_imaging_deconvolve_rsexecute_cmodel.fits" % self.dir,
            )

        if add_errors:
            self.vis_list = [
                rsexecute.execute(insert_unittest_errors)(self.vis_list[i])
                for i, _ in enumerate(self.frequency)
            ]

        #        self.vis_list = rsexecute.compute(self.vis_list, sync=True)
        self.vis_list = rsexecute.persist(self.vis_list)
        self.model_imagelist = rsexecute.scatter(self.model_imagelist)

        self.sensitivity_list = [
            rsexecute.execute(create_pb)(m, "LOW") for m in self.model_imagelist
        ]
        self.sensitivity_list = rsexecute.persist(self.sensitivity_list)

        self.model_imagelist = [
            rsexecute.execute(create_unittest_model, nout=freqwin)(
                self.vis_list[freqwin],
                self.image_pol,
                cellsize=cellsize,
                npixel=self.npixel,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

    def test_deconvolve_and_restore_cube_mmclean(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_list_rsexecute_workflow(
            self.vis_list,
            self.model_imagelist,
            context="ng",
            dopsf=False,
            normalise=True,
        )
        psf_imagelist = invert_list_rsexecute_workflow(
            self.vis_list,
            self.model_imagelist,
            context="ng",
            dopsf=True,
            normalise=True,
        )
        dirty_imagelist_trimmed = [
            rsexecute.execute(lambda x: x[0])(d) for d in dirty_imagelist
        ]
        psf_imagelist_trimmed = [
            rsexecute.execute(lambda x: x[0])(d) for d in psf_imagelist
        ]
        dec_imagelist = deconvolve_list_rsexecute_workflow(
            dirty_imagelist_trimmed,
            psf_imagelist_trimmed,
            self.model_imagelist,
            niter=100,
            fractional_threshold=0.01,
            scales=[0, 3],
            algorithm="mmclean",
            nmoment=2,
            nchan=self.freqwin,
            threshold=0.01,
            gain=0.7,
        )
        residual_imagelist = residual_list_rsexecute_workflow(
            self.vis_list, model_imagelist=dec_imagelist, context="ng"
        )
        restored_list = restore_list_rsexecute_workflow(
            model_imagelist=dec_imagelist,
            psf_imagelist=psf_imagelist,
            residual_imagelist=residual_imagelist,
            empty=self.model_imagelist,
        )

        restored = rsexecute.compute(restored_list, sync=True)
        restored = image_gather_channels(restored)

        self.save_and_check(
            34.025081369372046, -3.0862288540801703, restored, "mmclean"
        )

    def test_deconvolve_and_restore_cube_msclean(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_list_rsexecute_workflow(
            self.vis_list,
            self.model_imagelist,
            context="ng",
            dopsf=False,
            normalise=True,
        )
        psf_imagelist = invert_list_rsexecute_workflow(
            self.vis_list,
            self.model_imagelist,
            context="ng",
            dopsf=True,
            normalise=True,
        )
        dirty_imagelist_trimmed = [
            rsexecute.execute(lambda x: x[0])(d) for d in dirty_imagelist
        ]
        psf_imagelist_trimmed = [
            rsexecute.execute(lambda x: x[0])(d) for d in psf_imagelist
        ]
        dec_imagelist = deconvolve_list_rsexecute_workflow(
            dirty_imagelist_trimmed,
            psf_imagelist_trimmed,
            self.model_imagelist,
            niter=100,
            fractional_threshold=0.01,
            scales=[0, 3],
            algorithm="msclean",
            nchan=self.freqwin,
            threshold=0.01,
            gain=0.7,
        )
        residual_imagelist = residual_list_rsexecute_workflow(
            self.vis_list, model_imagelist=dec_imagelist, context="ng"
        )
        restored_list = restore_list_rsexecute_workflow(
            model_imagelist=dec_imagelist,
            psf_imagelist=psf_imagelist,
            residual_imagelist=residual_imagelist,
            empty=self.model_imagelist,
        )

        restored = rsexecute.compute(restored_list, sync=True)
        restored = image_gather_channels(restored)

        self.save_and_check(33.989223196919845, -3.376259696305599, restored, "msclean")

    def test_deconvolve_and_restore_cube_mmclean_facets(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_list_rsexecute_workflow(
            self.vis_list,
            self.model_imagelist,
            context="ng",
            dopsf=False,
            normalise=True,
        )
        psf_imagelist = invert_list_rsexecute_workflow(
            self.vis_list,
            self.model_imagelist,
            context="ng",
            dopsf=True,
            normalise=True,
        )
        dirty_imagelist_trimmed = [
            rsexecute.execute(lambda x: x[0])(d) for d in dirty_imagelist
        ]
        psf_imagelist_trimmed = [
            rsexecute.execute(lambda x: x[0])(d) for d in psf_imagelist
        ]
        dec_imagelist = deconvolve_list_rsexecute_workflow(
            dirty_imagelist_trimmed,
            psf_imagelist_trimmed,
            self.model_imagelist,
            niter=100,
            fractional_threshold=0.1,
            scales=[0, 3],
            algorithm="mmclean",
            nmoment=2,
            nchan=self.freqwin,
            threshold=0.01,
            gain=0.7,
            deconvolve_facets=4,
            deconvolve_overlap=8,
            deconvolve_taper="tukey",
        )
        residual_imagelist = residual_list_rsexecute_workflow(
            self.vis_list, model_imagelist=dec_imagelist, context="ng"
        )
        restored_list = restore_list_rsexecute_workflow(
            model_imagelist=dec_imagelist,
            psf_imagelist=psf_imagelist,
            residual_imagelist=residual_imagelist,
            empty=self.model_imagelist,
        )

        restored = rsexecute.compute(restored_list, sync=True)
        restored = image_gather_channels(restored)

        self.save_and_check(
            34.2025966873645, -3.8007230968610055, restored, "mmclean_facets"
        )

    def save_and_check(self, flux_max, flux_min, restored, tag):
        if self.persist:
            export_image_to_fits(
                restored,
                f"{self.dir}/test_imaging_deconvolve_rsexecute_{tag}_restored.fits",
            )
        qa = qa_image(restored)
        numpy.testing.assert_allclose(
            qa.data["max"], flux_max, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["min"], flux_min, atol=1e-7, err_msg=f"{qa}"
        )


if __name__ == "__main__":
    unittest.main()
