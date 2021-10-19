""" Unit tests for pipelines expressed via rsexecute
"""

import functools
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    export_image_to_fits,
    smooth_image,
    qa_image,
)
from rascil.processing_components.griddata.kernels import (
    create_awterm_convolutionfunction,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import (
    create_named_configuration,
    decimate_configuration,
)
from rascil.processing_components.simulation import (
    ingest_unittest_visibility,
    create_unittest_model,
    insert_unittest_errors,
    create_unittest_components,
)
from rascil.processing_components.skycomponent.operations import (
    find_skycomponents,
    find_nearest_skycomponent,
    insert_skycomponent,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import (
    zero_list_rsexecute_workflow,
    predict_list_rsexecute_workflow,
    invert_list_rsexecute_workflow,
    subtract_list_rsexecute_workflow,
    weight_list_rsexecute_workflow,
    residual_list_rsexecute_workflow,
    sum_invert_results_rsexecute,
    restore_list_rsexecute_workflow,
)
from processing_components.imaging.imaging_helpers import sum_invert_results

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestImaging(unittest.TestCase):
    def setUp(self):

        rsexecute.set_client(use_dask=True)

        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

    def tearDown(self):
        rsexecute.close()

    def actualSetUp(
        self,
        add_errors=False,
        freqwin=3,
        dospectral=True,
        dopol=False,
        zerow=False,
    ):

        self.npixel = 256
        self.low = create_named_configuration("LOWBD2", rmax=750.0)
        self.low = decimate_configuration(self.low, skip=6)
        self.freqwin = freqwin
        self.bvis_list = list()
        self.ntimes = 5
        self.cellsize = 0.0005
        # Choose the interval so that the maximum change in w is smallish
        integration_time = numpy.pi * (24 / (12 * 60))
        self.times = numpy.linspace(
            -integration_time * (self.ntimes // 2),
            integration_time * (self.ntimes // 2),
            self.ntimes,
        )

        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(
                freqwin * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.frequency = numpy.array([1.0e8])
            self.channelwidth = numpy.array([4e7])

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
        self.bvis_list = [
            rsexecute.execute(ingest_unittest_visibility)(
                self.low,
                numpy.array([self.frequency[freqwin]]),
                numpy.array([self.channelwidth[freqwin]]),
                self.times,
                self.vis_pol,
                self.phasecentre,
                zerow=zerow,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.model_list = [
            rsexecute.execute(create_unittest_model, nout=freqwin)(
                self.bvis_list[freqwin],
                self.image_pol,
                cellsize=self.cellsize,
                npixel=self.npixel,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.components_list = [
            rsexecute.execute(create_unittest_components)(
                self.model_list[freqwin],
                flux[freqwin, :][numpy.newaxis, :],
                single=False,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.components_list = rsexecute.compute(self.components_list, sync=True)

        self.model_list = [
            rsexecute.execute(insert_skycomponent, nout=1)(
                self.model_list[freqwin], self.components_list[freqwin]
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.model_list = rsexecute.compute(self.model_list, sync=True)

        self.bvis_list = [
            rsexecute.execute(dft_skycomponent_visibility)(
                self.bvis_list[freqwin], self.components_list[freqwin]
            )
            for freqwin, _ in enumerate(self.frequency)
        ]
        centre = self.freqwin // 2
        # Calculate the model convolved with a Gaussian.
        self.model = self.model_list[centre]

        self.cmodel = smooth_image(self.model)
        if self.persist:
            export_image_to_fits(
                self.model, "%s/test_imaging_model.fits" % self.results_dir
            )
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_imaging_cmodel.fits" % self.results_dir
            )

        if add_errors:
            self.bvis_list = [
                rsexecute.execute(insert_unittest_errors)(self.bvis_list[i])
                for i, _ in enumerate(self.frequency)
            ]

        self.components = self.components_list[centre]

        self.gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )

    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=1.0):
        comps = find_skycomponents(
            dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5
        )
        assert len(comps) == len(
            self.components
        ), "Different number of components found: original %d, recovered %d" % (
            len(self.components),
            len(comps),
        )
        cellsize = numpy.deg2rad(abs(dirty.image_acc.wcs.wcs.cdelt[0]))

        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(
                comp.direction, self.components
            )
            if separation / cellsize > positionthreshold:
                raise ValueError(
                    "Component differs in position %.3f pixels"
                    % (separation / cellsize)
                )

    def _predict_base(
        self, context="ng", do_wstacking=False, extra="", fluxthreshold=1.0, **kwargs
    ):
        centre = self.freqwin // 2

        vis_list = zero_list_rsexecute_workflow(self.bvis_list)
        vis_list = predict_list_rsexecute_workflow(
            vis_list, self.model_list, context=context, **kwargs
        )
        vis_list = subtract_list_rsexecute_workflow(self.bvis_list, vis_list)
        vis_list = rsexecute.compute(vis_list, sync=True)

        dirty = invert_list_rsexecute_workflow(
            vis_list,
            self.model_list,
            context=context,
            dopsf=False,
            normalise=True,
            **kwargs
        )
        dirty = rsexecute.compute(dirty, sync=True)[centre]

        assert numpy.max(numpy.abs(dirty[0]["pixels"].data)), "Residual image is empty"
        if self.persist:
            export_image_to_fits(
                dirty[0],
                "%s/test_imaging_predict_%s%s_%s_dirty.fits"
                % (self.results_dir, context, extra, rsexecute.type()),
            )

        maxabs = numpy.max(numpy.abs(dirty[0]["pixels"].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (
            maxabs,
            fluxthreshold,
        )

    def _invert_base(
        self,
        context,
        extra="",
        fluxthreshold=2.0,
        positionthreshold=1.0,
        check_components=True,
        gcfcf=None,
        dopsf=False,
        **kwargs
    ):

        centre = self.freqwin // 2
        dirty = invert_list_rsexecute_workflow(
            self.bvis_list,
            self.model_list,
            context=context,
            dopsf=dopsf,
            normalise=True,
            gcfcf=gcfcf,
            **kwargs
        )
        dirty = rsexecute.compute(dirty, sync=True)[centre]

        if self.persist:
            if dopsf:
                export_image_to_fits(
                    dirty[0],
                    "%s/test_imaging_invert_%s%s_%s_psf.fits"
                    % (self.results_dir, context, extra, rsexecute.type()),
                )
            else:
                export_image_to_fits(
                    dirty[0],
                    "%s/test_imaging_invert_%s%s_%s_dirty.fits"
                    % (self.results_dir, context, extra, rsexecute.type()),
                )

        assert numpy.max(numpy.abs(dirty[0]["pixels"].data)), "Image is empty"

        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold)

    def test_predict_2d(self):
        self.actualSetUp(zerow=True)
        self._predict_base(context="ng", do_wstacking=False, fluxthreshold=3.0)

    def test_predict_ng(self):
        self.actualSetUp()
        self._predict_base(context="ng", fluxthreshold=0.62)

    def test_predict_wprojection(self):
        self.actualSetUp()
        self._predict_base(
            context="ng",
            do_wstacking=False,
            extra="_wprojection",
            fluxthreshold=5.0,
            gcfcf=self.gcfcf,
        )

    def test_invert_2d(self):
        self.actualSetUp(zerow=True)
        self._invert_base(
            context="ng",
            do_wstacking=False,
            positionthreshold=2.0,
            check_components=False,
        )

    def test_invert_2d_psf(self):
        self.actualSetUp(zerow=True)
        self._invert_base(
            context="ng",
            do_wstacking=False,
            positionthreshold=2.0,
            check_components=False,
            dopsf=True,
        )

    def test_invert_2d_uniform(self):
        self.actualSetUp(zerow=True)
        self.bvis_list = weight_list_rsexecute_workflow(
            self.bvis_list, self.model_list, weighting="uniform"
        )
        self._invert_base(
            context="ng",
            do_wstacking=False,
            extra="_uniform",
            positionthreshold=2.0,
            check_components=False,
        )

    def test_invert_2d_robust(self):
        self.actualSetUp(zerow=True)
        self.bvis_list = weight_list_rsexecute_workflow(
            self.bvis_list, self.model_list, weighting="robust", robustness=0.0
        )
        self._invert_base(
            context="ng",
            do_wstacking=False,
            extra="_uniform",
            positionthreshold=2.0,
            check_components=False,
        )

    def test_invert_ng(self):
        self.actualSetUp()
        self._invert_base(context="ng", positionthreshold=2.0, check_components=True)

    def test_invert_wprojection(self):
        self.actualSetUp()
        # We increase the threshold on the position check because w projection does a particularly
        # poor job on this test. We could improve the precision with a more finely sampled w kernel
        # but the test would run longer. Give that ng is much faster, wprojection is not worth the
        # extra time testing.
        self._invert_base(
            context="ng",
            do_wstacking=False,
            extra="_wprojection",
            positionthreshold=2.1,
            gcfcf=self.gcfcf,
        )

    def test_zero_list(self):
        self.actualSetUp()

        centre = self.freqwin // 2
        vis_list = zero_list_rsexecute_workflow(self.bvis_list)
        vis_list = rsexecute.compute(vis_list, sync=True)

        assert numpy.max(numpy.abs(vis_list[centre].vis)) < 1e-15, numpy.max(
            numpy.abs(vis_list[centre].vis)
        )

        predicted_vis_list = [
            rsexecute.execute(dft_skycomponent_visibility)(
                vis_list[freqwin], self.components_list[freqwin]
            )
            for freqwin, _ in enumerate(self.frequency)
        ]
        predicted_vis_list = rsexecute.compute(predicted_vis_list, sync=True)
        assert (
            numpy.max(numpy.abs(predicted_vis_list[centre].vis.data)) > 0.0
        ), numpy.max(numpy.abs(predicted_vis_list[centre].vis.data))

        diff_vis_list = subtract_list_rsexecute_workflow(
            self.bvis_list, predicted_vis_list
        )
        diff_vis_list = rsexecute.compute(diff_vis_list, sync=True)

        assert numpy.max(numpy.abs(diff_vis_list[centre].vis.data)) < 1e-15, numpy.max(
            numpy.abs(diff_vis_list[centre].vis.data)
        )

    def test_residual_list(self):
        self.actualSetUp(zerow=True)

        centre = self.freqwin // 2
        residual_image_list = residual_list_rsexecute_workflow(
            self.bvis_list, self.model_list, context="ng", do_wstacking=False
        )
        residual_image_list = rsexecute.compute(residual_image_list, sync=True)
        qa = qa_image(residual_image_list[centre][0])
        assert numpy.abs(qa.data["max"] - 0.32584463456508744) < 1.0, str(qa)
        assert numpy.abs(qa.data["min"] + 0.4559162232699305) < 1.0, str(qa)

    def test_restored_list(self):
        self.actualSetUp(zerow=True)

        centre = self.freqwin // 2
        psf_image_list = invert_list_rsexecute_workflow(
            self.bvis_list,
            self.model_list,
            context="ng",
            do_wstacking=False,
            dopsf=True,
        )
        residual_image_list = residual_list_rsexecute_workflow(
            self.bvis_list, self.model_list, context="ng", do_wstacking=False
        )
        restored_image_list = restore_list_rsexecute_workflow(
            self.model_list, psf_image_list, residual_image_list
        )
        restored_image_list = rsexecute.compute(restored_image_list, sync=True)

        if self.persist:
            export_image_to_fits(
                restored_image_list[centre],
                "%s/test_imaging_invert_%s_restored.fits"
                % (self.results_dir, rsexecute.type()),
            )

        qa = qa_image(restored_image_list[centre])
        assert numpy.abs(qa.data["max"] - 100.00000000000163) < 1e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 5.555050064664968e-12) < 1e-7, str(qa)

    def test_restored_list_noresidual(self):
        self.actualSetUp(zerow=True)

        centre = self.freqwin // 2
        psf_image_list = invert_list_rsexecute_workflow(
            self.bvis_list,
            self.model_list,
            context="ng",
            do_wstacking=False,
            dopsf=True,
        )
        restored_image_list = restore_list_rsexecute_workflow(
            self.model_list, psf_image_list
        )
        restored_image_list = rsexecute.compute(restored_image_list, sync=True)
        if self.persist:
            export_image_to_fits(
                restored_image_list[centre],
                "%s/test_imaging_invert_%s_restored_noresidual.fits"
                % (self.results_dir, rsexecute.type()),
            )

        qa = qa_image(restored_image_list[centre])
        assert numpy.abs(qa.data["max"] - 100.0) < 1e-7, str(qa)
        assert numpy.abs(qa.data["min"]) < 1e-7, str(qa)

    def test_restored_list_facet(self):
        self.actualSetUp(zerow=True)

        def copy_image(im):
            return im.copy(deep=True)

        original_model_1 = [
            rsexecute.execute(copy_image, nout=1)(im) for im in self.model_list
        ]
        original_model_2 = [
            rsexecute.execute(copy_image, nout=1)(im) for im in self.model_list
        ]

        residual_image_list = residual_list_rsexecute_workflow(
            self.bvis_list, self.model_list, context="ng", do_wstacking=False
        )
        centre = self.freqwin // 2
        psf_image_list = invert_list_rsexecute_workflow(
            self.bvis_list,
            self.model_list,
            context="ng",
            do_wstacking=False,
            dopsf=True,
        )
        psf_image_list = rsexecute.compute(psf_image_list, sync=True)
        clean_beam = {"bmaj": 0.12, "bmin": 0.1, "bpa": -0.8257413937065491}

        restored_2facets_image_list = restore_list_rsexecute_workflow(
            original_model_1,
            psf_image_list,
            residual_image_list,
            restore_facets=4,
            restore_overlap=8,
            clean_beam=clean_beam,
        )
        restored_2facets_image_list = rsexecute.compute(
            restored_2facets_image_list, sync=True
        )

        restored_1facets_image_list = restore_list_rsexecute_workflow(
            original_model_2,
            psf_image_list,
            residual_image_list,
            restore_facets=1,
            restore_overlap=0,
            clean_beam=clean_beam,
        )

        restored_1facets_image_list = rsexecute.compute(
            restored_1facets_image_list, sync=True
        )

        if self.persist:
            export_image_to_fits(
                restored_1facets_image_list[0],
                "%s/test_imaging_invert_%s_restored_1facets.fits"
                % (self.results_dir, rsexecute.type()),
            )
            export_image_to_fits(
                restored_2facets_image_list[0],
                "%s/test_imaging_invert_%s_restored_2facets.fits"
                % (self.results_dir, rsexecute.type()),
            )

        qa = qa_image(restored_2facets_image_list[centre])
        assert numpy.abs(qa.data["max"] - 100.00000000000163) < 1e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 8.72234065230259e-12) < 1e-7, str(qa)

        restored_2facets_image_list[centre][
            "pixels"
        ].data -= restored_1facets_image_list[centre]["pixels"].data
        if self.persist:
            export_image_to_fits(
                restored_2facets_image_list[centre],
                "%s/test_imaging_invert_%s_restored_2facets_error.fits"
                % (self.results_dir, rsexecute.type()),
            )
        qa = qa_image(restored_2facets_image_list[centre])
        assert numpy.abs(qa.data["max"] - 0.012401241830647329) < 1e-7, str(qa)
        assert numpy.abs(qa.data["min"] + 0.0010159473717538114) < 1e-7, str(qa)

    def test_sum_invert_list(self):
        self.actualSetUp(zerow=True)

        residual_image_list = residual_list_rsexecute_workflow(
            self.bvis_list, self.model_list, context="ng", do_wstacking=False
        )
        residual_image_list = rsexecute.compute(residual_image_list, sync=True)
        route2 = sum_invert_results(residual_image_list)
        route1 = sum_invert_results_rsexecute(residual_image_list)
        route1 = rsexecute.compute(route1, sync=True)
        for r in route1, route2:
            assert len(r) == 2
            qa = qa_image(r[0])
            assert numpy.abs(qa.data["max"] - 0.15513038832438183) < 1.0, str(qa)
            assert numpy.abs(qa.data["min"] + 0.4607090445091728) < 1.0, str(qa)
            assert numpy.abs(r[1] - 11700.0) < 1e-7, r


if __name__ == "__main__":
    unittest.main()
