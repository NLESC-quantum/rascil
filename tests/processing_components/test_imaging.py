""" Unit tests for imaging functions


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
from rascil.data_models import get_parameter
from rascil.processing_components import weight_visibility
from rascil.processing_components.griddata.kernels import (
    create_awterm_convolutionfunction,
)
from rascil.processing_components.image.operations import (
    export_image_to_fits,
    smooth_image,
    qa_image,
)
from rascil.processing_components.imaging.imaging import (
    predict_blockvisibility,
    invert_blockvisibility,
)
from rascil.processing_components.imaging.dft import dft_skycomponent_visibility
from rascil.processing_components.imaging.primary_beams import create_pb_generic
from rascil.processing_components.simulation import (
    create_named_configuration,
    decimate_configuration,
)
from rascil.processing_components.simulation import (
    ingest_unittest_visibility,
    create_unittest_model,
    create_unittest_components,
)
from rascil.processing_components.skycomponent.operations import (
    find_skycomponents,
    find_nearest_skycomponent,
    insert_skycomponent,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestImaging2D(unittest.TestCase):
    def setUp(self):

        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

    def actualSetUp(
        self,
        freqwin=1,
        dospectral=True,
        image_pol=PolarisationFrame("stokesI"),
        zerow=False,
    ):

        self.npixel = 256
        self.low = create_named_configuration("LOWBD2", rmax=750.0)
        self.low = decimate_configuration(self.low, skip=3)
        self.freqwin = freqwin
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0

        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(
                freqwin * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.frequency = numpy.array([1e8])
            self.channelwidth = numpy.array([1e6])

        if image_pol == PolarisationFrame("stokesIQUV"):
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = image_pol
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        elif image_pol == PolarisationFrame("stokesIQ"):
            self.vis_pol = PolarisationFrame("linearnp")
            self.image_pol = image_pol
            f = numpy.array([100.0, 20.0])
        elif image_pol == PolarisationFrame("stokesIV"):
            self.vis_pol = PolarisationFrame("circularnp")
            self.image_pol = image_pol
            f = numpy.array([100.0, 20.0])
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
            ra=+180.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis = ingest_unittest_visibility(
            self.low,
            self.frequency,
            self.channelwidth,
            self.times,
            self.vis_pol,
            self.phasecentre,
            zerow=zerow,
        )

        self.model = create_unittest_model(
            self.vis, self.image_pol, npixel=self.npixel, nchan=freqwin
        )

        self.components = create_unittest_components(self.model, flux)

        self.model = insert_skycomponent(self.model, self.components)

        self.vis = dft_skycomponent_visibility(self.vis, self.components)

        # Calculate the model convolved with a Gaussian.

        self.cmodel = smooth_image(self.model)
        if self.persist:
            export_image_to_fits(
                self.model, "%s/test_imaging_model.fits" % self.results_dir
            )
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_imaging_cmodel.fits" % self.results_dir
            )

    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=0.1):
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
            assert separation / cellsize < positionthreshold, (
                "Component differs in position %.3f pixels" % separation / cellsize
            )

    def _predict_base(
        self,
        fluxthreshold=1.0,
        flux_max=0.0,
        flux_min=0.0,
        context="2d",
        gcfcf=None,
        **kwargs,
    ):

        if gcfcf is not None:
            context = "awprojection"

        vis = predict_blockvisibility(
            self.vis, self.model, context=context, gcfcf=gcfcf, **kwargs
        )

        vis["vis"].data = self.vis["vis"].data - vis["vis"].data
        dirty = invert_blockvisibility(
            vis,
            self.model,
            dopsf=False,
            normalise=True,
            context="2d",
        )

        if self.persist:
            export_image_to_fits(
                dirty[0],
                "%s/test_imaging_%s_residual.fits" % (self.results_dir, context),
            )
        for pol in range(dirty[0].image_acc.npol):
            assert numpy.max(
                numpy.abs(dirty[0]["pixels"].data[:, pol])
            ), "Residual image pol {} is empty".format(pol)

        maxabs = numpy.max(numpy.abs(dirty[0]["pixels"].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (
            maxabs,
            fluxthreshold,
        )
        qa = qa_image(dirty[0])
        numpy.testing.assert_allclose(
            qa.data["max"], flux_max, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["min"], flux_min, atol=1e-7, err_msg=f"{qa}"
        )

    def _invert_base(
        self,
        fluxthreshold=1.0,
        positionthreshold=1.0,
        check_components=True,
        flux_max=0.0,
        flux_min=0.0,
        context="2d",
        gcfcf=None,
        **kwargs,
    ):

        if gcfcf is not None:
            context = "awprojection"

        dirty = invert_blockvisibility(
            self.vis,
            self.model,
            dopsf=False,
            normalise=True,
            context=context,
            gcfcf=gcfcf,
            **kwargs,
        )

        if self.persist:
            export_image_to_fits(
                dirty[0], "%s/test_imaging_%s_dirty.fits" % (self.results_dir, context)
            )

        for pol in range(dirty[0].image_acc.npol):
            assert numpy.max(
                numpy.abs(dirty[0]["pixels"].data[:, pol])
            ), "Dirty image pol {} is empty".format(pol)
        for chan in range(dirty[0].image_acc.nchan):
            assert numpy.max(
                numpy.abs(dirty[0]["pixels"].data[chan])
            ), "Dirty image channel {} is empty".format(chan)

        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold)

        qa = qa_image(dirty[0])
        numpy.testing.assert_allclose(
            qa.data["max"], flux_max, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["min"], flux_min, atol=1e-7, err_msg=f"{qa}"
        )

    def test_predict_blockvisibility(self):
        self.actualSetUp(zerow=True)
        self._predict_base(
            name="predict_blockvisibility",
            flux_max=1.7506686178796016e-11,
            flux_min=-1.6386206755947555e-11,
        )

    def test_predict_blockvisibility_point(self):
        self.actualSetUp(zerow=True)
        self.model["pixels"].data[...] = 0.0
        nchan, npol, ny, nx = self.model.image_acc.shape
        self.model["pixels"].data[0, 0, ny // 2, nx // 2] = 1.0
        vis = predict_blockvisibility(self.vis, self.model, context="2d")
        assert numpy.max(numpy.abs(vis.vis - 1.0)) < 1e-12, numpy.max(
            numpy.abs(vis.vis - 1.0)
        )

    def test_predict_blockvisibility_point_IQUV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        self.model["pixels"].data[...] = 0.0
        nchan, npol, ny, nx = self.model.image_acc.shape
        self.model["pixels"].data[0, 0, ny // 2, nx // 2] = 1.0
        vis = predict_blockvisibility(self.vis, self.model, context="2d")
        assert numpy.max(numpy.abs(vis.vis[..., 0] - 1.0)) < 1e-12
        assert numpy.max(numpy.abs(vis.vis[..., 1])) < 1e-12
        assert numpy.max(numpy.abs(vis.vis[..., 2])) < 1e-12
        assert numpy.max(numpy.abs(vis.vis[..., 3] - 1.0)) < 1e-12

    def test_predict_blockvisibility_IQUV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        self._predict_base(
            name="predict_blockvisibility_IQUV",
            flux_max=1.7506197334688512e-11,
            flux_min=-1.6385712182817783e-11,
        )

    def test_predict_blockvisibility_IQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        self._predict_base(
            name="predict_blockvisibility_IQ",
            flux_max=1.7506197334688512e-11,
            flux_min=-1.6385712182817783e-11,
        )

    def test_predict_blockvisibility_IV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIV"))
        self._predict_base(
            name="predict_blockvisibility_IV",
            flux_max=1.7506197334688512e-11,
            flux_min=-1.6385712182817783e-11,
        )

    def test_invert_blockvisibility(self):
        self.actualSetUp(zerow=True)
        self._invert_base(
            name="blockvisibility",
            positionthreshold=2.0,
            check_components=False,
            context="ng",
            flux_max=100.92845444332372,
            flux_min=-8.116286458566002,
        )

    def test_invert_blockvisibility_IQUV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        self._invert_base(
            context="ng",
            name="invert_blockvisibility_IQUV",
            positionthreshold=2.0,
            check_components=True,
            flux_max=100.92845444332372,
            flux_min=-10.092845444332372,
        )

    def test_invert_blockvisibility_spec_I(self):
        self.actualSetUp(
            zerow=True,
            freqwin=4,
            image_pol=PolarisationFrame("stokesI"),
            dospectral=True,
        )
        self._invert_base(
            name="invert_blockvisibility_spec_I",
            context="ng",
            positionthreshold=2.0,
            check_components=True,
            flux_max=116.02263375798192,
            flux_min=-9.130114249590807,
        )

    def test_invert_blockvisibility_spec_IQUV(self):
        self.actualSetUp(
            zerow=True, freqwin=4, image_pol=PolarisationFrame("stokesIQUV")
        )
        self._invert_base(
            name="invert_blockvisibility_IQUV",
            positionthreshold=2.0,
            check_components=True,
            flux_max=115.83426630535374,
            flux_min=-11.583426630535378,
        )

    def test_invert_blockvisibility_IQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        self._invert_base(
            name="invert_blockvisibility_IQ",
            positionthreshold=2.0,
            check_components=True,
            context="ng",
            flux_max=100.92845444332372,
            flux_min=-8.116286458566002,
        )

    def test_invert_blockvisibility_IV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIV"))
        self._invert_base(
            name="invert_blockvisibility_IV",
            context="ng",
            positionthreshold=2.0,
            check_components=True,
            flux_max=100.92845444332372,
            flux_min=-8.116286458566002,
        )

    def test_predict_awterm(self):
        self.actualSetUp(zerow=False)
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            make_pb=make_pb,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        self._predict_base(
            fluxthreshold=62.0,
            name="predict_awterm",
            context="awprojection",
            gcfcf=gcfcf,
            flux_max=61.82267373099863,
            flux_min=-4.188093872633347,
        )

    def test_predict_awterm_spec(self):
        self.actualSetUp(zerow=False, freqwin=5)
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            make_pb=make_pb,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        self._predict_base(
            fluxthreshold=61.0,
            name="predict_awterm_spec",
            context="awprojection",
            gcfcf=gcfcf,
            flux_max=59.62485809400428,
            flux_min=-3.793824033959449,
        )

    def test_predict_awterm_spec_IQUV(self):
        self.actualSetUp(
            zerow=False, freqwin=5, image_pol=PolarisationFrame("stokesIQUV")
        )
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            make_pb=make_pb,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        self._predict_base(
            fluxthreshold=61.0,
            flux_max=59.62485809400428,
            flux_min=-5.9624858094004285,
            name="predict_awterm_spec_IQUV",
            gcfcf=gcfcf,
            context="awprojection",
        )

    def test_invert_awterm(self):
        self.actualSetUp(zerow=False)
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            make_pb=make_pb,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        self._invert_base(
            name="invert_awterm",
            positionthreshold=35.0,
            check_components=False,
            gcfcf=gcfcf,
            flux_max=96.69252147910645,
            flux_min=-6.110150403739334,
        )

    def test_invert_awterm_spec(self):
        self.actualSetUp(zerow=False, freqwin=5)
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            make_pb=make_pb,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        self._invert_base(
            name="invert_awterm_spec",
            positionthreshold=35.0,
            check_components=False,
            gcfcf=gcfcf,
            flux_max=110.98751973294647,
            flux_min=-8.991729415360501,
        )

    @unittest.skip("Too expensive for CI/CD")
    def test_invert_awterm_spec_IQUV(self):
        self.actualSetUp(
            zerow=False, freqwin=5, image_pol=PolarisationFrame("stokesIQUV")
        )
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            make_pb=make_pb,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        self._invert_base(
            name="invert_awterm_spec_IQUV",
            positionthreshold=35.0,
            check_components=False,
            gcfcf=gcfcf,
        )

    def test_predict_wterm(self):
        self.actualSetUp(zerow=False)
        gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        self._predict_base(
            fluxthreshold=5.0,
            name="predict_wterm",
            context="awprojection",
            gcfcf=gcfcf,
            flux_max=1.542478111903605,
            flux_min=-1.9124378846946475,
        )

    def test_invert_wterm(self):
        self.actualSetUp(zerow=False)
        gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        self._invert_base(
            name="invert_wterm",
            context="awprojection",
            positionthreshold=35.0,
            check_components=False,
            gcfcf=gcfcf,
            flux_max=100.29162257614617,
            flux_min=-8.34142746239203,
        )

    def test_invert_spec_wterm(self):

        self.actualSetUp(zerow=False, dospectral=True, freqwin=4)
        gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        self._invert_base(
            name="invert_spec_wterm",
            context="awprojection",
            positionthreshold=1.0,
            check_components=False,
            gcfcf=gcfcf,
            flux_max=114.50082196608498,
            flux_min=-9.16145050719757,
        )

    def test_invert_psf(self):
        self.actualSetUp(zerow=False)
        psf = invert_blockvisibility(self.vis, self.model, dopsf=True)
        error = numpy.max(psf[0]["pixels"].data) - 1.0
        assert abs(error) < 1.0e-12, error
        if self.persist:
            export_image_to_fits(
                psf[0], "%s/test_imaging_blockvisibility_psf.fits" % self.results_dir
            )

        assert numpy.max(numpy.abs(psf[0]["pixels"].data)), "Image is empty"

    def test_invert_psf_weighting(self):
        self.actualSetUp(zerow=False)
        for weighting in ["natural", "uniform", "robust"]:
            self.vis = weight_visibility(self.vis, self.model, weighting=weighting)
            psf = invert_blockvisibility(self.vis, self.model, dopsf=True)
            error = numpy.max(psf[0]["pixels"].data) - 1.0
            assert abs(error) < 1.0e-12, error
            if self.persist:
                export_image_to_fits(
                    psf[0],
                    "%s/test_imaging_blockvisibility_psf_%s.fits"
                    % (self.results_dir, weighting),
                )
            assert numpy.max(numpy.abs(psf[0]["pixels"].data)), "Image is empty"


if __name__ == "__main__":
    unittest.main()
