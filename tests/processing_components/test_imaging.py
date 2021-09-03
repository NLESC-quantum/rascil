""" Unit tests for pipelines expressed via dask.delayed


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
from rascil.processing_components import weight_visibility
from rascil.processing_components.griddata.kernels import (
    create_awterm_convolutionfunction,
)
from rascil.processing_components.image.operations import (
    export_image_to_fits,
    smooth_image,
    qa_image,
)
from rascil.processing_components.imaging.base import (
    predict_2d,
    invert_2d,
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

        self.dir = rascil_path("test_results")

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
        self.vis = list()
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
            export_image_to_fits(self.model, "%s/test_imaging_model.fits" % self.dir)
        if self.persist:
            export_image_to_fits(self.cmodel, "%s/test_imaging_cmodel.fits" % self.dir)

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
        self, fluxthreshold=1.0, name="predict_2d", flux_max=0.0, flux_min=0.0, **kwargs
    ):

        vis = predict_2d(self.vis, self.model, **kwargs)

        vis["vis"].data = self.vis["vis"].data - vis["vis"].data
        dirty = invert_2d(vis, self.model, dopsf=False, normalise=True)

        if self.persist:
            export_image_to_fits(
                dirty[0], "%s/test_imaging_%s_residual.fits" % (self.dir, name)
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
        name="invert_2d",
        flux_max=0.0,
        flux_min=0.0,
        **kwargs,
    ):

        dirty = invert_2d(self.vis, self.model, dopsf=False, normalise=True, **kwargs)

        if self.persist:
            export_image_to_fits(
                dirty[0], "%s/test_imaging_%s_dirty.fits" % (self.dir, name)
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

    def test_predict_2d(self):
        self.actualSetUp(zerow=True)
        self._predict_base(
            name="predict_2d",
            flux_max=0.07221939460567486,
            flux_min=-0.434717474892552754,
        )

    def test_predict_2d_point(self):
        self.actualSetUp(zerow=True)
        self.model["pixels"].data[...] = 0.0
        nchan, npol, ny, nx = self.model.image_acc.shape
        self.model["pixels"].data[0, 0, ny // 2, nx // 2] = 1.0
        vis = predict_2d(self.vis, self.model)
        assert numpy.max(numpy.abs(vis.vis - 1.0)) < 1e-12, numpy.max(
            numpy.abs(vis.vis - 1.0)
        )

    def test_predict_2d_point_IQUV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        self.model["pixels"].data[...] = 0.0
        nchan, npol, ny, nx = self.model.image_acc.shape
        self.model["pixels"].data[0, 0, ny // 2, nx // 2] = 1.0
        vis = predict_2d(self.vis, self.model)
        assert numpy.max(numpy.abs(vis.vis[..., 0] - 1.0)) < 1e-12
        assert numpy.max(numpy.abs(vis.vis[..., 1])) < 1e-12
        assert numpy.max(numpy.abs(vis.vis[..., 2])) < 1e-12
        assert numpy.max(numpy.abs(vis.vis[..., 3] - 1.0)) < 1e-12

    def test_predict_2d_IQUV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        self._predict_base(
            name="predict_2d_IQUV",
            flux_max=0.0722193946056726,
            flux_min=-0.4347174748925693,
        )

    def test_predict_2d_IQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        self._predict_base(
            name="predict_2d_IQ",
            flux_max=0.0722193946056726,
            flux_min=-0.434717474892569,
        )

    def test_predict_2d_IV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIV"))
        self._predict_base(
            name="predict_2d_IV",
            flux_max=0.0722193946056726,
            flux_min=-0.4347174748925693,
        )

    def test_invert_2d(self):
        self.actualSetUp(zerow=True)
        self._invert_base(
            name="invert_2d",
            positionthreshold=2.0,
            check_components=False,
            flux_max=100.9654697773242,
            flux_min=-8.103733961660813,
        )

    def test_invert_2d_IQUV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        self._invert_base(
            name="invert_2d_IQUV",
            positionthreshold=2.0,
            check_components=True,
            flux_max=100.96546977732424,
            flux_min=-10.096546977732427,
        )

    def test_invert_2d_spec_I(self):
        self.actualSetUp(
            zerow=True,
            freqwin=4,
            image_pol=PolarisationFrame("stokesI"),
            dospectral=True,
        )
        self._invert_base(
            name="invert_2d_spec_I",
            positionthreshold=2.0,
            check_components=True,
            flux_max=115.82172186951361,
            flux_min=-12.352221472031786,
        )

    def test_invert_2d_spec_IQUV(self):
        self.actualSetUp(
            zerow=True, freqwin=4, image_pol=PolarisationFrame("stokesIQUV")
        )
        self._invert_base(
            name="invert_2d_IQUV",
            positionthreshold=2.0,
            check_components=True,
            flux_max=115.82172186951371,
            flux_min=-12.352221472032312,
        )

    def test_invert_2d_IQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        self._invert_base(
            name="invert_2d_IQ",
            positionthreshold=2.0,
            check_components=True,
            flux_max=100.96546977732424,
            flux_min=-8.103733961660813,
        )

    def test_invert_2d_IV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIV"))
        self._invert_base(
            name="invert_2d_IV",
            positionthreshold=2.0,
            check_components=True,
            flux_max=100.96546977732424,
            flux_min=-8.103733961660813,
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
            gcfcf=gcfcf,
            flux_max=61.7962762969381,
            flux_min=-5.420056174913808,
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
            gcfcf=gcfcf,
            flux_max=59.59422693950642,
            flux_min=-5.0391706517957955,
        )

    @unittest.skip("Takes too long to run regularly")
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
            fluxthreshold=61.0, name="predict_awterm_spec_IQUV", gcfcf=gcfcf
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

    @unittest.skip("Takes too long to run regularly")
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
            gcfcf=gcfcf,
            flux_max=1.563253192648258,
            flux_min=-1.8992207460723078,
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
            positionthreshold=1.0,
            check_components=False,
            gcfcf=gcfcf,
            flux_max=114.50082196608498,
            flux_min=-9.16145050719757,
        )

    def test_invert_psf(self):
        self.actualSetUp(zerow=False)
        psf = invert_2d(self.vis, self.model, dopsf=True)
        error = numpy.max(psf[0]["pixels"].data) - 1.0
        assert abs(error) < 1.0e-12, error
        if self.persist:
            export_image_to_fits(psf[0], "%s/test_imaging_2d_psf.fits" % self.dir)

        assert numpy.max(numpy.abs(psf[0]["pixels"].data)), "Image is empty"

    def test_invert_psf_weighting(self):
        self.actualSetUp(zerow=False)
        for weighting in ["natural", "uniform", "robust"]:
            self.vis = weight_visibility(self.vis, self.model, weighting=weighting)
            psf = invert_2d(self.vis, self.model, dopsf=True)
            error = numpy.max(psf[0]["pixels"].data) - 1.0
            assert abs(error) < 1.0e-12, error
            if self.persist:
                export_image_to_fits(
                    psf[0], "%s/test_imaging_2d_psf_%s.fits" % (self.dir, weighting)
                )
            assert numpy.max(numpy.abs(psf[0]["pixels"].data)), "Image is empty"


if __name__ == "__main__":
    unittest.main()
