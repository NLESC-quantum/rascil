""" Unit tests for imaging using the WAGG GPU implementation of the nifty gridder

"""
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import (
    export_image_to_fits,
    smooth_image,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
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
from rascil.processing_components.visibility import copy_visibility


log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestImagingWG(unittest.TestCase):
    def setUp(self):

        from rascil.data_models.parameters import rascil_path

        self.test_dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

        self.verbosity = 0

    def actualSetUp(
        self,
        freqwin=1,
        dospectral=True,
        image_pol=PolarisationFrame("stokesI"),
        # zerow=True means zeroing w-terms when ingesting visibilities for unittests
        zerow=False,
        # mfs=True means multi-frequency synthesis
        mfs=False,
    ):

        self.npixel = 256
        self.low = create_named_configuration("LOWBD2", rmax=750.0)
        self.freqwin = freqwin
        self.blockvis = list()
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
            self.blockvis_pol = PolarisationFrame("linear")
            self.image_pol = image_pol
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        elif image_pol == PolarisationFrame("stokesIQ"):
            self.blockvis_pol = PolarisationFrame("linearnp")
            self.image_pol = image_pol
            f = numpy.array([100.0, 20.0])
        elif image_pol == PolarisationFrame("stokesIV"):
            self.blockvis_pol = PolarisationFrame("circularnp")
            self.image_pol = image_pol
            f = numpy.array([100.0, 20.0])
        else:
            self.blockvis_pol = PolarisationFrame("stokesI")
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
        self.blockvis = ingest_unittest_visibility(
            self.low,
            self.frequency,
            self.channelwidth,
            self.times,
            self.blockvis_pol,
            self.phasecentre,
            zerow=zerow,
        )

        self.model = create_unittest_model(
            self.blockvis, self.image_pol, npixel=self.npixel, nchan=freqwin
        )

        self.components = create_unittest_components(self.model, flux)

        self.model = insert_skycomponent(self.model, self.components)

        self.blockvis = dft_skycomponent_visibility(self.blockvis, self.components)

        # Calculate the model convolved with a Gaussian.

        self.cmodel = smooth_image(self.model)
        if self.persist:
            export_image_to_fits(
                self.model, "%s/test_imaging_wg_model.fits" % self.test_dir
            )
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_imaging_wg_cmodel.fits" % self.test_dir
            )

        if mfs:
            self.model = create_unittest_model(
                self.blockvis, self.image_pol, npixel=self.npixel, nchan=1
            )

    def _check_components(self, dirty, fluxthreshold=0.6, positionthreshold=0.1):
        comps = find_skycomponents(
            dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5
        )
        assert len(comps) == len(
            self.components
        ), "Different number of components found: original %d, recovered %d" % (
            len(self.components),
            len(comps),
        )
        cellsize = abs(dirty.image_acc.wcs.wcs.cdelt[0])

        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(
                comp.direction, self.components
            )
            assert separation / cellsize < positionthreshold, (
                "Component differs in position %.3f pixels" % separation / cellsize
            )

    def _predict_base(self, fluxthreshold=1.0, name="predict_wg", **kwargs):

        from rascil.processing_components.imaging.wg import predict_wg, invert_wg

        original_vis = copy_visibility(self.blockvis)
        vis = predict_wg(self.blockvis, self.model, verbosity=self.verbosity, **kwargs)
        vis["vis"].data = vis["vis"].data - original_vis["vis"].data
        dirty = invert_wg(
            vis,
            self.model,
            dopsf=False,
            normalise=True,
            verbosity=self.verbosity,
            **kwargs
        )

        if self.persist:
            export_image_to_fits(
                dirty[0], "%s/test_imaging_wg_%s_residual.fits" % (self.test_dir, name)
            )

        maxabs = numpy.max(numpy.abs(dirty[0]["pixels"].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (
            maxabs,
            fluxthreshold,
        )

    def _invert_base(
        self,
        fluxthreshold=1.0,
        positionthreshold=1.0,
        check_components=True,
        name="invert_wg",
        **kwargs
    ):

        from rascil.processing_components.imaging.wg import invert_wg

        dirty = invert_wg(
            self.blockvis,
            self.model,
            normalise=True,
            do_wstacking=True,
            verbosity=self.verbosity,
            **kwargs
        )

        if self.persist:
            export_image_to_fits(
                dirty[0], "%s/test_imaging_wg_%s_dirty.fits" % (self.test_dir, name)
            )

        assert numpy.max(numpy.abs(dirty[0]["pixels"].data)), "Image is empty"

        if check_components:
            self._check_components(dirty[0], fluxthreshold, positionthreshold)

    def test_predict_wg(self):
        self.actualSetUp()
        self._predict_base(name="predict_wg")

    def test_predict_wg_IQUV(self):
        self.actualSetUp(image_pol=PolarisationFrame("stokesIQUV"))
        self._predict_base(name="predict_wg_IQUV")

    def test_predict_wg_IQ(self):
        self.actualSetUp(image_pol=PolarisationFrame("stokesIQ"))
        self._predict_base(name="predict_wg_IQ")

    def test_predict_wg_IV(self):
        self.actualSetUp(image_pol=PolarisationFrame("stokesIV"))
        self._predict_base(name="predict_wg_IV")

    def test_invert_wg(self):
        self.actualSetUp()
        self._invert_base(
            name="invert_wg", positionthreshold=2.0, check_components=True
        )

    def test_invert_wg_psf(self):
        self.actualSetUp()
        self._invert_base(
            name="invert_wg_psf",
            positionthreshold=2.0,
            check_components=False,
            dopsf=True,
        )

    def test_invert_wg_IQUV(self):
        self.actualSetUp(image_pol=PolarisationFrame("stokesIQUV"))
        self._invert_base(
            name="invert_wg_IQUV", positionthreshold=2.0, check_components=True
        )

    def test_invert_wg_IQUV_psf(self):
        self.actualSetUp(image_pol=PolarisationFrame("stokesIQUV"))
        self._invert_base(
            name="invert_wg_IQUV_psf",
            positionthreshold=2.0,
            check_components=False,
            dopsf=True,
        )

    def test_invert_wg_IQ(self):
        self.actualSetUp(image_pol=PolarisationFrame("stokesIQ"))
        self._invert_base(
            name="invert_wg_IQ", positionthreshold=2.0, check_components=True
        )

    def test_invert_wg_IV(self):
        self.actualSetUp(image_pol=PolarisationFrame("stokesIV"))
        self._invert_base(
            name="invert_wg_IV", positionthreshold=2.0, check_components=True
        )

    def test_predict_wg_spec(self):
        self.actualSetUp(dospectral=True, freqwin=5)
        self._predict_base(name="predict_wg_spec")

    def test_invert_wg_spec(self):
        self.actualSetUp(dospectral=True, freqwin=5)
        self._invert_base(
            name="invert_wg_spec", positionthreshold=2.0, check_components=False
        )

    def test_invert_wg_spec_psf(self):
        self.actualSetUp(dospectral=True, freqwin=5)
        self._invert_base(
            name="invert_wg_spec_psf",
            positionthreshold=2.0,
            check_components=False,
            dopsf=True,
        )

    def test_predict_wg_spec_IQUV(self):
        self.actualSetUp(
            dospectral=True, freqwin=5, image_pol=PolarisationFrame("stokesIQUV")
        )
        self._predict_base(name="predict_wg_spec_IQUV")

    def test_invert_wg_spec_IQUV(self):
        self.actualSetUp(
            dospectral=True, freqwin=5, image_pol=PolarisationFrame("stokesIQUV")
        )
        self._invert_base(
            name="invert_wg_spec_IQUV", positionthreshold=2.0, check_components=False
        )

    def test_invert_wg_spec_IQUV_psf(self):
        self.actualSetUp(
            dospectral=True, freqwin=5, image_pol=PolarisationFrame("stokesIQUV")
        )
        self._invert_base(
            name="invert_wg_spec_IQUV_psf",
            positionthreshold=2.0,
            check_components=False,
            dopsf=True,
        )

    def test_invert_wg_mfs_IQUV(self):
        self.actualSetUp(
            dospectral=True,
            freqwin=5,
            image_pol=PolarisationFrame("stokesIQUV"),
            mfs=True,
        )
        self._invert_base(
            name="invert_wg_mfs_IQUV", positionthreshold=2.0, check_components=False
        )

    def test_invert_wg_mfs_IQUV_psf(self):
        self.actualSetUp(
            dospectral=True,
            freqwin=5,
            image_pol=PolarisationFrame("stokesIQUV"),
            mfs=True,
        )
        self._invert_base(
            name="invert_wg_mfs_IQUV_psf",
            positionthreshold=2.0,
            check_components=False,
            dopsf=True,
        )

    def test_predict_wg_spec_IQ(self):
        self.actualSetUp(
            dospectral=True, freqwin=5, image_pol=PolarisationFrame("stokesIQ")
        )
        self._predict_base(name="predict_wg_spec_IQ")

    def test_invert_wg_spec_IQ(self):
        self.actualSetUp(
            dospectral=True, freqwin=5, image_pol=PolarisationFrame("stokesIQ")
        )
        self._invert_base(
            name="invert_wg_spec_IQ", positionthreshold=2.0, check_components=False
        )


if __name__ == "__main__":
    unittest.main()
