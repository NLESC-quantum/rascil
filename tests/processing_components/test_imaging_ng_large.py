""" Unit tests for imaging using nifty gridder

"""
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import invert_ng
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
from rascil.processing_components.skycomponent.operations import (
    insert_skycomponent,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestImagingNGLarge(unittest.TestCase):
    def setUp(self):

        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

        self.verbosity = 0

    def actualSetUp(
        self,
        scale: int = 1,
    ):

        scale = int(scale)
        self.npixel = 1024 * scale
        rmax = scale * 750.0
        self.cellsize = 0.001 / scale

        self.low = create_named_configuration("LOWBD2", rmax=rmax)
        self.low = decimate_configuration(self.low, skip=6)
        self.times = numpy.array([0.0])

        self.frequency = numpy.array([1e8])
        self.channelwidth = numpy.array([1e6])

        self.blockvis_pol = PolarisationFrame("stokesI")
        self.image_pol = PolarisationFrame("stokesI")
        flux = numpy.array([[100.0]])

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
        )

        self.model = create_unittest_model(
            self.blockvis,
            self.image_pol,
            npixel=self.npixel,
            nchan=1,
            cellsize=self.cellsize,
        )

        self.components = create_unittest_components(self.model, flux)

        self.blockvis = dft_skycomponent_visibility(self.blockvis, self.components)

    def test_invert_ng(self):
        self.actualSetUp(scale=1)
        dirty = invert_ng(
            self.blockvis,
            self.model,
            normalise=True,
            verbosity=self.verbosity,
            threads=16,
        )[0]
        assert numpy.abs(dirty["pixels"]).any() > 0.0, str(dirty)


if __name__ == "__main__":
    unittest.main()
