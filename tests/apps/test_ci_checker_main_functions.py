"""Unit tests for individual functions in ci_checker_main.py

"""

import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_array_almost_equal

from rascil.apps.ci_checker_main import correct_primary_beam
from rascil.data_models.parameters import rascil_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.imaging.primary_beams import create_pb
from rascil.processing_components.image import (
    create_image,
    export_image_to_fits,
)
from rascil.processing_components.simulation import (
    create_mid_simulation_components,
    find_pb_width_null,
)
from rascil.processing_components.skycomponent import (
    apply_beam_to_skycomponent,
)


log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)


class TestCIChecker(unittest.TestCase):
    def setUp(self):

        self.dir = rascil_path("test_results")

        self.nchan = 8
        self.central_frequency = numpy.array([1e9])
        self.component_frequency = numpy.linspace(0.8e9, 1.2e9, self.nchan)
        self.phasecentre = SkyCoord(
            ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )

    def test_correct_primary_beam(self):

        hwhm_deg, null_az_deg, null_el_deg = find_pb_width_null(
            pbtype="MID", frequency=numpy.array([self.central_frequency])
        )

        hwhm = hwhm_deg * numpy.pi / 180.0
        fov_deg = 8.0 * 1.36e9 / self.central_frequency
        pb_npixel = 256
        d2r = numpy.pi / 180.0
        pb_cellsize = d2r * fov_deg / pb_npixel
        pbradius = 1.5
        pbradius = pbradius * hwhm
        flux_limit = 0.001

        self.components = create_mid_simulation_components(
            self.phasecentre,
            self.component_frequency,
            flux_limit,
            pbradius,
            pb_npixel,
            pb_cellsize,
            apply_pb=False,
        )[0]

        pbmodel = create_image(
            npixel=pb_npixel,
            cellsize=pb_cellsize,
            phasecentre=self.phasecentre,
            frequency=self.component_frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
        )

        pb = create_pb(pbmodel, "MID", pointingcentre=self.phasecentre, use_local=False)
        components_with_pb = apply_beam_to_skycomponent(self.components, pb)

        sensitivity_image = self.dir + "/test_ci_checker_functions_sensitivity.fits"
        export_image_to_fits(pb, sensitivity_image)

        reversed_comp = correct_primary_beam(
            None, sensitivity_image, components_with_pb, "MID"
        )

        orig_flux = [c.flux[self.nchan // 2][0] for c in self.components]
        reversed_flux = [c.flux[self.nchan // 2][0] for c in reversed_comp]

        assert_array_almost_equal(orig_flux, reversed_flux, decimal=3)


if __name__ == "__main__":
    unittest.main()
