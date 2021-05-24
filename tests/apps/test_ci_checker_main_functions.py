"""Unit tests for individual functions in ci_checker_main.py

"""

import logging
import unittest
import os
from unittest.mock import patch, Mock

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_array_almost_equal

from rascil.apps.ci_checker_main import (
    cli_parser,
    analyze_image,
    correct_primary_beam,
    read_skycomponent_from_txt,
)
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
    fit_skycomponent_spectral_index,
)


log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)


class TestCIChecker(unittest.TestCase):
    def setUp(self):

        # Generate mock objects for functions in ci_checker,
        # including skycomponents, primary beam image, restored image
        self.dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

        self.nchan = 8
        self.central_frequency = numpy.array([1e9])
        self.component_frequency = numpy.linspace(0.8e9, 1.2e9, self.nchan)
        self.phasecentre = SkyCoord(
            ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.npixel = 512
        self.cellsize = 0.0001
        self.bandwidth_array = numpy.full(self.nchan, 1e7)

        # Below are parameters used to generate primary beam
        hwhm_deg, null_az_deg, null_el_deg = find_pb_width_null(
            pbtype="MID", frequency=self.central_frequency
        )

        hwhm = hwhm_deg * numpy.pi / 180.0
        self.pbradius = 1.5 * hwhm
        self.pb_npixel = 256
        fov_deg = 8.0 * 1.36e9 / self.central_frequency
        d2r = numpy.pi / 180.0
        self.pb_cellsize = d2r * fov_deg / self.pb_npixel
        self.flux_limit = 0.001

        # List of skycomponents to test
        self.components = create_mid_simulation_components(
            self.phasecentre,
            self.component_frequency,
            self.flux_limit,
            self.pbradius,
            self.pb_npixel,
            self.pb_cellsize,
            apply_pb=False,
        )[0]

        # Test restored image
        self.multi_chan_image = create_image(
            npixel=self.npixel,
            cellsize=self.cellsize,
            polarisation_frame=PolarisationFrame("stokesI"),
            frequency=self.component_frequency,
            channel_bandwidth=self.bandwidth_array,
            phasecentre=self.phasecentre,
        )

        self.restored_image_multi = (
            self.dir + "/test_ci_checker_functions_nchan8_restored.fits"
        )
        export_image_to_fits(self.multi_chan_image, self.restored_image_multi)

        self.txtfile = self.dir + "/test_ci_checker_functions.txt"

        pbmodel = create_image(
            npixel=self.pb_npixel,
            cellsize=self.pb_cellsize,
            phasecentre=self.phasecentre,
            frequency=self.component_frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
        )

        self.pb = create_pb(
            pbmodel, "MID", pointingcentre=self.phasecentre, use_local=False
        )
        self.components_with_pb = apply_beam_to_skycomponent(self.components, self.pb)

        parser = cli_parser()
        self.args = parser.parse_args([])

    def test_correct_primary_beam_sensitivity(self):

        # Test using sensitivity image
        # After adding and dividing primary beam, the flux should stay roughly the same

        sensitivity_image = self.dir + "/test_ci_checker_functions_sensitivity.fits"
        export_image_to_fits(self.pb, sensitivity_image)

        reversed_comp = correct_primary_beam(
            None, sensitivity_image, self.components_with_pb, "MID"
        )

        orig_flux = [c.flux[self.nchan // 2][0] for c in self.components]
        reversed_flux = [c.flux[self.nchan // 2][0] for c in reversed_comp]

        assert_array_almost_equal(orig_flux, reversed_flux, decimal=3)

    def test_correct_primary_beam_restored(self):

        # Test using restored image
        reversed_comp_rest = correct_primary_beam(
            self.restored_image_multi, None, self.components_with_pb, telescope="MID"
        )
        orig_flux = [c.flux[self.nchan // 2][0] for c in self.components]
        reversed_flux_rest = [c.flux[self.nchan // 2][0] for c in reversed_comp_rest]

        assert_array_almost_equal(orig_flux, reversed_flux_rest, decimal=1)

    def test_correct_primary_beam_none(self):

        # If we don't feed either sensitivity or restored image, should return the same components back
        reversed_comp_none = correct_primary_beam(
            None, None, self.components_with_pb, "MID"
        )
        assert self.components_with_pb == reversed_comp_none

    def test_read_skycomponent_from_txt_multi(self):

        f = open(self.txtfile, "w")
        f.write(
            "# RA(deg), Dec(deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. freq. (Hz), Spectral index\n"
        )
        for cmp in self.components:
            coord_ra = cmp.direction.ra.degree
            coord_dec = cmp.direction.dec.degree
            spec_indx = fit_skycomponent_spectral_index(cmp)
            f.write(
                "%.6f, %.6f, %10.6e, %10.6e, %10.6e, %10.6e, %10.6e, %10.6e \n"
                % (
                    coord_ra,
                    coord_dec,
                    cmp.flux[self.nchan // 2],
                    0.0,
                    0.0,
                    0.0,
                    self.central_frequency,
                    spec_indx,
                )
            )
        f.close()

        components_read = read_skycomponent_from_txt(
            self.txtfile, self.component_frequency
        )
        assert len(components_read) == len(self.components)
        orig_ra = [c.direction.ra.degree for c in self.components]
        read_ra = [c.direction.ra.degree for c in components_read]

        assert_array_almost_equal(orig_ra, read_ra)

    def test_read_skycomponent_from_txt_single(self):

        # Test single channel components
        # The flux for each component should only have one value
        components_read_single = read_skycomponent_from_txt(
            self.txtfile, self.central_frequency
        )

        for c in components_read_single:
            assert c.flux.shape == (1, 1)

        orig_flux = [c.flux[self.nchan // 2][0] for c in self.components]
        read_flux = [c.flux[0][0] for c in components_read_single]

        assert_array_almost_equal(orig_flux, read_flux)

    def test_wrong_restored(self):

        # This part tests for no image input
        self.args.ingest_fitsname_restored = None
        self.assertRaises(FileNotFoundError, lambda: analyze_image(self.args))

    @patch("rascil.apps.ci_checker_main.ci_checker")
    def test_analyze_image_exceptions(self, mock_checker):
        mock_checker.return_value = Mock()
        self.args.ingest_fitsname_restored = self.restored_image_multi
        self.args.restart = False

        result = analyze_image(self.args)

        # call_args_list returns the input for function ci_checker
        # Assert using automatic beam size
        assert mock_checker.call_args_list[0][0][2] == (1.0, 1.0, 0.0)

        # Assert returning None values
        assert result == (None, None)


if __name__ == "__main__":
    unittest.main()
