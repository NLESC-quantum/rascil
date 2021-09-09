"""Unit tests for testing support


"""

import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import (
    export_image_to_fits,
    scale_and_rotate_image,
)
from rascil.processing_components.imaging.base import create_image_from_visibility
from rascil.processing_components import (
    create_pb,
    create_vp,
    convert_azelvp_to_radec,
    create_low_test_vp,
    qa_image,
    create_mid_allsky,
)
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


def check_max_min(im, flux_max, flux_min, context):
    qa = qa_image(im)
    numpy.testing.assert_allclose(
        qa.data["max"], flux_max, atol=1e-7, err_msg=f"{context} {qa}"
    )
    numpy.testing.assert_allclose(
        qa.data["min"], flux_min, atol=1e-7, err_msg=f"{context} {qa}"
    )


class TestPrimaryBeams(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

    def createVis(self, config="MID", dec=-35.0, rmax=1e3, freq=1.3e9):
        self.frequency = [freq]
        self.channel_bandwidth = [1e6]
        self.flux = numpy.array([[100.0]])
        self.phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.config = create_named_configuration(config)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        self.npixel = 1024
        self.fov = 8
        self.cellsize = numpy.pi * self.fov / (self.npixel * 180.0)
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants

        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(
            ra=+15 * u.deg, dec=dec * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis = create_blockvisibility(
            self.config,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
            polarisation_frame=PolarisationFrame("stokesI"),
        )

    def test_create_primary_beams_RADEC(self):
        self.createVis()
        for telescope, flux_max, flux_min in [
            ("VLA", 0.9896588738559976, 0.0),
            ("ASKAP", 0.9861593364197507, 0.0),
            ("MID", 1.0, 0.0),
            ("LOW", 1.0, 0.0),
        ]:
            model = create_image_from_visibility(
                self.vis,
                cellsize=self.cellsize,
                npixel=self.npixel,
                override_cellsize=False,
            )
            beam = create_pb(model, telescope=telescope, use_local=False)
            assert numpy.max(beam["pixels"].data) > 0.0, telescope
            if self.persist:
                export_image_to_fits(
                    beam,
                    "%s/test_primary_beam_RADEC_%s.fits" % (self.dir, telescope),
                )
            check_max_min(beam, flux_max, flux_min, telescope)

    def test_create_primary_beams_AZELGEO(self):
        self.createVis()
        for telescope, flux_max, flux_min in [
            ("VLA", 0.9896588738559976, 0.0),
            ("ASKAP", 0.9861593364197507, 0.0),
            ("MID", 1.0, 0.0),
            ("MID_GAUSS", 1.0, 0.0),
            ("MID_FEKO_B1", 1.0, 0.0),
            ("MID_FEKO_B2", 1.0, 0.0),
            ("MID_FEKO_Ku", 1.0, 0.0),
            ("LOW", 1.0, 0.0),
        ]:
            model = create_image_from_visibility(
                self.vis,
                cellsize=self.cellsize,
                npixel=self.npixel,
                override_cellsize=False,
            )
            beam = create_pb(model, telescope=telescope, use_local=True)
            if self.persist:
                export_image_to_fits(
                    beam,
                    "%s/test_primary_beam_AZELGEO_%s.fits" % (self.dir, telescope),
                )
            check_max_min(beam, flux_max, flux_min, telescope)

    def test_create_voltage_patterns(self):
        self.createVis()
        for telescope, flux_max, flux_min in [
            ("VLA", 0.9948159999999988 + 0j, -0.13737537430017216 + 0j),
            ("ASKAP", 0.9930555555555544 + 0j, -0.13906616397447308 + 0j),
            ("LOW", 0.9999999999999988 + 0j, -0.132279301631194 + 0j),
        ]:
            model = create_image_from_visibility(
                self.vis,
                cellsize=self.cellsize,
                npixel=self.npixel,
                override_cellsize=False,
            )
            beam = create_vp(model, telescope=telescope)
            assert numpy.max(numpy.abs(beam["pixels"].data.real)) > 0.0, telescope
            assert numpy.max(numpy.abs(beam["pixels"].data.imag)) < 1e-15, numpy.max(
                numpy.abs(beam["pixels"].data.imag)
            )
            check_max_min(beam, flux_max, flux_min, telescope)

    def test_create_voltage_patterns_MID_GAUSS(self):
        self.createVis()
        model = create_image_from_visibility(
            self.vis,
            npixel=self.npixel,
            cellsize=self.cellsize,
            override_cellsize=False,
        )
        telescope = "MID_GAUSS"
        beam = create_vp(model, telescope=telescope, padding=4)
        beam_data = beam["pixels"].data
        beam["pixels"].data = numpy.real(beam_data)
        check_max_min(beam, 1.0, -0.04908413672703686, telescope)
        if self.persist:
            export_image_to_fits(
                beam, "%s/test_voltage_pattern_real_%s.fits" % (self.dir, telescope)
            )
        beam["pixels"].data = numpy.imag(beam_data)
        check_max_min(beam, 0.0, 0.0, telescope)
        if self.persist:
            export_image_to_fits(
                beam, "%s/test_voltage_pattern_imag_%s.fits" % (self.dir, telescope)
            )

    def test_create_voltage_pattern_MID_allsky(self):
        self.createVis()
        telescope = "MID_GAUSS"
        beam = create_mid_allsky(self.vis.frequency)

        beam_data = beam["pixels"].data
        beam["pixels"].data = numpy.real(beam_data)
        check_max_min(beam, 1.0, -0.13220304339601227, telescope)
        if self.persist:
            export_image_to_fits(
                beam, "%s/test_voltage_pattern_real_mid_allsky.fits" % (self.dir)
            )
        beam["pixels"].data = numpy.imag(beam_data)
        check_max_min(beam, 0.0, 0.0, telescope)
        if self.persist:
            export_image_to_fits(
                beam, "%s/test_voltage_pattern_imag_mid_allsky.fits" % (self.dir)
            )

    def test_create_voltage_patterns_MID(self):
        self.createVis(freq=1.4e9)
        model = create_image_from_visibility(
            self.vis,
            npixel=self.npixel,
            cellsize=self.cellsize,
            override_cellsize=False,
        )
        for telescope, flux_max, flux_min in [
            ("MID", 1.0, -0.13227948455466312),
            ("MID_FEKO_B1", 1.0, -0.12867465048776316),
            ("MID_FEKO_B2", 0.9999995585656044, -0.14500584762256105),
            ("MID_FEKO_Ku", 1.0, -0.1162932799217111),
            ("MEERKAT_B2", 1.0, -0.09590146287902526),
        ]:
            beam = create_vp(model, telescope=telescope, padding=4)
            beam_data = beam["pixels"].data
            beam["pixels"].data = numpy.real(beam_data)
            beam.image_acc.wcs.wcs.crval[0] = 0.0
            beam.image_acc.wcs.wcs.crval[1] = 90.0
            if self.persist:
                export_image_to_fits(
                    beam,
                    "%s/test_voltage_pattern_real_zenith_%s.fits"
                    % (self.dir, telescope),
                )
            check_max_min(beam, flux_max, flux_min, telescope)

    def test_create_voltage_patterns_MID_rotate(self):
        self.createVis(freq=1.4e9)
        model = create_image_from_visibility(
            self.vis,
            npixel=self.npixel,
            cellsize=self.cellsize,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            override_cellsize=False,
        )
        for telescope, flux_max, flux_min in [
            ("MID_FEKO_B1", 1.000088108038104, -0.1286805307650631),
            ("MID_FEKO_B2", 1.0000627107836275, -0.1450148783145079),
            ("MID_FEKO_Ku", 0.9999126796425732, -0.11606096603175388),
        ]:
            beam = create_vp(telescope=telescope)
            beam = scale_and_rotate_image(beam, scale=[1.2, 0.8])

            if self.persist:
                export_image_to_fits(
                    beam,
                    "%s/test_voltage_pattern_real_prerotate_%s.fits"
                    % (self.dir, telescope),
                )
            beam_radec = convert_azelvp_to_radec(beam, model, numpy.pi / 4.0)

            beam_data = beam_radec["pixels"].data
            beam_radec["pixels"].data = numpy.real(beam_data)
            if self.persist:
                export_image_to_fits(
                    beam_radec,
                    "%s/test_voltage_pattern_real_rotate_%s.fits"
                    % (self.dir, telescope),
                )
            check_max_min(beam_radec, flux_max, flux_min, telescope)

    def test_create_voltage_patterns_LOW(self):
        self.createVis(freq=1e8)
        model = create_image_from_visibility(
            self.vis,
            npixel=self.npixel,
            cellsize=self.cellsize * 10.0,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            override_cellsize=False,
        )
        for az, el, flux_max, flux_min in [
            (60.0, 45.0, 0.9999999999999988 + 0j, -0.13227948739290077 + 0j),
            (-60.0, 45.0, 0.9999999999999988 + 0j, -0.13227948739290077 + 0j),
        ]:
            beam = create_low_test_vp(
                model,
                use_local=False,
                azel=(numpy.deg2rad(az), numpy.deg2rad(el)),
            )
            if self.persist:
                export_image_to_fits(
                    beam,
                    f"{self.dir}/test_voltage_pattern_low_real_az{az}_el{el}.fits",
                )
            check_max_min(beam, flux_max, flux_min, f"{az} {el}")


if __name__ == "__main__":
    unittest.main()
