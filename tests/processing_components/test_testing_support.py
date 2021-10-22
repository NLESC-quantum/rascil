"""Unit tests for testing support


"""
import os
import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.memory_data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import export_image_to_fits
from rascil.processing_components.imaging.primary_beams import create_low_test_beam
from rascil.processing_components.simulation import (
    create_test_image_from_s3,
    create_test_image,
    create_low_test_image_from_gleam,
    create_low_test_skycomponents_from_gleam,
    create_low_test_skymodel_from_gleam,
    create_test_skycomponents_from_s3,
)
from rascil.processing_components import concatenate_visibility
from rascil.processing_components.simulation import (
    create_named_configuration,
    decimate_configuration,
)
from rascil.processing_components.visibility.base import (
    create_blockvisibility,
    create_blockvisibility,
    copy_visibility,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestTesting_Support(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")
        self.persist = os.getenv("RASCIL_PERSIST", False)

        self.frequency = numpy.linspace(0.8e8, 1.2e8, 5)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.config = create_named_configuration("LOWBD2-CORE")
        self.config = decimate_configuration(self.config, skip=3)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants

    def createVis(self, config, dec=-35.0, rmax=None):
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

    def test_create_test_image(self):
        im = create_test_image()
        print(im)
        print(im.image_acc.wcs)
        assert len(im["pixels"].data.shape) == 4
        im = create_test_image(
            frequency=numpy.array([1e8]),
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        assert len(im["pixels"].data.shape) == 4
        assert im["pixels"].data.shape[0] == 1
        assert im["pixels"].data.shape[1] == 1
        im = create_test_image(
            frequency=numpy.array([1e8]),
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        assert len(im["pixels"].data.shape) == 4
        assert im["pixels"].data.shape[0] == 1
        assert im["pixels"].data.shape[1] == 4

    def test_create_low_test_skymodel_from_gleam(self):
        sm = create_low_test_skymodel_from_gleam(
            npixel=256,
            cellsize=0.001,
            frequency=self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            kind="cubic",
            flux_limit=0.3,
            flux_threshold=1.0,
            polarisation_frame=PolarisationFrame("stokesI"),
        )

        im = sm.image
        assert im["pixels"].data.shape[0] == 5
        assert im["pixels"].data.shape[1] == 1
        assert im["pixels"].data.shape[2] == 256
        assert im["pixels"].data.shape[3] == 256
        if self.persist:
            export_image_to_fits(
                im, "%s/test_test_support_low_gleam.fits" % (self.results_dir)
            )

        comp = sm.components
        assert len(comp) == 38, len(comp)
        assert comp[0].name == "GLEAM J005659-380954", comp[0].name
        assert comp[-1].name == "GLEAM J011412-321730", comp[-1].name

    def test_create_low_test_image_from_gleam(self):
        im = create_low_test_image_from_gleam(
            npixel=256,
            cellsize=0.001,
            channel_bandwidth=self.channel_bandwidth,
            frequency=self.frequency,
            phasecentre=self.phasecentre,
            kind="cubic",
            flux_limit=0.3,
        )
        assert im["pixels"].data.shape[0] == 5
        assert im["pixels"].data.shape[1] == 1
        assert im["pixels"].data.shape[2] == 256
        assert im["pixels"].data.shape[3] == 256
        if self.persist:
            export_image_to_fits(
                im, "%s/test_test_support_low_gleam.fits" % (self.results_dir)
            )

    def test_create_low_test_image_from_gleam_with_pb(self):
        im = create_low_test_image_from_gleam(
            npixel=256,
            cellsize=0.001,
            channel_bandwidth=self.channel_bandwidth,
            frequency=self.frequency,
            phasecentre=self.phasecentre,
            kind="cubic",
            applybeam=True,
            flux_limit=1.0,
        )
        assert im["pixels"].data.shape[0] == 5
        assert im["pixels"].data.shape[1] == 1
        assert im["pixels"].data.shape[2] == 256
        assert im["pixels"].data.shape[3] == 256
        if self.persist:
            export_image_to_fits(
                im, "%s/test_test_support_low_gleam_with_pb.fits" % (self.results_dir)
            )

    def test_create_low_test_skycomponents_from_gleam(self):
        sc = create_low_test_skycomponents_from_gleam(
            flux_limit=1.0,
            phasecentre=SkyCoord("17h20m31s", "-00d58m45s"),
            polarisation_frame=PolarisationFrame("stokesI"),
            frequency=self.frequency,
            kind="cubic",
            radius=0.001,
        )
        assert len(sc) == 1, "Only expected one source, actually found %d" % len(sc)
        assert sc[0].name == "GLEAM J172031-005845"
        self.assertAlmostEqual(sc[0].flux[0, 0], 357.2599499089219, 7)

    def test_create_test_skycomponents_from_s3(self):
        self.frequency = numpy.linspace(0.8e9, 1.2e9, 5)
        sc = create_test_skycomponents_from_s3(
            flux_limit=3.0,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
            frequency=self.frequency,
            radius=0.1,
        )
        assert len(sc) == 5, "Expected 5 sources, actually found %d" % len(sc)
        assert sc[0].name == "S3_36315789"
        self.assertAlmostEqual(sc[0].flux[0, 0], 3.6065651245943307, 7)

    @unittest.skip("Too expensive for CI/CD")
    def test_create_test_skycomponents_from_s3_deep(self):
        # Takes about 3 minutes to run so we keep it for running by hand.
        self.frequency = numpy.linspace(0.8e9, 1.2e9, 5)
        sc_unsorted = create_test_skycomponents_from_s3(
            flux_limit=1e-5,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
            frequency=self.frequency,
            radius=0.0003,
        )
        sc = sorted(sc_unsorted, key=lambda cmp: numpy.max(cmp.flux))

        assert len(sc) == 6, f"Expected 6 sources, actually found {len(sc)}: {sc}"
        assert sc[0].name == "S3_155080789"
        self.assertAlmostEqual(sc[0].flux[0, 0], 2.00858007e-05, 7)

    def test_create_test_image_from_s3_low(self):
        im = create_test_image_from_s3(
            npixel=1024,
            channel_bandwidth=numpy.array([1e6]),
            frequency=numpy.array([1e8]),
            phasecentre=self.phasecentre,
            fov=10,
        )
        assert im["pixels"].data.shape[0] == 1
        assert im["pixels"].data.shape[1] == 1
        assert im["pixels"].data.shape[2] == 1024
        assert im["pixels"].data.shape[3] == 1024
        if self.persist:
            export_image_to_fits(
                im, "%s/test_test_support_low_s3.fits" % (self.results_dir)
            )

    def test_create_test_image_from_s3_mid(self):
        im = create_test_image_from_s3(
            npixel=1024,
            channel_bandwidth=numpy.array([1e6]),
            frequency=numpy.array([1e9]),
            phasecentre=self.phasecentre,
            flux_limit=2e-3,
        )
        assert im["pixels"].data.shape[0] == 1
        assert im["pixels"].data.shape[1] == 1
        assert im["pixels"].data.shape[2] == 1024
        assert im["pixels"].data.shape[3] == 1024
        if self.persist:
            export_image_to_fits(
                im, "%s/test_test_support_mid_s3.fits" % (self.results_dir)
            )

    def test_create_test_image_s3_spectral(self):
        im = create_test_image_from_s3(
            npixel=1024,
            channel_bandwidth=numpy.array([1e6, 1e6, 1e6]),
            frequency=numpy.array([1e8 - 1e6, 1e8, 1e8 + 1e6]),
            phasecentre=self.phasecentre,
            fov=10,
            flux_limit=2e-3,
        )
        assert im["pixels"].data.shape[0] == 3
        assert im["pixels"].data.shape[1] == 1
        assert im["pixels"].data.shape[2] == 1024
        assert im["pixels"].data.shape[3] == 1024

    def test_create_low_test_image_s3_spectral_polarisation(self):

        im = create_test_image_from_s3(
            npixel=1024,
            channel_bandwidth=numpy.array([1e6, 1e6, 1e6]),
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            frequency=numpy.array([1e8 - 1e6, 1e8, 1e8 + 1e6]),
            fov=10,
        )
        assert im["pixels"].data.shape[0] == 3
        assert im["pixels"].data.shape[1] == 4
        assert im["pixels"].data.shape[2] == 1024
        assert im["pixels"].data.shape[3] == 1024
        if self.persist:
            export_image_to_fits(
                im, "%s/test_test_support_low_s3.fits" % (self.results_dir)
            )

    def test_create_low_test_beam(self):
        im = create_test_image(
            cellsize=0.002,
            frequency=numpy.array([1e8 - 5e7, 1e8, 1e8 + 5e7]),
            channel_bandwidth=numpy.array([5e7, 5e7, 5e7]),
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        bm = create_low_test_beam(model=im)
        if self.persist:
            export_image_to_fits(
                bm, "%s/test_test_support_low_beam.fits" % (self.results_dir)
            )

        bmshape = bm["pixels"].data.shape
        assert bmshape[0] == 3
        assert bmshape[1] == 4
        assert bmshape[2] == im["pixels"].data.shape[2]
        assert bmshape[3] == im["pixels"].data.shape[3]
        # Check to see if the beam scales as expected
        for i in [30, 40]:
            assert (
                numpy.max(
                    numpy.abs(
                        bm["pixels"].data[0, 0, 128, 128 - 2 * i]
                        - bm["pixels"].data[1, 0, 128, 128 - i]
                    )
                )
                < 0.02
            )
            assert (
                numpy.max(
                    numpy.abs(
                        bm["pixels"].data[0, 0, 128, 128 - 3 * i]
                        - bm["pixels"].data[2, 0, 128, 128 - i]
                    )
                )
                < 0.02
            )
            assert (
                numpy.max(
                    numpy.abs(
                        bm["pixels"].data[0, 0, 128 - 2 * i, 128]
                        - bm["pixels"].data[1, 0, 128 - i, 128]
                    )
                )
                < 0.02
            )
            assert (
                numpy.max(
                    numpy.abs(
                        bm["pixels"].data[0, 0, 128 - 3 * i, 128]
                        - bm["pixels"].data[2, 0, 128 - i, 128]
                    )
                )
                < 0.02
            )


if __name__ == "__main__":
    unittest.main()
