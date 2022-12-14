"""Unit tests for testing support


"""

import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import (
    create_blockvisibility,
    copy_visibility,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestConfigurations(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")

        self.frequency = numpy.linspace(0.8e8, 1.2e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0

    def createVis(self, config, dec=-35.0, rmax=None):
        self.config = create_named_configuration(config, rmax=rmax)
        if self.config.location.lat < 0.0:
            self.phasecentre = SkyCoord(
                ra=+15 * u.deg, dec=dec * u.deg, frame="icrs", equinox="J2000"
            )
        else:
            self.phasecentre = SkyCoord(
                ra=+15 * u.deg, dec=-dec * u.deg, frame="icrs", equinox="J2000"
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

    def test_named_configurations(self):
        for config in [
            "LOW",
            "LOWBD2",
            "LOWBD2-CORE",
            "ASKAP",
            "MID",
            "MIDR5",
            "MEERKAT+",
        ]:
            self.createVis(config)
            assert self.config.configuration_acc.size() > 0.0
            assert len(self.config["vp_type"]) == len(self.config["names"])

    def test_SKA_MID_configurations(self):
        for config in ["MID", "MIDR5"]:
            self.config = create_named_configuration(config)
            assert self.config.configuration_acc.size() > 0.0
            assert len(self.config["vp_type"]) == len(self.config["names"])
            assert "MEERKAT" in numpy.unique(self.config["vp_type"])
            assert "MID" in numpy.unique(self.config["vp_type"])

    def test_SKA_LOW_configurations(self):
        for config in ["LOW", "LOWR3", "LOWBD2-CORE"]:
            self.config = create_named_configuration(config)
            assert self.config.configuration_acc.size() > 0.0
            assert "LOW" in numpy.unique(self.config.vp_type)

    def test_clip_configuration(self):
        for rmax in [100.0, 3000.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]:
            self.config = create_named_configuration("LOW", rmax=rmax)
            assert self.config.configuration_acc.size() > 0.0

    def test_unknown_configuration(self):
        with self.assertRaises(ValueError):
            self.config = create_named_configuration("SKA1-OWL")


if __name__ == "__main__":
    unittest.main()
