"""Make plots of the configurations, along with typical uv coverage

"""

import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rascil.data_models import PolarisationFrame, rascil_path
from rascil.processing_components.simulation import (
    create_named_configuration,
    select_configuration,
    decimate_configuration,
)
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.processing_components.simulation import plot_uvcoverage, plot_configuration

log = logging.getLogger("logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))

CONFIG_SKIP = [
    ("LOW", 3),
    ("LOWBD2", 3),
    ("LOWBD2-CORE", 3),
    ("MID", 3),
    ("LOW-AA0.5", 1),
    ("MID-AA0.5", 1),
    ("ASKAP", 1),
    ("MEERKAT+", 1),
    ("LOFAR", 1),
    ("VLAA", 1),
    ("VLAA_north", 1),
]


class TestPlotConfigurations(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")

        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e7])
        self.flux = numpy.array([[100.0]])
        self.phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.times = numpy.linspace(-4 * 3600, 4 * 3600.0, 60) * numpy.pi / 43200.0

    def createVis(self, config, dec=-35.0, rmax=None, names=None, skip=1):
        self.config = create_named_configuration(config, rmax=rmax)
        self.config = select_configuration(self.config, names)
        self.config = decimate_configuration(self.config, skip=skip)
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

    def test_select_configurations(self):
        for config in ["MID"]:
            names = ["SKA057", "SKA062", "SKA072", "SKA071", "SKA002", "SKA049"]
            self.config = create_named_configuration(config)
            self.config = select_configuration(self.config, names)
            assert len(self.config.names) == len(names)
            assert self.config.configuration_acc.size() > 0.0
            plt.clf()
            plot_configuration(
                self.config,
                title=config,
                plot_file="{dir}/test_plot_select_{config}_configuration.png".format(
                    dir=rascil_path("test_results"), config=config
                ),
                label=True,
            )

    def test_plot_configurations(self):
        for config, skip in CONFIG_SKIP:
            self.config = create_named_configuration(config)
            self.config = decimate_configuration(self.config, skip=skip)
            assert self.config.configuration_acc.size() > 0.0
            plt.clf()
            plot_configuration(
                self.config,
                title=config,
                plot_file="{dir}/test_plot_{config}_configuration.png".format(
                    dir=rascil_path("test_results"), config=config
                ),
                label=skip == 1,
            )

    def test_plot_configurations_uvcoverage(self):
        for config, skip in CONFIG_SKIP:
            self.createVis(config, skip=skip)
            assert self.config.configuration_acc.size() > 0.0
            plt.clf()
            plot_uvcoverage(
                [self.vis],
                title=config,
                plot_file="{dir}/test_plot_{config}_uvcoverage.png".format(
                    dir=rascil_path("test_results"), config=config
                ),
            )


if __name__ == "__main__":
    unittest.main()
