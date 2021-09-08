""" Unit tests for simulation helpers

"""

import unittest

import astropy.units as u
import matplotlib.pyplot as plt
import xarray
import numpy
from astropy.coordinates import SkyCoord

from rascil.processing_components.simulation import (
    create_named_configuration,
    plot_pa,
    plot_azel,
    plot_uvcoverage,
    plot_uwcoverage,
    plot_vwcoverage,
)
from rascil.processing_components.visibility.base import (
    create_blockvisibility,
    create_blockvisibility,
)


class TestSimulationHelpers(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration("MID", rmax=10000.0)
        self.times = (numpy.pi / 12.0) * numpy.arange(-6.0, 6.0, 0.4)
        self.frequency = numpy.array([2.998e8])
        self.channel_bandwidth = numpy.array([1e8])

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(
            ra=+0.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
        )

    def test_plotazel(self):
        self.vis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
            elevation_limit=0.0,
        )
        plt.clf()
        plot_azel([self.vis])
        plt.show(block=False)

    def test_plotpa(self):
        self.vis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
            elevation_limit=0.0,
        )
        plt.clf()
        plot_pa([self.vis])
        plt.show(block=False)

    def test_plot_uvcoverage(self):
        self.vis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        plt.clf()
        plot_uvcoverage([self.vis])
        plt.clf()
        plot_uwcoverage([self.vis])
        plt.clf()
        plot_vwcoverage([self.vis])

    def test_plot_uvcoverage_flagged(self):
        self.vis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        plt.clf()
        before = self.vis["flags"].sum()
        self.vis["flags"] = xarray.where(
            self.vis["uvdist_lambda"] < 5000.0, self.vis["flags"], 1.0
        )
        after = self.vis["flags"].sum()
        plot_uvcoverage([self.vis])
        assert after > before


if __name__ == "__main__":
    unittest.main()
