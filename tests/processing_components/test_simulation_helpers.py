""" Unit tests for simulation helpers

"""

import unittest

import astropy.units as u
import matplotlib.pyplot as plt
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.memory_data_models import Skycomponent
from rascil.processing_components.simulation import create_named_configuration, plot_azel, plot_uvcoverage, \
    plot_uwcoverage, plot_vwcoverage
from rascil.processing_components.visibility.base import create_blockvisibility


class TestSimulationHelpers(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('MID')
        self.times = (numpy.pi / 12.0) * numpy.arange(-6.0, 6.0, 0.4)
        self.frequency = numpy.linspace(1.0e9, 1.1e9, 3)
        self.channel_bandwidth = numpy.array([1e8, 1e8, 1e8])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')
        self.comp = Skycomponent(direction=self.phasecentre, frequency=self.frequency, flux=self.flux)

    def test_plotazel(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          weight=1.0, elevation_limit=0.0)
        plt.clf()
        plot_azel([self.vis])
        plt.show(block=False)

    def test_plot_uvcoverage(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          weight=1.0)
        plt.clf()
        plot_uvcoverage([self.vis])
        plt.clf()
        plot_uwcoverage([self.vis])
        plt.clf()
        plot_vwcoverage([self.vis])


if __name__ == '__main__':
    unittest.main()
