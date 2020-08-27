"""Unit tests for testing support


"""

import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

from rascil.data_models import PolarisationFrame, rascil_path
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility, create_visibility
from rascil.processing_components.simulation import plot_uvcoverage
log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestPlotConfigurations(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e7])
        self.flux = numpy.array([[100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = numpy.linspace(-4*3600, 4*3600.0, 40) * numpy.pi / 43200.0
    
    def createVis(self, config, dec=-35.0, rmax=None):
        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.config, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))


    def test_plot_configurations(self):
        for config in ['LOW', 'LOWBD2', 'LOWBD2-CORE', 'ASKAP', 'MID', 'MEERKAT+']:
            self.createVis(config)
            assert self.config.size() > 0.0
            plt.clf()
            plot_uvcoverage([self.vis], title=config,
                            plot_file='{dir}/test_plot_{config}_uvcoverage.png'.format(
                                dir=rascil_path("test_results"), config=config))
            # print("Config ", config, " has centre", self.config.location.geodetic)
    
        for config in ['LOFAR', 'VLAA', 'VLAA_north']:
            self.createVis(config, +35.0)
            assert self.config.size() > 0.0
            assert len(self.config.vp_type) == len(self.config.names)
            plt.clf()
            plot_uvcoverage([self.vis], title=config, plot_file='{dir}/test_plot_{config}_uvcoverage.png'.format(
                                dir=rascil_path("test_results"), config=config))

if __name__ == '__main__':
    unittest.main()
