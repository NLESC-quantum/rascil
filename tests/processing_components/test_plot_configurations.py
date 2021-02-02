"""Unit tests for testing support


"""

import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rascil.data_models import PolarisationFrame, rascil_path
from rascil.processing_components.simulation import create_named_configuration, select_configuration
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.processing_components.simulation import plot_uvcoverage, plot_configuration
log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))



class TestPlotConfigurations(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e7])
        self.flux = numpy.array([[100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = numpy.linspace(-4*3600, 4*3600.0, 60) * numpy.pi / 43200.0
        # self.times = numpy.array([0])

    def createVis(self, config, dec=-35.0, rmax=None, names=None):
        self.config = create_named_configuration(config, rmax=rmax)
        self.config = select_configuration(self.config, names)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.config, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))

    def test_select_configurations(self):
        for config in ['MID']:
            names = ['SKA057', 'SKA062', 'SKA072', 'SKA071', 'SKA002', 'SKA049']
            self.createVis(config, rmax=2e2, names=names)
            assert self.config.configuration_acc.size() > 0.0
            plt.clf()
            plot_configuration(self.vis.configuration, title=config,
                               plot_file='{dir}/test_plot_{config}_configuration.png'.format(
                                   dir=rascil_path("test_results"), config=config),
                               label=True)

    def test_plot_configurations(self):
        for config in ['LOW', 'LOWBD2', 'LOWBD2-CORE', 'LLA', 'ASKAP', 'MID', 'MEERKAT+']:
            self.createVis(config)
            assert self.config.configuration_acc.size() > 0.0
            plt.clf()
            plot_configuration(self.vis.configuration, title=config,
                            plot_file='{dir}/test_plot_{config}_configuration.png'.format(
                                dir=rascil_path("test_results"), config=config))

        for config in ['LOFAR', 'VLAA', 'VLAA_north']:
            self.createVis(config, +35.0)
            assert self.config.configuration_acc.size() > 0.0
            plt.clf()
            plot_configuration(self.vis.configuration, title=config,
                               plot_file='{dir}/test_plot_{config}_configuration.png'.format(
                                   dir=rascil_path("test_results"), config=config))
        print("Config ", config, " has centre", self.config.location.geodetic)
    
    def test_plot_configurations_uvcoverage(self):
        for config in ['LOW', 'LOWBD2', 'LOWBD2-CORE','LLA', 'ASKAP', 'MID', 'MEERKAT+']:
            self.createVis(config)
            assert self.config.configuration_acc.size() > 0.0
            plt.clf()
            plot_uvcoverage([self.vis], title=config,
                            plot_file='{dir}/test_plot_{config}_uvcoverage.png'.format(
                                dir=rascil_path("test_results"), config=config))
            print("Config ", config, " has centre", self.config.location.geodetic)
    
        for config in ['LOFAR', 'VLAA', 'VLAA_north']:
            self.createVis(config, +35.0)
            assert self.config.configuration_acc.size() > 0.0
            assert len(self.config.vp_type) == len(self.config.names)
            plt.clf()
            plot_uvcoverage([self.vis], title=config, plot_file='{dir}/test_plot_{config}_uvcoverage.png'.format(
                dir=rascil_path("test_results"), config=config))


if __name__ == '__main__':
    unittest.main()
