""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models import Skycomponent, PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_visibility
from rascil.processing_components.visibility.vis_xarray import convert_visibility_to_xvisibility


class TestXVisibility(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = Skycomponent(direction=self.compreldirection, frequency=self.frequency, flux=self.flux)
    
    def test_create_visibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre,
                                     polarisation_frame=PolarisationFrame("linear"),
                                     weight=1.0)
        assert self.vis.nvis == len(self.vis.time)
    
    def test_convert_visibility_to_xarray(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre,
                                     polarisation_frame=PolarisationFrame("linear"),
                                     weight=1.0)
        assert self.vis.nvis == len(self.vis.time)
        xvis = convert_visibility_to_xvisibility(self.vis)
        print(xvis)
        assert xvis.sel({"polarisation":["XX", "YY"]}).data.shape[1] == 2

if __name__ == '__main__':
    unittest.main()
