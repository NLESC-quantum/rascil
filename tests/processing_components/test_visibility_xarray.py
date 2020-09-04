""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models import Skycomponent, PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.processing_components.visibility.vis_xarray import convert_blockvisibility_to_xvisibility


class TestXVisibility(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE', rmax=1000)
        self.times = (numpy.pi / 43200.0) * numpy.arange(-4 * 3600, +4 * 3600.0, 1800)
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
    
    def test_convert_blockvisibility_to_xvisibility(self):
        vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          integration_time=30.0,
                                          polarisation_frame=PolarisationFrame("linear"),
                                          weight=1.0)

        xvis = convert_blockvisibility_to_xvisibility(vis)
        print("\nInitial xvis")
        print(xvis)
        print("\nSlice of a DataArray")
        print(xvis.vis[100:110, 0:1])
        print("\nSelection of the Dataset by polarisation")
        print(xvis.sel({"polarisation": ["XY", "YX"]}))
        print("\nsel antenna1 yields smaller XVisibility")
        print(xvis.sel({"antenna1":10}))
        print("\nwhere antenna1 yields masked arrays")
        print(xvis.where(xvis.antenna1 == 10))
        print("\nBy uvdist yields masked arrays")
        print(xvis.where(xvis.uvdist < 40.0))
        print("\nBy time")
        print(xvis.where(xvis.datetime > numpy.datetime64("2020-01-01T23:00:00")))
        print("In bins of time")
        for result in xvis.groupby("datetime"):
            print(result[0], result[1].dims)
        print("In 3 bins of datetime")
        for result in xvis.groupby_bins(xvis.data["datetime"], bins=3):
            print(result[0], result[1].dims)
            
        # Selection of the Dataset by polarisation 
        newxvis = xvis.sel({"polarisation": ["XY", "YX"]})
        # sel antenna1 yields smaller XVisibility 
        newxvis = xvis.sel({"antenna1":10})
        # where antenna1 yields masked arrays 
        newxvis = xvis.where(xvis.antenna1 == 10)
        # By uvdist yields masked arrays 
        newxvis = xvis.where(xvis.uvdist < 40.0)
        # By time 
        newxvis = xvis.where(xvis.datetime > numpy.datetime64("2020-01-01T23:00:00"))
        # In bins of time 
        for result in xvis.groupby("datetime") :
            print(result[0], result[1].dims)
        # In 3 bins of datetime 
        for result in xvis.groupby_bins(xvis.data["datetime"], bins=3):
            print(result[0], result[1].dims)


if __name__ == '__main__':
    unittest.main()
