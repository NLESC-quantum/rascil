"""The data models used in RASCIL:

"""

__all__ = ['XVisibility']


import xarray
import logging
import sys
import warnings

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import FITSFixedWarning

warnings.simplefilter('ignore', FITSFixedWarning)
warnings.simplefilter('ignore', AstropyDeprecationWarning)

from rascil.data_models.polarisation import PolarisationFrame, ReceptorFrame

log = logging.getLogger('logger')

class XVisibility:
    """ XVisibility table class similar to Visibility but using xarray

    Visibility with uvw, time, integration_time, frequency, channel_bandwidth, a1, a2, vis, weight
    as separate columns in a numpy structured array, The fundemental unit is a complex vector of polarisation.

    XVisibility is defined to hold an observation with one direction.
    Polarisation frame is the same for the entire data set and can be stokes, circular, linear.
    The configuration is also an attribute.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.
    
    """
    
    def __init__(self,
                 data=None, frequency=None, channel_bandwidth=None,
                 phasecentre=None, configuration=None, uvw=None,
                 time=None, antenna1=None, antenna2=None, vis=None, flags=None,
                 weight=None, imaging_weight=None, integration_time=None,
                 polarisation_frame=PolarisationFrame('stokesI'),
                 source='anonymous', meta=None):
        """Visibility

        :param data: xarray.Dataset
        :param frequency: Frequency, one per row
        :param channel_bandwidth: Channel bamdwidth, one per row
        :param phasecentre: Phasecentre (skycoord)
        :param configuration: Configuration
        :param uvw: uvw coordinates [:,3]
        :param time: Time (UTC) per row
        :param antenna1: dish/station index
        :param antenna2: dish/station index
        :param vis: Complex visibility [:, npol]
        :param flags: Flags [:, npol]
        :param weight: Weight [: npol]
        :param imaging_weight: Imaging weight [:, npol]
        :param integration_time: Integration time, per row
        :param polarisation_frame: Polarisation Frame e.g. Polarisation_frame("linear")
        :param source: Source name
        :param meta: Meta info
        """
        if data is None and vis is not None:
            if imaging_weight is None:
                imaging_weight = weight
            nvis = vis.shape[0]
            assert len(time) == nvis
            assert len(frequency) == nvis
            assert len(channel_bandwidth) == nvis
            assert len(antenna1) == nvis
            assert len(antenna2) == nvis
            
            npol = polarisation_frame.npol

            dims = ("time", "polarisation", "spatial")

            coords = {"time": vis.time,
                      "polarisation": vis.polarisation_frame.names,
                      "spatial": numpy.zeros([3])}

            xvis_dict = {}
            xvis_dict["vis"] = xarray.DataArray(vis, dims=["time", "polarisation"])
            xvis_dict["uvw"] = xarray.DataArray(uvw, dims=["time", "spatial"])
            xvis_dict["antenna1"] = xarray.DataArray(antenna1, dims=["time"])
            xvis_dict["antenna2"] = xarray.DataArray(antenna2, dims=["time"])
            xvis_dict["datetime"] = \
                xarray.DataArray(Time(time / 86400.0, format='mjd', scale='utc').datetime64, dims=["time"])
            xvis_dict["weight"] = xarray.DataArray(weight, dims=["time", "polarisation"])
            xvis_dict["imaging_weight"] = xarray.DataArray(imaging_weight,
                                                           dims=["time", "polarisation"])
            xvis_dict["flags"] = xarray.DataArray(flags, dims=["time", "polarisation"])
            xvis_dict["frequency"] = xarray.DataArray(frequency, dims=["time"])
            xvis_dict["channel_bandwidth"] = xarray.DataArray(channel_bandwidth, dims=["time"])
            xvis_dict["integration_time"] = xarray.DataArray(integration_time, dims=["time"])
            data = xarray.Dataset(xvis_dict, coords=coords)
        
        self.data = data  # xarray
        self.phasecentre = phasecentre  # Phase centre of observation
        self.configuration = configuration  # Antenna/station configuration
        self.polarisation_frame = polarisation_frame
        self.frequency_map = None
        self.source = source
        self.meta = meta
    
    def __str__(self):
        """Default printer for Visibility

        """
        ufrequency = numpy.unique(self.frequency)
        uchannel_bandwidth = numpy.unique(self.channel_bandwidth)
        s = "Visibility:\n"
        s += "\tSource: %s\n" % self.source
        s += "\tNumber of visibilities: %s\n" % self.nvis
        s += "\tNumber of channels: %d\n" % len(ufrequency)
        s += "\tFrequency: %s\n" % str(ufrequency)
        s += "\tChannel bandwidth: %s\n" % str(uchannel_bandwidth)
        s += "\tNumber of polarisations: %s\n" % self.npol
        s += "\tVisibility shape: %s\n" % str(self.vis.shape)
        s += "\tNumber flags: %s\n" % numpy.sum(self.flags)
        s += "\tPolarisation Frame: %s\n" % self.polarisation_frame.type
        s += "\tPhasecentre: %s\n" % self.phasecentre
        s += "\tConfiguration: %s\n" % self.configuration.name
        s += "\tMetadata: %s\n" % self.meta
        
        return s
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        for col in self.data.dtype.fields.keys():
            size += self.data[col].nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return len(self.frequency)

    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.polarisation_frame.npol
    
    @property
    def nvis(self):
        """ Number of visibilities (i.e. rows)
        """
        return self.data['vis'].shape[0]
    
    @property
    def uvw(self):
        """ UVW coordinates (wavelengths) [nrows, 3]
        """
        return self.data['uvw']
    
    @property
    def u(self):
        """ u coordinate (wavelengths) [nrows]
        """
        return self.data['uvw'][:, 0]
    
    @property
    def v(self):
        """ v coordinate (wavelengths) [nrows]
        """
        return self.data['uvw'][:, 1]
    
    @property
    def w(self):
        """ w coordinate (wavelengths) [nrows]
        """
        return self.data['uvw'][:, 2]
    
    @property
    def uvdist(self):
        """ uv distance (wavelengths) [nrows]
        """
        return numpy.hypot(self.u, self.v)
    
    @property
    def uvwdist(self):
        """ uvw distance (wavelengths) [nrows]
        """
        return numpy.hypot(self.u, self.v, self.w)
    
    @property
    def time(self):
        """ Time (UTC) [nrows]
        """
        return self.data['time']
    
    @property
    def integration_time(self):
        """ Integration time [nrows]
        """
        return self.data['integration_time']
    
    @property
    def frequency(self):
        """ Frequency values
        """
        return self.data['frequency']
    
    @property
    def channel_bandwidth(self):
        """ Channel bandwidth values
        """
        return self.data['channel_bandwidth']
    
    @property
    def antenna1(self):
        """ Antenna index
        """
        return self.data['antenna1']
    
    @property
    def antenna2(self):
        """ Antenna index
        """
        return self.data['antenna2']

    @property
    def vis(self):
        """Complex visibility [:, npol]
        """
        return self.data['vis']

    @property
    def flagged_vis(self):
        """Flagged complex visibility [:, npol]
        """
        return self.data['vis'] * (1 - self.flags)

    @property
    def flags(self):
        """flags [:, npol]
        """
        return self.data['flags']

    @property
    def weight(self):
        """Weight [: npol]
        """
        return self.data['weight']

    @property
    def flagged_weight(self):
        """Weight [: npol]
        """
        return self.data['weight'] * (1 - self.data['flags'])

    @property
    def imaging_weight(self):
        """  Imaging weight [:, npol]
        """
        return self.data['imaging_weight']

    @property
    def flagged_imaging_weight(self):
        """  Flagged Imaging weight [:, npol]
        """
        return self.data['imaging_weight'] * (1 - self.data['flags'])

