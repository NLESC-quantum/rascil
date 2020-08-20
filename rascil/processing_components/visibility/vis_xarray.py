""" Visibility operations

"""
import xarray

__all__ = ['convert_visibility_to_xvisibility',
           'convert_xvisibility_to_visibility']

import logging
from typing import Union, List

import numpy 

from rascil.data_models.memory_data_models import Visibility, QA
from rascil.data_models.polarisation import PolarisationFrame

log = logging.getLogger('logger')

class XVisibility:
    """ Visibility as an Xarray

    visibility with uvw, time, integration_time, frequency, channel_bandwidth, pol,
    a1, a2, vis, weight Columns in a numpy structured array.

    visibility is defined to hold an observation with one direction.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.

    Polarisation frame is the same for the entire data set and can be stokesI, circular, linear.

    The configuration is also an attribute.
    """
    
    def __init__(self,
                 data=None, frequency=None, channel_bandwidth=None,
                 phasecentre=None, configuration=None, uvw=None,
                 time=None, vis=None, weight=None, integration_time=None,
                 flags=None,
                 polarisation_frame=PolarisationFrame('stokesI'),
                 imaging_weight=None, source='anonymous', meta=None):
        """visibility

        :param data: Structured data (used in copying)
        :param frequency: Frequency [nchan]
        :param channel_bandwidth: Channel bandwidth [nchan]
        :param phasecentre: Phasecentre (SkyCoord)
        :param configuration: Configuration
        :param uvw: UVW coordinates (m) [:, nant, nant, 3]
        :param time: Time (UTC) [:]
        :param vis: Complex visibility [:, nant, nant, nchan, npol]
        :param flags: Flags [:, nant, nant, nchan]
        :param weight: [:, nant, nant, nchan, npol]
        :param imaging_weight: [:, nant, nant, nchan, npol]
        :param integration_time: Integration time [:]
        :param polarisation_frame: Polarisation_Frame e.g. Polarisation_Frame("linear")
        :param source: Source name
        :param meta: Meta info
        """
        if meta is None:
            meta = dict()
        if data is None and vis is not None:
            ntimes, nants, _, nchan, npol = vis.shape
            assert vis.shape == weight.shape
            if isinstance(frequency, list):
                frequency = numpy.array(frequency)
            assert len(frequency) == nchan
            if isinstance(channel_bandwidth, list):
                channel_bandwidth = numpy.array(channel_bandwidth)
            assert len(channel_bandwidth) == nchan
            desc = [('index', 'i8'),
                    ('uvw', 'f8', (nants, nants, 3)),
                    ('time', 'f8'),
                    ('integration_time', 'f8'),
                    ('vis', 'c16', (nants, nants, nchan, npol)),
                    ('flags', 'i8', (nants, nants, nchan, npol)),
                    ('weight', 'f8', (nants, nants, nchan, npol)),
                    ('imaging_weight', 'f8', (nants, nants, nchan, npol))]
            data = numpy.zeros(shape=[ntimes], dtype=desc)
            data['index'] = list(range(ntimes))
            data['uvw'] = uvw
            data['time'] = time  # MJD in seconds
            data['integration_time'] = integration_time  # seconds
            data['vis'] = vis
            data['flags'] = flags
            data['weight'] = weight
            data['imaging_weight'] = imaging_weight
        
        self.data = data  # numpy structured array
        self.frequency = frequency
        self.channel_bandwidth = channel_bandwidth
        self.phasecentre = phasecentre  # Phase centre of observation
        self.configuration = configuration  # Antenna/station configuration
        self.polarisation_frame = polarisation_frame
        self.source = source
        self.meta = meta
    
    def __str__(self):
        """Default printer for visibility

        """
        s = "visibility:\n"
        s += "\tSource %s\n" % self.source
        s += "\tPhasecentre: %s\n" % self.phasecentre
        s += "\tNumber of visibility blocks: %s\n" % self.nvis
        s += "\tNumber of integrations: %s\n" % len(self.time)
        s += "\tVisibility shape: %s\n" % str(self.vis.shape)
        s += "\tNumber of flags: %s\n" % str(numpy.sum(self.flags))
        s += "\tNumber of channels: %d\n" % len(self.frequency)
        s += "\tFrequency: %s\n" % self.frequency
        s += "\tChannel bandwidth: %s\n" % self.channel_bandwidth
        s += "\tNumber of polarisations: %s\n" % self.npol
        s += "\tPolarisation Frame: %s\n" % self.polarisation_frame.type
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
        return self.data['vis'].shape[3]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.data['vis'].shape[4]
    
    @property
    def nants(self):
        """ Number of antennas
        """
        return self.data['vis'].shape[1]
    
    @property
    def uvw(self):
        """ UVW coordinates (metres) [nrows, nant, nant, 3]
        """
        return self.data['uvw']
    
    @property
    def u(self):
        """ u coordinate (metres) [nrows, nant, nant]
        """
        return self.data['uvw'][..., 0]
    
    @property
    def v(self):
        """ v coordinate (metres) [nrows, nant, nant]
        """
        return self.data['uvw'][..., 1]
    
    @property
    def w(self):
        """ w coordinate (metres) [nrows, nant, nant]
        """
        return self.data['uvw'][..., 2]
    
    @property
    def uvdist(self):
        """ uv distance (metres) [nrows, nant, nant]
        """
        return numpy.hypot(self.u, self.v)
    
    @property
    def uvwdist(self):
        """ uv distance (metres) [nrows, nant, nant]
        """
        return numpy.hypot(self.u, self.v, self.w)
    
    @property
    def vis(self):
        """ Complex visibility [nrows, nant, nant, ncha, npol]
        """
        return self.data['vis']
    
    @property
    def flagged_vis(self):
        """Flagged complex visibility [nrows, nant, nant, ncha, npol]
        """
        return self.data['vis'] * (1 - self.flags)
    
    @property
    def flags(self):
        """ Flags [nrows, nant, nant, nchan]
        """
        return self.data['flags']
    
    @property
    def weight(self):
        """ Weight[nrows, nant, nant, ncha, npol]
        """
        return self.data['weight']
    
    @property
    def flagged_weight(self):
        """Weight [: npol]
        """
        return self.data['weight'] * (1 - self.data['flags'])
    
    @property
    def imaging_weight(self):
        """ Imaging_weight[nrows, nant, nant, ncha, npol]
        """
        return self.data['imaging_weight']
    
    @property
    def flagged_imaging_weight(self):
        """ Flagged Imaging_weight[nrows, nant, nant, ncha, npol]
        """
        return self.data['imaging_weight'] * (1 - self.data['flags'])
    
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
    def nvis(self):
        """ Number of visibilities (in total)
        """
        return self.data.size


def convert_visibility_to_xvisibility(vis: Visibility) -> xarray.Dataset:
    """Convert visibility to XVisibility

    :param bvis:
    :param othervis:
    :return: XVisibility vis
    """
    coords = {"time": vis.time,
              "polarisation": vis.polarisation_frame.names}
    
    xvis_dict = {}
    xvis_dict["data"] = xarray.DataArray(vis.vis, dims=["time", "polarisation"])
    xvis_dict["weight"] = xarray.DataArray(vis.weight, dims=["time", "polarisation"])
    xvis_dict["imaging_weight"] = xarray.DataArray(vis.imaging_weight,
                                                   dims=["time", "polarisation"])
    xvis_dict["flags"] = xarray.DataArray(vis.flags, dims=["time", "polarisation"])
    xvis_dict["frequency"] = xarray.DataArray(vis.frequency, dims=["time"])
    xvis_dict["channel_bandwidth"] = xarray.DataArray(vis.channel_bandwidth, dims=["time"])
    xvis_dict["antenna1"] = xarray.DataArray(vis.antenna1, dims=["time"])
    xvis_dict["antenna2"] = xarray.DataArray(vis.antenna2, dims=["time"])
    xvis_dict["integration_time"] = xarray.DataArray(vis.integration_time, dims=["time"])
    xvis_xds = xarray.Dataset(xvis_dict, coords=coords)
    xvis_xds.attrs['source'] = vis.source
    xvis_xds.attrs['meta'] = vis.meta
    
    return xvis_xds
    # xvis_xds.attrs['xvis_name'] = xvis_name
    # xvis_xds.attrs['xvis_long'] = xvis_location['m0']['value']
    # xvis_xds.attrs['xvis_lat'] = xvis_location['m1']['value']
    # xvis_xds.attrs['xvis_elevation'] = xvis_location['m2']['value']
    # xvis_xds.attrs['long_units'] = xvis_location['m0']['unit']
    # xvis_xds.attrs['lat_units'] = xvis_location['m1']['unit']
    # xvis_xds.attrs['elevation_units'] = xvis_location['m2']['unit']
    # xvis_xds.attrs['coordinate_system'] = xvis_location['refer']


def convert_xvisibility_to_visibility(xvis: XVisibility) \
        -> Visibility:
    """Convert xvisibility to blockisibility

    :param vis:
    :param othervis:
    :return: Visibility vis
    """
    return Visibility()

