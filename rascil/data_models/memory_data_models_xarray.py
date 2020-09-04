"""The data models used in RASCIL:

"""

__all__ = ['XVisibility', 'XImage']


import xarray
import logging
import sys
import warnings
import copy

import numpy
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import FITSFixedWarning

warnings.simplefilter('ignore', FITSFixedWarning)
warnings.simplefilter('ignore', AstropyDeprecationWarning)

from rascil.data_models.polarisation import PolarisationFrame, ReceptorFrame

log = logging.getLogger('logger')

class XVisibility():
    """ XVisibility table class similar to Visibility but using xarray

    XVisibility is defined to hold an observation with one direction.
    Polarisation frame is the same for the entire data set and can be stokes, circular, linear.
    The configuration is also an attribute.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.
    
    Here is an example of an XVisibility
    
            <xarray.XVisibility>
        Dimensions:            (baseline: 10, frequency: 3, polarisation: 4, spatial: 3, time: 16)
        Coordinates:
          * time               (time) float64 5.085e+09 5.085e+09 ... 5.085e+09
          * baseline           (baseline) MultiIndex
          - antenna1           (baseline) int64 1 2 2 3 3 3 4 4 4 4
          - antenna2           (baseline) int64 0 0 1 0 1 2 0 1 2 3
          * frequency          (frequency) float64 1e+08 1.05e+08 1.1e+08
          * polarisation       (polarisation) <U2 'XX' 'XY' 'YX' 'YY'
        Dimensions without coordinates: spatial
        Data variables:
            vis                (time, baseline, frequency, polarisation) complex128 0...
            weight             (time, baseline, frequency, polarisation) float64 1.0 ...
            imaging_weight     (time, baseline, frequency, polarisation) float64 1.0 ...
            flags              (time, baseline, frequency, polarisation) int64 1 1 ... 1
            uvw                (time, baseline, spatial) float64 -32.05 ... -41.32
            uvdist             (time, spatial) float64 34.2 54.0 77.12 ... 32.43 46.31
            channel_bandwidth  (frequency) float64 1e+07 1e+07 1e+07
            integration_time   (time) float64 1.795e+03 1.795e+03 ... 1.795e+03
            datetime           (time) datetime64[ns] 2020-01-01T17:30:36.436666595 .....
        Attributes:
            phasecentre:         <SkyCoord (ICRS): (ra, dec) in deg\n    (180., -35.)>
            configuration:       Configuration:\n\nName: LOWBD2-CORE\n\tNumber of ant...
            polarisation_frame:  linear
            source:              anonymous
            meta:                None

    
    """
    
    def __init__(self, frequency=None, channel_bandwidth=None, phasecentre=None, configuration=None,
                 uvw=None, time=None, vis=None, flags=None, weight=None, baselines=None,
                 imaging_weight=None, integration_time=None, polarisation_frame=PolarisationFrame('stokesI'),
                 source='anonymous', meta=None):
        """Xarray format Visibility
        
        The coordinates for vis, weight, imaging_weight, and flags are (time, baseline, frequency, polarisation)

        :param frequency: Frequency array
        :param channel_bandwidth: Channel bandwidth
        :param phasecentre: Phasecentre (skycoord)
        :param configuration: Configuration
        :param uvw: uvw coordinates [ntimes, nbaselines, 3]
        :param time: Time (UTC)
        :param baselines: tuples of antenna1, antenna2
        :param vis: Complex visibility (time, baseline, frequency, polarisation)
        :param flags: Flags (time, baseline, frequency, polarisation)
        :param weight: Weight (time, baseline, frequency, polarisation)
        :param imaging_weight: Imaging weight (time, baseline, frequency, polarisation)
        :param integration_time: Integration time[time]
        :param polarisation_frame: Polarisation Frame e.g. Polarisation_frame("linear")
        :param source: Source name
        :param meta: Meta info
        """
        k = (frequency / const.c).value
        uvw_lambda = numpy.einsum("tbs,k->tbks", uvw, k)
        
        coords = {"time": time,
                  "baseline": baselines,
                  "frequency": frequency,
                  "polarisation": polarisation_frame.names,
                  "spatial": numpy.zeros([3], dtype='float')}
        
        datavars = dict()
        datavars["vis"] = xarray.DataArray(vis, dims=["time", "baseline", "frequency", "polarisation"])
        datavars["weight"] = xarray.DataArray(weight, dims=["time", "baseline", "frequency", "polarisation"])
        if imaging_weight is None:
            imaging_weight = weight
        datavars["imaging_weight"] = xarray.DataArray(imaging_weight,
                                                      dims=["time", "baseline", "frequency", "polarisation"])
        datavars["flags"] = xarray.DataArray(flags, dims=["time", "baseline", "frequency", "polarisation"])
        datavars["uvw"] = xarray.DataArray(uvw, dims=["time", "baseline", "spatial"])
        datavars["uvw_lambda"] = xarray.DataArray(uvw_lambda, dims=["time", "baseline", "spatial", "frequency"]),
        datavars["uvdist"] = xarray.DataArray(numpy.hypot(uvw[...,0], uvw[...,1]),
                                              dims=["time", "baseline"])
        datavars["uvdist_lambda"] = xarray.DataArray(numpy.hypot(uvw_lambda[...,0], uvw_lambda[...,1]),
                                              dims=["time", "baseline", "frequency"])
        datavars["channel_bandwidth"] = xarray.DataArray(channel_bandwidth, dims=["frequency"])
        datavars["integration_time"] = xarray.DataArray(integration_time, dims=["time"])
        
        datavars["datetime"] = xarray.DataArray(Time(time / 86400.0, format='mjd', scale='utc').datetime64,
                                                dims="time")

        self.phasecentre = phasecentre  # Phase centre of observation
        self.configuration = configuration  # Antenna/station configuration
        self.polarisation_frame = polarisation_frame
        self.source = source
        self.meta = meta

        self.data=xarray.Dataset(datavars, coords=coords)

    @property
    def nchan(self):
        """ Number of channels
        """
        return len(self.frequency)

    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.data.polarisation_frame.npol

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
        """ u coordinate (m) [nrows]
        """
        return self.data['uvw'][..., 0]

    @property
    def v(self):
        """ v coordinate (m) [nrows]
        """
        return self.data['uvw'][..., 1]

    @property
    def w(self):
        """ w coordinate (m) [nrows]
        """
        return self.data['uvw'][..., 2]

    @property
    def u_lambda(self):
        """ u coordinate (wavelengths) [nrows]
        """
        return self.data['uvw_lambda'][..., 0]

    @property
    def v_lambda(self):
        """ v coordinate (wavelengths) [nrows]
        """
        return self.data['uvw_lambda'][..., 1]

    @property
    def w_lambda(self):
        """ w coordinate (wavelengths) [nrows]
        """
        return self.data['uvw_lambda'][..., 2]

    @property
    def uvdist(self):
        """ uv distance (m) [nrows]
        """
        return self.data["uvdist"]

    @property
    def uvdist_lambda(self):
        """ uv distance (wavelengths) [nrows]
        """
        return self.data["uvdist_lambda"]

    @property
    def time(self):
        """ Time (UTC) [nrows]
        """
        return self.data['time']

    @property
    def datetime(self):
        """ Date Time (UTC) [nrows]
        """
        return self.data['datetime']

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


    def sel(self, *args, **kwargs):
        """ Use the xarray.sel operator to return selected XVisibility

        :param selection:
        :return:
        """
        newxvis = copy.deepcopy(self)
        newxvis.data = newxvis.data.sel(*args, **kwargs)
        return newxvis

    def where(self, *args, **kwargs):
        """ Use the xarray.where operator to return selected XVisibility, masked where condition fails

        :param condition:
        :return:
        """
        newxvis = copy.deepcopy(self)
        newxvis.data = newxvis.data.where(*args, **kwargs)
        return newxvis

    def groupby(self, *args, **kwargs):
        """ Use the xarray.groupby operator to return selected XVisibility

        :param condition:
        :return:
        """
        return [copy.deepcopy(newxvis) for newxvis in self.data.groupby(*args, **kwargs)]

    def groupby_bins(self, *args, **kwargs):
        """ Use the xarray.groupby_bins operator to return selected XVisibility

        :param condition:
        :return:
        """
        return [copy.deepcopy(newxvis) for newxvis in self.data.groupby_bins(*args, **kwargs)]
    

    def __str__(self):
        """Default printer for XVisibility

        """
        s = "XVisibility.data:\n"
        s += "{}\n".format(str(self.data))
        s += "XVisibility.attributes:\n"
        s += "\tSource: %s\n" % self.source
        s += "\tPolarisation Frame: %s\n" % self.polarisation_frame.type
        s += "\tPhasecentre: %s\n" % self.phasecentre
        s += "\tConfiguration: %s\n" % self.configuration.name
        s += "\tMetadata: %s\n" % self.meta
        
        return s



class XImage(xarray.DataArray):
    """Image class with Image data (as a numpy.array) and the AstroPy `implementation of
    a World Coodinate System <http://docs.astropy.org/en/stable/wcs>`_

    Many operations can be done conveniently using numpy processing_components on Image.data_models.

    Most of the imaging processing_components require an image in canonical format:
    - 4 axes: RA, DEC, POL, FREQ

    The conventions for indexing in WCS and numpy are opposite.
    - In astropy.wcs, the order is (longitude, latitude, polarisation, frequency)
    - in numpy, the order is (frequency, polarisation, latitude, longitude)

    .. warning::
        The polarisation_frame is kept in two places, the WCS and the polarisation_frame
        variable. The latter should be considered definitive.

    """
    
    def __init__(self, data, wcs=None, polarisation_frame=None):
        """ Create Image

        :param data: Data for image
        :param wcs: Astropy WCS object
        :param polarisation_frame: e.g. PolarisationFrame('stokesIQUV')
        """
        dims = {"l", "m", "frequency", "polarisation", "time"}
        
        npol, nchan, ny, nx = data.shape
        
        coords = {"l":range(nx), "m":range(ny),
                  "frequency": range(nchan),
                  "polarisation":polarisation_frame.names,
                  "time": None}

        datavars = dict()
        datavars["image"] = xarray.DataArray(data, coords=coords)

        attrs=dict()
        attrs['wcs'] = wcs
        attrs['polarisation_frame'] = polarisation_frame

        super().__init__(data=data, attrs=attrs)

def ximage_size(ximage):
    """ Return size in GB
    """
    size = 0
    size += ximage.data.nbytes
    return size / 1024.0 / 1024.0 / 1024.0

def ximage_phasecentre(ximage):
    """ Phasecentre (from WCS)
    """
    return SkyCoord(ximage.wcs.wcs.crval[0] * u.deg, ximage.wcs.wcs.crval[1] * u.deg)
