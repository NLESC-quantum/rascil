"""The data models used in RASCIL:

"""

__all__ = ['XVisibility', 'XImage']


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

class XVisibility(xarray.Dataset):
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
        super().__init__()
        if imaging_weight is None:
            imaging_weight = weight

        coords = {"time": time,
                  "baseline": baselines,
                  "frequency": frequency,
                  "polarisation": polarisation_frame.names}
        
        datavars = dict()
        datavars["vis"] = xarray.DataArray(vis, dims=["time", "baseline", "frequency", "polarisation"])
        datavars["weight"] = xarray.DataArray(weight, dims=["time", "baseline", "frequency", "polarisation"])
        datavars["imaging_weight"] = xarray.DataArray(imaging_weight, dims=["time", "baseline", "frequency", "polarisation"])
        datavars["flags"] = xarray.DataArray(flags, dims=["time", "baseline", "frequency", "polarisation"])
        
        datavars["uvw"] = xarray.DataArray(uvw, dims=["time", "baseline", "spatial"])
        datavars["uvdist"] = xarray.DataArray(numpy.hypot(uvw[..., 0], uvw[..., 1]),
                                              dims=["time", "baseline"])

        datavars["channel_bandwidth"] = xarray.DataArray(channel_bandwidth, dims=["frequency"])
        datavars["integration_time"] = xarray.DataArray(integration_time, dims=["time"])
        datavars["datetime"] = \
            xarray.DataArray(Time(time / 86400.0, format='mjd', scale='utc').datetime64,
                             dims=["time"])

        attrs = dict()
        attrs['phasecentre'] = phasecentre  # Phase centre of observation
        attrs['configuration'] = configuration  # Antenna/station configuration
        attrs['polarisation_frame'] = polarisation_frame
        attrs['source'] = source
        attrs['meta'] = meta

        super().__init__(datavars, coords=coords, attrs=attrs)
        
def xvisibility_size(xvis):
    """ Return size in GB
    """
    size = 0
    for col in xvis.data.dtype.fields.keys():
        size += xvis.data[col].nbytes
    return size / 1024.0 / 1024.0 / 1024.0


def xvisibility_nchan(xvis):
    """ Number of channels
    """
    return len(xvis.frequency)


def xvisibility_npol(xvis):
    """ Number of polarisations
    """
    return xvis.polarisation_frame.npol


def xvisibility_nvis(xvis):
    """ Number of visibilities (i.e. rows)
    """
    return xvis.data['vis'].shape[0]
    

def xvisibility_u(xvis):
    """ u coordinate (wavelengths) [nrows]
    """
    return xvis.data['uvw'][:, 0]


def xvisibility_v(xvis):
    """ v coordinate (wavelengths) [nrows]
    """
    return xvis.data['uvw'][:, 1]


def xvisibility_w(xvis):
    """ w coordinate (wavelengths) [nrows]
    """
    return xvis.data['uvw'][:, 2]


def xvisibility_uvdist(xvis):
    """ uv distance (wavelengths) [nrows]
    """
    return numpy.hypot(xvis.uvw[:,0], xvis.uvw[:1,])


def xvisibility_uvwdist(xvis):
    """ uvw distance (wavelengths) [nrows]
    """
    return numpy.hypot(xvis.u, xvis.v, xvis.w)


def xvisibility_flagged_vis(xvis):
    """Flagged complex visibility [:, npol]
    """
    return xvis.data['vis'] * (1 - xvis.flags)


def xvisibility_flagged_weight(xvis):
    """Weight [: npol]
    """
    return xvis.data['weight'] * (1 - xvis.data['flags'])


def xvisibility_flagged_imaging_weight(xvis):
    """  Flagged Imaging weight [:, npol]
    """
    return xvis.data['imaging_weight'] * (1 - xvis.data['flags'])


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


def ximage_nchan(ximage):
    """ Number of channels
    """
    return ximage.data.shape[0]


def ximage_npol(ximage):
    """ Number of polarisations
    """
    return ximage.data.shape[1]


def ximage_nheight(ximage):
    """ Number of pixels height i.e. y
    """
    return ximage.data.shape[2]


def ximage_nwidth(ximage):
    """ Number of pixels width i.e. x
    """
    return ximage.data.shape[3]


def ximage_frequency(ximage):
    """ Frequency values
    """
    w = ximage.wcs.sub(['spectral'])
    return w.wcs_pix2world(range(ximage.nchan), 0)[0]


def ximage_shape(ximage):
    """ Shape of data array
    """
    return ximage.data.shape

def ximage_phasecentre(ximage):
    """ Phasecentre (from WCS)
    """
    return SkyCoord(ximage.wcs.wcs.crval[0] * u.deg, ximage.wcs.wcs.crval[1] * u.deg)
