"""The data models used in RASCIL:

"""

__all__ = ['XBlockVisibility', 'XImage', 'XConfiguration']

import copy
import logging
import warnings

import numpy
import xarray
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import FITSFixedWarning

warnings.simplefilter('ignore', FITSFixedWarning)
warnings.simplefilter('ignore', AstropyDeprecationWarning)

from rascil.data_models.polarisation import PolarisationFrame, ReceptorFrame

log = logging.getLogger('logger')


class XConfiguration():
    """ Describe a XConfiguration as locations in x,y,z, mount type, diameter, names, and
        overall location
    """
    
    def __init__(self, name='', location=None,
                 names="%s", xyz=None, mount="alt-az", frame="",
                 receptor_frame=ReceptorFrame("linear"),
                 diameter=None, offset=None, stations="%s", vp_type=None):
        
        """Configuration object describing data for processing

        :param name: Name of configuration e.g. 'LOWR3'
        :param location: Location of array as an astropy EarthLocation
        :param names: Names of the dishes/stations
        :param xyz: Geocentric coordinates of dishes/stations
        :param mount: Mount types of dishes/stations 'altaz' | 'xy' | 'equatorial'
        :param frame: Reference frame of locations
        :param receptor_frame: Receptor frame
        :param diameter: Diameters of dishes/stations (m)
        :param offset: Axis offset (m)
        :param stations: Identifiers of the dishes/stations
        :param vp_type: Type of voltage pattern (string)
        """
        __slots__ = ()
        
        nants = len(names)
        if isinstance(stations, str):
            stations = [stations % ant for ant in range(nants)]
        if isinstance(names, str):
            names = [names % ant for ant in range(nants)]
        if isinstance(mount, str):
            mount = numpy.repeat(mount, nants)
        if offset is None:
            offset = numpy.zeros([nants, 3])
        if vp_type is None:
            vp_type = numpy.repeat("", nants)
        
        coords = {
            "id": range(nants),
            "spatial": numpy.zeros([3])
        }
        
        datavars = dict()
        datavars["names"] = xarray.DataArray(names, dims=["id"])
        datavars["xyz"] = xarray.DataArray(xyz, dims=["id", "spatial"])
        datavars["diameter"] = xarray.DataArray(diameter, dims=["id"])
        datavars["mount"] = xarray.DataArray(mount, dims=["id"])
        datavars["vp_type"] = xarray.DataArray(vp_type, dims=["id"])
        datavars["offset"] = xarray.DataArray(offset, dims=["id", "spatial"])
        datavars["stations"] = xarray.DataArray(stations, dims=["id"])
        
        self.name = name  # Name of configuration
        self.location = location  # EarthLocation
        self.receptor_frame = receptor_frame
        self.frame = frame
        
        self.data = xarray.Dataset(datavars, coords=coords)
    
    def __str__(self):
        """Default printer for Configuration

        """
        s = "XConfiguration:\n"
        s += "\tNumber of antennas/stations: %s\n" % len(self.names)
        s += "\tNames: %s\n" % self.names
        s += "\tDiameter: %s\n" % self.diameter
        s += "\tMount: %s\n" % self.mount
        s += "\tXYZ: %s\n" % self.xyz
        s += "\tAxis offset: %s\n" % self.offset
        s += "\tStations: %s\n" % self.stations
        s += "\tVoltage pattern type: %s\n" % self.vp_type
        
        return s
    
    def size(self):
        """ Return size in GB
        """
        size = self.data.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def names(self):
        """ Names of the dishes/stations"""
        return self.data['names']
    
    @property
    def vp_type(self):
        """ Names of the voltage pattern type"""
        return self.data['vp_type']
    
    @property
    def diameter(self):
        """ Diameter of dishes/stations (m)
        """
        return self.data['diameter']
    
    @property
    def xyz(self):
        """ XYZ locations of dishes/stations [:,3] (m)
        """
        return self.data['xyz']
    
    @property
    def mount(self):
        """ Mount types of dishes/stations ('azel' | 'equatorial'
        """
        return self.data['mount']
    
    @property
    def offset(self):
        """ Axis offset [:, 3] (m)
        """
        return self.data['offset']
    
    @property
    def stations(self):
        """ Station/dish identifier (may be the same as names)"""
        return self.data['stations']


class XBlockVisibility():
    """ XBlockVisibility table class similar to Visibility but using xarray

    XBlockVisibility is defined to hold an observation with one direction.
    Polarisation frame is the same for the entire data set and can be stokes, circular, linear.
    The configuration is also an attribute.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.
    
    Here is an example of an XBlockVisibility
    
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
                  "polarisation": polarisation_frame.names
                  }
        
        datavars = dict()
        datavars["vis"] = xarray.DataArray(vis, dims=["time", "baseline", "frequency", "polarisation"])
        datavars["weight"] = xarray.DataArray(weight, dims=["time", "baseline", "frequency", "polarisation"])
        if imaging_weight is None:
            imaging_weight = weight
        datavars["imaging_weight"] = xarray.DataArray(imaging_weight,
                                                      dims=["time", "baseline", "frequency", "polarisation"])
        datavars["flags"] = xarray.DataArray(flags, dims=["time", "baseline", "frequency", "polarisation"])
        datavars["u"] = xarray.DataArray(uvw[..., 0], dims=["time", "baseline"])
        datavars["v"] = xarray.DataArray(uvw[..., 1], dims=["time", "baseline"])
        datavars["w"] = xarray.DataArray(uvw[..., 2], dims=["time", "baseline"])
        datavars["u_lambda"] = xarray.DataArray(uvw_lambda[..., 0], dims=["time", "baseline", "frequency"])
        datavars["v_lambda"] = xarray.DataArray(uvw_lambda[..., 1], dims=["time", "baseline", "frequency"])
        datavars["w_lambda"] = xarray.DataArray(uvw_lambda[..., 2], dims=["time", "baseline", "frequency"])
        datavars["channel_bandwidth"] = xarray.DataArray(channel_bandwidth, dims=["frequency"])
        datavars["integration_time"] = xarray.DataArray(integration_time, dims=["time"])
        
        datavars["datetime"] = xarray.DataArray(Time(time / 86400.0, format='mjd', scale='utc').datetime64,
                                                dims="time")
        
        self.phasecentre = phasecentre  # Phase centre of observation
        self.configuration = configuration  # Antenna/station configuration
        self.polarisation_frame = polarisation_frame
        self.source = source
        self.meta = meta
        
        self.data = xarray.Dataset(datavars, coords=coords)
    
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
    def uvdist(self):
        """ uv distance (m)
        """
        return numpy.hypot(self.data["u"], self.data["v"])
    
    @property
    def uvdist_lambda(self):
        """ uv distance (wavelengths)
        """
        return numpy.hypot(self.data["u_lambda"], self.data["v_lambda"])
    
    @property
    def u(self):
        """ u (m)
        """
        return self.data["u"]
    
    @property
    def v(self):
        """ v (m)
        """
        return self.data["v"]
    
    @property
    def w(self):
        """ w (m)
        """
        return self.data["w"]
    
    @property
    def u_lambda(self):
        """ u (wavelengths)
        """
        return self.data["u_lambda"]
    
    @property
    def v_lambda(self):
        """ v (wavelengths)
        """
        return self.data["v_lambda"]
    
    @property
    def w_lambda(self):
        """ w (wavelengths)
        """
        return self.data["w_lambda"]
    
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
        s = "XVisibility.xarray.Dataset:\n"
        s += "{}\n".format(str(self.data))
        s += "XVisibility.attributes:\n"
        s += "\tSource: %s\n" % self.source
        s += "\tPolarisation Frame: %s\n" % self.polarisation_frame.type
        s += "\tPhasecentre: %s\n" % self.phasecentre
        s += "\tConfiguration: %s\n" % self.configuration.name
        s += "\tMetadata: %s\n" % self.meta
        
        return s


class XImage():
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
    
    def __init__(self, phasecentre, frequency, polarisation_frame=None,
                 dtype=None, data=None, wcs=None):
        """ Create an XImage

        :param axes:
        :param cellsize:
        :param frequency:
        :param phasecentre:
        :param polarisation_frame:
        :return: XImage
        """
        
        nx, ny = data.shape[-2], data.shape[-1]
        cellsize = numpy.abs(wcs.wcs.cdelt[1])
        cx = phasecentre.ra.to("deg").value
        cy = phasecentre.dec.to("deg").value
        
        dims = ["frequency", "polarisation", "lat", "lon"]
        coords = {
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "lat": numpy.linspace(cy - cellsize * ny // 2, cy + cellsize * ny // 2, ny),
            "lon": numpy.linspace(cx - cellsize * nx // 2, cx + cellsize * nx // 2, nx)
        }
        
        self.wcs = wcs
        self.polarisation_frame = polarisation_frame
        ramesh, decmesh = numpy.meshgrid(numpy.arange(ny), numpy.arange(nx))
        self.ra, self.dec = wcs.sub([1, 2]).wcs_pix2world(ramesh, decmesh, 0)
        
        nchan = len(frequency)
        npol = polarisation_frame.npol
        if dtype is None:
            dtype = "float"
        
        if data is None:
            data = numpy.zeros([nchan, npol, ny, nx], dtype=dtype)
        else:
            assert data.shape == (nchan, npol, ny, nx), \
                "Polarisation frame {} and data shape are incompatible".format(polarisation_frame.type)
        
        self.data = xarray.DataArray(data, dims=dims, coords=coords)
    
    def size(self):
        """ Return size in GB
        """
        size = self.data.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.data.shape[0]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.data.shape[1]
    
    @property
    def nheight(self):
        """ Number of pixels height i.e. y
        """
        return self.data.shape[2]
    
    @property
    def nwidth(self):
        """ Number of pixels width i.e. x
        """
        return self.data.shape[3]
    
    @property
    def frequency(self):
        """ Frequency values
        """
        w = self.wcs.sub(['spectral'])
        return w.wcs_pix2world(range(self.nchan), 0)[0]
    
    @property
    def shape(self):
        """ Shape of data array
        """
        return self.data.shape
    
    @property
    def phasecentre(self):
        """ Phasecentre (from WCS)
        """
        return SkyCoord(self.wcs.wcs.crval[0] * u.deg, self.wcs.wcs.crval[1] * u.deg)
    
    def __str__(self):
        """Default printer for Image

        """
        s = "Image:\n"
        s += "\tShape: %s\n" % str(self.data.shape)
        s += "\tData type: %s\n" % str(self.data.dtype)
        s += "\tWCS: %s\n" % self.wcs.__repr__()
        s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
        return s
