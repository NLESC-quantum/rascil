"""The data models used in RASCIL:

"""

__all__ = ['Configuration',
           'GainTable',
           'PointingTable',
           'Image',
           'GridData',
           'ConvolutionFunction',
           'Skycomponent',
           'SkyModel',
           'BlockVisibility',
           'FlagTable',
           'QA',
           'ScienceDataModel',
           'assert_same_chan_pol',
           'assert_vis_gt_compatible'
           ]

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

from rascil.data_models.polarisation import PolarisationFrame, ReceptorFrame

warnings.simplefilter('ignore', FITSFixedWarning)
warnings.simplefilter('ignore', AstropyDeprecationWarning)

log = logging.getLogger('rascil-logger')


class Configuration(xarray.Dataset):
    """ Describe a XConfiguration as locations in x,y,z, mount type, diameter, names, and
        overall location
    """
    
    def __init__(self, name='', location=None,
                 names=None, xyz=None, mount="alt-az", frame="",
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
        
        super().__init__()
        
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
            "id": list(range(nants)),
            "spatial": ["X", "Y", "Z"]
        }
        datavars = dict()
        datavars["names"] = xarray.DataArray(names, coords={"id": list(range(nants))}, dims=["id"])
        datavars["xyz"] = xarray.DataArray(xyz, coords=coords, dims=["id", "spatial"])
        datavars["diameter"] = xarray.DataArray(diameter, coords={"id": list(range(nants))}, dims=["id"])
        datavars["mount"] = xarray.DataArray(mount, coords={"id": list(range(nants))}, dims=["id"])
        datavars["vp_type"] = xarray.DataArray(vp_type, coords={"id": list(range(nants))}, dims=["id"])
        datavars["offset"] = xarray.DataArray(offset, coords=coords, dims=["id", "spatial"])
        datavars["stations"] = xarray.DataArray(stations, coords={"id": list(range(nants))}, dims=["id"])

        attrs = dict()

        attrs["name"] = name  # Name of configuration
        attrs["location"] = location  # EarthLocation
        attrs["receptor_frame"] = receptor_frame
        attrs["frame"] = frame
        
        super().__init__(datavars, coords=coords, attrs=attrs)
    
    # def check(self):
    #     """ Check that the internals are ok
    #
    #     :return:
    #     """
    #     assert isinstance(self.data, xarray.Dataset)
    #
    # def __str__(self):
    #     """Default printer for Configuration
    #
    #     """
    #     s = "Configuration:\n"
    #     s += "\nName: %s\n" % self.name
    #     s += "\tNumber of antennas/stations: %s\n" % len(self.names)
    #     s += "\tNames: %s\n" % self.names
    #     s += "\tDiameter: %s\n" % self.diameter
    #     s += "\tMount: %s\n" % self.mount
    #     s += "\tXYZ: %s\n" % self.xyz
    #     s += "\tAxis offset: %s\n" % self.offset
    #     s += "\tStations: %s\n" % self.stations
    #     s += "\tVoltage pattern type: %s\n" % self.vp_type
    #
    #     return s
    
    def size(self):
        """ Return size in GB
        """
        size = self.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    def datasizes(self):
        """
        Return sizes of data variables
        :return:
        """
        s = "Dataset size: {:.3f} GB\n".format(self.nbytes / 1024 / 1024 / 1024)
        for var in self.data.data_vars:
            s += "\t[{}]: \t{:.3f} GB\n".format(var, self[var].nbytes / 1024 / 1024 / 1024)
        return s
    
    @property
    def names(self):
        """ Names of the dishes/stations"""
        return self['names']
    
    @property
    def nants(self):
        """ Names of the dishes/stations"""
        return len(self['names'])
    
    @property
    def vp_type(self):
        """ Names of the voltage pattern type"""
        return self['vp_type']
    
    @property
    def diameter(self):
        """ Diameter of dishes/stations (m)
        """
        return self['diameter']
    
    @property
    def xyz(self):
        """ XYZ locations of dishes/stations [:,3] (m)
        """
        return self['xyz']
    
    @property
    def mount(self):
        """ Mount types of dishes/stations ('azel' | 'equatorial'
        """
        return self['mount']
    
    @property
    def offset(self):
        """ Axis offset [:, 3] (m)
        """
        return self['offset']
    
    @property
    def stations(self):
        """ Station/dish identifier (may be the same as names)"""
        return self['stations']


class GainTable(xarray.Dataset):
    """ Gain table with data_models: time, antenna, gain[:, chan, rec, rec], weight columns

    The weight is usually that output from gain solvers.
    """
    
    def __init__(self, gain: numpy.array = None, time: numpy.array = None, interval=None,
                 weight: numpy.array = None, residual: numpy.array = None, frequency: numpy.array = None,
                 receptor_frame: ReceptorFrame = ReceptorFrame("linear"), phasecentre=None, configuration=None):
        """ Create a gaintable from arrays

        The definition of gain is:

            Vobs = g_i g_j^* Vmodel

        :param gain: Complex gain [nrows, nants, nchan, nrec, nrec]
        :param time: Centroid of solution [nrows]
        :param interval: Interval of validity
        :param weight: Weight of gain [nrows, nchan, nrec, nrec]
        :param residual: Residual of fit [nchan, nrec, nrec]
        :param frequency: Frequency [nchan]
        :param receptor_frame: Receptor frame
        :param phasecentre: Phasecentre (SkyCoord)
        :param configuration: Configuration
        """
        
        super().__init__()
        
        ntimes, nants, nchan, nrec, _ = gain.shape
        antennas = range(nants)
        coords = {
            "time": time,
            "antenna": antennas,
            "frequency": frequency,
            "receptor1": receptor_frame.names,
            "receptor2": receptor_frame.names,
        }
        
        datavars = dict()
        datavars["gain"] = xarray.DataArray(gain, dims=["time", "antenna", "frequency", "receptor1", "receptor2"])
        datavars["weight"] = xarray.DataArray(weight, dims=["time", "antenna", "frequency", "receptor1", "receptor2"])
        datavars["residual"] = xarray.DataArray(residual, dims=["time", "frequency", "receptor1", "receptor2"])
        datavars["interval"] = xarray.DataArray(interval, dims=["time"])
        datavars["datetime"] = xarray.DataArray(Time(time / 86400.0, format='mjd', scale='utc').datetime64,
                                                dims="time")
        attrs = dict()
        
        attrs["receptor_frame"] = receptor_frame
        attrs["phasecentre"] = phasecentre
        attrs["configuration"] = configuration

        super().__init__(datavars, coords=coords, attrs=attrs)
        
    
    def size(self):
        """ Return size in GB
        """
        return self.nbytes / 1024.0 / 1024.0 / 1024.0
    
    def datasizes(self):
        """
        Return sizes of data variables
        :return:
        """
        s = "Dataset size: {:.3f} GB\n".format(self.nbytes / 1024 / 1024 / 1024)
        for var in self.data_vars:
            s += "\t[{}]: \t{:.3f} GB\n".format(var, self[var].nbytes / 1024 / 1024 / 1024)
        return s
    
    @property
    def time(self):
        """ Centroid of solution [ntimes]
        """
        return self['time']
    
    @property
    def interval(self):
        """ Interval of validity [ntimes]
        """
        return self['interval']
    
    @property
    def gain(self):
        """ Complex gain [ntimes, nants, nchan, nrec, nrec]
        """
        return self['gain']
    
    @property
    def frequency(self):
        """ Frequency [nchan]
        """
        return self['frequency']
    
    @property
    def receptor1(self):
        """ Receptor name
        """
        return self['receptor1']
    
    @property
    def receptor2(self):
        """ Receptor name
        """
        return self['receptor2']
    
    @property
    def weight(self):
        """ Weight of gain [ntimes, nants, nchan, nrec, nrec]

        """
        return self['weight']
    
    @property
    def residual(self):
        """ Residual of fit [nchan, nrec, nrec]
        """
        return self['residual']
    
    @property
    def ntimes(self):
        """ Number of times (i.e. rows) in this table
        """
        return self['gain'].shape[0]
    
    @property
    def nants(self):
        """ Number of dishes/stations
        """
        return self['gain'].shape[1]
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self['gain'].shape[2]
    
    @property
    def nrec(self):
        """ Number of receivers

        """
        return len(self['receptor1'])
    
    @property
    def receptors(self):
        """ Receptors

        """
        return self['receptor1']
    
    # def __str__(self):
    #     """Default printer for GainTable
    # 
    #     """
    #     s = "GainTable:\n"
    #     s += "Dataset: {}".format(self.data)
    #     s += "\tTimes: %s\n" % str(self.ntimes)
    #     s += "\tData shape: %s\n" % str(self["pixels"].data.shape)
    #     s += "\tReceptor frames: %s\n" % str(self.receptors)
    #     s += "\tPhasecentre: %s\n" % str(self.phasecentre)
    #     
    #     return s


class PointingTable(xarray.Dataset):
    """ Pointing table with data_models: time, antenna, offset[:, chan, rec, 2], weight columns

    The weight is usually that output from gain solvers.
    """
    
    def __init__(self, data=None, pointing: numpy.array = None, nominal: numpy.array = None,
                 time: numpy.array = None, interval=None,
                 weight: numpy.array = None, residual: numpy.array = None, frequency: numpy.array = None,
                 receptor_frame: ReceptorFrame = ReceptorFrame("linear"), pointing_frame: str = "local",
                 pointingcentre=None, configuration=None):
        """ Create a pointing table from arrays

        :param data: Structured data (used in copying)
        :param pointing: Pointing (rad) [:, nants, nchan, nrec, 2]
        :param nominal: Nominal pointing (rad) [:, nants, nchan, nrec, 2]
        :param time: Centroid of solution [:]
        :param interval: Interval of validity
        :param weight: Weight [: nants, nchan, nrec]
        :param residual: Residual [: nants, nchan, nrec, 2]
        :param frequency: [nchan]
        :param receptor_frame: e.g. Receptor_frame("linear")
        :param pointing_frame: Pointing frame
        :param pointingcentre: SkyCoord
        :param configuration: Configuration
        """
        super().__init__()

        ntimes, nants, nchan, nrec, _ = pointing.shape
        antennas = range(nants)
        coords = {
            "time": time,
            "antenna": antennas,
            "frequency": frequency,
            "receptor": receptor_frame.names,
            "angle": ["az", "el"]
        }
        
        datavars = dict()
        datavars["pointing"] = xarray.DataArray(pointing, dims=["time", "antenna", "frequency", "receptor", "angle"])
        datavars["nominal"] = xarray.DataArray(nominal, dims=["time", "antenna", "frequency", "receptor", "angle"])
        datavars["weight"] = xarray.DataArray(weight, dims=["time", "antenna", "frequency", "receptor", "angle"])
        datavars["residual"] = xarray.DataArray(residual, dims=["time", "frequency", "receptor", "angle"])
        datavars["interval"] = xarray.DataArray(interval, dims=["time"])
        datavars["datetime"] = xarray.DataArray(Time(time / 86400.0, format='mjd', scale='utc').datetime64, dims="time")

        attrs = dict()
        attrs["frequency"] = frequency
        attrs["receptor_frame"] = receptor_frame
        attrs["pointing_frame"] = pointing_frame
        attrs["pointingcentre"] = pointingcentre
        attrs["configuration"] = configuration

        super().__init__(datavars, coords=coords, attrs=attrs)
        
    
    def size(self):
        """ Return size in GB
        """
        return self.data.nbytes / 1024.0 / 1024.0 / 1024.0
    
    def datasizes(self):
        """
        Return sizes of data variables
        :return:
        """
        s = "Dataset size: {:.3f} GB\n".format(self.data.nbytes / 1024 / 1024 / 1024)
        for var in self.data.data_vars:
            s += "\t[{}]: \t{:.3f} GB\n".format(var, self[var].nbytes / 1024 / 1024 / 1024)
        return s
    
    @property
    def time(self):
        """ Time (s UTC) [:]
        """
        return self['time']
    
    @property
    def interval(self):
        """ Interval of validity (s) [:]
        """
        return self['interval']
    
    @property
    def nominal(self):
        """ Nominal pointing (rad) [:, nants, nchan, nrec, 2]
        """
        return self['nominal']
    
    @property
    def pointing(self):
        """ Pointing (rad) [:, nants, nchan, nrec, 2]
        """
        return self['pointing']
    
    @property
    def weight(self):
        """ Weight [: nants, nchan, nrec]
        """
        return self['weight']
    
    @property
    def residual(self):
        """ Residual [: nants, nchan, nrec, 2]
        """
        return self['residual']
    
    @property
    def ntimes(self):
        """ Number of time (i.e. rows in table)"""
        return self['pointing'].shape[0]
    
    @property
    def nants(self):
        """ Number of dishes/stations
        """
        return self['pointing'].shape[1]
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self['pointing'].shape[2]
    
    @property
    def nrec(self):
        """ Number of receptors
        """
        return self.receptor_frame.nrec
    
    # def __str__(self):
    #     """Default printer for PointingTable
    #
    #     """
    #     s = "PointingTable:\n"
    #     s += "Dataset: {}".format(self.data)
    #     s += "\tTimes: %s\n" % str(self.ntimes)
    #     s += "\tData shape: %s\n" % str(self["pixels"].data.shape)
    #     s += "\tReceptor frame: %s\n" % str(self.receptor_frame.type)
    #     s += "\tPointing centre: %s\n" % str(self.pointingcentre)
    #     s += "\tConfiguration: %s\n" % str(self.configuration)
    #     s += "Data: {}".format(self.data)
    #     return s
    #

class Image(xarray.Dataset):
    """Image class with Image data (as an xarray.DataArray) and the AstroPy `implementation of
    a World Coodinate System <http://docs.astropy.org/en/stable/wcs>`_

    Many operations can be done conveniently using numpy processing_components on Image.data.

    Most of the imaging processing_components require an image in canonical format:
    - 4 axes: RA, DEC, POL, FREQ

    The conventions for indexing in WCS and numpy are opposite.
    - In astropy.wcs, the order is (longitude, latitude, polarisation, frequency)
    - in numpy, the order is (frequency, polarisation, latitude, longitude)

    .. warning::
        The polarisation_frame is kept in two places, the WCS and the polarisation_frame
        variable. The latter should be considered definitive.

    """

    __slots__ = ['__dict__']

    def __init__(self, data, phasecentre, frequency, polarisation_frame=None, wcs=None):
        """ Create an XImage

        :param frequency:
        :param phasecentre:
        :param polarisation_frame:
        :return: Image
        """
        super().__init__()

        nx, ny = data.shape[-2], data.shape[-1]
        cellsize = numpy.abs(wcs.wcs.cdelt[1])
        cx = phasecentre.ra.to("deg").value
        cy = phasecentre.dec.to("deg").value
        
        assert cellsize > 0.0, "Cellsize must be positive"
        
        dims = ["frequency", "polarisation", "m", "l"]
        coords = {
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "m": numpy.linspace(cy - cellsize * ny / 2, cy + cellsize * ny / 2, ny),
            "l": numpy.linspace(cx - cellsize * nx / 2, cx + cellsize * nx / 2, nx)
        }
        
        anchan = len(frequency)
        anpol = polarisation_frame.npol
        
        if len(data.shape) == 2:
            data = data.reshape([anchan, anpol, ny, nx])
        
        assert data.shape[0] == anchan, \
            "Number of frequency channels {} and data shape {} are incompatible" \
                .format(len(frequency), data.shape)
        assert data.shape[1] == anpol, \
            "Polarisation frame {} and data shape {} are incompatible".format(polarisation_frame.type, data.shape)
        assert coords["l"][0] != coords["l"][-1]
        assert coords["m"][0] != coords["m"][-1]
        
        assert len(coords["m"]) == ny
        assert len(coords["l"]) == nx

        data_vars = dict()
        data_vars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)
        attrs = {"phasecentre": phasecentre, "wcs":wcs, "polarisation_frame":polarisation_frame}
        
        super().__init__(data_vars, coords=coords, attrs=attrs)
    
    @property
    def shape(self):
        """ Shape of array
        
        :return:
        """
        return self["pixels"].data.shape
        
    def check(self):
        """ Check that the internals are ok

        :return:
        """
        assert isinstance(self, xarray.DataArray)
    
    def size(self):
        """ Return size in GB
        """
        size = self.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.shape[0]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.shape[1]
    
    @property
    def frequency(self):
        """ Frequency values
        """
        w = self.attrs["wcs"].sub(['spectral'])
        return w.wcs_pix2world(range(self.nchan), 0)[0]
    
    @property
    def phasecentre(self):
        """ Phasecentre (from WCS)
        """
        return SkyCoord(self.attrs["wcs"].wcs.crval[0] * u.deg, self.attrs["wcs"].wcs.crval[1] * u.deg)
    
    @property
    def ra_dec_mesh(self):
        """ RA, Dec mesh

        :return:
        """
        ny = self.shape[-2]
        nx = self.shape[-1]
        ramesh, decmesh = numpy.meshgrid(numpy.arange(ny), numpy.arange(nx))
        return self.attrs["wcs"].sub([1, 2]).wcs_pix2world(ramesh, decmesh, 0)

    @property
    def wcs(self):
        """

        :return:
        """
        return self.attrs["wcs"]

    @property
    def polarisation_frame(self):
        """

        :return:
        """
        return self.attrs["polarisation_frame"]


class GridData(xarray.Dataset):
    """Class to hold Gridded data for Fourier processing
    - Has four or more coordinates: [chan, pol, z, y, x] where x can be u, l; y can be v, m; z can be w, n

    The conventions for indexing in WCS and numpy are opposite.
    - In astropy.wcs, the order is (longitude, latitude, polarisation, frequency)
    - in numpy, the order is (frequency, polarisation, depth, latitude, longitude)

    .. warning::
        The polarisation_frame is kept in two places, the WCS and the polarisation_frame
        variable. The latter should be considered definitive.

    """
    
    def __init__(self, polarisation_frame=None,
                 dtype=None, data=None, grid_wcs=None, projection_wcs=None):
        """ Create a GridData

        :param polarisation_frame:
        :return: GridData
        """
        
        super().__init__()

        
        nchan, npol, nw, ny, nx = data.shape
        frequency = grid_wcs.sub(['spectral']).wcs_pix2world(range(nchan), 0)[0]
        wrange = grid_wcs.sub([3]).wcs_pix2world(range(nw), 0)[0]
        
        assert npol == polarisation_frame.npol
        cellsize = numpy.abs(projection_wcs.wcs.cdelt[1])
        cellsize_rad = numpy.deg2rad(cellsize)
        du = 1.0 / cellsize_rad
        dv = 1.0 / cellsize_rad
        cu = grid_wcs.wcs.crval[0]
        cv = grid_wcs.wcs.crval[1]
        
        assert cellsize > 0.0, "Cellsize must be positive"
        
        dims = ["frequency", "polarisation", "w", "v", "u"]
        coords = {
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "w": wrange,
            "v": numpy.linspace(cv - dv * ny / 2, cv + dv * ny / 2, ny),
            "u": numpy.linspace(cu - du * nx / 2, cu + du * nx / 2, nx)
        }
        
        assert coords["u"][0] != coords["u"][-1]
        assert coords["v"][0] != coords["v"][-1]
        
        attrs = dict()
        attrs["grid_wcs"] = grid_wcs
        attrs["projection_wcs"] = projection_wcs
        attrs["polarisation_frame"] = polarisation_frame
        
        nchan = len(frequency)
        npol = polarisation_frame.npol
        if dtype is None:
            dtype = "float"
        
        if data is None:
            data = numpy.zeros([nchan, npol, nw, ny, nx], dtype=dtype)
        else:
            if data.shape == (ny, nx):
                data = data.reshape([1, 1, nw, ny, nx])
            assert data.shape == (nchan, npol, nw, ny, nx), \
                "Polarisation frame {} and data shape {} are incompatible".format(polarisation_frame.type,
                                                                                  data.shape)
        
        data_vars = dict()
        data_vars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)
        super().__init__(data_vars, coords=coords, attrs=attrs)

    
    def check(self):
        """ Check that the internals are ok
        
        :return:
        """
        assert isinstance(self.data, xarray.DataArray)
    
    def size(self):
        """ Return size in GB
        """
        size = self.data.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self["pixels"].data.shape[0]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self["pixels"].data.shape[1]
    
    @property
    def frequency(self):
        """ Frequency values
        """
        w = self.grid_wcs.sub(['spectral'])
        return w.wcs_pix2world(range(self.nchan), 0)[0]
    
    @property
    def shape(self):
        """ Shape of data array
        """
        return self["pixels"].data.shape
    
    @property
    def phasecentre(self):
        """ Phasecentre (from WCS)
        """
        return SkyCoord(self.grid_wcs.wcs.crval[0] * u.deg, self.grid_wcs.wcs.crval[1] * u.deg)
    
    @property
    def ra_dec_mesh(self):
        """ RA, Dec mesh

        :return:
        """
        ny = self["pixels"].data.shape[-2]
        nx = self["pixels"].data.shape[-1]
        ramesh, decmesh = numpy.meshgrid(numpy.arange(ny), numpy.arange(nx))
        return self.projection_wcs.sub([1, 2]).wcs_pix2world(ramesh, decmesh, 0)
    
    # def __str__(self):
    #     """Default printer for GridData
    # 
    #     """
    #     s = "GridData:\n"
    #     s += "{}\n".format(str(self.data))
    #     s += "\tShape: %s\n" % str(self["pixels"].data.shape)
    #     s += "\tData type: %s\n" % str(self.data.dtype)
    #     s += "\tGrid WCS: %s\n" % self.grid_wcs.__repr__()
    #     s += "\tProjection WCS: %s\n" % self.projection_wcs.__repr__()
    #     s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
    #     return s
    # 

class ConvolutionFunction(xarray.Dataset):
    """Class to hold Convolution function for Fourier processing
    - Has four or more coordinates: [chan, pol, z, y, x] where x can be u, l; y can be v, m; z can be w, n

    The cf has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes

    The axes UU,VV have the same physical stride as the image, The axes DUU, DVV are subsampled.

    Convolution function holds the original sky plane projection in the projection_wcs.

    """
    
    def __init__(self, data, grid_wcs=None, projection_wcs=None, polarisation_frame=None):
        """Create ConvolutionFunction

        :param data: Data for cf
        :param grid_wcs: Astropy WCS object for the grid
        :param projection_wcs: Astropy WCS object for the projection
        :param polarisation_frame: Polarisation_frame e.g. PolarisationFrame('linear')
        """
        
        super().__init__()

        nchan, npol, nw, oversampling, _, support, _ = data.shape
        frequency = grid_wcs.sub(['spectral']).wcs_pix2world(range(nchan), 0)[0]
        
        assert npol == polarisation_frame.npol
        cellsize = numpy.abs(projection_wcs.wcs.cdelt[1])
        cellsize_rad = numpy.deg2rad(cellsize)
        du = 1.0 / cellsize_rad
        dv = 1.0 / cellsize_rad
        ddu = 1.0 / cellsize_rad / oversampling
        ddv = 1.0 / cellsize_rad / oversampling
        cu = grid_wcs.wcs.crval[0]
        cv = grid_wcs.wcs.crval[1]
        cdu = oversampling // 2
        cdv = oversampling // 2
        
        assert cellsize > 0.0, "Cellsize must be positive"
        
        dims = ["frequency", "polarisation", "w", "dv", "du", "v", "u"]
        coords = {
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "w": numpy.zeros([nw]),
            "dv": numpy.linspace(cdv - ddv * oversampling / 2, cdv + ddv * oversampling / 2, oversampling,
                                 endpoint=False),
            "du": numpy.linspace(cdu - ddu * oversampling / 2, cdu + ddu * oversampling / 2, oversampling,
                                 endpoint=False),
            "v": numpy.linspace(cv - dv * support / 2, cv + dv * support / 2, support,
                                endpoint=False),
            "u": numpy.linspace(cu - du * support / 2, cu + du * support / 2, support,
                                endpoint=False)
        }
        
        assert coords["u"][0] != coords["u"][-1]
        assert coords["v"][0] != coords["v"][-1]
        
        attrs = dict()
        attrs["grid_wcs"] = grid_wcs
        attrs["projection_wcs"] = projection_wcs
        attrs["polarisation_frame"] = polarisation_frame
        
        nchan = len(frequency)
        npol = polarisation_frame.npol
        if data is None:
            data = numpy.zeros([nchan, npol, nw, oversampling, oversampling, support, support], dtype='complex')
        else:
            assert data.shape == (nchan, npol, nw, oversampling, oversampling, support, support), \
                "Polarisation frame {} and data shape {} are incompatible".format(polarisation_frame.type,
                                                                                  data.shape)
        data_vars = dict()
        data_vars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)
        super().__init__(data_vars, coords=coords, attrs=attrs)

    
    def check(self):
        """ Check that the internals are ok

        :return:
        """
        assert isinstance(self.data, xarray.DataArray)
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    def datasizes(self):
        """
        Return sizes of data variables
        :return:
        """
        s = "Dataset size: {:.3f} GB\n".format(self.data.nbytes / 1024 / 1024 / 1024)
        for var in self.data.data_vars:
            s += "\t[{}]: \t{:.3f} GB\n".format(var, self[var].nbytes / 1024 / 1024 / 1024)
        return s
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self["pixels"].data.shape[0]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self["pixels"].data.shape[1]
    
    @property
    def ndepth(self):
        """ Number of pixels deep i.e. z
        """
        return self.data.shape[4]
    
    @property
    def frequency(self):
        """ Frequency values
        """
        w = self.grid_wcs.sub(['spectral'])
        return w.wcs_pix2world(range(self.nchan), 0)[0]
    
    @property
    def shape(self):
        """ Shape of data array
        """
        assert len(self.data.shape) == 7
        return self.data.shape
    
    @property
    def phasecentre(self):
        """ Phasecentre (from projection WCS)
        """
        return SkyCoord(self.projection_wcs.wcs.crval[0] * u.deg, self.projection_wcs.wcs.crval[1] * u.deg)
    
    # def __str__(self):
    #     """Default printer for ConvolutionFunction
    #
    #     """
    #     s = "Convolution function:\n"
    #     s += "{}\n".format(str(self.data))
    #     s += "\tShape: %s\n" % str(self.data.shape)
    #     s += "\tGrid WCS: %s\n" % self.grid_wcs
    #     s += "\tProjection WCS: %s\n" % self.projection_wcs
    #     s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
    #     return s


class Skycomponent:
    """Skycomponents are used to represent compact sources on the sky. They possess direction,
    flux as a function of frequency and polarisation, shape (with params), and polarisation frame.

    For example, the following creates and predicts the visibility from a collection of point sources
    drawn from the GLEAM catalog::

        sc = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                    polarisation_frame=PolarisationFrame("stokesIQUV"),
                                                    frequency=frequency, kind='cubic',
                                                    phasecentre=phasecentre,
                                                    radius=0.1)
        model = create_image_from_visibility(vis, cellsize=0.001, npixel=512, frequency=frequency,
                                            polarisation_frame=PolarisationFrame('stokesIQUV'))

        bm = create_low_test_beam(model=model)
        sc = apply_beam_to_skycomponent(sc, bm)
        vis = dft_skycomponent_visibility(vis, sc)
    """
    
    def __init__(self,
                 direction=None, frequency=None, name=None, flux=None, shape='Point',
                 polarisation_frame=PolarisationFrame('stokesIQUV'), params=None):
        """ Define the required structure

        :param direction: SkyCoord
        :param frequency: numpy.array [nchan]
        :param name: user friendly name
        :param flux: numpy.array [nchan, npol]
        :param shape: str e.g. 'Point' 'Gaussian'
        :param polarisation_frame: Polarisation_frame e.g. PolarisationFrame('stokesIQUV')
        :param params: numpy.array shape dependent parameters
        """
        
        self.direction = direction
        self.frequency = numpy.array(frequency)
        self.name = name
        self.flux = numpy.array(flux)
        self.shape = shape
        if params is None:
            params = {}
        self.params = params
        self.polarisation_frame = polarisation_frame
        
        assert len(self.frequency.shape) == 1, frequency
        assert len(self.flux.shape) == 2, flux
        assert self.frequency.shape[0] == self.flux.shape[0], \
            "Frequency shape %s, flux shape %s" % (self.frequency.shape, self.flux.shape)
        assert polarisation_frame.npol == self.flux.shape[1], \
            "Polarisation is %s, flux shape %s" % (polarisation_frame.type, self.flux.shape)
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.flux.shape[0]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.flux.shape[1]
    
    def __str__(self):
        """Default printer for Skycomponent

        """
        s = "Skycomponent:\n"
        s += "\tName: %s\n" % self.name
        s += "\tFlux: %s\n" % self.flux
        s += "\tFrequency: %s\n" % self.frequency
        s += "\tDirection: %s\n" % self.direction
        s += "\tShape: %s\n" % self.shape
        
        s += "\tParams: %s\n" % self.params
        s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
        return s


class SkyModel:
    """ A model for the sky, including an image, components, gaintable and a mask
    """
    
    def __init__(self, image=None, components=None, gaintable=None, mask=None, fixed=False):
        """ A model of the sky as an image, components, gaintable and a mask

        Use copy_skymodel to make a proper copy of skymodel
        :param image: Image
        :param components: List of components
        :param gaintable: Gaintable for this skymodel
        :param mask: Mask for the image
        :param fixed: Is this model fixed?
        """
        if components is None:
            components = []
        
        #if image is not None:
            #assert isinstance(image, Image), image
        self.image = image
        
        if components is not None:
            assert isinstance(components, list)
            for comp in components:
                assert isinstance(comp, Skycomponent), comp
        self.components = [sc for sc in components]
        
        if gaintable is not None:
            assert isinstance(gaintable, GainTable), gaintable
        self.gaintable = gaintable
        
        #if mask is not None:
        #    assert isinstance(mask, Image), mask
        self.mask = mask
        
        self.fixed = fixed
    
    def __str__(self):
        """Default printer for SkyModel

        """
        s = "SkyModel: fixed: %s\n" % self.fixed
        for i, sc in enumerate(self.components):
            s += str(sc)
        s += "\n"
        
        s += str(self.image)
        s += "\n"
        
        s += str(self.mask)
        s += "\n"
        
        s += str(self.gaintable)
        
        return s


class BlockVisibility(xarray.Dataset):
    """ BlockVisibility table class

    BlockVisibility with uvw, time, integration_time, frequency, channel_bandwidth, pol,
    a1, a2, vis, weight Columns in an xarray DataSet which is available as .data

    BlockVisibility is defined to hold an observation with one direction.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.

    Polarisation frame is the same for the entire data set and can be stokesI, circular, circularpnp, linear, linearnp

    The configuration is stored as an attribute..
    """
    
    def __init__(self, frequency=None, channel_bandwidth=None, phasecentre=None, configuration=None, uvw=None,
                 time=None, vis=None, weight=None, integration_time=None, flags=None, baselines=None,
                 polarisation_frame=PolarisationFrame('stokesI'), imaging_weight=None, source='anonymous', meta=None,
                 low_precision="float32"):
        """BlockVisibility

        :param frequency: Frequency [nchan]
        :param channel_bandwidth: Channel bandwidth [nchan]
        :param phasecentre: Phasecentre (SkyCoord)
        :param configuration: Configuration
        :param uvw: UVW coordinates (m) [:, nant, nant, 3]
        :param time: Time (UTC) [:]
        :param baselines: List of baselines
        :param flags: Flags [:, nant, nant, nchan]
        :param weight: [:, nant, nant, nchan, npol]
        :param imaging_weight: [:, nant, nant, nchan, npol]
        :param integration_time: Integration time [:]
        :param polarisation_frame: Polarisation_Frame e.g. Polarisation_Frame("linear")
        :param source: Source name
        :param meta: Meta info
        """
        
        super().__init__()
        
        if imaging_weight is None:
            imaging_weight = weight
        if integration_time is None:
            integration_time = numpy.ones_like(time)
        
        k = (frequency / const.c).value
        if len(frequency) == 1:
            uvw_lambda = (uvw * k)[..., numpy.newaxis, :]
        else:
            uvw_lambda = numpy.einsum("tbs,k->tbks", uvw, k)
        
        coords = {
            "time": time,
            "baseline": baselines,
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "uvw_index": ["u", "v", "w"]
        }
        
        datavars = dict()
        datavars["integration_time"] = xarray.DataArray(integration_time.astype(low_precision),
                                                        dims=["time"], attrs={"units": "s"})
        datavars["datetime"] = xarray.DataArray(Time(time / 86400.0, format='mjd', scale='utc').datetime64,
                                                dims=["time"], attrs={"units": "s"})
        datavars["vis"] = xarray.DataArray(vis, dims=["time", "baseline", "frequency", "polarisation"],
                                           attrs={"units": "Jy"})
        datavars["weight"] = xarray.DataArray(weight.astype(low_precision),
                                              dims=["time", "baseline", "frequency", "polarisation"])
        datavars["imaging_weight"] = xarray.DataArray(imaging_weight.astype(low_precision),
                                                      dims=["time", "baseline", "frequency", "polarisation"])
        datavars["flags"] = xarray.DataArray(flags.astype(low_precision),
                                             dims=["time", "baseline", "frequency", "polarisation"])
        
        datavars["uvw"] = xarray.DataArray(uvw, dims=["time", "baseline", "uvw_index"], attrs={"units": "m"})
        
        datavars["uvw_lambda"] = xarray.DataArray(uvw_lambda, dims=["time", "baseline", "frequency", "uvw_index"],
                                                  attrs={"units": "lambda"})
        datavars["uvdist_lambda"] = xarray.DataArray(numpy.hypot(uvw_lambda[..., 0], uvw_lambda[..., 1]),
                                                     dims=["time", "baseline", "frequency"], attrs={"units": "lambda"})
        
        datavars["channel_bandwidth"] = xarray.DataArray(channel_bandwidth, dims=["frequency"], attrs={"units": "Hz"})
        
        attrs = dict()
        attrs["phasecentre"] = phasecentre  # Phase centre of observation
        attrs["configuration"] = configuration  # Antenna/station configuration
        attrs["polarisation_frame"] = polarisation_frame
        attrs["source"] = source
        attrs["meta"] = meta
        
        super().__init__(datavars, coords=coords, attrs=attrs)
    
    def check(self):
        """ Check that the internals are ok

        :return:
        """
        assert isinstance(self, xarray.Dataset)
    
    def size(self):
        """ Return size in GB
        """
        return self.nbytes / 1024.0 / 1024.0 / 1024.0
    
    def datasizes(self):
        """
        Return sizes of data variables
        :return:
        """
        s = "Dataset size: {:.3f} GB\n".format(self.nbytes / 1024 / 1024 / 1024)
        for var in self.data_vars:
            s += "\t[{}]: \t{:.3f} GB\n".format(var, self[var].nbytes / 1024 / 1024 / 1024)
        return s
    
    @property
    def rows(self):
        """ Rows
        """
        return range(len(self.time))
    
    @property
    def ntimes(self):
        """ Number of times (i.e. rows) in this table
        """
        return len(self['time'])
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return len(self['frequency'])
    
    @property
    def frequency(self):
        """ Number of channels
        """
        return self['frequency']
    
    @property
    def channel_bandwidth(self):
        """ Number of channels
        """
        return self['channel_bandwidth']
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.polarisation_frame.npol
    
    @property
    def nants(self):
        """ Number of antennas
        """
        return self.configuration.nants
    
    @property
    def baselines(self):
        """ Baselines
        """
        return self["baseline"]
    
    @property
    def nbaselines(self):
        """ Number of Baselines
        """
        return len(self["baseline"])
    
    @property
    def uvw(self):
        """ UVW coordinates (metres) [nrows, nbaseline, 3]
        """
        return self['uvw']
    
    @property
    def uvw_lambda(self):
        """ UVW coordinates (wavelengths) [nrows, nbaseline, nchan, 3]
        """
        return self['uvw_lambda']
    
    @property
    def u(self):
        """ u coordinate (metres) [nrows, nbaseline]
        """
        return self['uvw'][..., 0]
    
    @property
    def v(self):
        """ v coordinate (metres) [nrows, nbaseline]
        """
        return self['uvw'][..., 1]
    
    @property
    def w(self):
        """ w coordinate (metres) [nrows, nbaseline]
        """
        return self['uvw'][..., 2]
    
    @property
    def u_lambda(self):
        """ u coordinate (wavelengths) [nrows, nbaseline]
        """
        return self['uvw_lambda'][..., 0]
    
    @property
    def v_lambda(self):
        """ v coordinate (wavelengths) [nrows, nbaseline]
        """
        return self['uvw_lambda'][..., 1]
    
    @property
    def w_lambda(self):
        """ w coordinate (wavelengths) [nrows, nbaseline]
        """
        return self['uvw_lambda'][..., 2]
    
    @property
    def uvdist(self):
        """ uv distance (metres) [nrows, nbaseline]
        """
        return numpy.hypot(self.u, self.v)
    
    @property
    def uvdist_lambda(self):
        """ uv distance (metres) [nrows, nbaseline]
        """
        return numpy.hypot(self.u_lambda, self.v_lambda)
    
    @property
    def uvwdist(self):
        """ uv distance (metres) [nrows, nbaseline]
        """
        return numpy.hypot(self.u, self.v, self.w)
    
    @property
    def vis(self):
        """ Complex visibility [nrows, nbaseline, ncha, npol]
        """
        return self['vis']
    
    @property
    def flagged_vis(self):
        """Flagged complex visibility [nrows, nbaseline, ncha, npol]
        """
        return self['vis'] * (1 - self['flags'])
    
    @property
    def flags(self):
        """ Flags [nrows, nbaseline, nchan]
        """
        return self['flags']
    
    @property
    def weight(self):
        """ Weight[nrows, nbaseline, nchan, npol]
        """
        return self['weight']
    
    @property
    def flagged_weight(self):
        """Weight [: npol]
        """
        return self['weight'] * (1 - self['flags'])
    
    @property
    def imaging_weight(self):
        """ Imaging_weight[nrows, nbaseline, nchan, npol]
        """
        return self['imaging_weight']
    
    @property
    def flagged_imaging_weight(self):
        """ Flagged Imaging_weight[nrows, nbaseline, nchan, npol]
        """
        return self['imaging_weight'] * (1 - self['flags'])
    
    @property
    def time(self):
        """ Time (UTC) [nrows]
        """
        return self['time']
    
    @property
    def datetime(self):
        """ Time (UTC) [nrows]
        """
        return self['datetime']
    
    @property
    def integration_time(self):
        """ Integration time [nrows]
        """
        return self['integration_time']
    
    @property
    def nvis(self):
        """ Number of visibilities (in total)
        """
        return numpy.product(self.vis.shape)


class FlagTable(xarray.Dataset):
    """ Flag table class

    Flags, time, integration_time, frequency, channel_bandwidth, pol,
    in an xarray.

    The configuration is also an attribute
    """
    
    def __init__(self, baselines=None, flags=None,
                 frequency=None, channel_bandwidth=None,
                 configuration=None, time=None, integration_time=None,
                 polarisation_frame=None):
        """FlagTable

        :param frequency: Frequency [nchan]
        :param channel_bandwidth: Channel bandwidth [nchan]
        :param configuration: Configuration
        :param time: Time (UTC) [ntimes]
        :param flags: Flags [ntimes, nbaseline, nchan]
        :param integration_time: Integration time [ntimes]
        """
        super().__init__()
        
        coords = {
            "time": time,
            "baseline": baselines,
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
        }
        
        datavars = dict()
        datavars["flags"] = xarray.DataArray(flags, dims=["time", "baseline", "frequency", "polarisation"])
        datavars["integration_time"] = xarray.DataArray(integration_time, dims=["time"])
        datavars["channel_bandwidth"] = xarray.DataArray(channel_bandwidth, dims=["frequency"])
        datavars["datetime"] = xarray.DataArray(Time(time / 86400.0, format='mjd', scale='utc').datetime64,
                                                dims="time")
        
        attrs = dict()
        attrs["polarisation_frame"] = polarisation_frame
        attrs["configuration"] = configuration  # Antenna/station configuration
        
        super().__init__(datavars, coords=coords, attrs=attrs)
    
    # def __str__(self):
    #     """Default printer for FlagTable
    #
    #     """
    #     s = "FlagTable:\n"
    #     s += "{}\n".format(str(self.data))
    #     s += "\tNumber of integrations: %s\n" % len(self.time)
    #     s += "\tFlags shape: %s\n" % str(self.flags.shape)
    #     s += "\tNumber of channels: %d\n" % len(self.frequency)
    #     s += "\tFrequency: %s\n" % self.frequency
    #     s += "\tChannel bandwidth: %s\n" % self.channel_bandwidth
    #     s += "\tNumber of polarisations: %s\n" % self.npol
    #     s += "\tPolarisation Frame: %s\n" % self.polarisation_frame.type
    #     s += "\tConfiguration: %s\n" % self.configuration.name
    #
    #     return s
    
    def size(self):
        """ Return size in GB
        """
        return self.nbytes / 1024.0 / 1024.0 / 1024.0
    
    def datasizes(self):
        """
        Return sizes of data variables
        :return:
        """
        s = "Dataset size: {:.3f} GB\n".format(self.nbytes / 1024 / 1024 / 1024)
        for var in self.data_vars():
            s += "\t[{}]: \t{:.3f} GB\n".format(var, self[var].nbytes / 1024 / 1024 / 1024)
        return s
    
    @property
    def time(self):
        """ Time
        """
        return self['time']
    
    @property
    def baseline(self):
        """ Baselines
        """
        return self['baseline']
    
    @property
    def datetime(self):
        """ DateTime
        """
        return self['datetime']
    
    @property
    def flags(self):
        """ Flags [nrows, nbaseline, ncha, npol]
        """
        return self['flags']
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return len(self['frequency'])
    
    @property
    def frequency(self):
        """ Number of channels
        """
        return self['frequency']
    
    @property
    def channel_bandwidth(self):
        """ Number of channels
        """
        return self['channel_bandwidth']
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.polarisation_frame.npol
    
    @property
    def nants(self):
        """ Number of antennas
        """
        return self.configuration.nants
    
    @property
    def baselines(self):
        """ Baselines
        """
        return self["baseline"]
    
    @property
    def nbaselines(self):
        """ Number of Baselines
        """
        return len(self["baseline"])


class QA:
    """ Quality assessment

    :param origin: str, origin e.g. "continuum_imaging_pipeline"
    :param data: Data containing standard fields
    :param context: Context of QA e.g. "Cycle 5"

    """
    
    def __init__(self, origin=None, data=None, context=None):
        """QA

        :param origin:
        :param data:
        :param context:
        """
        self.origin = origin  # Name of function originating QA assessment
        self.data = data  # Dictionary containing standard fields
        self.context = context  # Context string
    
    def __str__(self):
        """Default printer for QA

        """
        s = "Quality assessment:\n"
        s += "\tOrigin: %s\n" % self.origin
        s += "\tContext: %s\n" % self.context
        s += "\tData:\n"
        for dataname in self.data.keys():
            s += "\t\t%s: %r\n" % (dataname, str(self[dataname]))
        return s


class ScienceDataModel:
    """ Science Data Model: not defined yet"""
    
    def __init__(self):
        pass
    
    def __str__(self):
        """ Default printer for Science Data Model
        """
        return ""


def assert_same_chan_pol(o1, o2):
    """
    Assert that two entities indexed over channels and polarisations
    have the same number of them.

    :param o1: Object 1 e.g. BlockVisibility
    :param o2: Object 1 e.g. BlockVisibility
    :return: Bool
    """
    assert o1.npol == o2.npol, \
        "%s and %s have different number of polarisations: %d != %d" % \
        (type(o1).__name__, type(o2).__name__, o1.npol, o2.npol)
    if isinstance(o1, BlockVisibility) and isinstance(o2, BlockVisibility):
        assert o1.nchan == o2.nchan, \
            "%s and %s have different number of channels: %d != %d" % \
            (type(o1).__name__, type(o2).__name__, o1.nchan, o2.nchan)


def assert_vis_gt_compatible(vis: BlockVisibility, gt: GainTable):
    """ Check if visibility and gaintable are compatible

    :param vis:
    :param gt:
    :return: Bool
    """
    pass
