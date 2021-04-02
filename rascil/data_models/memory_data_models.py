"""The data models used in RASCIL:

"""

__all__ = [
    "Configuration",
    "GainTable",
    "PointingTable",
    "Image",
    "GridData",
    "ConvolutionFunction",
    "Skycomponent",
    "SkyModel",
    "BlockVisibility",
    "FlagTable",
    "QA",
    "ScienceDataModel",
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

from rascil.data_models.polarisation import (
    PolarisationFrame,
    ReceptorFrame,
    polarisation_frame_from_names,
)
from rascil.data_models.xarray_coordinate_support import image_wcs, griddata_wcs, cf_wcs

warnings.simplefilter("ignore", FITSFixedWarning)
warnings.simplefilter("ignore", AstropyDeprecationWarning)

log = logging.getLogger("rascil-logger")


class XarrayAccessorMixin:
    """Convenience methods to access the fields of the xarray"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def size(self):
        """Return size in GB"""
        size = self._obj.nbytes
        return size / 1024.0 / 1024.0 / 1024.0

    def datasizes(self):
        """Return string describing sizes of data variables
        :return: string
        """
        s = "Dataset size: {:.3f} GB\n".format(self._obj.nbytes / 1024 / 1024 / 1024)
        for var in self._obj.data_vars:
            s += "\t[{}]: \t{:.3f} GB\n".format(
                var, self._obj[var].nbytes / 1024 / 1024 / 1024
            )
        return s


class Configuration(xarray.Dataset):
    """A Configuration describes an array configuration.

    Here is an example::

        <xarray.Configuration>
        Dimensions:   (id: 115, spatial: 3)
        Coordinates:
          * id        (id) int64 0 1 2 3 4 5 6 7 8 ... 107 108 109 110 111 112 113 114
          * spatial   (spatial) <U1 'X' 'Y' 'Z'
        Data variables:
            names     (id) <U6 'M000' 'M001' 'M002' ... 'SKA102' 'SKA103' 'SKA104'
            xyz       (id, spatial) float64 -0.0 9e-05 1.053e+03 ... -810.3 1.053e+03
            diameter  (id) float64 13.5 13.5 13.5 13.5 13.5 ... 15.0 15.0 15.0 15.0 15.0
            mount     (id) <U4 'azel' 'azel' 'azel' 'azel' ... 'azel' 'azel' 'azel'
            vp_type   (id) <U7 'MEERKAT' 'MEERKAT' 'MEERKAT' ... 'MID' 'MID' 'MID'
            offset    (id, spatial) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
            stations  (id) <U3 '0' '1' '2' '3' '4' '5' ... '110' '111' '112' '113' '114'
        Attributes:
            rascil_data_model:  Configuration
            name:               MID
            location:           (5109237.71471275, 2006795.66194638, -3239109.1838011...
            receptor_frame:     <rascil.data_models.polarisation.ReceptorFrame object...
            frame:

    """

    __slots__ = ()

    def __init__(
        self,
        name="",
        location=None,
        names=None,
        xyz=None,
        mount="alt-az",
        frame="",
        receptor_frame=ReceptorFrame("linear"),
        diameter=None,
        offset=None,
        stations="%s",
        vp_type=None,
    ):

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

        coords = {"id": list(range(nants)), "spatial": ["X", "Y", "Z"]}

        datavars = dict()
        datavars["names"] = xarray.DataArray(
            names, coords={"id": list(range(nants))}, dims=["id"]
        )
        datavars["xyz"] = xarray.DataArray(xyz, coords=coords, dims=["id", "spatial"])
        datavars["diameter"] = xarray.DataArray(
            diameter, coords={"id": list(range(nants))}, dims=["id"]
        )
        datavars["mount"] = xarray.DataArray(
            mount, coords={"id": list(range(nants))}, dims=["id"]
        )
        datavars["vp_type"] = xarray.DataArray(
            vp_type, coords={"id": list(range(nants))}, dims=["id"]
        )
        datavars["offset"] = xarray.DataArray(
            offset, coords=coords, dims=["id", "spatial"]
        )
        datavars["stations"] = xarray.DataArray(
            stations, coords={"id": list(range(nants))}, dims=["id"]
        )

        attrs = dict()
        attrs["rascil_data_model"] = "Configuration"
        attrs["name"] = name  # Name of configuration
        attrs["location"] = location  # EarthLocation
        attrs["receptor_frame"] = receptor_frame
        attrs["frame"] = frame

        super().__init__(datavars, coords=coords, attrs=attrs)


@xarray.register_dataset_accessor("configuration_acc")
class ConfigurationAccessor(XarrayAccessorMixin):
    """Convenience methods to access the fields of the Configuration"""

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def nants(self):
        """ Names of the dishes/stations"""
        return len(self._obj["names"])


class GainTable(xarray.Dataset):
    """Gain table with: time, antenna,  weight columns

    The weight is usually that output from gain solvers.

    Here is an example::

        <xarray.GainTable>
        Dimensions:    (antenna: 115, frequency: 3, receptor1: 2, receptor2: 2, time: 3)
        Coordinates:
          * time       (time) float64 5.085e+09 5.085e+09 5.085e+09
          * antenna    (antenna) int64 0 1 2 3 4 5 6 7 ... 108 109 110 111 112 113 114
          * frequency  (frequency) float64 1e+08 1.05e+08 1.1e+08
          * receptor1  (receptor1) <U1 'X' 'Y'
          * receptor2  (receptor2) <U1 'X' 'Y'
        Data variables:
            gain       (time, antenna, frequency, receptor1, receptor2) complex128 (0...
            weight     (time, antenna, frequency, receptor1, receptor2) float64 1.0 ....
            residual   (time, frequency, receptor1, receptor2) float64 0.0 0.0 ... 0.0
            interval   (time) float32 99.72697 99.72697 99.72697
            datetime   (time) datetime64[ns] 2000-01-01T03:54:07.843184299 ... 2000-0...
        Attributes:
            rascil_data_model:  GainTable
            receptor_frame:     <rascil.data_models.polarisation.ReceptorFrame object...
            phasecentre:        <SkyCoord (ICRS): (ra, dec) in deg    (180., -35.)>
            configuration:      <xarray.Configuration> Dimensions:   (id: 115, spati...

    """

    __slots__ = ()

    def __init__(
        self,
        gain: numpy.array = None,
        time: numpy.array = None,
        interval=None,
        weight: numpy.array = None,
        residual: numpy.array = None,
        frequency: numpy.array = None,
        receptor_frame: ReceptorFrame = ReceptorFrame("linear"),
        phasecentre=None,
        configuration=None,
    ):
        """Create a gaintable from arrays

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
        datavars["gain"] = xarray.DataArray(
            gain, dims=["time", "antenna", "frequency", "receptor1", "receptor2"]
        )
        datavars["weight"] = xarray.DataArray(
            weight, dims=["time", "antenna", "frequency", "receptor1", "receptor2"]
        )
        datavars["residual"] = xarray.DataArray(
            residual, dims=["time", "frequency", "receptor1", "receptor2"]
        )
        datavars["interval"] = xarray.DataArray(interval, dims=["time"])
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64, dims="time"
        )
        attrs = dict()
        attrs["rascil_data_model"] = "GainTable"
        attrs["receptor_frame"] = receptor_frame
        attrs["phasecentre"] = phasecentre
        attrs["configuration"] = configuration

        super().__init__(datavars, coords=coords, attrs=attrs)


@xarray.register_dataset_accessor("gaintable_acc")
class GainTableAccessor(XarrayAccessorMixin):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def ntimes(self):
        """Number of times (i.e. rows) in this table"""
        return self._obj["gain"].shape[0]

    @property
    def nants(self):
        """Number of dishes/stations"""
        return self._obj["gain"].shape[1]

    @property
    def nchan(self):
        """Number of channels"""
        return self._obj["gain"].shape[2]

    @property
    def nrec(self):
        """Number of receivers"""
        return len(self._obj["receptor1"])

    @property
    def receptors(self):
        """Receptors"""
        return self._obj["receptor1"]


class PointingTable(xarray.Dataset):
    """Pointing table with data_models: time, antenna, offset[:, chan, rec, 2], weight columns

    Here is an example::

        <xarray.PointingTable>
        Dimensions:    (angle: 2, antenna: 115, frequency: 3, receptor: 2, time: 3)
        Coordinates:
          * time       (time) float64 5.085e+09 5.085e+09 5.085e+09
          * antenna    (antenna) int64 0 1 2 3 4 5 6 7 ... 108 109 110 111 112 113 114
          * frequency  (frequency) float64 1e+08 1.05e+08 1.1e+08
          * receptor   (receptor) <U1 'X' 'Y'
          * angle      (angle) <U2 'az' 'el'
        Data variables:
            pointing   (time, antenna, frequency, receptor, angle) float64 -0.0002627...
            nominal    (time, antenna, frequency, receptor, angle) float64 -3.142 ......
            weight     (time, antenna, frequency, receptor, angle) float64 1.0 ... 1.0
            residual   (time, frequency, receptor, angle) float64 0.0 0.0 ... 0.0 0.0
            interval   (time) float64 99.73 99.73 99.73
            datetime   (time) datetime64[ns] 2000-01-01T03:54:07.843184299 ... 2000-0...
        Attributes:
            rascil_data_model:  PointingTable
            receptor_frame:     <rascil.data_models.polarisation.ReceptorFrame object...
            pointing_frame:     azel
            pointingcentre:     <SkyCoord (ICRS): (ra, dec) in deg    (180., -35.)>
            configuration:      <xarray.Configuration> Dimensions:   (id: 115, spati...

    """

    __slots__ = ()

    def __init__(
        self,
        data=None,
        pointing: numpy.array = None,
        nominal: numpy.array = None,
        time: numpy.array = None,
        interval=None,
        weight: numpy.array = None,
        residual: numpy.array = None,
        frequency: numpy.array = None,
        receptor_frame: ReceptorFrame = ReceptorFrame("linear"),
        pointing_frame: str = "local",
        pointingcentre=None,
        configuration=None,
    ):
        """Create a pointing table from arrays

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
            "angle": ["az", "el"],
        }

        datavars = dict()
        datavars["pointing"] = xarray.DataArray(
            pointing, dims=["time", "antenna", "frequency", "receptor", "angle"]
        )
        datavars["nominal"] = xarray.DataArray(
            nominal, dims=["time", "antenna", "frequency", "receptor", "angle"]
        )
        datavars["weight"] = xarray.DataArray(
            weight, dims=["time", "antenna", "frequency", "receptor", "angle"]
        )
        datavars["residual"] = xarray.DataArray(
            residual, dims=["time", "frequency", "receptor", "angle"]
        )
        datavars["interval"] = xarray.DataArray(interval, dims=["time"])
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64, dims="time"
        )

        attrs = dict()
        attrs["rascil_data_model"] = "PointingTable"
        attrs["receptor_frame"] = receptor_frame
        attrs["pointing_frame"] = pointing_frame
        attrs["pointingcentre"] = pointingcentre
        attrs["configuration"] = configuration

        super().__init__(datavars, coords=coords, attrs=attrs)


@xarray.register_dataset_accessor("pointingtable_acc")
class PointingTableAccessor(XarrayAccessorMixin):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def nants(self):
        """Number of dishes/stations"""
        return self._obj["pointing"].shape[1]

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj["pointing"].frequency)

    @property
    def nrec(self):
        """Number of receptors"""
        return self.receptor_frame.nrec


class Image(xarray.Dataset):
    """Image class with pixels as an xarray.DataArray and the AstroPy`implementation of
    a World Coodinate System <http://docs.astropy.org/en/stable/wcs>`_

    The actual image values are kept in a data_var of the xarray.Dataset called "pixels"

    Many operations can be done conveniently using xarray processing_components on Image or on
    numpy operations on Image["pixels"].data. If the "pixels" data variable is chunked then
    Dask is automatically used whereever possible to distribute processing.

    Here is an example::

        <xarray.Image>
        Dimensions:       (chan: 3, pol: 4, x: 256, y: 256)
        Coordinates:
            frequency     (chan) float64 1e+08 1.01e+08 1.02e+08
            polarisation  (pol) <U1 'I' 'Q' 'U' 'V'
          * y             (y) float64 -35.11 -35.11 -35.11 ... -34.89 -34.89 -34.89
          * x             (x) float64 179.9 179.9 179.9 179.9 ... 180.1 180.1 180.1
            ra            (x, y) float64 180.1 180.1 180.1 180.1 ... 179.9 179.9 179.9
            dec           (x, y) float64 -35.11 -35.11 -35.11 ... -34.89 -34.89 -34.89
        Dimensions without coordinates: chan, pol
        Data variables:
            pixels        (chan, pol, y, x) float64 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
        Attributes:
            rascil_data_model:  Image
            frame:              icrs


    """

    __slots__ = ()

    def __init__(self, data, polarisation_frame=None, wcs=None, clean_beam=None):
        """Create an Image

        :param data: pixel values
        :param polarisation_frame: as a PolarisationFrame object
        :param wcs: WCS object
        :param clean_beam: dict e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
        :return: Image (i.e. xarray.Dataset)
        """
        super().__init__()

        nchan, npol, ny, nx = data.shape

        frequency = wcs.sub([4]).wcs_pix2world(range(nchan), 0)[0]
        cellsize = numpy.deg2rad(numpy.abs(wcs.wcs.cdelt[1]))
        cx = numpy.deg2rad(wcs.wcs.crval[0])
        cy = numpy.deg2rad(wcs.wcs.crval[1])

        lmesh, mmesh = numpy.meshgrid(numpy.arange(ny), numpy.arange(nx))
        try:
            ra, dec = wcs.sub([1, 2]).wcs_pix2world(lmesh, mmesh, 0)
            ra = numpy.deg2rad(ra)
            dec = numpy.deg2rad(dec)
        except:
            log.warning(
                "Coordinates not RA, Dec pair: ra, dec coordinates are x, y meshes"
            )
            ra = lmesh
            dec = mmesh

        # Define the dimensions
        dims = ["frequency", "polarisation", "y", "x"]

        # Define the coordinates on these dimensions
        coords = {
            "frequency": ("frequency", frequency),
            "polarisation": ("polarisation", polarisation_frame.names),
            "y": numpy.linspace(
                cy - cellsize * ny / 2, cy + cellsize * ny / 2, ny, endpoint=False
            ),
            "x": numpy.linspace(
                cx - cellsize * nx / 2, cx + cellsize * nx / 2, nx, endpoint=False
            ),
            "ra": (("x", "y"), ra, {"units": "rad"}),
            "dec": (("x", "y"), dec, {"units": "rad"}),
        }

        assert (
            data.shape[0] == nchan
        ), "Number of frequency channels {} and data shape {} are incompatible".format(
            len(frequency), data.shape
        )
        assert (
            data.shape[1] == npol
        ), "Polarisation frame {} and data shape {} are incompatible".format(
            polarisation_frame.type, data.shape
        )
        assert coords["x"][0] != coords["x"][-1]
        assert coords["y"][0] != coords["y"][-1]

        assert len(coords["y"]) == ny
        assert len(coords["x"]) == nx

        data_vars = dict()
        data_vars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)

        if isinstance(clean_beam, dict):
            for key in ["bmaj", "bmin", "bpa"]:
                if key not in clean_beam.keys():
                    raise KeyError(f"Image: clean_beam must have key {key}")
            
        attrs = {
            "rascil_data_model": "Image",
            "_polarisation_frame": polarisation_frame.type,
            "_projection": (wcs.wcs.ctype[0], wcs.wcs.ctype[1]),
            "spectral_type": wcs.wcs.ctype[3],
            "clean_beam": clean_beam,
        }

        super().__init__(data_vars, coords=coords, attrs=attrs)


@xarray.register_dataset_accessor("image_acc")
class ImageAccessor(XarrayAccessorMixin):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def shape(self):
        """Shape of array

        :return:
        """
        return self._obj["pixels"].data.shape

    @property
    def nchan(self):
        """Number of channels

        :return: Number of channels
        """
        return len(self._obj.frequency)

    @property
    def npol(self):
        """Number of polarisations

        :return: Number of polarisations
        """
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"]).npol

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)

        :return:
        """
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    @property
    def projection(self):
        """Projection (from coords)

        :return:
        """
        return self._obj.attrs["_projection"]

    @property
    def phasecentre(self):
        """Return the phasecentre as a SkyCoord

        :return:
        """
        cx = len(self._obj.coords["x"].data) // 2
        cy = len(self._obj.coords["y"].data) // 2
        return SkyCoord(
            numpy.rad2deg(self._obj.coords["ra"][cy, cx].values) * u.deg,
            numpy.rad2deg(self._obj.coords["dec"][cy, cx].values) * u.deg,
            frame="icrs",
            equinox="J2000",
        )

    @property
    def wcs(self):
        """Return the equivalent WCS

        :return:
        """
        return image_wcs(self._obj)


class GridData(xarray.Dataset):
    """Class to hold Gridded data for Fourier processing
    - Has four or more coordinates: [chan, pol, z, y, x] where x can be u, l; y can be v, m; z can be w, n

    The conventions for indexing in WCS and numpy are opposite.
    - In astropy.wcs, the order is (longitude, latitude, polarisation, frequency)
    - in numpy, the order is (frequency, polarisation, depth, latitude, longitude)

    .. warning::
        The polarisation_frame is kept in two places, the WCS and the polarisation_frame
        variable. The latter should be considered definitive.

    Here is an example::

        <xarray.Image>
        Dimensions:       (chan: 3, l: 256, m: 256, pol: 4)
        Coordinates:
            frequency     (chan) float64 1e+08 1.01e+08 1.02e+08
            polarisation  (pol) <U1 'I' 'Q' 'U' 'V'
          * m             (m) float64 -35.11 -35.11 -35.11 ... -34.89 -34.89 -34.89
          * l             (l) float64 179.9 179.9 179.9 179.9 ... 180.1 180.1 180.1
            ra            (l, m) float64 180.1 180.1 180.1 180.1 ... 179.9 179.9 179.9
            dec           (l, m) float64 -35.11 -35.11 -35.11 ... -34.89 -34.89 -34.89
        Dimensions without coordinates: chan, pol
        Data variables:
            pixels        (chan, pol, m, l) float64 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
        Attributes:
            rascil_data_model:  Image


    """

    __slots__ = ()

    def __init__(self, data, polarisation_frame=None, grid_wcs=None):
        """Create a GridData

        :param polarisation_frame:
        :return: GridData
        """

        super().__init__()

        nchan, npol, nv, nu = data.shape

        frequency = grid_wcs.sub([4]).wcs_pix2world(range(nchan), 0)[0]

        assert npol == polarisation_frame.npol
        cu = grid_wcs.wcs.crval[0]
        cv = grid_wcs.wcs.crval[1]
        du = grid_wcs.wcs.cdelt[0]
        dv = grid_wcs.wcs.cdelt[1]

        dims = ["frequency", "polarisation", "u", "v"]

        # Define the coordinates on these dimensions
        coords = {
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "v": numpy.linspace(cv - dv * nv / 2, cv + dv * nv / 2, nv, endpoint=False),
            "u": numpy.linspace(cu - du * nu / 2, cu + du * nu / 2, nu, endpoint=False),
        }

        attrs = dict()

        attrs["rascil_data_model"] = "GridData"
        attrs["_polarisation_frame"] = polarisation_frame.type

        data_vars = dict()
        data_vars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)
        super().__init__(data_vars, coords=coords, attrs=attrs)


@xarray.register_dataset_accessor("griddata_acc")
class GridDataAccessor(XarrayAccessorMixin):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj.frequency)

    def npol(self):
        """Number of polarisations

        :return: Number of polarisations
        """
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"]).npol

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)

        :return:
        """
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    @property
    def shape(self):
        """Shape of data array"""
        return self._obj["pixels"].data.shape

    @property
    def griddata_wcs(self):
        """Return the equivalent WCS

        :return:
        """
        return griddata_wcs(self._obj)

    @property
    def projection_wcs(self):
        """Return the equivalent WCS

        :return:
        """
        return image_wcs(self._obj)


class ConvolutionFunction(xarray.Dataset):
    """Class to hold Convolution function for Fourier processing
    - Has four or more coordinates: [chan, pol, z, y, x] where x can be u, l; y can be v, m; z can be w, n

    The cf has axes [chan, pol, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes

    The axes UU,VV have the same physical stride as the image, The axes DUU, DVV are subsampled.

    Convolution function holds the original sky plane projection in the projection_wcs.

    Here is an example::

        <xarray.ConvolutionFunction>
        Dimensions:       (du: 8, dv: 8, frequency: 1, polarisation: 1, u: 16, v: 16, w: 1)
        Coordinates:
          * frequency     (frequency) float64 1e+08
          * polarisation  (polarisation) <U1 'I'
          * w             (w) float64 0.0
          * dv            (dv) float64 -1.031e+05 -7.735e+04 ... 5.157e+04 7.735e+04
          * du            (du) float64 -1.031e+05 -7.735e+04 ... 5.157e+04 7.735e+04
          * v             (v) float64 -1.65e+06 -1.444e+06 ... 1.238e+06 1.444e+06
          * u             (u) float64 -1.65e+06 -1.444e+06 ... 1.238e+06 1.444e+06
        Data variables:
            pixels        (frequency, polarisation, w, dv, du, v, u) complex128 0j .....
        Attributes:
            rascil_data_model:   ConvolutionFunction
            grid_wcs:            WCS Keywords Number of WCS axes: 7 CTYPE : 'UU' ...
            projection_wcs:      WCS Keywords Number of WCS axes: 4 CTYPE : 'RA--...
            polarisation_frame:  stokesI


    """

    __slots__ = ()

    def __init__(self, data, cf_wcs=None, polarisation_frame=None):
        """Create ConvolutionFunction

        :param data: Data for cf
        :param cf_wcs: Astropy WCS object for the grid
        :param polarisation_frame: Polarisation_frame e.g. PolarisationFrame('linear')
        """

        super().__init__()

        nchan, npol, nw, oversampling, _, support, _ = data.shape
        frequency = cf_wcs.sub(["spectral"]).wcs_pix2world(range(nchan), 0)[0]

        assert (
            npol == polarisation_frame.npol
        ), "Mismatch between requested image polarisation and actual visibility polarisation"

        du = cf_wcs.wcs.cdelt[0]
        dv = cf_wcs.wcs.cdelt[1]
        ddu = cf_wcs.wcs.cdelt[0] / oversampling
        ddv = cf_wcs.wcs.cdelt[1] / oversampling
        cu = cf_wcs.wcs.crval[0]
        cv = cf_wcs.wcs.crval[1]
        cdu = oversampling // 2
        cdv = oversampling // 2

        wstep = numpy.abs(cf_wcs.wcs.cdelt[4])

        coords = {
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "dv": numpy.linspace(
                cdv - ddv * oversampling / 2,
                cdv + ddv * oversampling / 2,
                oversampling,
                endpoint=False,
            ),
            "du": numpy.linspace(
                cdu - ddu * oversampling / 2,
                cdu + ddu * oversampling / 2,
                oversampling,
                endpoint=False,
            ),
            "w": numpy.linspace(-wstep * nw / 2, wstep * nw / 2, nw, endpoint=False),
            "v": numpy.linspace(
                cv - dv * support / 2, cv + dv * support / 2, support, endpoint=False
            ),
            "u": numpy.linspace(
                cu - du * support / 2, cu + du * support / 2, support, endpoint=False
            ),
        }

        if nw == 1:
            coords["w"]: numpy.zeros([1])

        dims = ["frequency", "polarisation", "w", "dv", "du", "u", "v"]

        assert coords["u"][0] != coords["u"][-1]
        assert coords["v"][0] != coords["v"][-1]

        attrs = dict()
        attrs["rascil_data_model"] = "ConvolutionFunction"
        attrs["_polarisation_frame"] = polarisation_frame.type

        nchan = len(frequency)
        npol = polarisation_frame.npol
        if data is None:
            data = numpy.zeros(
                [nchan, npol, nw, oversampling, oversampling, support, support],
                dtype="complex",
            )
        else:
            assert data.shape == (
                nchan,
                npol,
                nw,
                oversampling,
                oversampling,
                support,
                support,
            ), "Polarisation frame {} and data shape {} are incompatible".format(
                polarisation_frame.type, data.shape
            )
        data_vars = dict()
        data_vars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)
        super().__init__(data_vars, coords=coords, attrs=attrs)


@xarray.register_dataset_accessor("convolutionfunction_acc")
class ConvolutionFunctionAccessor(XarrayAccessorMixin):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj.frequency)

    @property
    def npol(self):
        """Number of polarisations"""
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"]).npol

    @property
    def cf_wcs(self):
        """Return the equivalent WCS

        :return:
        """
        return cf_wcs(self._obj)

    @property
    def shape(self):
        """Shape of data array"""
        return self._obj["pixels"].data.shape

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)

        :return:
        """
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])


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

    def __init__(
        self,
        direction=None,
        frequency=None,
        name=None,
        flux=None,
        shape="Point",
        polarisation_frame=PolarisationFrame("stokesIQUV"),
        params=None,
    ):
        """Define the required structure

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
        assert (
            self.frequency.shape[0] == self.flux.shape[0]
        ), "Frequency shape %s, flux shape %s" % (self.frequency.shape, self.flux.shape)
        assert (
            polarisation_frame.npol == self.flux.shape[1]
        ), "Polarisation is %s, flux shape %s" % (
            polarisation_frame.type,
            self.flux.shape,
        )

    @property
    def nchan(self):
        """Number of channels"""
        return self.flux.shape[0]

    @property
    def npol(self):
        """Number of polarisations"""
        return self.flux.shape[1]

    def __str__(self):
        """Default printer for Skycomponent"""
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
    """A model for the sky, including an image, components, gaintable and a mask"""

    def __init__(
        self, image=None, components=None, gaintable=None, mask=None, fixed=False
    ):
        """A model of the sky as an image, components, gaintable and a mask

        Use copy_skymodel to make a proper copy of skymodel
        :param image: Image
        :param components: List of components
        :param gaintable: Gaintable for this skymodel
        :param mask: Mask for the image
        :param fixed: Is this model fixed?
        """
        if components is None:
            components = []

        # if image is not None:
        ##assert isinstance(image, Image), image
        self.image = image

        self.components = [sc for sc in components]
        self.gaintable = gaintable

        # if mask is not None:
        #    #assert isinstance(mask, Image), mask
        self.mask = mask

        self.fixed = fixed

    def __str__(self):
        """Default printer for SkyModel"""
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
    """BlockVisibility xarray Dataset class

    BlockVisibility is defined to hold an observation with one direction.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre.

    Polarisation frame is the same for the entire data set and can be stokesI, circular, circularnp, linear, linearnp

    The configuration is stored as an attribute.

    Here is an example::

        <xarray.BlockVisibility>
        Dimensions:            (baselines: 6670, frequency: 3, polarisation: 4, time: 3, uvw_index: 3)
        Coordinates:
          * time               (time) float64 5.085e+09 5.085e+09 5.085e+09
          * baselines          (baselines) MultiIndex
          - antenna1           (baselines) int64 0 0 0 0 0 0 ... 112 112 112 113 113 114
          - antenna2           (baselines) int64 0 1 2 3 4 5 ... 112 113 114 113 114 114
          * frequency          (frequency) float64 1e+08 1.05e+08 1.1e+08
          * polarisation       (polarisation) <U2 'XX' 'XY' 'YX' 'YY'
          * uvw_index          (uvw_index) <U1 'u' 'v' 'w'
        Data variables:
            integration_time   (time) float32 99.72697 99.72697 99.72697
            datetime           (time) datetime64[ns] 2000-01-01T03:54:07.843184299 .....
            vis                (time, baselines, frequency, polarisation) complex128 ...
            weight             (time, baselines, frequency, polarisation) float32 0.0...
            imaging_weight     (time, baselines, frequency, polarisation) float32 0.0...
            flags              (time, baselines, frequency, polarisation) float32 0.0...
            uvw                (time, baselines, uvw_index) float64 0.0 0.0 ... 0.0 0.0
            uvw_lambda         (time, baselines, frequency, uvw_index) float64 0.0 .....
            uvdist_lambda      (time, baselines, frequency) float64 0.0 0.0 ... 0.0 0.0
            channel_bandwidth  (frequency) float64 1e+07 1e+07 1e+07
        Attributes:
            phasecentre:         <SkyCoord (ICRS): (ra, dec) in deg    (180., -35.)>
            configuration:       <xarray.Configuration>Dimensions:   (id: 115, spat...
            polarisation_frame:  linear
            source:              unknown
            meta:                None


    """

    __slots__ = ()

    def __init__(
        self,
        frequency=None,
        channel_bandwidth=None,
        phasecentre=None,
        configuration=None,
        uvw=None,
        time=None,
        vis=None,
        weight=None,
        integration_time=None,
        flags=None,
        baselines=None,
        polarisation_frame=PolarisationFrame("stokesI"),
        imaging_weight=None,
        source="anonymous",
        meta=None,
        low_precision="float32",
    ):
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

        if weight is None:
            weight = numpy.ones(vis.shape)
        else:
            assert weight.shape == vis.shape

        if imaging_weight is None:
            imaging_weight = weight
        else:
            assert imaging_weight.shape == vis.shape

        if integration_time is None:
            integration_time = numpy.ones_like(time)
        else:
            assert len(integration_time) == len(time)

        k = (frequency / const.c).value
        if len(frequency) == 1:
            uvw_lambda = (uvw * k)[..., numpy.newaxis, :]
        else:
            uvw_lambda = numpy.einsum("tbs,k->tbks", uvw, k)

        nchan = len(frequency)
        npol = polarisation_frame.npol
        # Define the names of the dimensions

        coords = {
            "time": time,
            "baselines": baselines,
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "spatial": ["u", "v", "w"],
        }

        datavars = dict()
        datavars["integration_time"] = xarray.DataArray(
            integration_time.astype(low_precision), dims=["time"], attrs={"units": "s"}
        )
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64,
            dims=["time"],
            attrs={"units": "s"},
        )
        datavars["vis"] = xarray.DataArray(
            vis,
            dims=["time", "baselines", "frequency", "polarisation"],
            attrs={"units": "Jy"},
        )
        datavars["weight"] = xarray.DataArray(
            weight.astype(low_precision),
            dims=["time", "baselines", "frequency", "polarisation"],
        )
        datavars["imaging_weight"] = xarray.DataArray(
            imaging_weight.astype(low_precision),
            dims=["time", "baselines", "frequency", "polarisation"],
        )
        datavars["flags"] = xarray.DataArray(
            flags.astype(low_precision),
            dims=["time", "baselines", "frequency", "polarisation"],
        )
        datavars["uvw"] = xarray.DataArray(
            uvw, dims=["time", "baselines", "spatial"], attrs={"units": "m"}
        )

        datavars["uvw_lambda"] = xarray.DataArray(
            uvw_lambda,
            dims=["time", "baselines", "frequency", "spatial"],
            attrs={"units": "lambda"},
        )
        datavars["uvdist_lambda"] = xarray.DataArray(
            numpy.hypot(uvw_lambda[..., 0], uvw_lambda[..., 1]),
            dims=["time", "baselines", "frequency"],
            attrs={"units": "lambda"},
        )

        datavars["channel_bandwidth"] = xarray.DataArray(
            channel_bandwidth, dims=["frequency"], attrs={"units": "Hz"}
        )

        attrs = dict()
        attrs["configuration"] = configuration  # Antenna/station configuration
        attrs["source"] = source
        attrs["phasecentre"] = phasecentre
        attrs["_polarisation_frame"] = polarisation_frame.type
        attrs["meta"] = meta

        super().__init__(datavars, coords=coords, attrs=attrs)


@xarray.register_dataset_accessor("blockvisibility_acc")
class BlockVisibilityAccessor(XarrayAccessorMixin):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def rows(self):
        """Rows"""
        return range(len(self._obj.time))

    @property
    def ntimes(self):
        """Number of times (i.e. rows) in this table"""
        return len(self._obj["time"])

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj["frequency"])

    @property
    def npol(self):
        """Number of polarisations"""
        return len(self._obj.polarisation)

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)

        :return:
        """
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    @property
    def nants(self):
        """Number of antennas"""
        return self._obj.configuration.configuration_acc.nants

    @property
    def nbaselines(self):
        """Number of Baselines"""
        return len(self._obj["baselines"])

    @property
    def u(self):
        """u coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 0]

    @property
    def v(self):
        """v coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 1]

    @property
    def w(self):
        """w coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 2]

    @property
    def u_lambda(self):
        """u coordinate (wavelengths) [nrows, nbaseline]"""
        return self._obj["uvw_lambda"][..., 0]

    @property
    def v_lambda(self):
        """v coordinate (wavelengths) [nrows, nbaseline]"""
        return self._obj["uvw_lambda"][..., 1]

    @property
    def w_lambda(self):
        """w coordinate (wavelengths) [nrows, nbaseline]"""
        return self._obj["uvw_lambda"][..., 2]

    @property
    def uvdist(self):
        """uv distance (metres) [nrows, nbaseline]"""
        return numpy.hypot(self.u, self.v)

    @property
    def uvdist_lambda(self):
        """uv distance (metres) [nrows, nbaseline]"""
        return numpy.hypot(self.u_lambda, self.v_lambda)

    @property
    def uvwdist(self):
        """uv distance (metres) [nrows, nbaseline]"""
        return numpy.hypot(self.u, self.v, self.w)

    @property
    def flagged_vis(self):
        """Flagged complex visibility [nrows, nbaseline, nchan, npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["vis"].data * (1 - self._obj["flags"].data)

    @property
    def flagged_weight(self):
        """Weight [: npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["weight"].data * (1 - self._obj["flags"].data)

    @property
    def flagged_imaging_weight(self):
        """Flagged Imaging_weight[nrows, nbaseline, nchan, npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["imaging_weight"].data * (1 - self._obj["flags"].data)

    @property
    def nvis(self):
        """Number of visibilities (in total)"""
        return numpy.product(self._obj.vis.shape)


class FlagTable(xarray.Dataset):
    """Flag table class

    Flags, time, integration_time, frequency, channel_bandwidth, pol,
    in an xarray.

    The configuration is also an attribute
    """

    __slots__ = ()

    def __init__(
        self,
        baselines=None,
        flags=None,
        frequency=None,
        channel_bandwidth=None,
        configuration=None,
        time=None,
        integration_time=None,
        polarisation_frame=None,
    ):
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
            "baselines": baselines,
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
        }

        datavars = dict()
        datavars["flags"] = xarray.DataArray(
            flags, dims=["time", "baselines", "frequency", "polarisation"]
        )
        datavars["integration_time"] = xarray.DataArray(integration_time, dims=["time"])
        datavars["channel_bandwidth"] = xarray.DataArray(
            channel_bandwidth, dims=["frequency"]
        )
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64, dims="time"
        )

        attrs = dict()
        attrs["_polarisation_frame"] = polarisation_frame.type
        attrs["configuration"] = configuration  # Antenna/station configuration

        super().__init__(datavars, coords=coords, attrs=attrs)


@xarray.register_dataset_accessor("flagtable_acc")
class FlagTableAccessor(XarrayAccessorMixin):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj["frequency"])

    @property
    def npol(self):
        """Number of polarisations"""
        return self.polarisation_frame.npol

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)

        :return:
        """
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    @property
    def nants(self):
        """Number of antennas"""
        return self.attrs["configuration"].configuration_acc.nants

    @property
    def nbaselines(self):
        """Number of Baselines"""
        return len(self["baselines"])


class QA:
    """Quality assessment

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
        """Default printer for QA"""
        s = "Quality assessment:\n"
        s += "\tOrigin: %s\n" % self.origin
        s += "\tContext: %s\n" % self.context
        s += "\tData:\n"
        for dataname in self.data.keys():
            s += "\t\t%s: %r\n" % (dataname, str(self.data[dataname]))
        return s


class ScienceDataModel:
    """ Science Data Model: not defined yet"""

    def __init__(self):
        pass

    def __str__(self):
        """Default printer for Science Data Model"""
        return ""
