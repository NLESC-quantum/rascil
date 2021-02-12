"""Configuration definitions. A Configuration definition is read from a number of different formats.

"""

__all__ = ['create_configuration_from_file',
           'create_configuration_from_MIDfile',
           'create_configuration_from_SKAfile',
           'create_configuration_from_LLAfile',
           'create_LOFAR_configuration',
           'create_named_configuration',
           'limit_rmax',
           'find_vptype_from_name',
           'select_configuration']

import numpy
from typing import Union
from astropy import units as u
from astropy.coordinates import EarthLocation

from rascil.data_models.memory_data_models import Configuration
from rascil.data_models.parameters import rascil_data_path, get_parameter
from rascil.processing_components.util.installation_checks import check_data_directory
from rascil.processing_components.util.coordinate_support import xyz_at_latitude
from rascil.processing_components.util import lla_to_ecef,ecef_to_enu,eci_to_enu

import logging

log = logging.getLogger('rascil-logger')

def find_vptype_from_name(names, match: Union[str, dict] = "unknown"):
    """Determine voltage pattern type from name using a dictionary
    
    There ae two modes:
    
    If match is a dict, then the antenna/station names are matched. An example of match
    would be: d={"M0":"MEERKAT", "SKA":"MID"} The test if whether the
    key e.g. M0 is in the antenna/station name e.g. M053
    
    If match is a str then the returned array is filled with that value.
    
    :param names:
    :param match:
    :return:
    """
    if isinstance(match, dict):
        vp_types = numpy.repeat("unknown", len(names))
        for item in match:
            for i, name in enumerate(names):
                if item in name:
                    vp_types[i] = match.get(item)
    elif isinstance(match, str):
        vp_types = numpy.repeat(match, len(names))
    else:
        raise ValueError("match must be str or dict")
    
    return vp_types
    
def create_configuration_from_file(antfile: str, location: EarthLocation = None,
                                   mount: str = 'azel',
                                   names: str = "%d",
                                   vp_type: Union[str, dict] = "Unknown",
                                   diameter=35.0,
                                   rmax=None, name='', skip=1,
                                   ecef = True) -> Configuration:
    """ Define configuration from a text file

    :param antfile: Antenna file name
    :param location: Earthlocation of array
    :param mount: mount type: 'azel', 'xy', 'equatorial'
    :param names: Antenna names e.g. "VLA%d"
    :param vp_type: string or rule to map name to voltage pattern type
    :param diameter: Effective diameter of station or antenna
    :param rmax: Maximum distance from array centre (m)
    :param name: Name of array
    :return: Configuration
    """
    check_data_directory()

    antxyz = numpy.genfromtxt(antfile, delimiter=",")
    assert antxyz.shape[1] == 3, ("Antenna array has wrong shape %s" % antxyz.shape)

    nants = antxyz.shape[0]
    if ecef:
        antxyz = ecef_to_enu(location, antxyz)

    diameters = diameter * numpy.ones(nants)
    anames = [names % ant for ant in range(nants)]
    mounts = numpy.repeat(mount, nants)
    antxyz, diameters, anames, mounts = limit_rmax(antxyz, diameters, anames, mounts, rmax)

    antxyz = antxyz[::skip]
    diameters = diameters[::skip]
    anames = anames[::skip]
    mounts = mounts[::skip]

    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz,
                       vp_type=find_vptype_from_name(anames, vp_type),
                       diameter=diameters, name=name)
    return fc


def create_configuration_from_SKAfile(antfile: str,
                                      mount: str = 'azel',
                                      names: str = "%d",
                                      vp_type: Union[str, dict] = "Unknown",
                                      rmax=None, name='', location=None,
                                      skip=1) -> Configuration:
    """ Define configuration from a SKA format file

    :param antfile: Antenna file name
    :param location: Earthlocation of array
    :param mount: mount type: 'azel', 'xy', 'equatorial'
    :param names: Antenna names e.g. "VLA%d"
    :param rmax: Maximum distance from array centre (m)
    :param name: Name of array
    :return: Configuration
    """
    check_data_directory()

    antdiamlonglat = numpy.genfromtxt(antfile, usecols=[0, 1, 2], delimiter="\t")
    
    assert antdiamlonglat.shape[1] == 3, ("Antenna array has wrong shape %s" % antdiamlonglat.shape)
    antxyz = numpy.zeros([antdiamlonglat.shape[0] - 1, 3])
    diameters = numpy.zeros([antdiamlonglat.shape[0] - 1])
    for ant in range(antdiamlonglat.shape[0] - 1):
        loc = EarthLocation(lon=antdiamlonglat[ant, 1], lat=antdiamlonglat[ant, 2], height=0.0).geocentric
        antxyz[ant] = [loc[0].to(u.m).value, loc[1].to(u.m).value, loc[2].to(u.m).value]
        diameters[ant] = antdiamlonglat[ant, 0]

    nants = antxyz.shape[0]
    anames = [names % ant for ant in range(nants)]
    mounts = numpy.repeat(mount, nants)
    antxyz, diameters, anames, mounts = limit_rmax(antxyz, diameters, anames, mounts, rmax)

    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz,
                       vp_type=find_vptype_from_name(names, vp_type),
                       diameter=diameters, name=name)
    return fc


def create_configuration_from_MIDfile(antfile: str, location=None,
                                      mount: str = 'azel',
                                      vp_type: Union[str, dict] = "Unknown",
                                      rmax=None, name='',
                                      skip=1,
                                      ecef = True) -> Configuration:
    """ Define configuration from a SKA MID format file

    :param antfile: Antenna file name
    :param mount: mount type: 'azel', 'xy'
    :param rmax: Maximum distance from array centre (m)
    :param name: Name of array
    :param ecef: Configuration file format: ECEF or local-xyz
    :return: Configuration
    """
    check_data_directory()

    antxyz = numpy.genfromtxt(antfile, skip_header=5, usecols=[0, 1, 2], delimiter=" ")


    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    if ecef:
        antxyz = ecef_to_enu(location, antxyz)

    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=5, usecols=[4], delimiter=" ")
    mounts = numpy.repeat(mount, nants)
    diameters = numpy.genfromtxt(antfile, dtype='str', skip_header=5, usecols=[3], delimiter=" ").astype('float')

    antxyz, diameters, anames, mounts = limit_rmax(antxyz, diameters, anames, mounts, rmax)
    
    antxyz = antxyz[::skip]
    diameters = diameters[::skip]
    anames = anames[::skip]
    mounts = mounts[::skip]
    
    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz,
                       vp_type=find_vptype_from_name(anames, vp_type),
                       diameter=diameters, name=name)

    return fc

def create_configuration_from_LLAfile(antfile: str, location: EarthLocation = None,
                                   mount: str = 'azel',
                                   names: str = "%d",
                                   vp_type: Union[str, dict] = "Unknown",
                                   diameter=35.0, alt=0.0,
                                   rmax=None, name='', skip=1,
                                   ecef=False) -> Configuration:
    """ Define configuration from a longitude-latitude file

    :param antfile: Antenna file name
    :param location: Earthlocation of array
    :param mount: mount type: 'azel', 'xy', 'equatorial'
    :param names: Antenna names e.g. "VLA%d"
    :param vp_type: string or rule to map name to voltage pattern type
    :param diameter: Effective diameter of station or antenna
    :param height: The altitude assumed
    :param rmax: Maximum distance from array centre (m)
    :param name: Name of array
    :param ecef: Configuration file format: ECEF or local-xyz
    :return: Configuration
    """
    check_data_directory()

    antxyz = numpy.genfromtxt(antfile, delimiter=",")

    nants = antxyz.shape[0]
    if antxyz.shape[1] == 2: #if no altitude data
        alts = alt * numpy.ones(nants)

    lon, lat = antxyz[:,0], antxyz[:,1]
    x,y,z = lla_to_ecef(lat*u.deg, lon*u.deg, alt)
    antxyz = numpy.stack((x,y,z),axis=1)
    
    if ecef:
        antxyz = ecef_to_enu(location, antxyz)

    diameters = diameter * numpy.ones(nants)
    anames = [names % ant for ant in range(nants)]
    mounts = numpy.repeat(mount, nants)
    antxyz, diameters, anames, mounts = limit_rmax(antxyz, diameters, anames, mounts, rmax)

    antxyz = antxyz[::skip]
    diameters = diameters[::skip]
    anames = anames[::skip]
    mounts = mounts[::skip]

    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz,
                       vp_type=find_vptype_from_name(anames, vp_type),
                       diameter=diameters, name=name)
    return fc


def limit_rmax(antxyz, diameters, names, mounts, rmax):
    """ Select antennas with radius from centre < rmax
    
    :param antxyz: Geocentric coordinates
    :param diameters: diameters in metres
    :param names: Names
    :param mounts: Mount types
    :param rmax: Maximum radius (m)
    :return:
    """
    if rmax is not None:
        lantxyz = antxyz - numpy.average(antxyz, axis=0)
        r = numpy.sqrt(lantxyz[:, 0] ** 2 + lantxyz[:, 1] ** 2 + lantxyz[:, 2] ** 2)

        antxyz = antxyz[r < rmax]
        log.debug('create_configuration_from_file: Maximum radius %.1f m includes %d antennas/stations' %
                  (rmax, antxyz.shape[0]))
        diameters = diameters[r < rmax]
        names = numpy.array(names)[r < rmax]
        mounts = numpy.array(mounts)[r<rmax]
    else:
        log.debug('create_configuration_from_file: %d antennas/stations' % (antxyz.shape[0]))
    return antxyz, diameters, names, mounts


def create_LOFAR_configuration(antfile: str, location,
                               rmax=1e6, skip=1) -> Configuration:
    """ Define configuration from the LOFAR configuration file

    :param antfile:
    :param location: EarthLocation
    :param rmax: Maximum distance from array centre (m)
    :return: Configuration
    """
    check_data_directory()

    antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1, 2, 3], delimiter=",")
    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    antxyz = ecef_to_enu(location, antxyz)
    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
    mounts = numpy.repeat('XY', nants)
    diameters = numpy.repeat(35.0, nants)
    
    antxyz, diameters, mounts, anames = limit_rmax(antxyz, diameters, anames, mounts, rmax)

    antxyz = antxyz[::skip]
    diameters = diameters[::skip]
    anames = anames[::skip]
    mounts = mounts[::skip]

    vp_type = {"HBA":"HBA", "LBA":"LBA"}
    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz,
                       vp_type=find_vptype_from_name(anames, vp_type),
                       diameter=diameters, name='LOFAR')
    return fc


def create_named_configuration(name: str = 'LOWBD2', **kwargs) -> Configuration:
    """ Create standard configurations e.g. LOWBD2, MIDBD2

    Possible configurations are::
        LOWBD2
        LOWBD2-core
        LOW == LOWR3
        MID == MIDR5
        MEERKAT+
        ASKAP
        LOFAR
        VLAA
        VLAA_north

    :param name: name of Configuration MID, LOW, LOFAR, VLAA, ASKAP
    :param rmax: Maximum distance of station from the average (m)
    :return:
    
    For LOWBD2, setting rmax gives the following number of stations
    100.0       13
    300.0       94
    1000.0      251
    3000.0      314
    10000.0     398
    30000.0     476
    100000.0    512
    """
    
    check_data_directory()

    low_location = EarthLocation(lon=116.76444824*u.deg, lat=-26.824722084*u.deg, height=300.0)
    mid_location = EarthLocation(lon=21.443803*u.deg, lat=-30.712925*u.deg, height=1053.000000)
    meerkat_location = EarthLocation(lon=21.44388889*u.deg, lat=-30.7110565*u.deg, height=1086.6)
    if name == 'LOWBD2':
        location = low_location
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_data_path("configurations/LOWBD2.csv"),
                                            location=location, mount='XY', names='LOWBD2_%d',
                                            vp_type="LOW",
                                            diameter=35.0, name=name, ecef=False, **kwargs)
    elif name == 'LOWBD2-CORE':
        location = low_location
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_data_path("configurations/LOWBD2-CORE.csv"),
                                            vp_type="LOW",
                                            location=location, mount='XY', names='LOWBD2_%d',
                                            diameter=35.0, name=name, ecef=False, **kwargs)
    elif (name == 'LOW') or (name == 'LOWR3'):
        location = low_location
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_MIDfile(antfile=rascil_data_path("configurations/ska1low.cfg"),
                                               vp_type="LOW",
                                          mount='XY', name=name, location=location, **kwargs)
    elif (name == 'MID') or (name == "MIDR5"):
        location = mid_location
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_MIDfile(antfile=rascil_data_path("configurations/ska1mid.cfg"),
                                               vp_type={"M0":"MEERKAT", "SKA":"MID"},
            mount='azel', name=name, location=location, **kwargs)
    elif name == 'MEERKAT+':
        location = meerkat_location
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_MIDfile(antfile=rascil_data_path("configurations/mkatplus.cfg"),
                                               vp_type={"m0": "MEERKAT", "s0": "MID"},
                                               mount='ALT-AZ', name=name, location=location, **kwargs)
    elif name == 'ASKAP':
        location = EarthLocation(lon=+116.6356824*u.deg, lat=-26.7013006*u.deg, height=377.0*u.m)
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_MIDfile(antfile=rascil_data_path("configurations/askap.cfg"),
                                            vp_type="ASKAP",
                                            mount='equatorial', name=name, location=location, **kwargs)
    elif name == 'LOFAR':
        location = EarthLocation(x=3826923.9 * u.m, y=460915.1 * u.m, z=5064643.2 * u.m)
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        assert get_parameter(kwargs, "meta", False) is False
        fc = create_configuration_from_MIDfile(antfile=rascil_data_path("configurations/lofar.cfg"), location=location,
                                               mount="XY", vp_type="LOFAR", name=name, **kwargs)
    elif name == 'VLAA':
        location = EarthLocation(lon=-107.6184*u.deg, lat=34.0784*u.deg, height=2124.0)
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_data_path("configurations/vlaa_local.csv"),
                                            location=location,
                                            mount='AZEL',
                                            names='VLA_%d',
                                            vp_type="VLA",
                                            diameter=25.0,
                                            name=name, ecef=False,  **kwargs)
    elif name == 'VLAA_north':
        location = EarthLocation(lon=-107.6184*u.deg, lat=90.000*u.deg, height=0.0)
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_data_path("configurations/vlaa_local.csv"),
                                            location=location,
                                            mount='AZEL',
                                            names='VLA_%d',
                                            vp_type="VLA",
                                            diameter=25.0,
                                            name=name, ecef=False, **kwargs)
    elif name == 'LLA':
        location = low_location
        log.debug("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_LLAfile(antfile=rascil_data_path("configurations/LOW_SKA-TEL-SKO-0000422_Rev3.txt"),
                                            location=location, mount='XY', names='LOW_%d',
                                            vp_type="LOW",diameter=38.0, alt=300.0, 
                                            name=name, ecef=True, **kwargs)

    else:
        raise ValueError("No such Configuration %s" % name)
    return fc

def select_configuration(config, names=None):
    """ Select a subset of antennas/names
    
    :param config:
    :param names:
    :return:
    """
    
    if names is None:
        return config
    
    ind = []
    for iname, name in enumerate(config.names):
        if name in names:
            ind.append(iname)
            
    assert len(ind) > 0, "No antennas selected using names {}".format(names)
    
    fc = Configuration(location=config.location,
                       names=config.names[ind],
                       mount=config.mount[ind],
                       xyz=config.xyz[ind],
                       vp_type=config.vp_type[ind],
                       diameter=config.diameter[ind],
                       name=config.name)
    return fc
