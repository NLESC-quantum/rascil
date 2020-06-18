""" geometry

"""

__all__ = ['calculate_transit_time', 'calculate_hourangles', 'calculate_parallactic_angles',
           'calculate_azel']

import logging

import numpy

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Angle

log = logging.getLogger('logger')


def angle_to_quanta(angle):
    return {"value": angle.rad, "unit": "rad"}


def calculate_parallactic_angles(location, utc_time, direction):
    """ Return hour angles for location, utc_time, and direction

    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: Angle
    """
    
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)
    
    from astroplan import Observer
    site = Observer(location=location)
    return site.target_hour_angle(utc_time, direction).wrap_at('180d')


def calculate_hourangles(location, utc_time, direction):
    """ Return hour angles for location, utc_time, and direction

    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: Angle
    """
    
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)
    
    
    from astroplan import Observer
    site = Observer(location=location)
    return site.target_hour_angle(utc_time, direction).wrap_at('180d')


def calculate_transit_time(location, utc_time, direction, fraction_day=0.01):
    """ Find the UTC time of the nearest transit

    :param fraction_day: Step in this fraction of day to find transit
    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: astropy Time
    """
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)

    from astroplan import Observer
    site = Observer(location)
    return site.target_meridian_transit_time(utc_time, direction, which="next", n_grid_points=100)


def calculate_azel(location, utc_time, direction):
    """ Return az el for a location, utc_time, and direction

    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: astropy Angle, Angle
    """
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)


    from astroplan import Observer
    site = Observer(location=location)
    altaz = site.altaz(utc_time, direction)
    return altaz.az.wrap_at('180d'), altaz.alt
