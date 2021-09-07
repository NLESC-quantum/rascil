""" geometry

"""

__all__ = [
    "calculate_transit_time",
    "calculate_hourangles",
    "calculate_parallactic_angles",
    "calculate_azel",
    "utc_to_ms_epoch",
]

import logging

from astroplan import Observer

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Angle

log = logging.getLogger("rascil-logger")


def calculate_parallactic_angles(location, utc_time, direction):
    """Return hour angles for location, utc_time, and direction

    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: Angle
    """

    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)

    site = Observer(location=location)
    return site.target_hour_angle(utc_time, direction).wrap_at("180d")


def calculate_hourangles(location, utc_time, direction):
    """Return hour angles for location, utc_time, and direction

    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: Angle
    """

    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)

    site = Observer(location=location)
    return site.target_hour_angle(utc_time, direction).wrap_at("180d")


def calculate_transit_time(location, utc_time, direction, fraction_day=1e-7):
    """Find the UTC time of the nearest transit

    :param fraction_day: Step in this fraction of day to find transit
    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: astropy Time
    """
    site = Observer(location)
    return site.target_meridian_transit_time(
        utc_time, direction, which="next", n_grid_points=100
    )


def calculate_azel(location, utc_time, direction):
    """Return az el for a location, utc_time, and direction

    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: astropy Angle, Angle
    """
    site = Observer(location=location)
    altaz = site.altaz(utc_time, direction)
    return altaz.az.wrap_at("180d"), altaz.alt


def utc_to_ms_epoch(ts):
    """Convert an timestamp to seconds (epoch values)
        epoch suitable for using in a Measurement Set

    :param ts:  An astropy Time object.
    :result: The epoch time ``t`` in seconds suitable for fields in measurement sets.
    """
    # Use astropy Time
    epoch_s = ts.mjd * 24 * 60 * 60.0
    return epoch_s
