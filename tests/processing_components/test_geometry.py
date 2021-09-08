""" Unit tests for coordinate calculations

"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from rascil.processing_components.util.geometry import (
    calculate_azel,
    calculate_hourangles,
    calculate_transit_time,
    calculate_parallactic_angles,
    utc_to_ms_epoch,
)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.location = EarthLocation(
            lon=116.76444824 * u.deg, lat=-26.824722084 * u.deg, height=300.0
        )
        self.times = (numpy.pi / 43200.0) * numpy.arange(-43200, +43200, 3600.0)
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.utc_time = Time(["2020-01-01T00:00:00"], format="isot", scale="utc")

    def test_azel(self):
        utc_times = Time(
            numpy.arange(0.0, 1.0, 0.1) + self.utc_time.mjd, format="mjd", scale="utc"
        )
        azel = calculate_azel(self.location, utc_times, self.phasecentre)
        numpy.testing.assert_array_almost_equal(azel[0][0].deg, -113.964241)
        numpy.testing.assert_array_almost_equal(azel[1][0].deg, 57.715754)
        numpy.testing.assert_array_almost_equal(azel[0][-1].deg, -171.470433)
        numpy.testing.assert_array_almost_equal(azel[1][-1].deg, 81.617363)

    def test_hourangles(self):
        ha = calculate_hourangles(self.location, self.utc_time, self.phasecentre)
        numpy.testing.assert_array_almost_equal(ha[0].deg, 36.881315)

    def test_parallacticangles(self):
        pa = calculate_parallactic_angles(
            self.location, self.utc_time, self.phasecentre
        )
        numpy.testing.assert_array_almost_equal(pa[0].deg, 85.756057)

    def test_transit_time(self):
        transit_time = calculate_transit_time(
            self.location, self.utc_time, self.phasecentre
        )
        numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895866)

    def test_transit_time_below_horizon(self):
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=+80.0 * u.deg, frame="icrs", equinox="J2000"
        )
        transit_time = calculate_transit_time(
            self.location, self.utc_time, self.phasecentre
        )
        numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895804)

    def test_utc_to_ms_epoch(self):
        utc_time = Time("2020-01-01T00:00:00", format="isot", scale="utc")
        ms_epoch = utc_to_ms_epoch(utc_time)
        numpy.testing.assert_array_almost_equal(ms_epoch, 5084553600.0)


if __name__ == "__main__":
    unittest.main()
