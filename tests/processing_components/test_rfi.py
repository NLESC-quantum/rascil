""" Unit tests for RFI simulation

"""

import logging
import math
import unittest
from unittest.mock import patch

import astropy.units as u
import numpy
import numpy.testing
from astropy.coordinates import SkyCoord, EarthLocation

from rascil.data_models import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation.rfi import (
    create_propagators,
    calculate_rfi_at_station,
    calculate_station_correlation_rfi,
    simulate_DTV_prop,
    create_propagators_prop,
    simulate_rfi_block_prop,
)
from rascil.processing_components.visibility.base import create_blockvisibility

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)


class TestRFISim(unittest.TestCase):
    def setUp(self):
        """Initialise common elements"""
        # Set the random number so that we lways get the same answers
        numpy.random.seed(1805550721)

        sample_freq = 3e4
        self.nchannels = 1000
        self.frequency = 170.5e6 + numpy.arange(self.nchannels) * sample_freq

        integration_time = 0.5
        self.ntimes = 100
        self.times = numpy.arange(self.ntimes) * integration_time

        rmax = 1000.0
        self.antskip = 33
        self.low = create_named_configuration("LOWR3", rmax=rmax, skip=self.antskip)
        self.nants = len(self.low.names)

        # Perth transmitter
        self.transmitter_dict = {
            "location": [115.8605, -31.9505],
            "power": 50000.0,
            "height": 175,
            "freq": 177.5,
            "bw": 7,
        }

        # Info. for dummy BlockVisibility
        ftimes = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        ffrequency = numpy.linspace(1.4e8, 1.9e8, 5)
        channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
        polarisation_frame = PolarisationFrame("linear")
        phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.bvis = create_blockvisibility(
            self.low,
            ftimes,
            ffrequency,
            channel_bandwidth=channel_bandwidth,
            polarisation_frame=polarisation_frame,
            phasecentre=phasecentre,
            weight=1.0,
        )

    def test_rfi_correlation(self):
        """Calculate the value of the correlated RFI using nominal attenuation, check for regression"""
        # Perth from Google for the moment
        perth = EarthLocation(lon=115.8605 * u.deg, lat=-31.9505 * u.deg, height=0.0)

        # Calculate the power spectral density of the DTV station: Watts/Hz
        emitter, channel_range = simulate_DTV_prop(
            self.frequency, self.times, power=50e3, time_variable=False
        )

        # Calculate the propagators for signals from Perth to the stations in low
        # These are fixed in time but vary with frequency. The ad hoc attenuation
        # is set to produce signal roughly equal to noise at LOW
        attenuation = 1e-5
        propagators = create_propagators(
            self.low, perth, frequency=self.frequency, attenuation=attenuation
        )
        assert propagators.shape == (self.nants, self.nchannels), propagators.shape

        # Now calculate the RFI at the stations, based on the emitter and the propagators
        rfi_at_station = calculate_rfi_at_station(propagators, emitter)
        assert rfi_at_station.shape == (
            self.ntimes,
            self.nants,
            self.nchannels,
        ), rfi_at_station.shape

        # Calculate the rfi correlation. The return value is in Jy
        correlation = calculate_station_correlation_rfi(
            rfi_at_station, self.bvis.baselines
        )
        numpy.testing.assert_almost_equal(
            numpy.max(numpy.abs(correlation)), 0.8810092775005085
        )
        assert correlation.shape == (
            self.ntimes,
            len(self.bvis.baselines),
            self.nchannels,
            1,
        ), correlation.shape

    def test_rfi_propagators(self):
        """Calculate the value of the propagator from Perth to array, check for regression"""
        nants_start = len(self.low.names)

        # Calculate the power spectral density of the DTV station using the transmitter bandwidth and frequency
        # : Watts/Hz
        emitter, channel_range = simulate_DTV_prop(
            self.frequency,
            self.times,
            power=self.transmitter_dict["power"],
            freq_cen=self.transmitter_dict["freq"] * 1e6,
            bw=self.transmitter_dict["bw"] * 1e6,
            time_variable=False,
            frequency_variable=False,
        )
        numpy.testing.assert_almost_equal(
            numpy.max(numpy.abs(emitter)), 0.05980416299307147
        )
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (self.ntimes, self.nchannels)

        # Calculate the propagators for signals from Perth to the stations in low
        # These are fixed in time but vary with frequency. The attenuation and beam gain
        # are set per frequency channel covered by the transmitter bandwidth.

        propagators = create_propagators_prop(
            self.low,
            self.frequency,
            nants_start,
            station_skip=self.antskip,
            attenuation=1e-9,
            beamgainval=1e-8,
            trans_range=channel_range,
        )
        assert propagators.shape == (self.nants, self.nchannels), propagators.shape

    def test_simulate_dtv_prop(self):
        """
        simulate_DTV_prop correctly calculates the power spectral density [W/Hz] of
        the DTV station in Perth using the transmitter bandwidth and frequency

        Do all time_variable, frequency_variable cases
        """
        emitter, channel_range = simulate_DTV_prop(
            self.frequency,
            self.times,
            power=self.transmitter_dict["power"],
            freq_cen=self.transmitter_dict["freq"] * 1e6,
            bw=self.transmitter_dict["bw"] * 1e6,
            time_variable=False,
            frequency_variable=False,
        )

        numpy.testing.assert_almost_equal(
            numpy.max(numpy.abs(emitter)), 0.05980416299307147
        )
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (self.ntimes, self.nchannels)

        emitter, channel_range = simulate_DTV_prop(
            self.frequency,
            self.times,
            power=self.transmitter_dict["power"],
            freq_cen=self.transmitter_dict["freq"] * 1e6,
            bw=self.transmitter_dict["bw"] * 1e6,
            time_variable=False,
            frequency_variable=True,
        )

        numpy.testing.assert_almost_equal(
            numpy.max(numpy.abs(emitter)), 0.20291900444429292
        )
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (self.ntimes, self.nchannels)

        emitter, channel_range = simulate_DTV_prop(
            self.frequency,
            self.times,
            power=self.transmitter_dict["power"],
            freq_cen=self.transmitter_dict["freq"] * 1e6,
            bw=self.transmitter_dict["bw"] * 1e6,
            time_variable=True,
            frequency_variable=False,
        )

        numpy.testing.assert_almost_equal(
            numpy.max(numpy.abs(emitter)), 0.15779614478384324
        )
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (self.ntimes, self.nchannels)

        emitter, channel_range = simulate_DTV_prop(
            self.frequency,
            self.times,
            power=self.transmitter_dict["power"],
            freq_cen=self.transmitter_dict["freq"] * 1e6,
            bw=self.transmitter_dict["bw"] * 1e6,
            time_variable=True,
            frequency_variable=True,
        )

        numpy.testing.assert_almost_equal(
            numpy.max(numpy.abs(emitter)), 0.28001147137009325
        )
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (self.ntimes, self.nchannels)

    def test_create_propagators_prop(self):
        """
        Calculate the propagators for signals from Perth to the stations in low
        These are fixed in time but vary with frequency. The attenuation and beam gain
        are set per frequency channel covered by the transmitter bandwidth.
        """
        channel_range = (117, 350)

        propagators = create_propagators_prop(
            self.low,
            self.frequency,
            self.nants,
            station_skip=self.antskip,
            attenuation=1e-9,
            beamgainval=1e-8,
            trans_range=channel_range,
        )

        assert propagators.shape == (self.nants, self.nchannels)

        # if we don't want the exact value, because it's hard to calculate
        # maybe we can test that there are values different than 1.0 in the array,
        # which were supposed to be added by the function
        assert numpy.abs(propagators).min() != 1.0
        assert numpy.abs(propagators).max() == 1.0

        assert round(numpy.abs(propagators).min(), 16) == 3.1622777e-9

    @patch("rascil.processing_components.simulation.rfi.get_file_strings")
    def test_simulate_rfi_block_prop(self, mock_get_file_string): #
        """
        regression to test that simulate_rfi_block_prop correctly updates the
        block visibility data with RFI signal using the default
        transmitter/beamgain/attenuation information
        """
        nants_start = self.nants
        bvis = self.bvis.copy()

        rfi_at_station = numpy.zeros((len(bvis.time), nants_start, len(bvis.frequency)), dtype=complex)
        rfi_at_station[:, :, 3] += 0.044721359549995794
        rfi_at_station[:, :, 4] += 0.044721359549995794
        mock_get_file_string.return_value = (rfi_at_station, 1.0)

        starting_visibility = bvis["vis"].data.copy()

        simulate_rfi_block_prop(
            bvis,
            nants_start,
            self.antskip,
            attenuation_state=None,
            beamgain_state=None,
            use_pole=False,
            transmitter_list=None,
            frequency_variable=False,
            time_variable=False,
        )

        # original block visibility doesn't have any signal in it
        # len(bvis["frequency"]) => 5
        for i in range(5):
            assert (starting_visibility[:, :, i, :] == 0).all()

        # rfi signal is expected in the 4th and 5th channels (index 3, 4)
        for i in range(3):
            assert (bvis["vis"].data[:, :, i, :] == 0).all()

        assert (bvis["vis"].data[:, :, 3, :] != 0).all()
        assert (bvis["vis"].data[:, :, 4, :] != 0).all()

        # checking some values to make sure results don't change
        # index: time=0, baseline=1, channel=3, pol=all
        assert (abs(bvis["vis"].data[0, 1, 3, :]) == 200000000000000016777216.00).all()
        # index: time=0, baseline=2, channel=3, pol=all
        assert (abs(bvis["vis"].data[0, 2, 3, :]) == 199999999999999983222784.00).all()
        # index: time=4, baseline=4, channel=4, pol=all
        assert (abs(bvis["vis"].data[4, 4, 4, :]) == 199999999999999983222784.00).all()


if __name__ == "__main__":
    unittest.main()
