""" Unit tests for RFI simulation

"""

import logging
import unittest
from unittest.mock import patch

import astropy.units as u
import numpy
import numpy.testing
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation.rfi import (
    apply_beam_gain_for_low,
    calculate_station_correlation_rfi,
    simulate_rfi_block_prop,
    match_frequencies,
)
from rascil.processing_components.visibility.base import create_blockvisibility

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)


class TestRFISim(unittest.TestCase):
    def setUp(self):
        pass

    def setup_telescope(self, telescope):
        """Initialise common elements"""
        # Set the random number so that we lways get the same answers
        numpy.random.seed(1805550721)

        self.nchannels = 1000

        integration_time = 0.5
        self.ntimes = 100

        rmax = 1000.0
        antskip = 33
        self.configuration = create_named_configuration(
            telescope, rmax=rmax, skip=antskip
        )
        self.nants = len(self.configuration.names)

        self.apparent_power = numpy.ones((self.ntimes, self.nants, self.nchannels))

        # Info. for dummy BlockVisibility
        ftimes = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        if telescope == "MID":
            ffrequency = numpy.linspace(1.4e9, 1.9e9, 5)
            channel_bandwidth = numpy.array([1e8, 1e8, 1e8, 1e8, 1e8])
        else:
            ffrequency = numpy.linspace(1.3, 1.5e8, 5)
            channel_bandwidth = numpy.array([4e6, 4e6, 4e6, 4e6, 4e6])

        polarisation_frame = PolarisationFrame("linear")
        # Set the phasecentre so as to point roughly towards Perth at transit
        phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-62.8 * u.deg, frame="icrs", equinox="J2000"
        )
        self.bvis = create_blockvisibility(
            self.configuration,
            ftimes,
            ffrequency,
            channel_bandwidth=channel_bandwidth,
            polarisation_frame=polarisation_frame,
            phasecentre=phasecentre,
            weight=1.0,
        )

    def test_match_frequencies_one_to_one_match(self):
        """
        One RFI channel matches one bvis channel, and there are two bvis channels,
        which will contain zero RFI signal.
        """
        rfi_signal = numpy.ones((2, 3, 4))  # 4 channels
        rfi_chans = numpy.linspace(200, 800, rfi_signal.shape[-1])
        bvis_chans = numpy.linspace(10, 1000, 6)  # 6 channels
        bvis_bandwidth = numpy.ones(6) * 20.0

        result = match_frequencies(rfi_signal, rfi_chans, bvis_chans, bvis_bandwidth)

        assert result.shape[-1] == len(bvis_chans)

        # no signal in the first and last bvis channels
        assert (result[:, :, 0] == result[:, :, -1]).all()
        assert (result[:, :, 0] == 0.0).all()

        # signal in the middle four bvis channels
        for i in range(1, 5):
            assert (result[:, :, i] == 1.0).all()

    def test_match_frequencies_multi_rfi_in_single_bvis_chan(self):
        """
        Multiple RFI channels match a single bvis channel.
        Use the median value.
        """
        self.setup_telescope("LOW")
        rfi_signal = self.apparent_power[:, :, :-10]
        # these will provide data for bvis chan 3
        rfi_signal[0, 0, [28, 29, 30, 31, 32]] = numpy.array([2.0, 2.0, 4.0, 5.0, 5.0])
        rfi_signal[1, 1, [28, 29, 30, 31, 32]] = numpy.array([1.0, 2.0, 5.0, 5.0, 5.0])
        # these will provide data for bvis chan 4
        rfi_signal[:, :, [38, 39, 40, 41, 42]] = numpy.array([1.0, 3.0, 6.0, 7.0, 8.0])

        rfi_chans = numpy.linspace(10, 1000, rfi_signal.shape[-1])
        bvis_chans = numpy.linspace(10, 10000, self.nchannels)
        bvis_bandwidth = numpy.ones(self.nchannels) * 5.0

        result = match_frequencies(rfi_signal, rfi_chans, bvis_chans, bvis_bandwidth)

        assert result.shape[-1] == len(bvis_chans)
        assert result[0, 0, 3] == 4.0
        assert result[1, 1, 3] == 5.0
        assert (result[:, :, 4] == 6.0).all()

    def test_calculate_rfi_at_station_single_beam_gain(self):
        self.setup_telescope("LOW")

        apparent_power = numpy.ones((2, 3, 4))  # 2 times, 3 antennas, 4 channels
        beam_gain_value = 9
        beam_gain_ctx = "bg_value"

        result = apply_beam_gain_for_low(
            apparent_power, beam_gain_value, beam_gain_ctx, None
        )
        # function multiplies the given apparent power with the sqrt of beam gain
        assert (result == apparent_power * 3).all()

    @patch("rascil.processing_components.simulation.rfi.numpy.loadtxt")
    def test_calculate_rfi_at_station_beam_gain_array(self, mock_load):
        self.setup_telescope("LOW")
        apparent_power = numpy.ones((2, 3, 4))  # 2 times, 3 antennas, 4 channels
        beam_gain_value = "some-file"
        beam_gain_ctx = "bg_file"
        # file contains a value of beam gain per channel
        mock_load.return_value = numpy.array([9, 4, 16, 25])

        result = apply_beam_gain_for_low(
            apparent_power, beam_gain_value, beam_gain_ctx, None
        )
        # function multiplies the given apparent power with the sqrt of beam gain
        assert (result[:, :, 0] == apparent_power[:, :, 0] * 3).all()
        assert (result[:, :, 1] == apparent_power[:, :, 1] * 2).all()
        assert (result[:, :, 2] == apparent_power[:, :, 2] * 4).all()
        assert (result[:, :, 3] == apparent_power[:, :, 3] * 5).all()

    def test_rfi_correlation(self):
        self.setup_telescope("LOW")
        """Calculate the value of the correlated RFI using nominal emitter power, check for regression"""
        apparent_power = self.apparent_power * 1.0e-10
        beam_gain_value = 3.0e-8
        beam_gain_ctx = "bg_value"
        # Now calculate the RFI at the stations, based on the emitter and the propagators
        rfi_at_station = apply_beam_gain_for_low(
            apparent_power, beam_gain_value, beam_gain_ctx, None
        )
        assert rfi_at_station.shape == (
            self.ntimes,
            self.nants,
            self.nchannels,
        ), rfi_at_station.shape

        # Calculate the rfi correlation. The return value is in Jy
        correlation = calculate_station_correlation_rfi(
            rfi_at_station, self.bvis.baselines
        )
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(correlation)), 0.03)
        assert correlation.shape == (
            self.ntimes,
            len(self.bvis.baselines),
            self.nchannels,
            1,
        ), correlation.shape

    def test_simulate_rfi_block_prop_use_pol(self):
        """
        regression to test that simulate_rfi_block_prop correctly updates the
        block visibility data with RFI signal.

        RFI signal is for the same frequency channels as the BlockVisibility has
        """
        self.setup_telescope("MID")
        nants_start = self.nants
        bvis = self.bvis.copy()

        emitter_power = numpy.zeros(
            (1, len(bvis.time), nants_start, len(bvis.frequency)), dtype=complex
        )  # one source
        # only add signal to the 4th and 5th channels (for testing purposes)
        emitter_power[:, :, :, 3] = 1.0e-10
        emitter_power[:, :, :, 4] = 5.0e-10
        emitter_coordinates = numpy.ones(
            (1, len(bvis.time), nants_start, 3),
        )
        # azimuth, elevation, distance
        emitter_coordinates[:, :, :, 0] = -170.0
        emitter_coordinates[:, :, :, 1] = 0.03
        emitter_coordinates[:, :, :, 2] = 600000.0

        starting_visibility = bvis["vis"].data.copy()

        simulate_rfi_block_prop(
            bvis,
            emitter_power,
            emitter_coordinates,
            ["source1"],
            bvis.frequency.values,
            beam_gain_state=None,
            use_pole=False,
        )

        # original block visibility doesn't have any signal in it
        # len(bvis["frequency"]) => 5
        for i in range(5):
            assert (starting_visibility[:, :, i, :] == 0).all()

        # rfi signal is expected in the 4th and 5th channels (index 3, 4)
        assert (bvis["vis"].data[:, :, 3, :] != 0).all()
        assert (bvis["vis"].data[:, :, 4, :] != 0).all()
        # assert (abs(bvis["vis"].data[:, :, 3, 0] / 1e6).round(1) == 1.0).all()
        # assert (abs(bvis["vis"].data[:, :, 4, 0] / 2.5e7).round(1) == 1.0).all()


if __name__ == "__main__":
    unittest.main()
