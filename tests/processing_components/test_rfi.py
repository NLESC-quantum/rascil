""" Unit tests for RFI simulation

"""

import logging
import unittest

import astropy.units as u
import numpy
import numpy.testing
from astropy.coordinates import SkyCoord, EarthLocation

from rascil.data_models import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation.rfi import create_propagators, calculate_rfi_at_station, \
    calculate_station_correlation_rfi, simulate_DTV_prop, create_propagators_prop
from rascil.processing_components.visibility.base import create_blockvisibility

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)


class TestRFISim(unittest.TestCase):
    def setUp(self):
        """ Initialise common elements

        """
        # Set the random number so that we lways get the same answers
        numpy.random.seed(1805550721)
        
        pass
    
    def test_rfi_correlation(self):
        """ Calculate the value of the correlated RFI using nominal attenuation, check for regression """
        sample_freq = 3e4
        nchannels = 1000
        frequency = 170.5e6 + numpy.arange(nchannels) * sample_freq
        
        ntimes = 100
        integration_time = 0.5
        times = numpy.arange(ntimes) * integration_time
        
        # Perth from Google for the moment
        perth = EarthLocation(lon=115.8605 * u.deg, lat=-31.9505 * u.deg, height=0.0)
        
        rmax = 1000.0
        low = create_named_configuration('LOWR3', rmax=rmax, skip=33)
        nants = len(low.names)
        
        # Info. for dummy BlockVisibility
        ftimes = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        ffrequency = numpy.linspace(0.8e8, 1.2e8, 5)
        channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
        polarisation_frame = PolarisationFrame("linear")
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        bvis = create_blockvisibility(low, ftimes, ffrequency, channel_bandwidth=channel_bandwidth,
                                      polarisation_frame=polarisation_frame, phasecentre=phasecentre, weight=1.0)
        
        # Calculate the power spectral density of the DTV station: Watts/Hz
        emitter, channel_range = simulate_DTV_prop(frequency, times, power=50e3, timevariable=False)
        
        # Calculate the propagators for signals from Perth to the stations in low
        # These are fixed in time but vary with frequency. The ad hoc attenuation
        # is set to produce signal roughly equal to noise at LOW
        attenuation = 1e-5
        propagators = create_propagators(low, perth, frequency=frequency, attenuation=attenuation)
        assert propagators.shape == (nants, nchannels), propagators.shape
        
        # Now calculate the RFI at the stations, based on the emitter and the propagators
        rfi_at_station = calculate_rfi_at_station(propagators, emitter)
        assert rfi_at_station.shape == (ntimes, nants, nchannels), rfi_at_station.shape
        
        # Calculate the rfi correlation. The return value is in Jy
        correlation = calculate_station_correlation_rfi(rfi_at_station, bvis.baselines)
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(correlation)), 10.143272111897195)
        assert correlation.shape == (ntimes, len(bvis.baselines), nchannels, 1), correlation.shape
    
    def test_rfi_propagators(self):
        """ Calculate the value of the propagator from Perth to array, check for regression """
        sample_freq = 3e4
        nchannels = 1000
        frequency = 170.5e6 + numpy.arange(nchannels) * sample_freq
        
        ntimes = 100
        integration_time = 0.5
        times = numpy.arange(ntimes) * integration_time
        
        rmax = 1000.0
        antskip = 33
        low = create_named_configuration('LOWR3', rmax=rmax, skip=antskip)
        nants_start = len(low.names)
        nants = len(low.names)
        
        # Perth transmitter
        tx_name = 'Perth'
        transmitter_dict = {'location': [115.8605, -31.9505], 'power': 50000.0, 'height': 175, 'freq': 177.5,
                            'bw': 7}
        
        # Calculate the power spectral density of the DTV station using the transmitter bandwidth and frequency
        # : Watts/Hz
        emitter, channel_range = simulate_DTV_prop(frequency, times, power=transmitter_dict['power'],
                                                   freq_cen=transmitter_dict['freq'] * 1e6,
                                                   bw=transmitter_dict['bw'] * 1e6,
                                                   timevariable=False, frequency_variable=False)
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(emitter)), 0.05980416299307147)
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (ntimes, nchannels)
        
        # Calculate the propagators for signals from Perth to the stations in low
        # These are fixed in time but vary with frequency. The attenuation and beam gain
        # are set per frequency channel covered by the transmitter bandwidth.
        
        propagators = create_propagators_prop(low, frequency, nants_start, station_skip=antskip, attenuation=1e-9,
                                              beamgainval=1e-8, trans_range=channel_range)
        assert propagators.shape == (nants, nchannels), propagators.shape
    
    def test_simulate_dtv_prop(self):
        """
        simulate_DTV_prop correctly calculates the power spectral density [W/Hz] of
        the DTV station in Perth using the transmitter bandwidth and frequency
    
        Do all timevariable, frequency_variable cases
        """
        sample_freq = 3e4
        nchannels = 1000
        frequency = 170.5e6 + numpy.arange(nchannels) * sample_freq
        
        ntimes = 100
        integration_time = 0.5
        times = numpy.arange(ntimes) * integration_time
        
        # Perth transmitter
        transmitter_dict = {'location': [115.8605, -31.9505],
                            'power': 50000.0, 'height': 175,
                            'freq': 177.5, 'bw': 7}
        
        emitter, channel_range = simulate_DTV_prop(frequency, times, power=transmitter_dict['power'],
                                                   freq_cen=transmitter_dict['freq'] * 1e6,
                                                   bw=transmitter_dict['bw'] * 1e6,
                                                   timevariable=False, frequency_variable=False)
        
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(emitter)), 0.05980416299307147)
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (ntimes, nchannels)
        
        emitter, channel_range = simulate_DTV_prop(frequency, times, power=transmitter_dict['power'],
                                                   freq_cen=transmitter_dict['freq'] * 1e6,
                                                   bw=transmitter_dict['bw'] * 1e6,
                                                   timevariable=False, frequency_variable=True)
        
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(emitter)), 0.20291900444429292)
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (ntimes, nchannels)
        
        emitter, channel_range = simulate_DTV_prop(frequency, times, power=transmitter_dict['power'],
                                                   freq_cen=transmitter_dict['freq'] * 1e6,
                                                   bw=transmitter_dict['bw'] * 1e6,
                                                   timevariable=True, frequency_variable=False)
        
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(emitter)), 0.15779614478384324)
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (ntimes, nchannels)
        
        emitter, channel_range = simulate_DTV_prop(frequency, times, power=transmitter_dict['power'],
                                                   freq_cen=transmitter_dict['freq'] * 1e6,
                                                   bw=transmitter_dict['bw'] * 1e6,
                                                   timevariable=True, frequency_variable=True)
        
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(emitter)), 0.28001147137009325)
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (ntimes, nchannels)
    
    def test_create_propagators_prop(self):
        """
        Calculate the propagators for signals from Perth to the stations in low
        These are fixed in time but vary with frequency. The attenuation and beam gain
        are set per frequency channel covered by the transmitter bandwidth.
        """
        sample_freq = 3e4
        nchannels = 1000
        frequency = 170.5e6 + numpy.arange(nchannels) * sample_freq
        
        rmax = 1000.0
        antskip = 33
        low = create_named_configuration('LOWR3', rmax=rmax, skip=antskip)
        
        nants = len(low.names)
        
        channel_range = (117, 350)
        
        propagators = create_propagators_prop(low, frequency, nants, station_skip=antskip, attenuation=1e-9,
                                              beamgainval=1e-8, trans_range=channel_range)
        
        assert propagators.shape == (nants, nchannels)
        
        # if we don't want the exact value, because it's hard to calculate
        # maybe we can test that there are values different than 1.0 in the array,
        # which were supposed to be added by the function
        assert numpy.abs(propagators).min() != 1.0
        assert numpy.abs(propagators).max() == 1.0
        
        assert round(numpy.abs(propagators).min(), 16) == 3.1622777e-9


if __name__ == '__main__':
    unittest.main()
