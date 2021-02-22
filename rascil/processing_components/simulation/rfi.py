"""Functions used to simulate RFI. Developed as part of SP-122/SIM.

The scenario is:
* There is a TV station at a remote location (e.g. Perth), emitting a broadband signal (7MHz) of known power (50kW).
* The emission from the TV station arrives at LOW stations with phase delay and attenuation. Neither of these are
well known but they are probably static.
* The RFI enters LOW stations in a side-lobe of the main beam. Calculations by Fred Dulwich indicate that this
provides attenuation of about 55 - 60dB for a source close to the horizon.
* The RFI enters each LOW station with fixed delay and zero fringe rate (assuming no e.g. ionospheric ducting)
* In tracking a source on the sky, the signal from one station is delayed and fringe-rotated to stop the fringes for
one direction on the sky.
* The fringe rotation stops the fringe from a source at the phase tracking centre but phase rotates the RFI, which
now becomes time-variable.
* The correlation data are time- and frequency-averaged over a timescale appropriate for the station field of view.
This averaging de-correlates the RFI signal.
* We want to study the effects of this RFI on statistics of the images: on source and at the pole.
"""

__all__ = ['simulate_DTV_prop', 'create_propagators', 'create_propagators_prop',
           'calculate_averaged_correlation', 'calculate_rfi_at_station',
           'simulate_rfi_block_prop', 'calculate_station_correlation_rfi']

import copy
import logging

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord, EarthLocation

from rascil import phyconst
from rascil.processing_components.util.array_functions import average_chunks2
from rascil.processing_components.util.compass_bearing import calculate_initial_compass_bearing
from rascil.processing_components.util.coordinate_support import simulate_point
from rascil.processing_components.util.coordinate_support import skycoord_to_lmn, azel_to_hadec
from rascil.processing_components.visibility.visibility_geometry import calculate_blockvisibility_hourangles

log = logging.getLogger("rascil-logger")

def simulate_DTV_prop(frequency, times, power=50e3, freq_cen=177.5e06, bw=7e06, time_variable=False,
                      frequency_variable=False):
    """ Calculate DTV sqrt(power) as a function of time and frequency

    :param frequency: (sample frequencies)
    :param times: sample times (s)
    :param power: DTV emitted power W
    :param freq_cen: central frequency of DTV
    :param bw: bandwidth of DTV
    :param timevariable:
    :param frequency_variable:
    :return: Complex array [ntimes, nchan]
    
    """
    
    # find frequency range for DTV
    DTVrange = numpy.where((frequency <= freq_cen + (bw / 2.)) & (frequency >= freq_cen - (bw / 2.)))
    if len(DTVrange[0]) == 0:
        raise ValueError(f"No overlap between DTV band: centre {freq_cen}Hz, bandwidth"
                         f" {bw}Hz and specified frequency range {frequency}")
    
    # idx = (np.abs(frequency - freq_cen)).argmin()
    # print(frequency, freq_cen, bw, DTVrange)
    echan = DTVrange[0].max()
    bchan = DTVrange[0].min()
    nchan = len(frequency)
    ntimes = len(times)
    shape = [ntimes, nchan]
    # bchan = nchan // 4
    # echan = 3 * nchan // 4
    amp = numpy.sqrt(power / ( 2.0 * (frequency[echan] - frequency[bchan])))

    signal = numpy.zeros(shape, dtype='complex')
    sub_channel_range = (echan + 1) - bchan
    if time_variable:
        if frequency_variable:
            sshape = [ntimes, sub_channel_range]
            signal[:, bchan:echan + 1] += numpy.random.normal(0.0, amp, sshape) \
                                            + 1j * numpy.random.normal(0.0, amp, sshape)
        else:
            sshape = [ntimes, 1]
            signal[:, bchan:echan + 1] += numpy.random.normal(0.0, amp, sshape) \
                                          + 1j * numpy.random.normal(0.0, amp, sshape)
    else:
        if frequency_variable:
            sshape = [1, sub_channel_range]
            signal[:, bchan:echan + 1] += (numpy.random.normal(0.0, amp, sshape)
                                           + 1j * numpy.random.normal(0.0, amp, sshape))
        else:
            signal[:, bchan:echan + 1] = amp

    apower = numpy.std(numpy.abs(signal))
    assert apower > 0.0, apower
    return signal, [bchan, echan]


def create_propagators(config, interferer, frequency, attenuation=1e-9):
    """ Create a set of propagators
    
    :param config: RASCIL Configuration
    :param interferer: EarthLocation for interferer
    :param attenuation: Attenuation of signal
    :return: Complex array [nants, ntimes]
    """
    nchannels = len(frequency)
    nants = len(config['names'].data)
    interferer_xyz = [interferer.geocentric[0].value, interferer.geocentric[1].value, interferer.geocentric[2].value]
    propagators = numpy.zeros([nants, nchannels], dtype='complex')
    for iant, ant_xyz in enumerate(config.xyz.data):
        vec = ant_xyz - interferer_xyz
        # This ignores the Earth!
        r = numpy.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        k = 2.0 * numpy.pi * frequency / phyconst.c_m_s
        propagators[iant, :] = numpy.exp(- 1.0j * k * r) / r
    return propagators * attenuation


def create_propagators_prop(config, frequency, nants_start, station_skip=1, attenuation=1e-9, beamgainval=0.,
                            trans_range=None):
    """ Create a set of propagators
    
    :param config: configuration
    :param frequency: frequencies
    :param attenuation: generic attenuation value to use if no transmitter specified, else filename to load
    :param beamgainval: float generic beam gain value to use if no transmitter specified, else filename to load
    :param nants_start: limiting station to use determined by use of rmax
    :param station_skip: number of stations to skip
    :param trans_range: array start and stop channels for applying the attenuation and beam gain
    :return: Complex array [nants, ntimes]
    """
    
    nchannels = len(frequency)
    nants = len(config['names'])
    propagation = numpy.ones([nants, nchannels], dtype='complex')
    if isinstance(attenuation, str):
        propagation_trans = numpy.power(10, -1 * numpy.load(attenuation) / 10.)
    else:
        propagation_trans = attenuation
    if isinstance(beamgainval, str):
        beamgain_trans = numpy.loadtxt(beamgainval)
    else:
        beamgain_trans = beamgainval
    propagation_trans *= beamgain_trans
    if type(propagation_trans) is numpy.ndarray:
        propagation_trans = propagation_trans[:nants_start]
        propagation_trans = propagation_trans[::station_skip]
    if trans_range is None:
        propagation[...] = propagation_trans
    else:
        # print(propagation.shape, propagation_trans.shape)
        propagation[:, trans_range[0]:trans_range[1] + 1] = propagation_trans
    
    propagation = numpy.sqrt(propagation)
    # print(propagation.type)
    return propagation


def calculate_rfi_at_station(propagators, emitter):
    """ Calculate the rfi at each station
    
    :param propagators: [nstations, nchannels]
    :param emitter: [ntimes, nchannels]
    :return: Complex array [ntimes, nstations, nchannels]
    """
    rfi_at_station = emitter[:, numpy.newaxis, ...] * propagators[numpy.newaxis, ...]
    # rfi_at_station[numpy.abs(rfi_at_station)<1e-15] = 0.
    return rfi_at_station


def calculate_station_correlation_rfi(rfi_at_station, baselines):
    """ Form the correlation from the rfi at the station

    :param rfi_at_station: [btimes, nchan, nants, nants]
    :param baselines: BlockVisibility baselines object
    :return: correlation(ntimes, nbaselines, nchan] in Jy
    """
    ntimes, nants, nchan = rfi_at_station.shape
    correlationt = numpy.zeros([ntimes, nchan, nants, nants], dtype='complex')
    correlationb = numpy.zeros([ntimes, nchan, len(baselines)], dtype='complex')
    rfit = numpy.transpose(rfi_at_station, axes=[0, 2, 1])
    rfitc = numpy.conjugate(rfit)
    for itime in range(ntimes):
        for chan in range(nchan):
            correlationt[itime, chan, ...] = numpy.outer(rfit[itime, chan, :],
                                                         rfitc[itime, chan, :])
            # reshape station axes to baseline for xarray
            for ibaseline, (a1, a2) in enumerate(baselines.data):
                correlationb[itime, chan, ibaseline, ...] = correlationt[itime, chan, a1, a2]
    # correlation = numpy.transpose(correlationt, axes=[0, 2, 3, 1])
    correlation = numpy.transpose(correlationb, axes=[0, 2, 1])
    return correlation[..., numpy.newaxis] * 1e26


def calculate_averaged_correlation(correlation, time_width, channel_width):
    """ Average the correlation in time and frequency
    :param correlation: Correlation(ntimes, nant, nants, nchan]
    :param channel_width: Number of channels to average
    :param time_width: Number of integrations to average
    :return:
    """
    wts = numpy.ones(correlation.shape, dtype='float')
    return average_chunks2(correlation, wts, (time_width, channel_width))[0]


def check_prop_parms(attenuation_state, beamgain_state, transmitter_list):
    """ Extract and check attenuation and beam gain parameters
    
    :param attenuation_state:
    :param beamgain_state:
    :param transmitter_list:
    :return:
    """
    if attenuation_state is None:
        attenuation_value = 1.0
        att_context = 'att_value'
    else:
        attenuation_value = attenuation_state[0]
        att_context = attenuation_state[1]
    if beamgain_state is None:
        beamgain_value = 1.0
        bg_context = 'bg_value'
    else:
        beamgain_value = beamgain_state[0]
        bg_context = beamgain_state[1]
    if not transmitter_list:
        transmitter_list = {'Test_DTV': {'location': [115.8605, -31.9505], 'power': 50000.0, 'height': 175},
                            'freq': 177.5, 'bw': 7}
    
    return attenuation_value, att_context, beamgain_value, bg_context, transmitter_list


def get_file_strings(attenuation_value, att_context, beamgain_value, bg_context, trans):
    if att_context == 'att_dir':
        attenuation = attenuation_value + 'Attenuation_' + trans + '.npy'
    elif att_context == 'att_file':
        attenuation = attenuation_value
    else:
        attenuation = attenuation_value

    if bg_context == 'bg_dir':
        beamgain = beamgain_value + trans + '_beam_gain_TIME_SEP_CHAN_SEP_CROSS_POWER_AMP_I_I.txt'
    elif bg_context == 'bg_file':
        beamgain = beamgain_value
    else:
        beamgain = beamgain_value

    return attenuation, beamgain


def simulate_rfi_block_prop(bvis, nants_start, station_skip, attenuation_state=None,
                            beamgain_state=None, use_pole=False, transmitter_list=None,
                            frequency_variable=False, time_variable=False):
    """ Simulate RFI in a BlockVisility
    
    :param time_variable: Is the signal to be simulated as variable in time?
    :param frequency_variable: Is the signal to be simulated as variable in frequency?
    :param transmitter_list: dictionary of transmitters
    :param beamgain_state: beam gains to apply to the signal or file containing values and flag to declare which
    :param attenuation_state: Attenuation to be applied to signal or file containing values and flag to declare which
    :param use_pole: Set the emitter to nbe at the southern celestial pole
    :return: BlockVisibility
    """
    
    # sort inputs
    attenuation_value, att_context, beamgain_value, bg_context, transmitter_list = check_prop_parms(attenuation_state,
                                                                                                    beamgain_state,
                                                                                                    transmitter_list)
    
    # temporary copy to calculate contribution for each transmitter
    bvis_data_copy = copy.copy(bvis['vis'].data)
    
    for trans in transmitter_list:
        
        # print('Processing transmitter', trans)
        emitter_power = transmitter_list[trans]['power']
        emitter_location = EarthLocation(lon=transmitter_list[trans]['location'][0],
                                         lat=transmitter_list[trans]['location'][1],
                                         height=transmitter_list[trans]['height'])
        emitter_freq = transmitter_list[trans]['freq'] * 1e06
        emitter_bw = transmitter_list[trans]['bw'] * 1e06
        
        attenuation, beamgain = get_file_strings(attenuation_value, att_context, beamgain_value, bg_context, trans)
        
        # Calculate the power spectral density of the DTV station: Watts/Hz
        emitter, DTV_range = simulate_DTV_prop(bvis.frequency, bvis.time,
                                               power=emitter_power, freq_cen=emitter_freq, bw=emitter_bw,
                                               frequency_variable=frequency_variable,
                                               time_variable=time_variable)
        
        # Calculate the propagators for signals from Perth to the stations in low
        # These are fixed in time but vary with frequency. The ad hoc attenuation
        # is set to produce signal roughly equal to noise at LOW
        propagators = create_propagators_prop(bvis.configuration,
                                              bvis.frequency, nants_start=nants_start,
                                              station_skip=station_skip, attenuation=attenuation,
                                              beamgainval=beamgain, trans_range=DTV_range)
        # Now calculate the RFI at the stations, based on the emitter and the propagators
        rfi_at_station = calculate_rfi_at_station(propagators, emitter)
        
        # Calculate the rfi correlation using the fringe rotation and the rfi at the station
        # [ntimes, nants, nants, nchan, npol]
        
        bvis_data_copy[...] = calculate_station_correlation_rfi(rfi_at_station, baselines=bvis.baselines)
        
        ntimes, nbaselines, nchan, npol = bvis.vis.shape
        
        k = numpy.array(bvis.frequency) / phyconst.c_m_s
        uvw = bvis.uvw.data[..., numpy.newaxis] * k
        
        if use_pole:
            # Calculate phasor needed to shift from the phasecentre to the pole
            pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')
            l, m, n = skycoord_to_lmn(pole, bvis.phasecentre)
            phasor = numpy.ones([ntimes, nbaselines, nchan, npol], dtype='complex')
            for chan in range(nchan):
                phasor[:, :, chan, :] = simulate_point(uvw[..., chan], l, m)[..., numpy.newaxis]
            
            # Now add this into the BlockVisibility
            bvis['vis'].data += bvis_data_copy * phasor
        else:
            # We know where the emitter is. Calculate the bearing to the emitter from
            # the site, generate az, el, and convert to ha, dec. ha, dec is static.
            site = bvis.configuration.location
            site_tup = (site.lat.deg, site.lon.deg)
            emitter_tup = (emitter_location.lat.deg, emitter_location.lon.deg)
            # Compass bearing is in range [0,360.0]
            az = calculate_initial_compass_bearing(site_tup, emitter_tup) * numpy.pi / 180.0
            el = 0.0
            hadec = azel_to_hadec(az, el, site.lat.rad)
            r2d = 180.0 / numpy.pi
            log.info(
                f"simulate_rfi_block: Emitter at az, el {az * r2d:.3}, {el * r2d:.3} " + \
                f"appears at ha, dec {hadec[0] * r2d:.3}, {hadec[1] * r2d:.3}")
            # Now step through the time stamps, calculating the effective
            # sky position for the emitter, and performing phase rotation
            # appropriately
            hourangles = calculate_blockvisibility_hourangles(bvis)
            for iha, ha in enumerate(hourangles.data):
                ra = - hadec[0] + ha
                dec = hadec[1]
                emitter_sky = SkyCoord(ra * u.rad, dec * u.rad)
                l, m, n = skycoord_to_lmn(emitter_sky, bvis.phasecentre)
                
                phasor = numpy.ones([nbaselines, nchan, npol], dtype='complex')
                for chan in range(nchan):
                    phasor[:, chan, :] = simulate_point(uvw[iha, ..., chan], l, m)[..., numpy.newaxis]
                
                # Now fill this into the BlockVisibility
                bvis['vis'].data[iha, ...] += bvis_data_copy[iha, ...] * phasor
            

    return bvis
