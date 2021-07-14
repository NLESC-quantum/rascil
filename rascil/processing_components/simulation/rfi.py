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

__all__ = [
    "calculate_averaged_correlation",
    "simulate_rfi_block_prop",
    "calculate_station_correlation_rfi",
]

import copy
import logging

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord, EarthLocation

from rascil import phyconst
from rascil.processing_components.util.array_functions import average_chunks2
from rascil.processing_components.util.compass_bearing import (
    calculate_initial_compass_bearing,
)
from rascil.processing_components.util.coordinate_support import simulate_point
from rascil.processing_components.util.coordinate_support import (
    skycoord_to_lmn,
    azel_to_hadec,
)
from rascil.processing_components.visibility.visibility_geometry import (
    calculate_blockvisibility_hourangles,
)

log = logging.getLogger("rascil-logger")


def calculate_station_correlation_rfi(rfi_at_station, baselines):
    """Form the correlation from the rfi at the station

    :param rfi_at_station: [btimes, nchan, nants, nants]
    :param baselines: BlockVisibility baselines object
    :return: correlation(ntimes, nbaselines, nchan] in Jy
    """
    ntimes, nants, nchan = rfi_at_station.shape
    correlationt = numpy.zeros([ntimes, nchan, nants, nants], dtype="complex")
    correlationb = numpy.zeros([ntimes, nchan, len(baselines)], dtype="complex")
    rfit = numpy.transpose(rfi_at_station, axes=[0, 2, 1])
    rfitc = numpy.conjugate(rfit)
    for itime in range(ntimes):
        for chan in range(nchan):
            correlationt[itime, chan, ...] = numpy.outer(
                rfit[itime, chan, :], rfitc[itime, chan, :]
            )
            # reshape station axes to baseline for xarray
            for ibaseline, (a1, a2) in enumerate(baselines.data):
                correlationb[itime, chan, ibaseline, ...] = correlationt[
                    itime, chan, a1, a2
                ]

    correlation = numpy.transpose(correlationb, axes=[0, 2, 1])
    return correlation[..., numpy.newaxis] * 1e26


def calculate_averaged_correlation(correlation, time_width, channel_width):
    """Average the correlation in time and frequency
    :param correlation: Correlation(ntimes, nant, nants, nchan]
    :param channel_width: Number of channels to average
    :param time_width: Number of integrations to average
    :return:
    """
    wts = numpy.ones(correlation.shape, dtype="float")
    return average_chunks2(correlation, wts, (time_width, channel_width))[0]


def match_frequencies(rfi_signal, rfi_frequencies, bvis_freq_channels, bvis_bandwidth):
    """
    The RFI signal data needs to provide information to all of the frequency
    channels of the block visibility (bvis), which is updated with the RFI signal.

    This function compares the channels for the RFI data and the channels of the bvis,
    and adds 0s where there is no RFI signal in a bvis channel, and adds the signal
    where there is.

    :param rfi_signal: RFI signal, numpy array [ntimes x nantennas x nchannels]
    :param rfi_frequencies: numpy array of the frequency channels of the RFI signal (length == nchannels)
    :param bvis_freq_channels: numpy array of the frequency channels in the block visibility
    :param bvis_bandwidth: numpy array of bandwidths of each freq. channel in the block vis.
    """
    index = []
    for i, chan in enumerate(bvis_freq_channels):
        ind_left = numpy.where(rfi_frequencies >= chan - bvis_bandwidth[i] / 2.0)
        ind_right = numpy.where(rfi_frequencies <= chan + bvis_bandwidth[i] / 2.0)
        index.append(numpy.intersect1d(ind_left, ind_right))

    # append one line of zeros to the channel column of rfi_signal for each channel
    # in frequency_channels that is empty in rfi_frequencies
    to_append = numpy.zeros((rfi_signal.shape[0], rfi_signal.shape[1], 1))

    new_array = to_append.copy()
    for ind in index:
        if ind.size == 0:  # empty array --> need to add zeros at this index
            new_array = numpy.append(new_array, to_append, axis=2)
        elif ind.size > 1:
            new_array = numpy.append(
                new_array,
                numpy.median(rfi_signal[:, :, ind], axis=2, keepdims=True),
                axis=2,
            )
        else:
            new_array = numpy.append(new_array, rfi_signal[:, :, ind], axis=2)

    # there is an extra row at the beginning of the frequency column,
    # which we need to cut off to get the right size
    return new_array[:, :, 1:]


def calculate_rfi_at_station(
    isotropic_emitter_power,
    beam_gain_value,
    beam_gain_context,
    rfi_source,
):
    """
    Apply the beam gain to the apparent emitter power, to determine
    the received RFI signal at SKA station(s).

    :param isotropic_emitter_power: apparent emitter power level as received by an isotropic antenna
                [ntimes x nantennas x nchannels]
    :param beam_gain_value: either a directory or a file containing beam gain values, or an actual beam gain value
    :param beam_gain_context: this specifies the type of `beamgain_value`
    :param rfi_source: name or ID of the RFI source
    """
    if beam_gain_context == "bg_dir":
        # specific to OSKAR-produced data
        beamgain = (
            beam_gain_value
            + rfi_source
            + "_beam_gain_TIME_SEP_CHAN_SEP_CROSS_POWER_AMP_I_I.txt"
        )
    elif beam_gain_context == "bg_file":
        beamgain = beam_gain_value
    else:
        beamgain = beam_gain_value

    if isinstance(beamgain, str):
        beamgain = numpy.loadtxt(beamgain)

    # apply the beamgain to the input apparent emitter power
    rfi_at_station = isotropic_emitter_power.copy() * numpy.sqrt(
        beamgain
    )  # shape of rfi_at_station = [ntimes, nants, n_rfi_channels]

    return rfi_at_station


def simulate_rfi_block_prop(
    bvis,
    apparent_emitter_power,
    apparent_emitter_coordinates,
    rfi_sources,
    rfi_frequencies,
    beam_gain_state=None,
    use_pole=False,
):
    """Simulate RFI in a BlockVisility

    :param bvis: input BlockVisibility, to be updated with RFI
    :param apparent_emitter_power: RFI emitter power as received by an isotropic SKA antenna
                [nrfi_sources x ntimes x nantennas x nchannels]
    :param apparent_emitter_coordinates: azimuth, elevation, distance information of RFI emitters
                [nrfi_sources x ntimes x nantennas x 3]
    :param rfi_sources: RFI source names or IDs
    :param rfi_frequencies: frequency channels where there is RFI information
                length = nchannels
    :param beam_gain_state: tuple of beam gains to apply to the signal or file containing values,
                           and flag to declare which
    :param use_pole: Set the emitter to nbe at the southern celestial pole
    :return: BlockVisibility
    """

    if beam_gain_state is None:
        beamgain_value = 1.0
        bg_context = "bg_value"
    else:
        beamgain_value = beam_gain_state[0]
        bg_context = beam_gain_state[1]

    # temporary copy to calculate contribution for each RFI source
    bvis_data_copy = copy.copy(bvis["vis"].data)
    ntimes, nbaselines, nchan, npol = bvis.vis.shape

    for i, source in enumerate(
        rfi_sources
    ):  # this will tell us the index of the source in the data
        rfi_at_station = calculate_rfi_at_station(
            apparent_emitter_power[i],
            beamgain_value,
            bg_context,
            source,
        )

        rfi_at_station_all_chans = match_frequencies(
            rfi_at_station,
            rfi_frequencies,
            bvis.frequency.values,
            bvis.channel_bandwidth.values,
        )

        # Calculate the RFI correlation using the fringe rotation and the RFI at the station
        # [ntimes, nants, nants, nchan, npol]
        bvis_data_copy[...] = calculate_station_correlation_rfi(
            rfi_at_station_all_chans, baselines=bvis.baselines
        )

        k = numpy.array(bvis.frequency) / phyconst.c_m_s
        uvw = bvis.uvw.data[..., numpy.newaxis] * k

        if use_pole:
            # Calculate phasor needed to shift from the phasecentre to the pole
            pole = SkyCoord(
                ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame="icrs", equinox="J2000"
            )
            l, m, n = skycoord_to_lmn(pole, bvis.phasecentre)
            phasor = numpy.ones([ntimes, nbaselines, nchan, npol], dtype="complex")
            for chan in range(nchan):
                phasor[:, :, chan, :] = simulate_point(uvw[..., chan], l, m)[
                    ..., numpy.newaxis
                ]

            # Now add this into the BlockVisibility
            bvis["vis"].data += bvis_data_copy * phasor

        else:
            # emitter_location = [
            #     EarthLocation(
            #         lon=116.061666666667,
            #         lat=-32.0127777777778,
            #         height=175,
            #     ),
            #     EarthLocation(
            #         lon=114.121111111111,
            #         lat=-21.9186111111111,
            #         height=34,
            #     ),
            # ]
            #
            # site = bvis.configuration.location
            # site_tup = (site.lat.deg, site.lon.deg)
            # emitter_tup = (emitter_location[i].lat.deg, emitter_location[i].lon.deg)
            # # Compass bearing is in range [0,360.0]
            # az = (
            #     calculate_initial_compass_bearing(site_tup, emitter_tup)
            #     * numpy.pi
            #     / 180.0
            # )
            # el = 0.0
            # hadec = azel_to_hadec(az, el, site.lat.rad)
            # r2d = 180.0 / numpy.pi
            # log.info(
            #     f"simulate_rfi_block: Emitter at az, el {az * r2d:.3}, {el * r2d:.3} "
            #     + f"appears at ha, dec {hadec[0] * r2d:.3}, {hadec[1] * r2d:.3}"
            # )
            # # Now step through the time stamps, calculating the effective
            # # sky position for the emitter, and performing phase rotation
            # # appropriately
            # hourangles = calculate_blockvisibility_hourangles(bvis)
            # for iha, ha in enumerate(hourangles.data):
            #     ra = -hadec[0] + ha
            #     dec = hadec[1]
            #     emitter_sky = SkyCoord(ra * u.rad, dec * u.rad)
            #     l, m, n = skycoord_to_lmn(emitter_sky, bvis.phasecentre)
            #
            #     phasor = numpy.ones([nbaselines, nchan, npol], dtype="complex")
            #     for chan in range(nchan):
            #         phasor[:, chan, :] = simulate_point(uvw[iha, ..., chan], l, m)[
            #             ..., numpy.newaxis
            #         ]
            #
            #     # Now fill this into the BlockVisibility
            #     bvis["vis"].data[iha, ...] += bvis_data_copy[iha, ...] * phasor

            # We know where the emitter is.
            site = bvis.configuration.location

            # apparent_emitter_coordinates [nsource x ntimes x nantennas x [az,el,dist]
            # azimuth is index [:, :, :, 0]; elevation is index [:, :, :, 1]
            az = apparent_emitter_coordinates[i, :, :, 0]
            el = apparent_emitter_coordinates[i, :, :, 1]

            hadec = azel_to_hadec(az, el, site.lat.rad)

            # Now step through the time stamps, calculating the effective
            # sky position for the emitter, and performing phase rotation
            # appropriately
            hourangles = calculate_blockvisibility_hourangles(bvis)
            for iha, ha in enumerate(hourangles.data):
                for j, (station1, station2) in enumerate(bvis.baselines.values):
                    # station 1
                    ra1 = -hadec[0][iha, station1] + ha
                    dec1 = hadec[1][iha, station1]
                    emitter_sky1 = SkyCoord(ra1 * u.rad, dec1 * u.rad)
                    l1, m1, n1 = skycoord_to_lmn(emitter_sky1, bvis.phasecentre)

                    # station 2
                    ra2 = -hadec[0][iha, station2] + ha
                    dec2 = hadec[1][iha, station2]
                    emitter_sky2 = SkyCoord(ra2 * u.rad, dec2 * u.rad)
                    l2, m2, n2 = skycoord_to_lmn(emitter_sky2, bvis.phasecentre)

                    phasor = numpy.ones([nbaselines, nchan, npol], dtype="complex")
                    for chan in range(nchan):
                        phasor[j, chan, :] = (
                            simulate_point(uvw[iha, j, :, chan], l1, m1)
                            - simulate_point(uvw[iha, j, :, chan], l2, m2)
                        )[..., numpy.newaxis]

                    # Now fill this into the BlockVisibility
                    bvis["vis"].data[iha, j, ...] += (
                        bvis_data_copy[iha, j, ...] * phasor
                    )

    return bvis
