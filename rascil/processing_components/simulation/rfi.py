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
from astropy.coordinates import SkyCoord
from astropy.time import Time

from rascil import phyconst
from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components.util.array_functions import average_chunks2
from rascil.processing_components.util.geometry import calculate_azel

from rascil.processing_components.imaging.primary_beams import (
    create_vp,
    create_mid_allsky,
)
from rascil.processing_components.image.operations import (
    create_image,
    export_image_to_fits,
)
from rascil.processing_components.util.coordinate_support import (
    simulate_point_antenna,
    xyz_to_uvw,
    skycoord_to_lmn,
    azel_to_hadec,
)
from rascil.processing_components.visibility.visibility_geometry import (
    calculate_blockvisibility_hourangles,
)
from rascil.processing_components.skycomponent.operations import create_skycomponent
from rascil.processing_components.calibration.operations import apply_gaintable
from rascil.processing_components.calibration.pointing import (
    create_pointingtable_from_blockvisibility,
)

from rascil.processing_components.simulation.pointing import (
    simulate_gaintable_from_pointingtable,
)
from rascil.processing_components.visibility.operations import copy_visibility

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


def apply_beam_gain_for_low(
    isotropic_emitter_power,
    beam_gain_for_low,
    rfi_source,
):
    """
    Apply the beam gain to the apparent emitter power, to determine
    the received RFI signal at SKA LOW station(s).

    :param isotropic_emitter_power: apparent emitter power level as received by an isotropic antenna
                [ntimes x nantennas x nchannels]
    :param beam_gain_for_low: beam gain data / information
                if None, beam_gain = 1.0 is used
                if provided, it is either a single value,
                or a numpy array with dimensions [nsources x nstations x nchannels]
    :param rfi_source: integer indicating where in the beam array the correct data for the source can be found
    """
    if beam_gain_for_low is None:
        log.warning(
            "Beam gain wasn't provided for Low calculations. Not applying beam gain."
        )
        beam_gain_for_low = 1.0

    if isinstance(beam_gain_for_low, numpy.ndarray):
        # get the right values for the current source
        beam_gain = beam_gain_for_low[rfi_source, ...]
    else:
        beam_gain = beam_gain_for_low

    # apply the beam gain to the input apparent emitter power
    rfi_at_station = isotropic_emitter_power.copy() * numpy.sqrt(
        beam_gain
    )  # shape of rfi_at_station = [ntimes, nants, n_rfi_channels]

    return rfi_at_station


def apply_beam_gain_for_mid(
    bvis,
    sub_vis,
    voltage_pattern,
    azimuth,
    elevation,
    phase_centre,
):
    """
    Apply the beam gain to the apparent emitter power, to determine
    the received RFI signal at SKA MID antenna(s).

    :param bvis: original block visibility
    :param sub_vis: a copy and subset of the block visibility, already containing the RFI signal
    :param voltage_pattern: voltage pattern object for SKA Mid
    :param azimuth: azimuth of the RFI source at a given time, in degrees
    :param elevation: elevation of the RFI source at a given time, in degrees
    :param phase_centre: astropy.coordinates.sky_coordinate.SkyCoord object
    """
    flux = numpy.zeros((bvis.vis.shape[2], bvis.vis.shape[3]))  # [nchan, npol]

    pointing_table = create_pointingtable_from_blockvisibility(sub_vis)

    # there is always one time index in pt --> the first index is always 0
    # Axes are [time, ant, frequency, receptor, angle]. All receptors and
    # frequencies have the same pointing.
    utc_time = Time(
        [numpy.average(pointing_table["time"]) / 86400.0], format="mjd", scale="utc"
    )
    az_centre, el_centre = calculate_azel(
        sub_vis.configuration.location, utc_time, sub_vis.phasecentre
    )
    az_centre = az_centre.to("rad").value
    el_centre = el_centre.to("rad").value

    # The nominal pointing is the phasecentre in az, el. This is the same value
    # calculated by simulate_gaintable_from_pointingtable
    pointing_table["nominal"][0, :, :, :, 0] = az_centre[
        :, numpy.newaxis, numpy.newaxis
    ]
    pointing_table["nominal"][0, :, :, :, 1] = el_centre[
        :, numpy.newaxis, numpy.newaxis
    ]
    # The pointing is the interferer in az el as seen from each antenna
    pointing_table["pointing"][0, :, :, :, 0] = numpy.deg2rad(
        azimuth[:, numpy.newaxis, numpy.newaxis]
    )
    pointing_table["pointing"][0, :, :, :, 1] = numpy.deg2rad(
        elevation[:, numpy.newaxis, numpy.newaxis]
    )
    # Pointing tables define pointing to be relative to the nominal
    pointing_table["pointing"] -= pointing_table["nominal"]
    pointing_table["pointing"][0, :, :, :, 0] *= numpy.cos(
        pointing_table["nominal"][0, :, :, :, 1]
    )

    # Update the location of the emitter
    emitter_comp = create_skycomponent(
        direction=phase_centre,
        flux=flux,
        frequency=bvis.frequency.data,
        polarisation_frame=PolarisationFrame(bvis._polarisation_frame),
    )
    # Now calculate and apply the gains
    gt = simulate_gaintable_from_pointingtable(
        vis=sub_vis,
        pt=pointing_table,
        sc=[emitter_comp],
        vp=voltage_pattern,
        elevation_limit=0.0,
        jones_type="B",
    )

    # apply the beam gain --> it updates subvis in place
    apply_gaintable(sub_vis, gt[0], inverse=True)


def _get_uvw_per_station(xyz, ha, dec):
    """arrays"""
    uvw = [xyz_to_uvw(xyz[i], ha[i], dec[i]) for i in range(len(ha))]

    return numpy.array(uvw)


def simulate_rfi_block_prop(
    bvis,
    apparent_emitter_power,
    apparent_emitter_coordinates,
    rfi_sources,
    rfi_frequencies,
    low_beam_gain=None,
    apply_primary_beam=True,
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
    :param low_beam_gain: beam gain data / information for Low.
                If provided, it is either a single value,
                or a numpy array with dimensions [nrfi_sources x nstations x nchannels];
                for Mid, use None
    :param apply_primary_beam: Apply the primary beam, not used for Low
    :return: BlockVisibility
    """

    mid_or_low = bvis.configuration.name
    if "MID" not in mid_or_low and "LOW" not in mid_or_low:
        raise ValueError(
            "Telescope configuration is neither for SKA Mid nor for SKA Low."
            "Please specify correct configuration."
        )

    # temporary copy to calculate contribution for each RFI source
    bvis_data_copy = copy.copy(bvis["vis"].data)
    ntimes, nbaselines, nchan, npol = bvis.vis.shape

    if apply_primary_beam and "MID" in mid_or_low:
        vp = create_mid_allsky(bvis.frequency)

    for i, source in enumerate(
        rfi_sources
    ):  # this will tell us the index of the source in the data

        if "LOW" in mid_or_low:
            # Apply beam gain for Low RFI data
            rfi_at_station = apply_beam_gain_for_low(
                apparent_emitter_power[i], low_beam_gain, i
            )
            rfi_at_station_all_chans = match_frequencies(
                rfi_at_station,
                rfi_frequencies,
                bvis.frequency.values,
                bvis.channel_bandwidth.values,
            )

        else:
            rfi_at_station_all_chans = match_frequencies(
                apparent_emitter_power[i],
                rfi_frequencies,
                bvis.frequency.values,
                bvis.channel_bandwidth.values,
            )

        # Calculate the RFI correlation using the fringe rotation and the RFI at the station
        # [ntimes, nants, nants, nchan, npol]
        bvis_data_copy[...] = calculate_station_correlation_rfi(
            rfi_at_station_all_chans, baselines=bvis.baselines
        )

        k = bvis.frequency.data / phyconst.c_m_s

        # We know where the emitter is.
        site = bvis.configuration.location
        latitude = site.lat.rad

        # apparent_emitter_coordinates [nsource x ntimes x nantennas x [az,el,dist]
        # azimuth is index [:, :, :, 0]; elevation is index [:, :, :, 1]
        az = apparent_emitter_coordinates[i, :, :, 0]
        el = apparent_emitter_coordinates[i, :, :, 1]

        ha_emitter, dec_emitter = azel_to_hadec(
            numpy.deg2rad(az), numpy.deg2rad(el), latitude
        )

        # Now step through the time stamps, calculating the effective
        # sky position for the emitter, and performing phase rotation
        # appropriately
        hourangles = calculate_blockvisibility_hourangles(bvis)
        final_bvis = copy_visibility(bvis, zero=True)
        for iha, (time, subvis) in enumerate(final_bvis.groupby("time", squeeze=False)):
            ha_phase_ctr = hourangles[iha]
            # calculate station-based arrays
            ant_uvw = _get_uvw_per_station(
                bvis.configuration.xyz.data,
                ha_emitter[iha, :] + ha_phase_ctr.rad,
                dec_emitter[iha, :],
            )

            ant_ra = -ha_emitter[iha, :] + ha_phase_ctr.rad
            ant_dec = dec_emitter[iha, :]

            emitter_sky = SkyCoord(ant_ra * u.rad, ant_dec * u.rad)
            l, m, n = skycoord_to_lmn(emitter_sky, bvis.phasecentre)

            phasor = numpy.ones([nbaselines, nchan, npol], dtype="complex")
            for j, (station1, station2) in enumerate(subvis.baselines.values):
                for chan in range(nchan):
                    phasor[j, chan, :] = (
                        simulate_point_antenna(
                            k[chan] * ant_uvw[station1], l[station1], m[station1]
                        )
                        * numpy.conjugate(
                            simulate_point_antenna(
                                k[chan] * ant_uvw[station2],
                                l[station2],
                                m[station2],
                            )
                        )
                    )[..., numpy.newaxis]

            # Now fill this into the BlockVisibility
            subvis["vis"].data[0, ...] = bvis_data_copy[iha, ...] * phasor

            if apply_primary_beam and "MID" in mid_or_low:
                # Apply beam gain for Mid
                apply_beam_gain_for_mid(
                    bvis,
                    subvis,
                    vp,
                    az[iha],
                    el[iha],
                    bvis.phasecentre,
                )

        # Now accumulate over all sources
        bvis["vis"].data += final_bvis["vis"].data

    return bvis
