"""
Functions that Flagging Visibility and BlockVisibility.

The flags of BlockVisibility has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes.


"""

__all__ = ["flagging_blockvisibility", "flagging_blockvisibility_with_bl"]

import logging

import numpy
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import BlockVisibility, FlagTable

log = logging.getLogger("rascil-logger")


def flagging_blockvisibility(bvis, antenna=[], channel=[], polarization=[]):
    """Flagging BlockVisibility

    :param bvis: BlockVisibility
    :param antenna: The list of antenna number
    :param channel: The list of Channel number
    :param polarization:
    :return: BlockVisibility
    """

    # assert isinstance(bvis, BlockVisibility), bvis
    # assert isinstance(antenna, list)
    if len(antenna) == 0:
        flagged_baselines = list(range(len(bvis.baselines)))
    else:
        flagged_baselines = []
        for ibaseline, (a1, a2) in enumerate(bvis.baselines.data):
            if a1 in antenna:
                flagged_baselines.append(ibaseline)
            if a2 in antenna:
                flagged_baselines.append(ibaseline)

    if len(channel) == 0 and len(polarization) == 0:
        for ibaseline in flagged_baselines:
            bvis["flags"].data[:, ibaseline, ...] = 1
            bvis["flags"].data[:, ibaseline, ...] = 1
    elif len(channel) == 0:
        for ibaseline in flagged_baselines:
            for pol in polarization:
                bvis["flags"].data[:, ibaseline, ..., pol] = 1
                bvis["flags"].data[:, ibaseline, ..., pol] = 1
    elif len(polarization) == 0:
        for ibaseline in flagged_baselines:
            for ch in channel:
                bvis["flags"].data[:, ibaseline, ch, :] = 1
                bvis["flags"].data[:, ibaseline, ch, :] = 1
    else:
        for ibaseline in flagged_baselines:
            for ch in channel:
                for pol in polarization:
                    bvis["flags"].data[:, ibaseline, ch, pol] = 1
                    bvis["flags"].data[:, ibaseline, ch, pol] = 1

    return bvis


def flagging_blockvisibility_with_bl(bvis, baseline=[]):
    """Flagging BlockVisibility with Baseline (nant, nant, channel, pol)

    :param bvis: BlockVisibility
    :param polarization:
    :return: BlockVisibility
    """

    # assert isinstance(bvis, BlockVisibility), bvis
    # assert isinstance(baseline, list)
    if len(baseline) == 0:
        for ant1, ant2 in baseline:
            ibaseline = bvis.baselines.get_loc((ant1, ant2))
            bvis["flags"].data[:, ibaseline, :, :] = 1
    return bvis
