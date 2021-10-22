"""
Functions that Flagging Visibility and BlockVisibility.

The flags of BlockVisibility has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes.


"""

__all__ = ["flagging_blockvisibility"]

import logging

import numpy
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import BlockVisibility, FlagTable

log = logging.getLogger("rascil-logger")


def flagging_blockvisibility(
    bvis, baselines=None, antennas=None, channels=None, polarisations=None
):
    """Flagging BlockVisibility

    :param bvis: BlockVisibility
    :param baselines: The list of baseline numbers to flag
    :param antennas: The list of antenna number to flag
    :param channels: The list of Channel number to flag
    :param polarisations: The list of polarisations to flag
    :return: BlockVisibility
    """
    if polarisations is None:
        polarisations = []
    if antennas is None:
        antennas = []
    if channels is None:
        channels = []
    if baselines is None:
        baselines = []

    for baseline in baselines:
        bvis["flags"].data[:, baseline, ...] = 1
    for channel in channels:
        bvis["flags"].data[..., channel, :] = 1
    for pol in polarisations:
        bvis["flags"].data[..., pol] = 1
    for ibaseline, (a1, a2) in enumerate(bvis.baselines.data):
        if a1 in antennas or a2 in antennas:
            bvis["flags"].data[:, ibaseline, ...] = 1

    return bvis
