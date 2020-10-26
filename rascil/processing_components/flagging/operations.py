"""
Functions that Flagging Visibility and BlockVisibility.

The flags of BlockVisibility has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes.


"""

__all__ = ["flagging_blockvisibility", "flagging_blockvisibility_with_bl"]

import logging

import numpy
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import BlockVisibility, Visibility, FlagTable
from rascil.data_models.memory_data_models import QA
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import create_image_from_array

log = logging.getLogger("logger")


def flagging_blockvisibility(bvis, antenna=[], channel=[], polarization=[]):
    """Flagging BlockVisibility

    :param bvis: BlockVisibility
    :param antenna: The list of antenna number
    :param channel: The list of Channel number
    :param polarization:
    :return: BlockVisibility
    """

    assert isinstance(bvis, BlockVisibility), bvis
    assert isinstance(antenna, list)
    if len(channel) == 0 and len(polarization) == 0:
        for ant in antenna:
            bvis.data["flags"][:, ant, ...] = 1
            bvis.data["flags"][:, :, ant, ...] = 1
    elif len(channel) == 0:
        for ant in antenna:
            for pol in polarization:
                bvis.data["flags"][:, ant, :, :, pol] = 1
                bvis.data["flags"][:, :, ant, :, pol] = 1
    elif len(polarization) == 0:
        for ant in antenna:
            for ch in channel:
                bvis.data["flags"][:, ant, :, ch, :] = 1
                bvis.data["flags"][:, :, ant, ch, :] = 1
    else:
        for ant in antenna:
            for ch in channel:
                for pol in polarization:
                    bvis.data["flags"][:, ant, :, ch, pol] = 1
                    bvis.data["flags"][:, :, ant, ch, pol] = 1

    return bvis


def flagging_blockvisibility_with_bl(bvis, baseline=[]):
    """Flagging BlockVisibility with Baseline (nant, nant, channel, pol)

    :param baseline:
    :param bvis: BlockVisibility
    :return: BlockVisibility
    """

    assert isinstance(bvis, BlockVisibility), bvis
    assert isinstance(baseline, list)
    if len(baseline) == 0:
        for ant1, ant2 in baseline:
            bvis.data["flags"][:, ant1, ant2, :, :] = 1
    return bvis
