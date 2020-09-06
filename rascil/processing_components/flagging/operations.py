"""
Functions that Flagging Visibility and BlockVisibility.

The flags of BlockVisibility has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes.


"""

__all__ = ['flagging_blockvisibility']

import logging

import numpy
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import BlockVisibility, Visibility, FlagTable
from rascil.data_models.memory_data_models import QA
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import create_image_from_array

log = logging.getLogger('logger')

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
            bvis.data['flags'][:, ant, ...] = 0
            bvis.data['flags'][:, :, ant, ...] = 0
    elif len(channel) == 0:
        for ant in antenna:
            for pol in polarization:
                bvis.data['flags'][:, ant, :, :, pol] = 0
                bvis.data['flags'][:, :, ant, :, pol] = 0
    elif len(polarization) == 0:
        for ant in antenna:
            for ch in channel:
                bvis.data['flags'][:, ant, :, ch, :] = 0
                bvis.data['flags'][:, :, ant, ch, :] = 0
    else:
        for ant in antenna:
            for ch in channel:
                for pol in polarization:
                    bvis.data['flags'][:, ant, :, ch, pol] = 0
                    bvis.data['flags'][:, :, ant, ch, pol] = 0

    return bvis
