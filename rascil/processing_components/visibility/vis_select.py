""" Visibility selectors for a BlockVisibility.


"""
__all__ = ['vis_select_uvrange', 'vis_select_wrange']

import logging

import numpy

from rascil.data_models.memory_data_models import BlockVisibility

log = logging.getLogger('logger')

def vis_select_uvrange(vis: BlockVisibility, uvmin=0.0, uvmax=numpy.infty):
    """Return rows in valid region
    
    :param vis: blockvisibility
    :param uvmin:
    :param uvmax:
    :return: Boolean array of valid rows
    """

    assert isinstance(vis, BlockVisibility)

    uvdist = numpy.sqrt(vis.u**2+vis.v**2)
    rows = (uvmin < uvdist) & (uvdist <= uvmax)
    return rows


def vis_select_wrange(vis: BlockVisibility, wmax=numpy.infty):
    """Return rows in valid region

    :param vis: blockvisibility
    :param wmax: w max in wavelengths
    :return: Boolean array of valid rows
    """
    assert isinstance(vis, BlockVisibility)

    absw = numpy.abs(vis.w)
    rows = (wmax >= absw)
    return rows
