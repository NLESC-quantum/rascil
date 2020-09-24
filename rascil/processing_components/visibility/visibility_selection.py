"""
Simple flagging operations (Still in development)
"""

__all__ = ['blockvisibility_select',
           'blockvisibility_groupby',
           'blockvisibility_groupby_bins',
           'blockvisibility_where']

import copy
import logging

import numpy

from rascil.data_models.memory_data_models import BlockVisibility

log = logging.getLogger('logger')

def blockvisibility_select(bvis, selection):
    """ Select subset of BlockVisibility using xarray syntax

    :param bvis:
    :param selection:
    :return:
    """
    newbvis = copy.copy(bvis)
    newbvis.data = bvis.data.sel(selection)
    return newbvis


def blockvisibility_where(bvis, condition, **kwargs):
    """ Select where a condition holds of BlockVisibility using xarray syntax

    :param bvis:
    :param condition:
    :return:
    """
    newbvis = copy.copy(bvis)
    newbvis.data = bvis.data.where(condition, **kwargs)
    return newbvis


def blockvisibility_groupby(bvis, coordinate, **kwargs):
    """ Group bu a coordinate condition holds of BlockVisibility using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of bvis

    :param bvis:
    :param coordinate:
    :return:
    """
    for group in bvis.data.groupby(coordinate, **kwargs):
        newbvis = copy.copy(bvis)
        newbvis.data = group[1]
        yield group[0], newbvis


def blockvisibility_groupby_bins(bvis, coordinate, bins, **kwargs):
    """ Group bu a coordinate condition holds of BlockVisibility using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of bvis

    :param bvis:
    :param coordinate:
    :return:
    """
    for group in bvis.data.groupby_bins(coordinate, bins=bins, **kwargs):
        newbvis = copy.copy(bvis)
        newbvis.data = group[1]
        yield group[0], newbvis
