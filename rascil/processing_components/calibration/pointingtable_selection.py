"""
Simple pointingtable operations (Still in development)
"""

__all__ = ['pointingtable_select',
           'pointingtable_groupby',
           'pointingtable_groupby_bins',
           'pointingtable_where']

import copy
import logging

from rascil.data_models.memory_data_models import PointingTable

log = logging.getLogger('rascil-logger')


def pointingtable_select(pt, selection):
    """ Select subset of pointingtable using xarray syntax

    :param pt:
    :param selection:
    :return:
    """
    newpt = copy.copy(pt)
    newpt.data = pt.data.sel(selection)
    return newpt


def pointingtable_where(pt, condition, **kwargs):
    """ Select where a condition holds of pointingtable using xarray syntax

    :param pt:
    :param condition:
    :return:
    """
    newpt = copy.copy(pt)
    newpt.data = pt.data.where(condition, **kwargs)
    return newpt


def pointingtable_groupby(pt, coordinate, **kwargs):
    """ Group bu a coordinate condition holds of pointingtable using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of pt

    :param pt:
    :param coordinate:
    :return:
    """
    for group in pt.data.groupby(coordinate, **kwargs):
        newpt = copy.copy(pt)
        newpt.data = group[1]
        yield group[0], newpt


def pointingtable_groupby_bins(pt, coordinate, bins, **kwargs):
    """ Group bu a coordinate condition holds of pointingtable using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of pt

    :param pt:
    :param coordinate:
    :return:
    """
    for group in pt.data.groupby_bins(coordinate, bins=bins, **kwargs):
        newpt = copy.copy(pt)
        newpt.data = group[1]
        yield group[0], newpt
