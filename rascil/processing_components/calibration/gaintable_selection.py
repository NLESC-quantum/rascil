"""
Simple gaintable operations (Still in development)
"""

__all__ = ['gaintable_select',
           'gaintable_groupby',
           'gaintable_groupby_bins',
           'gaintable_where']

import copy
import logging

import numpy

from rascil.data_models.memory_data_models import GainTable

log = logging.getLogger('logger')

def gaintable_select(gt, selection):
    """ Select subset of gaintable using xarray syntax

    :param gt:
    :param selection:
    :return:
    """
    newgt = copy.copy(gt)
    newgt.data = gt.data.sel(selection)
    return newgt


def gaintable_where(gt, condition, **kwargs):
    """ Select where a condition holds of gaintable using xarray syntax

    :param gt:
    :param condition:
    :return:
    """
    newgt = copy.copy(gt)
    newgt.data = gt.data.where(condition, **kwargs)
    return newgt


def gaintable_groupby(gt, coordinate, **kwargs):
    """ Group bu a coordinate condition holds of gaintable using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of gt

    :param gt:
    :param coordinate:
    :return:
    """
    for group in gt.data.groupby(coordinate, **kwargs):
        newgt = copy.copy(gt)
        newgt.data = group[1]
        yield group[0], newgt


def gaintable_groupby_bins(gt, coordinate, bins, **kwargs):
    """ Group bu a coordinate condition holds of gaintable using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of gt

    :param gt:
    :param coordinate:
    :return:
    """
    for group in gt.data.groupby_bins(coordinate, bins=bins, **kwargs):
        newgt = copy.copy(gt)
        newgt.data = group[1]
        yield group[0], newgt
