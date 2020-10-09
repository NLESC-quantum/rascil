"""
Simple flagging operations (Still in development)
"""

__all__ = ['image_select',
           'image_iselect',
           'image_groupby',
           'image_groupby_bins',
           'image_where']

import copy
import logging

import numpy
import xarray

from rascil.data_models.memory_data_models import Image

log = logging.getLogger('rascil-logger')

def image_select(im, selection):
    """ Select subset of image using xarray syntax

    :param im:
    :param selection:
    :return:
    """
    newim = copy.copy(im)
    newim.data = im.data.sel(selection)
    return newim


def image_iselect(im, selection):
    """ Select subset of image using xarray syntax

    :param im:
    :param selection:
    :return:
    """
    newim = copy.copy(im)
    newim.data = im.data.isel(selection)
    return newim


def image_where(im, condition, make_copy=True, **kwargs):
    """ Select where a condition holds of image using xarray syntax

    :param im:
    :param condition:
    :return:
    """
    if make_copy:
        newim = copy.copy(im)
        newim.data = im.data.where(condition, **kwargs)
        return newim
    else:
        im.data = im.data.where(condition, **kwargs)
        return im

def image_groupby(im, coordinate, **kwargs):
    """ Group bu a coordinate condition holds of image using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of im

    :param im:
    :param coordinate:
    :return:
    """
    for group in im.data.groupby(coordinate, **kwargs):
        newim = copy.copy(im)
        newim.data = group[1]
        yield newim


def image_groupby_bins(im, coordinate, bins, **kwargs):
    """ Group bu a coordinate condition holds of image using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of im

    :param im:
    :param coordinate:
    :return:
    """
    for group in im.data.groupby_bins(coordinate, bins=bins, **kwargs):
        newim = copy.copy(im)
        newim.data = group[1]
        yield newim
