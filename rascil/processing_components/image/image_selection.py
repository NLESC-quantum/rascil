"""
Simple flagging operations (Still in development)
"""

__all__ = ['image_select',
           'image_iselect',
           'image_groupby',
           'image_groupby_bins',
           'image_concat',
           'image_where']

import copy
import logging

import numpy
import xarray

from rascil.data_models.memory_data_models import Image
from rascil.processing_components.image.operations import copy_image
log = logging.getLogger('rascil-logger')

def image_select(im, selection, **kwargs):
    """ Select subset of image using xarray syntax

    :param im:
    :param selection:
    :return:
    """
    newim = copy.copy(im)
    newim.data = im.data.sel(selection, **kwargs)
    return newim


def image_iselect(im, selection, **kwargs):
    """ Select subset of image using xarray syntax

    :param im:
    :param selection:
    :return:
    """
    newim = copy.copy(im)
    newim.data = im.data.isel(selection, **kwargs)
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

def image_concat(im_list, dim, **kwargs):
    """ Concatenate a list of images

    :param im_list: List of images
    :param dim: Dimension to concatenate along e.g. "frequency"
    :return: Image
    """
    for im in im_list:
        assert not numpy.isnan(numpy.sum(im.data.values)), \
            "NaNs present in input image {}".format(im)

    newim = copy_image(im_list[0])
    newim.data = xarray.concat([im.data for im in im_list], dim=dim, **kwargs)
    
    assert newim.shape[0] == len(im_list), \
        "Input {} images, output {} channel image".format(len(im_list), newim.shape[0])
    assert newim.shape[-2:] == im_list[0].shape[-2:], \
        "Input shape {}, output shape {}".format(im_list[0].shape[-2:], newim.shape[-2:])

    assert not numpy.isnan(numpy.sum(newim.data.values)), \
        "NaNs present in output image {}".format(newim)

    return newim


