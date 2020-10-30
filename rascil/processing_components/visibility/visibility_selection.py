"""
Simple flagging operations (Still in development)
"""

__all__ = ['blockvisibility_select',
           'blockvisibility_iselect',
           'blockvisibility_groupby',
           'blockvisibility_groupby_bins',
           'blockvisibility_where']

import copy
import logging

log = logging.getLogger('rascil-logger')


def blockvisibility_select(bvis, selection):
    """ Select subset of BlockVisibility using xarray syntax

    :param bvis:
    :param selection:
    :return:
    """
    newbvis = copy.copy(bvis)
    newbvis.data = bvis.data.sel(selection)
    return newbvis


def blockvisibility_iselect(bvis, selection):
    """ Select subset of BlockVisibility using xarray syntax by index

    :param bvis:
    :param selection:
    :return:
    """
    newbvis = copy.copy(bvis)
    newbvis.data = bvis.data.isel(selection)
    return newbvis

def blockvisibility_fillna(bvis):
    """ Fill Nan's
    
    Using xarray.where introduces Nan's into a dataset. Weflag these points
    by setting the flags point appropriately and also zero the other
    data variables.
    :param bvis:
    :return:
    """
    bvis.data.flags.fillna(1.0)
    for value in ["uvw", "uvw_lambda", "uvdist_lambda", "time", "integration_time", "channel_bandwidth"]:
        bvis.data[value].fillna(0)
    from datetime import datetime
    bvis.datetime.fillna(datetime(1970, 1, 1))
    return bvis

def blockvisibility_where(bvis, condition, make_copy=True, flagnans=True, **kwargs):
    """ Select where a condition holds of BlockVisibility using xarray syntax

    :param bvis:
    :param condition:
    :param make_copy: Make a copy instead of modifying in place
    :return:
    """
    if make_copy:
        newbvis = copy.copy(bvis)
        newbvis.data = bvis.data.where(condition, **kwargs)
        if flagnans:
            newbvis = blockvisibility_fillna(newbvis)
        return newbvis
    else:
        bvis.data = bvis.data.where(condition, **kwargs)
        if flagnans:
            bvis = blockvisibility_fillna(bvis)
        return bvis


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
    :param bins: Number of bins or lis of bins
    :return:
    """
    for group in bvis.data.groupby_bins(coordinate, bins=bins, **kwargs):
        newbvis = copy.copy(bvis)
        newbvis.data = group[1]
        yield group[0], newbvis
