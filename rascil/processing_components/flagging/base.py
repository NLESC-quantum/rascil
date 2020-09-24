"""
Simple flagging operations (Still in development)
"""

__all__ = ['flagtable_summary',
           'copy_flagtable',
           'create_flagtable_from_blockvisibility',
           'flagtable_select',
           'flagtable_groupby',
           'flagtable_groupby_bins',
           'flagtable_where',
           'qa_flagtable']

import copy
import logging

import numpy

from rascil.data_models.memory_data_models import BlockVisibility, FlagTable, QA

log = logging.getLogger('logger')


def flagtable_summary(ft: FlagTable):
    """Return string summarizing the FlagTable

    :param ft: FlagTable
    :return: string
    """
    return "%d rows" % (len(ft.data))


def copy_flagtable(ft: FlagTable, zero=False) -> FlagTable:
    """Copy a flagtable

    :param ft: FlagTable
    Performs a deepcopy of the data array
    :param zero: Zero the flags
    :returns: FlagTable

    """
    assert isinstance(ft, FlagTable), ft

    newft = copy.copy(ft)
    newft.data = numpy.copy(ft.data)
    if zero:
        ft.data['flags'][...] = 0.0
    return newft


def create_flagtable_from_blockvisibility(bvis: BlockVisibility, **kwargs) -> FlagTable:
    """ Create FlagTable matching BlockVisibility

    :param bvis:
    :param kwargs:
    :return:
    """
    return FlagTable(flags=bvis.flags, frequency=bvis.frequency, channel_bandwidth=bvis.channel_bandwidth,
                     configuration=bvis.configuration, time=bvis.time,
                     integration_time=bvis.integration_time,
                     polarisation_frame=bvis.polarisation_frame)


def flagtable_select(ft, selection):
    """ Select subset of FlagTable using xarray syntax

    :param ft:
    :param selection:
    :return:
    """
    newft = copy.copy(ft)
    newft.data = ft.data.sel(selection)
    return newft


def flagtable_where(ft, condition, **kwargs):
    """ Select where a condition holds of FlagTable using xarray syntax

    :param ft:
    :param condition:
    :return:
    """
    newft = copy.copy(ft)
    newft.data = ft.data.where(condition, **kwargs)
    return newft


def flagtable_groupby(ft, coordinate, **kwargs):
    """ Group bu a coordinate condition holds of FlagTable using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of ft

    :param ft:
    :param coordinate:
    :return:
    """
    for group in ft.data.groupby(coordinate, **kwargs):
        newft = copy.copy(ft)
        newft.data = group[1]
        yield group[0], newft

def flagtable_groupby_bins(ft, coordinate, bins, **kwargs):
    """ Group bu a coordinate condition holds of FlagTable using xarray syntax

    Returns a sequence of (value, group) pairs where the value is that of the
    coordinate, and group is the part of ft

    :param ft:
    :param coordinate:
    :return:
    """
    for group in ft.data.groupby_bins(coordinate, bins=bins, **kwargs):
        newft = copy.copy(ft)
        newft.data = group[1]
        yield group[0], newft


def qa_flagtable(ft: FlagTable, context=None) -> QA:
    """Assess the quality of FlagTable

    :param context:
    :param ft: FlagTable to be assessed
    :return: QA
    """
    assert isinstance(ft, FlagTable), ft

    aflags = numpy.abs(ft.flags)
    data = {'maxabs': numpy.max(aflags),
            'minabs': numpy.min(aflags),
            'mean': numpy.mean(aflags),
            'sum': numpy.sum(aflags),
            'medianabs': numpy.median(aflags)}
    qa = QA(origin='qa_flagtable',
            data=data,
            context=context)
    return qa
