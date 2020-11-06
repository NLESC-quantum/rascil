"""
Simple flagging operations (Still in development)

"""

__all__ = ['flagtable_summary',
           'copy_flagtable',
           'create_flagtable_from_blockvisibility',
           'qa_flagtable']

import copy
import logging

import numpy

from rascil.data_models.memory_data_models import BlockVisibility, FlagTable, QA

log = logging.getLogger('rascil-logger')


def flagtable_summary(ft: FlagTable):
    """Return string summarizing the FlagTable

    :param ft: FlagTable
    :return: string
    """
    return "%d rows" % (len(ft.data))


def copy_flagtable(ft: FlagTable, zero=False) -> FlagTable:
    """Copy a flagtable

    :param ft: FlagTable
    :param zero: Zero the flags
    :returns: FlagTable

    """
    ##assert isinstance(ft, FlagTable), ft

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


def qa_flagtable(ft: FlagTable, context=None) -> QA:
    """Assess the quality of FlagTable

    :param context:
    :param ft: FlagTable to be assessed
    :return: QA
    """
    ##assert isinstance(ft, FlagTable), ft

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
