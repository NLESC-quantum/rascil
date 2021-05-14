""" Workflows for handling blockvisibility data

"""

__all__ = [
    "create_blockvisibility_from_ms_rsexecute",
    "concatenate_blockvisibility_frequency_rsexecute",
    "concatenate_blockvisibility_time_rsexecute",
]

import logging

from rascil.processing_components import concatenate_visibility

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.processing_components import (
    create_blockvisibility_from_ms,
    concatenate_blockvisibility_frequency,
)

log = logging.getLogger("rascil-logger")


def create_blockvisibility_from_ms_rsexecute(
    msname, nchan_per_blockvis, nout, dds, average_channels=False
):
    """Graph for reading from a MeasurementSet into a list of blockvisibility

    :param msname:
    :param nchan_per_blockvis:
    :param nout:
    :param dds:
    :param average_channels:
    :return:
    """

    # Read the MS into RASCIL BlockVisibility objects
    log.info("create_blockvisibility_from_ms_rsexecute: Defining graph")
    log.info(
        "create_blockvisibility_from_ms_rsexecute: will load MS data descriptors {dds} into {n} BlockVisibility's of {nchan} channels".format(
            dds=dds, n=nout * len(dds), nchan=nchan_per_blockvis
        )
    )

    bvis_list = [
        [
            rsexecute.execute(create_blockvisibility_from_ms, nout=nout)(
                msname=msname,
                selected_dds=[dd],
                start_chan=chan_block * nchan_per_blockvis,
                end_chan=(1 + chan_block) * nchan_per_blockvis - 1,
                average_channels=average_channels,
            )[0]
            for dd in dds
            for chan_block in range(nout)
        ]
    ]

    # This is a list of lists so we flatten it to a list
    bvis_list = [item for sublist in bvis_list for item in sublist]

    return bvis_list


def concatenate_blockvisibility_frequency_rsexecute(bvis_list, split=2):
    """Concatenate a list of blockvisibility's, ordered in frequency

    :param bvis_list: List of Blockvis, ordered in frequency
    :param split: Split into
    :return: BlockVis
    """
    if len(bvis_list) > split:
        centre = len(bvis_list) // split
        result = [
            concatenate_blockvisibility_frequency_rsexecute(bvis_list[:centre]),
            concatenate_blockvisibility_frequency_rsexecute(bvis_list[centre:]),
        ]
        return rsexecute.execute(concatenate_blockvisibility_frequency, nout=2)(result)
    else:
        return rsexecute.execute(concatenate_blockvisibility_frequency, nout=2)(
            bvis_list
        )


def concatenate_blockvisibility_time_rsexecute(bvis_list, split=2):
    """Concatenate a list of blockvisibility's, ordered in time

    :param bvis_list: List of Blockvis, ordered in frequency
    :param split: Split into
    :return: BlockVis
    """
    if len(bvis_list) > split:
        centre = len(bvis_list) // split
        result = [
            concatenate_blockvisibility_time_rsexecute(bvis_list[:centre]),
            concatenate_blockvisibility_time_rsexecute(bvis_list[centre:]),
        ]
        return rsexecute.execute(concatenate_visibility, nout=2)(result)
    else:
        return rsexecute.execute(concatenate_visibility, nout=2)(bvis_list)
