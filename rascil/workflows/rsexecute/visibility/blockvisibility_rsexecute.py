""" Workflows for handling blockvisibility data

"""

__all__ = ['create_blockvisibility_from_ms_rsexecute']

import logging

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.processing_components import create_blockvisibility_from_ms

log = logging.getLogger('rascil-logger')

def create_blockvisibility_from_ms_rsexecute(msname, nchan_per_blockvis, nout, dds,
                      average_channels=False):
    """ Graph for reading from a MeasurementSet into a list of blockvisibility
    
    :param msname:
    :param nchan_per_blockvis:
    :param nout:
    :param dds:
    :param average_channels:
    :return:
    """

    # Read the MS into RASCIL BlockVisibility objects
    log.info("Loading MS into {n} BlockVisibility's of {nchan} channels"
             .format(n=nout * len(dds), nchan=nchan_per_blockvis))

    bvis_list = [[rsexecute.execute(create_blockvisibility_from_ms, nout=nout)
                  (msname=msname,
                   selected_dds=[dd],
                   start_chan=chan_block * nchan_per_blockvis,
                   end_chan=(1 + chan_block) * nchan_per_blockvis - 1,
                   average_channels=average_channels)[0]
                  for dd in dds for chan_block in range(nout)]]

    # This is a list of lists so we flatten it to a list
    bvis_list = [item for sublist in bvis_list for item in sublist]

    return bvis_list

