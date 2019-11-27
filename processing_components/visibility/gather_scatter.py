""" Visibility iterators for iterating through a BlockVisibility or Visibility.

A typical use would be to make a sequence of snapshot visibilitys::

    for rows in vis_timeslice_iter(vt, vis_slices=vis_slices):
        visslice = create_visibility_from_rows(vt, rows)
        dirtySnapshot = create_visibility_from_visibility(visslice, npixel=512, cellsize=0.001, npol=1)
        dirtySnapshot, sumwt = invert_2d(visslice, dirtySnapshot)


"""

import logging
from typing import List

import numpy

from data_models.memory_data_models import Visibility, BlockVisibility
from ..visibility.base import create_visibility_from_rows
from ..visibility.iterators import vis_timeslice_iter, vis_wslice_iter

log = logging.getLogger(__name__)


def visibility_scatter(vis: Visibility, vis_iter, vis_slices=1) -> List[Visibility]:
    """Scatter a visibility into a list of subvisibilities
    
    If vis_iter is over time then the type of the outvisibilities will be the same as inout
    If vis_iter is over w then the type of the output visibilities will always be Visibility

    :param vis: Visibility
    :param vis_iter: visibility iterator
    :param vis_slices: Number of slices to be made
    :return: list of subvisibilitys
    """
    
    assert vis is not None
    
    if vis_slices == 1:
        return [vis]
    
    visibility_list = list()
    for i, rows in enumerate(vis_iter(vis, vis_slices=vis_slices)):
        subvis = create_visibility_from_rows(vis, rows)
        visibility_list.append(subvis)
    
    return visibility_list


def visibility_gather(visibility_list: List[Visibility], vis: Visibility, vis_iter, vis_slices=None) -> Visibility:
    """Gather a list of subvisibilities back into a visibility
    
    The iterator setup must be the same as used in the scatter.

    :param visibility_list: List of subvisibilities
    :param vis: Output visibility
    :param vis_iter: visibility iterator
    :param vis_slices: Number of slices to be gathered (optional)
    :return: vis
    """
    
    if vis_slices == 1:
        return visibility_list[0]
    
    if vis_slices is None:
        vis_slices = len(visibility_list)
    
    rowses = []
    for i, rows in enumerate(vis_iter(vis, vis_slices=vis_slices)):
        rowses.append(rows)

    for i, rows in enumerate(rowses):
        assert i < len(visibility_list), "Gather not consistent with scatter for slice %d" % i
        if visibility_list[i] is not None and numpy.sum(rows):
            assert numpy.sum(rows) == visibility_list[i].nvis, \
                "Mismatch in number of rows (%d, %d) in gather for slice %d" % \
            (numpy.sum(rows), visibility_list[i].nvis, i)
            vis.data[rows] = visibility_list[i].data[...]
    
    return vis


def visibility_scatter_w(vis: Visibility, vis_slices=1) -> List[Visibility]:
    assert isinstance(vis, Visibility), vis
    return visibility_scatter(vis, vis_iter=vis_wslice_iter, vis_slices=vis_slices)

def visibility_scatter_time(vis: Visibility, vis_slices=1) -> List[Visibility]:
    return visibility_scatter(vis, vis_iter=vis_timeslice_iter, vis_slices=vis_slices)


def visibility_gather_w(visibility_list: List[Visibility], vis: Visibility, vis_slices=1) -> Visibility:
    assert isinstance(vis, Visibility), vis
    return visibility_gather(visibility_list, vis, vis_iter=vis_wslice_iter, vis_slices=vis_slices)


def visibility_gather_time(visibility_list: List[Visibility], vis: Visibility, vis_slices=1) -> Visibility:
    return visibility_gather(visibility_list, vis, vis_iter=vis_timeslice_iter, vis_slices=vis_slices)

def visibility_scatter_channel(vis: BlockVisibility) -> List[Visibility]:
    """ Scatter channels to separate visibilities
    
    :param vis:
    :return:
    """
    assert isinstance(vis, BlockVisibility), vis

    def extract_channel(v, chan):
        vis_shape = numpy.array(v.data['vis'].shape)
        vis_shape[3] = 1
        
        vis = BlockVisibility(data=None,
                              frequency=numpy.array([v.frequency[chan]]),
                              channel_bandwidth=numpy.array([v.channel_bandwidth[chan]]),
                              phasecentre=v.phasecentre,
                              configuration=v.configuration,
                              uvw=v.uvw,
                              time=v.time,
                              vis=v.vis[..., chan, :][..., numpy.newaxis, :],
                              weight=v.weight[..., chan, :][..., numpy.newaxis, :],
                              imaging_weight=v.imaging_weight[..., chan, :][..., numpy.newaxis, :],
                              integration_time=v.integration_time,
                              polarisation_frame=v.polarisation_frame,
                              source=v.source,
                              meta=v.meta)
        return vis
    
    return [extract_channel(vis, channel) for channel, _ in enumerate(vis.frequency)]


def visibility_gather_channel(vis_list: List[BlockVisibility], vis: BlockVisibility = None):
    """ Gather a visibility by channel
    
    :param vis_list:
    :param vis:
    :return:
    """
    
    cols = ['vis', 'weight']
    
    if vis is None:
        
        vis_shape = numpy.array(vis_list[0].vis.shape)
        vis_shape[-2] = len(vis_list)
        for v in vis_list:
            assert len(v.frequency) == 1
            assert len(v.channel_bandwidth) == 1
            
        vis = BlockVisibility(data=None,
                              frequency=numpy.array([v.frequency[0] for v in vis_list]),
                              channel_bandwidth=numpy.array([v.channel_bandwidth[0] for v in vis_list]),
                              phasecentre=vis_list[0].phasecentre,
                              configuration=vis_list[0].configuration,
                              uvw=vis_list[0].uvw,
                              time=vis_list[0].time,
                              vis=numpy.zeros(vis_shape, dtype=vis_list[0].vis.dtype),
                              weight=numpy.ones(vis_shape, dtype=vis_list[0].weight.dtype),
                              imaging_weight=numpy.ones(vis_shape, dtype=vis_list[0].weight.dtype),
                              integration_time=vis_list[0].integration_time,
                              polarisation_frame=vis_list[0].polarisation_frame,
                              source=vis_list[0].source,
                              meta=vis_list[0].meta)
    
    assert len(vis.frequency) == len(vis_list)
    
    for chan, _ in enumerate(vis_list):
        subvis = vis_list[chan]
        assert abs(subvis.frequency[0] - vis.frequency[chan]) < 1e-15
        for col in cols:
            vis.data[col][..., chan, :] = subvis.data[col][..., 0, :]
        vis.frequency[chan] = subvis.frequency[0]
    
    nchan = vis.vis.shape[-2]
    assert nchan == len(vis.frequency)
    
    return vis
