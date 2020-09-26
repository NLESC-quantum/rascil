""" Iterators for iterating through a BlockVisibility.

A typical use would be to make a sequence of snapshot images::

    for rows in vis_timeslice_iter(vt):
        visslice = create_blockvisibility_from_rows(vt, rows)
        dirtySnapshot = create_image_from_visibility(visslice, npixel=512, cellsize=0.001, npol=1)
        dirtySnapshot, sumwt = invert_2d(visslice, dirtySnapshot)

"""

__all__ = ['vis_null_iter', 'vis_timeslice_iter', 'vis_timeslices', 'vis_wslice_iter', 'vis_wslices']

import logging

import numpy

from rascil.data_models.memory_data_models import BlockVisibility

log = logging.getLogger('logger')


def vis_null_iter(vis: BlockVisibility, vis_slices=1) -> numpy.ndarray:
    """Null iterator returning true for all rows
    
    :param vis:
    :param vis_slices:
    :return:
    """
    assert vis is not None
    assert isinstance(vis, BlockVisibility), vis
    yield numpy.ones_like(vis.time, dtype=bool)


def vis_timeslice_iter(vis: BlockVisibility, vis_slices=None) -> numpy.ndarray:
    """ Time slice iterator

    :param vis:
    :param vis_slices: Number of time slices
    :return: Boolean array with selected rows=True
    """
    assert vis is not None
    assert isinstance(vis, BlockVisibility), vis
    timemin = numpy.min(vis.time)
    timemax = numpy.max(vis.time)
    
    if isinstance(vis, BlockVisibility) and (vis_slices == "auto" or vis_slices is None):
        for ib in range(vis.nvis):
            boxes = numpy.zeros([vis.ntimes], dtype='bool')
            boxes[ib] = True
            yield boxes
    else:
        
        if vis_slices is None:
            vis_slices = vis_timeslices(vis, 'auto')
            print("Found {} visslices".format(vis_slices))
        
        boxes = numpy.linspace(timemin, timemax, vis_slices)
        if vis_slices > 1:
            timeslice = boxes[1] - boxes[0]
        else:
            timeslice = timemax - timemin
        
        for box in boxes:
            rows = numpy.abs(vis.time - box) <= 0.5 * timeslice
            yield rows


def vis_timeslices(vis: BlockVisibility, timeslice='auto') -> int:
    """ Calculate number of time slices in a visibility

    :param vis: blockvisibility
    :param timeslice: 'auto' or float (seconds)
    :return: Number of slices
    """
    assert isinstance(vis, BlockVisibility), vis

    timemin = numpy.min(vis.time)
    timemax = numpy.max(vis.time)
    
    assert timeslice == "auto", timeslice
    
    if isinstance(vis, BlockVisibility):
        if timeslice == 'auto':
            return len(vis.time)

    if timeslice == 'auto':
        return len(numpy.unique(vis.time))
    else:
        return numpy.ceil(timemax - timemin) / timeslice


def vis_wslices(vis: BlockVisibility, wslice=10.0) -> int:
    """ Calculate number of w slices (or stack) in a visibility

    :param vis: blockvisibility
    :param wslice: width of w slice (in lambda)
    :return: Number of slices
    """
    assert isinstance(vis, BlockVisibility), vis
    wmaxabs = numpy.max(numpy.abs(vis.w))
    
    return 1 + 2 * numpy.round(wmaxabs / wslice).astype('int')

def vis_wslice_iter(vis: BlockVisibility, vis_slices=1) -> numpy.ndarray:
    """ W slice iterator

    :param vis:
    :param vis_slices: Number of slices
    :return: Boolean array with selected rows=True
    """
    assert isinstance(vis, BlockVisibility), vis
    wmaxabs = numpy.max(numpy.abs(vis.w))
    
    boxes = numpy.linspace(- wmaxabs, +wmaxabs, vis_slices)
    if vis_slices > 1:
        wstack = boxes[1] - boxes[0]
    else:
        wstack = 2 * wmaxabs
    
    for box in boxes:
        rows = numpy.abs(vis.w - box) < 0.5 * wstack
        yield rows
