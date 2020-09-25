""" Functions for calibration, including creation of gaintables, application of gaintables, and
merging gaintables.

"""

__all__ = ['gaintable_summary', 'qa_gaintable', 'apply_gaintable', 'append_gaintable',
           'create_gaintable_from_blockvisibility', 'create_gaintable_from_blockvisibility',
           'gaintable_select', 'create_gaintable_from_rows', 'copy_gaintable']

import copy
import logging
from typing import Union

import matplotlib.pyplot as plt
import numpy.linalg
#from astropy.visualization import time_support
from astropy.time import Time

from rascil.data_models.memory_data_models import GainTable, BlockVisibility, QA, assert_vis_gt_compatible
from rascil.data_models.polarisation import ReceptorFrame
log = logging.getLogger('logger')


def apply_gaintable(vis: BlockVisibility, gt: GainTable, inverse=False, **kwargs) -> BlockVisibility:
    """Apply a gain table to a block visibility

    The corrected visibility is::

        V_corrected = {g_i * g_j^*}^-1 V_obs

    If the visibility data are polarised e.g. polarisation_frame("linear") then the inverse operator
    represents an actual inverse of the gains.

    :param vis: blockvisibility to have gains applied
    :param gt: Gaintable to be applied
    :param inverse: Apply the inverse (default=False)
    :return: input vis with gains applied

    """
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis
    assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt
    
    ntimes, nants, nchan, nrec, _ = gt.gain.shape
    
    assert_vis_gt_compatible(vis, gt)
    
    if inverse:
        log.debug('apply_gaintable: Apply inverse gaintable')
    else:
        log.debug('apply_gaintable: Apply gaintable')
    
    is_scalar = gt.gain.shape[-2:] == (1, 1)
    if vis.npol == 1:
        log.debug('apply_gaintable: scalar gains')
    
    # row_numbers = numpy.array(list(range(len(vis.time))), dtype='int')
    row_numbers = numpy.arange(len(vis.time))
    done = numpy.zeros(len(row_numbers), dtype='int')
    from rascil.processing_components import blockvisibility_select, gaintable_select

    for row in range(ntimes):
        vis_rows = numpy.abs(vis.time.values - gt.time.values[row]) < gt.interval.values[row] / 2.0
        vis_rows = row_numbers[vis_rows]
        if len(vis_rows) > 0:
            
            # Lookup the gain for this set of visibilities
            gain = gt.data['gain'].values[row]
            cgain = numpy.conjugate(gt.data['gain'].values[row])
            gainwt = gt.data['weight'].values[row]
            
            # The shape of the mueller matrix is
            nant, nchan, nrec, _ = gain.shape
            baselines = vis.baselines.values
            
            original = vis.vis.values[vis_rows]
            applied = copy.copy(vis.vis.values[vis_rows])
            appliedwt = copy.copy(vis.weight.values[vis_rows])
            if vis.npol == 1:
                if inverse:
                    # lgain = numpy.ones_like(gain)
                    # lgain[numpy.abs(gain) > 0.0] = 1.0 / gain[numpy.abs(gain) > 0.0]
                    lgain= numpy.ones_like(gain)
                    numpy.putmask(lgain,numpy.abs(gain) > 0.0, 1.0 / gain)
                else:
                    lgain = gain

                # tlgain = lgain.T
                # tclgain = numpy.conjugate(tlgain)
                # smueller = numpy.ones([nchan, nant, nant], dtype='complex')
                # for chan in range(nchan):
                #     smueller[chan, :, :] = numpy.ma.outer(tlgain[0, 0, chan, :],
                #                                           tclgain[0, 0, chan, :]).reshape([nant, nant])
                # numpy.testing.assert_allclose(smueller,smueller1,rtol=1e-5)

                # Original Code with Loop
                # for sub_vis_row in range(original.shape[0]):
                #     for chan in range(nchan):
                #         applied[sub_vis_row, :, :, chan, 0] = \
                #             original[sub_vis_row, :, :, chan, 0] * smueller[chan, :, :]
                #         antantwt = numpy.outer(gainwt[:, chan, 0, 0], gainwt[:, chan, 0, 0])
                #         appliedwt[sub_vis_row, :, :, chan, 0] = antantwt
                #         applied[sub_vis_row, :, :, chan, 0][antantwt == 0.0] = 0.0

                # Optimized (SIM-423)
                # smueller1 = numpy.ones([nchan, nant, nant], dtype='complex')
                smueller1 = numpy.einsum('ijlm,kjlm->jik', lgain, numpy.conjugate(lgain))

                for sub_vis_row in range(original.shape[0]):
                    for ibaseline, (a1, a2) in enumerate(baselines):
                        for chan in range(nchan):
                            applied[sub_vis_row, ibaseline, chan, 0] = \
                                original[sub_vis_row, ibaseline, chan, 0] * smueller1[chan, a1, a2]
                            antantwt = gainwt[a1, chan, 0, 0] * gainwt[a2, chan, 0, 0]
                            appliedwt[sub_vis_row, ibaseline, chan, 0] = \
                                gainwt[a1, chan, 0, 0] * gainwt[a2, chan, 0, 0]
                            #applied[sub_vis_row, ibaseline, chan, 0][antantwt == 0.0] = 0.0

                # smueller1 = numpy.einsum('ijlm,kjlm->ikj', lgain, numpy.conjugate(lgain))
                # for sub_vis_row in range(original.shape[0]):
                #     applied[sub_vis_row, :, :, :, 0] = \
                #         original[sub_vis_row, :, :, :, 0] * smueller1[:, :, :]
                #     antantwt = numpy.einsum('ik,jk->ijk',gainwt[:, :, 0, 0], gainwt[:, :, 0, 0])
                #     appliedwt[sub_vis_row, :, :, :, 0] = antantwt
                #     numpy.putmask(applied[sub_vis_row, :, :, :, 0], antantwt[:,:,:] == 0.0, 0.0)

            elif vis.npol == 2:
                has_inverse_ant = numpy.zeros([nant, nchan], dtype='bool')
                if inverse:
                    igain = gain.copy()
                    cigain = cgain.copy()
                    for a1 in range(nants):
                        for chan in range(nchan):
                            try:
                                igain[a1, chan, :, :] = numpy.linalg.inv(gain[a1, chan, :, :])
                                cigain[a1, chan, :, :] = numpy.conjugate(igain[a1, chan, :, :])
                                has_inverse_ant[a1, chan] = True
                            except numpy.linalg.linalg.LinAlgError:
                                has_inverse_ant[a1, chan] = False
        
                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, (a1, a2) in enumerate(baselines):
                            for chan in range(nchan):
                                if has_inverse_ant[a1, chan] and has_inverse_ant[a2, chan]:
                                    cfs = numpy.diag(original[sub_vis_row, ibaseline, chan, ...])
                                    applied[sub_vis_row, ibaseline, chan, ...] = \
                                        numpy.diag(igain[a1, chan, :, :] @ \
                                                   cfs @ cigain[a2, chan, :, :]).reshape([2])
                else:
                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, (a1, a2) in enumerate(baselines):
                            for chan in range(nchan):
                                    cfs = numpy.diag(original[sub_vis_row, ibaseline, chan, ...])
                                    applied[sub_vis_row, ibaseline, chan, ...] = \
                                        numpy.diag(gain[a1, chan, :, :] @ cfs @ cgain[a2, chan, :, :]).reshape([2])

            elif vis.npol == 4:
                has_inverse_ant = numpy.zeros([nant, nchan], dtype='bool')
                if inverse:
                    igain = gain.copy()
                    cigain = cgain.copy()
                    for a1 in range(nants):
                        for chan in range(nchan):
                            try:
                                igain[a1, chan, :, :] = numpy.linalg.inv(gain[a1, chan, :, :])
                                cigain[a1, chan, :, :] = numpy.conjugate(igain[a1, chan, :, :])
                                has_inverse_ant[a1, chan] = True
                            except numpy.linalg.linalg.LinAlgError:
                                has_inverse_ant[a1, chan] = False
               
                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, baseline in enumerate(baselines):
                            for chan in range(nchan):
                                if has_inverse_ant[baseline[0], chan] and has_inverse_ant[baseline[1], chan]:
                                    cfs = original[sub_vis_row, ibaseline, chan, ...].reshape([2,2])
                                    applied[sub_vis_row, ibaseline, chan, ...] = \
                                        (igain[baseline[0], chan, :, :] @ cfs @ cigain[baseline[1], chan, :, :]).reshape([4])
                else:
                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, baseline in enumerate(baselines):
                            for chan in range(nchan):
                                cfs = original[sub_vis_row, ibaseline, chan, ...].reshape([2, 2])
                                applied[sub_vis_row, ibaseline, chan, ...] = \
                                    (gain[baseline[0], chan, :, :] @ cfs @ cgain[baseline[1], chan, :, :]).reshape([4])
            
            else:
                times = Time(vis.time / 86400.0, format='mjd', scale='utc')
                print("No row in gaintable for visibility time range  {} to {}".format(times[0].isot, times[-1].isot))
                log.warning("No row in gaintable for visibility row, time range  {} to {}".format(times[0].isot, times[-1].isot))

            vis.vis.values[vis_rows] = applied

    return vis


def gaintable_summary(gt: GainTable):
    """Return string summarizing the Gaintable

    :param gt: Gaintable
    :returns: string

    """
    return "%.3f GB" % (gt.size())


def create_gaintable_from_blockvisibility(vis: BlockVisibility, timeslice=None,
                                          frequencyslice: float = None, **kwargs) -> GainTable:
    """ Create gain table from visibility.
    
    This makes an empty gain table consistent with the BlockVisibility.
    
    :param vis: BlockVisibilty
    :param timeslice: Time interval between solutions (s)
    :param frequencyslice: Frequency solution width (Hz) (NYI)
    :return: GainTable
    
    """
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis
    
    nants = vis.nants
    
    if timeslice is None or timeslice == 'auto':
        utimes = numpy.unique(vis.time)
        gain_interval = vis.integration_time
    else:
        utimes = vis.time.values[0] + timeslice * numpy.unique(numpy.round((vis.time.values - vis.time.values[0]) / timeslice))
        gain_interval = timeslice * numpy.ones_like(utimes)
    
    ntimes = len(utimes)
    
    #    log.debug('create_gaintable_from_blockvisibility: times are %s' % str(utimes))
    #    log.debug('create_gaintable_from_blockvisibility: intervals are %s' % str(gain_interval))
    
    ntimes = len(utimes)
    ufrequency = numpy.unique(vis.frequency)
    nfrequency = len(ufrequency)
    
    receptor_frame = ReceptorFrame(vis.polarisation_frame.type)
    nrec = receptor_frame.nrec
    
    gainshape = [ntimes, nants, nfrequency, nrec, nrec]
    gain = numpy.ones(gainshape, dtype='complex')
    if nrec > 1:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0
    
    gain_weight = numpy.ones(gainshape)
    gain_time = utimes
    gain_frequency = ufrequency
    gain_residual = numpy.zeros([ntimes, nfrequency, nrec, nrec])
    
    gt = GainTable(gain=gain, time=gain_time, interval=gain_interval, weight=gain_weight, residual=gain_residual,
                   frequency=gain_frequency, receptor_frame=receptor_frame, phasecentre=vis.phasecentre,
                   configuration=vis.configuration)
    
    assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt
    assert_vis_gt_compatible(vis, gt)
    
    return gt


def append_gaintable(gt: GainTable, othergt: GainTable) -> GainTable:
    """Append othergt to gt

    :param gt:
    :param othergt:
    :return: GainTable gt + othergt
    """
    assert gt.receptor_frame == othergt.receptor_frame
    gt.data = numpy.hstack((gt.data, othergt.data))
    return gt


def copy_gaintable(gt: GainTable, zero=False):
    """Copy a GainTable

    Performs a deepcopy of the data array
    """
    
    if gt is None:
        return gt
    
    assert isinstance(gt, GainTable), gt
    
    newgt = copy.copy(gt)
    newgt.data = copy.deepcopy(gt.data)
    if zero:
        newgt.data['gt'][...] = 0.0
    return newgt


def gaintable_select(gt, selection):
    """ Select subset of GainTable using xarray syntax

    :param ft:
    :param selection:
    :return:
    """
    newgt = copy.copy(gt)
    newgt.data = gt.data.sel(selection)
    return newgt

def create_gaintable_from_rows(gt: GainTable, rows: numpy.ndarray, makecopy=True) \
        -> Union[GainTable, None]:
    """ Create a GainTable from selected rows

    :param gt: GainTable
    :param rows: Boolean array of row selection
    :param makecopy: Make a deep copy (True)
    :return: GainTable
    """
    
    if rows is None or numpy.sum(rows) == 0:
        return None
    
    assert len(rows) == gt.ntimes, "Length of rows does not agree with length of GainTable"
    
    assert isinstance(gt, GainTable), gt
    
    if makecopy:
        newgt = copy_gaintable(gt)
        newgt.data = copy.deepcopy(gt.data[rows])
        return newgt
    else:
        gt.data = copy.deepcopy(gt.data[rows])
        
        return gt


def qa_gaintable(gt: GainTable, context=None) -> QA:
    """Assess the quality of a gaintable

    :param gt:
    :return: QA
    """
    agt = numpy.abs(gt.gain.values[gt.weight.values > 0.0])
    pgt = numpy.angle(gt.gain.values[gt.weight.values > 0.0])
    rgt = gt.residual.values[numpy.sum(gt.weight.values, axis=1) > 0.0]
    data = {'shape': gt.gain.shape,
            'maxabs-amp': numpy.max(agt),
            'minabs-amp': numpy.min(agt),
            'rms-amp': numpy.std(agt),
            'medianabs-amp': numpy.median(agt),
            'maxabs-phase': numpy.max(pgt),
            'minabs-phase': numpy.min(pgt),
            'rms-phase': numpy.std(pgt),
            'medianabs-phase': numpy.median(pgt),
            'residual': numpy.max(rgt)
            }
    return QA(origin='qa_gaintable', data=data, context=context)
