"""
Functions that implement prediction of and imaging from visibilities using the nifty gridder.

https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

This performs all necessary w term corrections, to high precision.

Note that nifty gridder doesn't like some null data such as all w = 0 and do_wstacking=True.
Also true of the visibilities.

"""

__all__ = ['predict_ng', 'invert_ng']

import copy
import logging

import numpy
import nifty_gridder as ng
from rascil.data_models.memory_data_models import BlockVisibility, \
    Image
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import convert_pol_frame
from rascil.processing_components.image.operations import image_is_canonical
from rascil.processing_components.imaging.base import shift_vis_to_image, \
    normalize_sumwt
from rascil.processing_components.visibility.base import copy_visibility

log = logging.getLogger('rascil-logger')

def predict_ng(bvis: BlockVisibility, model: Image, **kwargs) -> BlockVisibility:
    """ Predict using convolutional degridding.
    
    Nifty-gridder version. https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

    In the imaging and pipeline workflows, this may be invoked using context='ng'.

    :param bvis: BlockVisibility to be predicted
    :param model: model image
    :return: resulting BlockVisibility (in place works)
    """
    
    # assert isinstance(bvis, BlockVisibility), bvis
    assert image_is_canonical(model)
    
    if model is None:
        return bvis
    
    nthreads = get_parameter(kwargs, "threads", 4)
    epsilon = get_parameter(kwargs, "epsilon", 1e-12)
    do_wstacking = get_parameter(kwargs, "do_wstacking", True)
    verbosity = get_parameter(kwargs, "verbosity", 0)
    
    newbvis = copy_visibility(bvis, zero=True)
    
    # Extracting data from BlockVisibility
    freq = bvis.frequency.data  # frequency, Hz
    nrows, nbaselines, vnchan, vnpol = bvis.vis.shape
    
    uvw = newbvis.uvw.data
    uvw = uvw.reshape([nrows * nbaselines, 3])
    uvw = numpy.nan_to_num(uvw)
    vist = numpy.zeros([vnpol, vnchan, nbaselines * nrows], dtype='complex')
    
    # Get the image properties
    m_nchan, m_npol, ny, nx = model["pixels"].data.shape
    # Check if the number of frequency channels matches in bvis and a model
    #        assert (m_nchan == v_nchan)
    assert (m_npol == vnpol)
    
    fuvw = copy.deepcopy(uvw)
    # We need to flip the u and w axes. The flip in w is equivalent to the conjugation of the
    # convolution function grid_visibility to griddata
    fuvw[:, 0] *= -1.0
    fuvw[:, 2] *= -1.0
    
    # Find out the image size/resolution
    pixsize = numpy.abs(numpy.radians(model.image_acc.wcs.wcs.cdelt[0]))
    
    # Make de-gridding over a frequency range and pol fields
    vis_to_im = numpy.round(model.image_acc.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
    
    mfs = m_nchan == 1
    
    if mfs:
        for vpol in range(vnpol):
            vist[vpol, :, :] = ng.dirty2ms(fuvw.astype(numpy.float64),
                                           bvis.frequency.data.astype(numpy.float64),
                                           model["pixels"].data[0, vpol, :, :].T.astype(numpy.float64),
                                           pixsize_x=pixsize,
                                           pixsize_y=pixsize,
                                           epsilon=epsilon,
                                           do_wstacking=do_wstacking,
                                           nthreads=nthreads,
                                           verbosity=verbosity).T
    
    else:
        for vpol in range(vnpol):
            for vchan in range(vnchan):
                imchan = vis_to_im[vchan]
                vist[vpol, vchan, :] = ng.dirty2ms(fuvw.astype(numpy.float64),
                                                   numpy.array(freq[vchan:vchan + 1]).astype(numpy.float64),
                                                   model["pixels"].data[imchan, vpol, :, :].T.astype(numpy.float64),
                                                   pixsize_x=pixsize,
                                                   pixsize_y=pixsize,
                                                   epsilon=epsilon,
                                                   do_wstacking=do_wstacking,
                                                   nthreads=nthreads,
                                                   verbosity=verbosity)[:, 0]
    
    vis = convert_pol_frame(vist.T, model.image_acc.polarisation_frame, bvis.blockvisibility_acc.polarisation_frame,
                            polaxis=2)
    
    vis = vis.reshape([nrows, nbaselines, vnchan, vnpol])
    newbvis["vis"].data = vis
    
    # Now we can shift the visibility from the image frame to the original visibility frame
    return shift_vis_to_image(newbvis, model, tangent=True, inverse=True)


def invert_ng(bvis: BlockVisibility, model: Image, dopsf: bool = False,
              normalize: bool = True,
              **kwargs) -> (Image, numpy.ndarray):
    """ Invert using nifty-gridder module
    
    https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

    Use the image im as a template. Do PSF in a separate call.

    In the imaging and pipeline workflows, this may be invoked using context='ng'. It is the default

    :param dopsf: Make the PSF instead of the dirty image
    :param bvis: BlockVisibility to be inverted
    :param im: image template (not changed)
    :param normalize: Normalize by the sum of weights (True)
    :return: (resulting image, sum of the weights for each frequency and polarization)

    """
    assert image_is_canonical(model)
    
    # assert isinstance(bvis, BlockVisibility), bvis
    
    im = model.copy(deep=True)
    
    nthreads = get_parameter(kwargs, "threads", 4)
    epsilon = get_parameter(kwargs, "epsilon", 1e-12)
    do_wstacking = get_parameter(kwargs, "do_wstacking", True)
    verbosity = get_parameter(kwargs, "verbosity", 0)
    
    sbvis = copy_visibility(bvis)
    sbvis = shift_vis_to_image(sbvis, im, tangent=True, inverse=False)
    
    freq = sbvis.frequency.data  # frequency, Hz
    
    nrows, nbaselines, vnchan, vnpol = sbvis.vis.shape
    # if dopsf:
    #     sbvis = fill_vis_for_psf(sbvis)
    
    ms = sbvis.blockvisibility_acc.flagged_vis.data
    ms = ms.reshape([nrows * nbaselines, vnchan, vnpol])
    ms = convert_pol_frame(ms, bvis.blockvisibility_acc.polarisation_frame,
                           im.image_acc.polarisation_frame, polaxis=2).astype("c16")
    
    uvw = copy.deepcopy(sbvis.uvw.data)
    uvw = uvw.reshape([nrows * nbaselines, 3])
    
    wgt = sbvis.blockvisibility_acc.flagged_imaging_weight.data.astype("f8")
    wgt = wgt.reshape([nrows * nbaselines, vnchan, vnpol])
    
    # if epsilon > 5.0e-6:
    #     ms = ms.astype("c8")
    #     wgt = wgt.astype("f4")
    
    # Find out the image size/resolution
    npixdirty = im["pixels"].data.shape[-1]
    pixsize = numpy.abs(numpy.radians(im.image_acc.wcs.wcs.cdelt[0]))
    
    fuvw = copy.deepcopy(uvw)
    # We need to flip the u and w axes.
    fuvw[:, 0] *= -1.0
    fuvw[:, 2] *= -1.0
    
    nchan, npol, ny, nx = im["pixels"].data.shape
    im["pixels"].data[...] = 0.0
    sumwt = numpy.zeros([nchan, npol])
    
    # There's a latent problem here with the weights.
    # wgt = numpy.real(convert_pol_frame(wgt, bvis.blockvisibility_acc.polarisation_frame, im.image_acc.polarisation_frame, polaxis=2))
    
    # Set up the conversion from visibility channels to image channels
    vis_to_im = numpy.round(model.image_acc.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
    
    # Nifty gridder likes to receive contiguous arrays so we transpose at the beginning
    
    mfs = nchan == 1 and vnchan > 1
    mst = ms.T
    wgtt = wgt.T
    if dopsf:
        mst[...] = 0.0
        mst[0, ...] = 1.0
    
    if mfs:
        for pol in range(npol):
            lms = numpy.ascontiguousarray(mst[pol, :, :].T)
            if numpy.max(numpy.abs(lms)) > 0.0:
                lwt = numpy.ascontiguousarray(wgtt[pol, :, :].T)
                dirty = ng.ms2dirty(fuvw,
                                    bvis.frequency.data,
                                    lms,
                                    lwt,
                                    npixdirty, npixdirty, pixsize, pixsize, epsilon,
                                    do_wstacking=do_wstacking,
                                    nthreads=nthreads, verbosity=verbosity)
                im["pixels"].data[0, pol] += dirty.T
            sumwt[0, pol] += numpy.sum(wgtt[pol, :, :].T)
    else:
        for pol in range(npol):
            for vchan in range(vnchan):
                ichan = vis_to_im[vchan]
                frequency = numpy.array(freq[vchan:vchan + 1]).astype(numpy.float64)
                lms = numpy.ascontiguousarray(mst[pol, vchan, :, numpy.newaxis])
                if numpy.max(numpy.abs(lms)) > 0.0:
                    lwt = numpy.ascontiguousarray(wgtt[pol, vchan, :, numpy.newaxis])
                    dirty = ng.ms2dirty(fuvw,
                                        frequency,
                                        lms, lwt,
                                        npixdirty, npixdirty, pixsize, pixsize,
                                        epsilon, nthreads,
                                        do_wstacking, verbosity=verbosity)
                    im["pixels"].data[ichan, pol] += dirty.T
                sumwt[ichan, pol] += numpy.sum(wgtt[pol, vchan, :].T, axis=0)
    
    if normalize:
        im = normalize_sumwt(im, sumwt)
    
    return im, sumwt
