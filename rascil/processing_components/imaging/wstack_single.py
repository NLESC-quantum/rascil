"""
The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is
approximated as:

.. math::

    V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected.
"""

__all__ = ['predict_wstack_single', 'invert_wstack_single']

import logging

import numpy

from rascil.data_models.memory_data_models import Visibility, Image
from rascil.processing_components.image.operations import copy_image, create_w_term_like, image_is_canonical
from rascil.processing_components.imaging.base import predict_2d, invert_2d
from rascil.processing_components.visibility.base import copy_visibility

log = logging.getLogger('logger')


def predict_wstack_single(vis, model, remove=True, gcfcf=None, **kwargs) -> Visibility:
    """ Predict using a single w slices.
    
    This processes a single w plane, rotating out the w beam for the average w

    The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is
    approximated as:

    .. math::

        V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

    If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected.

    :param vis: Visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """

    assert isinstance(vis, Visibility), vis
    assert image_is_canonical(model)

    vis.data['vis'][...] = 0.0

    log.debug("predict_wstack_single: predicting using single w slice")

    # We might want to do wprojection so we remove the average w
    w_average = numpy.average(vis.w)
    if remove:
        vis.data['uvw'][..., 2] -= w_average
    tempvis = copy_visibility(vis)

    # Calculate w beam and apply to the model. The imaginary part is not needed
    workimage = copy_image(model)
    w_beam = create_w_term_like(model, w_average, vis.phasecentre)
    
    # Do the real part
    workimage.data = w_beam.data.real * model.data
    vis = predict_2d(vis, workimage, gcfcf=gcfcf, **kwargs)
    
    # and now the imaginary part
    workimage.data = w_beam.data.imag * model.data
    tempvis = predict_2d(tempvis, workimage, gcfcf=gcfcf, **kwargs)
    vis.data['vis'] -= 1j * tempvis.data['vis']
    
    if remove:
        vis.data['uvw'][..., 2] += w_average

    return vis


def invert_wstack_single(vis: Visibility, im: Image, dopsf, normalize=True, remove=True,
                         gcfcf=None, **kwargs) -> (Image, numpy.ndarray):
    """Process single w slice
    
    The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is
    approximated as:

    .. math::

        V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

    If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :returns: image, sum of weights
    """
    assert image_is_canonical(im)

    log.debug("invert_wstack_single: predicting using single w slice")
    
    kwargs['imaginary'] = True
    
    assert isinstance(vis, Visibility), vis
    
    # We might want to do wprojection so we remove the average w
    w_average = numpy.average(vis.w)
    if remove:
        vis.data['uvw'][..., 2] -= w_average
    
    reWorkimage, sumwt, imWorkimage = invert_2d(vis, im, dopsf, normalize=normalize, gcfcf=gcfcf, **kwargs)
    
    if remove:
        vis.data['uvw'][..., 2] += w_average

    # Calculate w beam and apply to the model. The imaginary part is not needed
    w_beam = create_w_term_like(im, w_average, vis.phasecentre)
    reWorkimage.data = w_beam.data.real * reWorkimage.data - w_beam.data.imag * imWorkimage.data
    
    return reWorkimage, sumwt
