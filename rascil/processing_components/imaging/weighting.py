"""
Functions that aid weighting the visibility data prior to imaging.

There are two classes of functions:
    - Changing the weight dependent on noise level or sample density or a combination
    - Tapering the weihght spatially to avoid effects of sharp edges or to emphasize a given scale size in the image

"""

__all__ = ['weight_visibility', 'taper_visibility_gaussian', 'taper_visibility_tukey']


import numpy

import logging

import astropy.constants as constants

from rascil.data_models.memory_data_models import BlockVisibility
from rascil.processing_components.griddata.gridding import grid_blockvisibility_weight_to_griddata, \
    griddata_blockvisibility_reweight
from rascil.processing_components.griddata.kernels import create_pswf_convolutionfunction
from rascil.processing_components.griddata.operations import create_griddata_from_image
from rascil.processing_components.image.operations import image_is_canonical
from rascil.processing_components.util.array_functions import tukey_filter

log = logging.getLogger('rascil-logger')


def weight_visibility(vis, model, gcfcf=None, weighting='uniform', robustness=0.0, **kwargs):
    """ Weight the visibility data

    This is done collectively so the weights are summed over all vis_lists and then
    corrected

    :param vis_list:
    :param model_imagelist: Model required to determine weighting parameters
    :param weighting: Type of weighting (uniform or robust or natural)
    :param robustness: Robustness parameter
    :param kwargs: Parameters for functions in graphs
    :return: Reweighted vis
   """

    assert image_is_canonical(model)
    if gcfcf is None:
        gcfcf = create_pswf_convolutionfunction(model, polarisation_frame=vis.polarisation_frame)

    griddata = create_griddata_from_image(model, polarisation_frame=vis.polarisation_frame)
    griddata, sumwt = grid_blockvisibility_weight_to_griddata(vis, griddata, gcfcf[1])
    vis = griddata_blockvisibility_reweight(vis, griddata, gcfcf[1], weighting=weighting, robustness=robustness)

    return vis


def weight_blockvisibility(vis, model, gcfcf=None, weighting="uniform", robustness=0.0, **kwargs):
    """ Weight the visibility data

    This is done collectively so the weights are summed over all vis_lists and then
    corrected

    :param vis_list:
    :param model_imagelist: Model required to determine weighting parameters
    :param weighting: Type of weighting
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
   """
    log.info("weight_blockvisibility: is deprecated, use weight_visibility")
    return weight_visibility(vis, model, gcfcf, weighting, robustness, **kwargs)


def taper_visibility_gaussian(vis, beam=None):
    """ Taper the visibility weights

    These are cumulative. If You can reset the imaging_weights
    using :py:mod:`processing_components.imaging.weighting.weight_visibility`

    :param vis: blockvisibility with imaging_weight's to be tapered
    :param beam: desired resolution (Full width half maximum, radians)
    :return: visibility with imaging_weight column modified
    """
    #assert isinstance(vis, BlockVisibility), vis
    
    if beam is None:
        raise ValueError("Beam size not specified for Gaussian taper")
    
    #assert isinstance(vis, BlockVisibility), vis
    # See http://mathworld.wolfram.com/FourierTransformGaussian.html
    scale_factor = numpy.pi ** 2 * beam ** 2 / (4.0 * numpy.log(2.0))

    for chan, freq in enumerate(vis.frequency.values):
        wave = constants.c.to('m s^-1').value / freq
        uvdistsq = (vis.blockvisibility_acc.u.values ** 2 +
                    vis.blockvisibility_acc.v.values ** 2) / wave**2
        wt = numpy.exp(-scale_factor * uvdistsq)
        vis['imaging_weight'].data[..., chan, :] = vis.blockvisibility_acc.flagged_imaging_weight.data[..., chan, :] * \
                                                   wt[..., numpy.newaxis]

    return vis


def taper_visibility_tukey(vis, tukey=0.1):
    """ Taper the visibility weights
    
    This algorithm is present in WSClean.

    See https://sourceforge.net/p/wsclean/wiki/Tapering

    tukey, a circular taper that smooths the outer edge set by -maxuv-l
    inner-tukey, a circular taper that smooths the inner edge set by -minuv-l
    edge-tukey, a square-shaped taper that smooths the edge set by the uv grid and -taper-edge.

    These are cumulative. If You can reset the imaging_weights
    using :py:mod:`processing_components.imaging.weighting.weight_visibility`

    :param vis: blockvisibility with imaging_weight's to be tapered
    :return: visibility with imaging_weight column modified
    """

    #assert isinstance(vis, BlockVisibility), vis

    oshape = vis['imaging_weight'].data[..., 0, 0].shape
    for chan, freq in enumerate(vis.frequency.values):
        wave = constants.c.to('m s^-1').value / freq
        uvdist = numpy.sqrt(vis.blockvisibility_acc.u.data ** 2 +
                            vis.blockvisibility_acc.v.data ** 2)
        uvdist = uvdist.flatten() / wave
        uvdistmax = numpy.max(uvdist)
        uvdist /= uvdistmax
        wt = numpy.array([tukey_filter(uv, tukey) for uv in uvdist]).reshape(oshape)
        vis['imaging_weight'].data[..., chan, :] = vis.blockvisibility_acc.flagged_imaging_weight.data[..., chan, :] \
                                                        * wt[..., numpy.newaxis]

    return vis

