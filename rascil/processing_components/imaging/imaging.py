"""
Functions for predicting blockvisibility from a model image, and invert a blockvisibility to
make an (image, sumweights) tuple. These redirect to specific versions.
"""

__all__ = [
    "predict_blockvisibility",
    "invert_blockvisibility",
]

import logging

import numpy

from rascil.data_models.memory_data_models import BlockVisibility, Image
from rascil.processing_components.imaging.base import (
    predict_awprojection,
    invert_awprojection,
)
from rascil.processing_components.imaging.ng import predict_ng, invert_ng
from rascil.processing_components.imaging.wg import predict_wg, invert_wg

log = logging.getLogger("rascil-logger")


def predict_blockvisibility(
    vis: BlockVisibility, model: Image, context="ng", gcfcf=None, **kwargs
) -> BlockVisibility:
    """Predict blockvisibility from an image

    For awprojection, the gridding details must be supplied via a tuple of
    (gridding correction function, convolution function) or a partial
    to calculate it.

    :param vis: blockvisibility to be predicted
    :param model: model image
    :param context: Type: 2d or awprojection, ng or wg (default: ng)
    :param gcfcf: Tuple of (grid correction function, convolution function) or partial function
    :return: resulting visibility (in place works)
    """
    if context == "awprojection":
        return predict_awprojection(vis, model, gcfcf=gcfcf, **kwargs)
    elif context == "2d":
        return predict_ng(vis, model, do_wstacking=False, **kwargs)
    elif context == "ng":
        return predict_ng(vis, model, **kwargs)
    elif context == "wg":
        return predict_wg(vis, model, **kwargs)
    else:
        raise ValueError(f"Unknown imaging context {context}")


def invert_blockvisibility(
    vis: BlockVisibility,
    im: Image,
    dopsf: bool = False,
    normalise: bool = True,
    context="ng",
    gcfcf=None,
    **kwargs,
) -> (Image, numpy.ndarray):
    """Invert blockvisibility to make an (image, sum weights) tuple

    Use the image im as a template. Do PSF in a separate call.

    For awprojection, the gridding details must be supplied via a tuple of
    (gridding correction function, convolution function) or a partial
    to calculate it.

    :param vis: blockvisibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image (default: False)
    :param normalise: normalise by the sum of weights (default: True)
    :param context: Type: 2d or awprojection, ng or wg (default: ng)
    :param gcfcf: Tuple of (grid correction function, convolution function) or partial function
    :return: (resulting image, sum of weights)
    """

    if context == "awprojection":
        return invert_awprojection(
            vis, im, dopsf=dopsf, normalise=normalise, gcfcf=gcfcf, **kwargs
        )
    elif context == "2d":
        return invert_ng(
            vis, im, dopsf=dopsf, normalise=normalise, do_wstacking=False, **kwargs
        )
    elif context == "ng":
        return invert_ng(vis, im, dopsf=dopsf, normalise=normalise, **kwargs)
    elif context == "wg":
        return invert_wg(vis, im, dopsf=dopsf, normalise=normalise, **kwargs)
     
    else:
        raise ValueError(f"Unknown imaging context {context}")
