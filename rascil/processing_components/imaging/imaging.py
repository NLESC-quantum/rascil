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

log = logging.getLogger("rascil-logger")


def predict_blockvisibility(
    vis: BlockVisibility, model: Image, context="ng", **kwargs
) -> BlockVisibility:
    """Predict blockvisibility from an image

    :param vis: blockvisibility to be predicted
    :param model: model image
    :param context: Type: 2d or awprojection or ng (ng)
    :return: resulting visibility (in place works)
    """
    if context == "awprojection":
        return predict_awprojection(vis, model, **kwargs)
    elif context == "2d":
        return predict_ng(vis, model, do_wstacking=False, **kwargs)
    elif context == "ng":
        return predict_ng(vis, model, **kwargs)
    else:
        raise ValueError(f"Unknown imaging context {context}")


def invert_blockvisibility(
    vis: BlockVisibility,
    im: Image,
    dopsf: bool = False,
    normalise: bool = True,
    context="ng",
    **kwargs,
) -> (Image, numpy.ndarray):
    """Invert blockvisibility to make an (image, sum weights) tuple

    Use the image im as a template. Do PSF in a separate call.

    :param vis: blockvisibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image (False)
    :param normalise: normalise by the sum of weights (True)
    :param context: Type: 2d or awprojection or ng (ng)
    :return: (resulting image, sum of weights)
    """

    if context == "awprojection":
        return invert_awprojection(vis, im, dopsf=dopsf, normalise=normalise, **kwargs)
    elif context == "2d":
        return invert_ng(
            vis, im, dopsf=dopsf, normalise=normalise, do_wstacking=False, **kwargs
        )
    elif context == "ng":
        return invert_ng(vis, im, dopsf=dopsf, normalise=normalise, **kwargs)
    else:
        raise ValueError(f"Unknown imaging context {context}")
