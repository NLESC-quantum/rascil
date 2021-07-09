"""Workflows for imaging, including predict, invert, residual, restore, deconvolve, weight, taper, zero, subtract and sum results from invert

"""

__all__ = [
    "predict_list_serial_workflow",
    "invert_list_serial_workflow",
    "weight_list_serial_workflow",
    "taper_list_serial_workflow",
    "zero_list_serial_workflow",
    "subtract_list_serial_workflow",
]

import collections
import logging

import numpy

from rascil.data_models.memory_data_models import Image, BlockVisibility
from rascil.data_models.parameters import get_parameter
from rascil.processing_components.image.operations import create_empty_image_like
from rascil.workflows.shared.imaging import imaging_context
from rascil.workflows.shared.imaging import (
    sum_invert_results,
    remove_sumwt,
    sum_predict_results,
    threshold_list,
)
from rascil.processing_components.griddata import (
    griddata_merge_weights,
    grid_blockvisibility_weight_to_griddata,
    griddata_blockvisibility_reweight,
)
from rascil.processing_components.griddata import create_pswf_convolutionfunction
from rascil.processing_components.griddata import create_griddata_from_image
from rascil.processing_components.image import deconvolve_cube, restore_cube
from rascil.processing_components.image import (
    image_scatter_facets,
    image_gather_facets,
    image_scatter_channels,
    image_gather_channels,
)
from rascil.processing_components.image.taylor_terms import (
    calculate_image_frequency_moments,
)
from rascil.processing_components.imaging import normalise_sumwt
from rascil.processing_components.imaging import taper_visibility_gaussian
from rascil.processing_components.visibility import copy_visibility
from rascil.processing_components import fit_psf

log = logging.getLogger("rascil-logger")


def predict_list_serial_workflow(vis_list, model_imagelist, context="ng", **kwargs):
    """Predict, iterating over both the scattered vis_list and image

    The visibility and image are scattered, the visibility is predicted on each part, and then the
    parts are assembled.

    :param vis_list: list of vis
    :param model_imagelist: Model used to determine image parameters
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
    """

    assert len(vis_list) == len(
        model_imagelist
    ), "Model must be the same length as the vis_list"

    # Predict_2d does not clear the vis so we have to do it here.
    vis_list = zero_list_serial_workflow(vis_list)

    c = imaging_context(context)
    predict = c["predict"]

    # Loop over all windows
    predict_results = [
        predict(vis, model_imagelist[ivis], **kwargs)
        for ivis, vis in enumerate(vis_list)
    ]

    return predict_results


def invert_list_serial_workflow(
    vis_list,
    template_model_imagelist,
    dopsf=False,
    normalise=True,
    context="ng",
    **kwargs
):
    """Sum results from invert, iterating over the scattered image and vis_list

    :param vis_list: list of vis
    :param template_model_imagelist: list of template models
    :param dopsf: Make the PSF instead of the dirty image
    :param facets: Number of facets
    :param normalise: normalise by sumwt
    :param vis_slices: Number of slices
    :param context: Imaging context
    :param kwargs: Parameters for functions in components
    :return: List of (image, sumwt) tuples, one per vis in vis_list

    For example::

        model_list = [create_image_from_visibility
            (v, npixel=npixel, cellsize=cellsize, polarisation_frame=pol_frame)
            for v in vis_list]

        dirty_list = invert_list_serial_workflow(vis_list, template_model_imagelist=model_list, context='wstack',
                                                    vis_slices=51)
        dirty, sumwt = dirty_list[centre]

    """

    if not isinstance(template_model_imagelist, collections.abc.Iterable):
        template_model_imagelist = [template_model_imagelist]

    c = imaging_context(context)
    invert = c["invert"]

    assert len(template_model_imagelist) == len(vis_list)
    invert_results = [
        invert(
            vis,
            template_model_imagelist[ivis],
            dopsf=dopsf,
            normalise=normalise,
            **kwargs
        )
        for ivis, vis in enumerate(vis_list)
    ]

    return invert_results


def weight_list_serial_workflow(
    vis_list, model_imagelist, weighting="uniform", **kwargs
):
    """Weight the visibility data

    This is done collectively so the weights are summed over all vis_lists and then
    corrected

    :param vis_list:
    :param model_imagelist: Model required to determine weighting parameters
    :param weighting: Type of weighting
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs
    """
    centre = len(model_imagelist) // 2

    def grid_wt(vis, model):
        if vis is not None:
            if model is not None:
                griddata = create_griddata_from_image(
                    model, polarisation_frame=vis.blockvisibility_acc.polarisation_frame
                )
                griddata = grid_blockvisibility_weight_to_griddata(vis, griddata)
                return griddata
            else:
                return None
        else:
            return None

    weight_list = [
        grid_wt(vis_list[i], model_imagelist[i]) for i in range(len(vis_list))
    ]

    merged_weight_grid = griddata_merge_weights(weight_list)

    def re_weight(vis, model, gd):
        if gd is not None:
            if vis is not None:
                # Ensure that the griddata has the right axes so that the convolution
                # function mapping works
                agd = create_griddata_from_image(
                    model, polarisation_frame=vis.blockvisibility_acc.polarisation_frame
                )
                agd["pixels"].data = gd[0]["pixels"].data
                vis = griddata_blockvisibility_reweight(vis, agd, weighting=weighting)
                return vis
            else:
                return None
        else:
            return vis

    return [
        re_weight(v, model_imagelist[i], merged_weight_grid)
        for i, v in enumerate(vis_list)
    ]


def taper_list_serial_workflow(vis_list, size_required):
    """Taper to desired size

    :param vis_list: List of vis
    :param size_required: Size in radians
    :return: List of vis
    """
    return [taper_visibility_gaussian(v, beam=size_required) for v in vis_list]


def zero_list_serial_workflow(vis_list):
    """Initialise vis to zero: creates new data holders

    :param vis_list: List of vis
    :return: List of vis
    """

    def zero(vis):
        if vis is not None:
            zerovis = copy_visibility(vis)
            zerovis["vis"].data[...] = 0.0
            return zerovis
        else:
            return None

    return [zero(v) for v in vis_list]


def subtract_list_serial_workflow(vis_list, model_vislist):
    """Initialise vis to zero

    :param vis_list: List of vis
    :param model_vislist: Model to be subtracted
    :return: List of vis
    """

    def subtract_vis(vis, model_vis):
        if vis is not None and model_vis is not None:
            assert (
                vis.blockvisibility_acc.flagged_vis.shape
                == model_vis.blockvisibility_acc.flagged_vis.shape
            )
            subvis = copy_visibility(vis)
            subvis["vis"].data[...] -= model_vis["vis"].data[...]
            return subvis
        else:
            return None

    return [
        subtract_vis(vis=vis_list[i], model_vis=model_vislist[i])
        for i in range(len(vis_list))
    ]
