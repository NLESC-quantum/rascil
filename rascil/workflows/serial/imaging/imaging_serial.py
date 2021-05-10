"""Workflows for imaging, including predict, invert, residual, restore, deconvolve, weight, taper, zero, subtract and sum results from invert

"""

__all__ = [
    "predict_list_serial_workflow",
    "invert_list_serial_workflow",
    "residual_list_serial_workflow",
    "restore_list_serial_workflow",
    "deconvolve_list_serial_workflow",
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
from rascil.processing_components.image import calculate_image_frequency_moments
from rascil.processing_components.imaging import normalize_sumwt
from rascil.processing_components.imaging import taper_visibility_gaussian
from rascil.processing_components.visibility import copy_visibility
from rascil.processing_components import fit_psf

log = logging.getLogger("rascil-logger")


def predict_list_serial_workflow(
    vis_list, model_imagelist, context="ng", gcfcf=None, **kwargs
):
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
    if isinstance(gcfcf, collections.abc.Iterable) and len(gcfcf) > 2:
        predict_results = [
            predict(vis, model_imagelist[ivis], gcfcf=gcfcf[ivis], **kwargs)
            for ivis, vis in enumerate(vis_list)
        ]
    else:
        predict_results = [
            predict(vis, model_imagelist[ivis], gcfcf=gcfcf[0], **kwargs)
            for ivis, vis in enumerate(vis_list)
        ]

    return predict_results


def invert_list_serial_workflow(
    vis_list,
    template_model_imagelist,
    dopsf=False,
    normalize=True,
    context="ng",
    gcfcf=None,
    **kwargs
):
    """Sum results from invert, iterating over the scattered image and vis_list

    :param vis_list: list of vis
    :param template_model_imagelist: list of template models
    :param dopsf: Make the PSF instead of the dirty image
    :param facets: Number of facets
    :param normalize: Normalize by sumwt
    :param vis_slices: Number of slices
    :param context: Imaging context
    :param gcfcg: tuple containing grid correction and convolution function
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
    if isinstance(gcfcf, collections.abc.Iterable) and len(gcfcf) > 2:
        assert len(gcfcf) == len(vis_list)
        invert_results = [
            invert(
                vis,
                template_model_imagelist[ivis],
                dopsf=dopsf,
                normalise=normalize,
                gcfcf=gcfcf[ivis],
                **kwargs
            )
            for ivis, vis in enumerate(vis_list)
        ]
    else:
        invert_results = [
            invert(
                vis,
                template_model_imagelist[ivis],
                dopsf=dopsf,
                normalise=normalize,
                gcfcf=gcfcf[0],
                **kwargs
            )
            for ivis, vis in enumerate(vis_list)
        ]

    return invert_results


def residual_list_serial_workflow(
    vis, model_imagelist, context="2d", gcfcf=None, **kwargs
):
    """Create a graph to calculate residual image

    :param vis: List of vis
    :param model_imagelist: Model used to determine image parameters
    :param context: Imaging context e.g. '2d', 'wstack'
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: list of (image, sumwt) tuples
    """
    model_vis = zero_list_serial_workflow(vis)
    model_vis = predict_list_serial_workflow(
        model_vis, model_imagelist, context=context, gcfcf=gcfcf, **kwargs
    )
    residual_vis = subtract_list_serial_workflow(vis, model_vis)
    result = invert_list_serial_workflow(
        residual_vis,
        model_imagelist,
        dopsf=False,
        normalize=True,
        context=context,
        gcfcf=gcfcf,
        **kwargs
    )
    return result


def restore_list_serial_workflow(
    model_imagelist,
    psf_imagelist,
    residual_imagelist=None,
    restore_facets=1,
    restore_overlap=0,
    restore_taper="tukey",
    **kwargs
):
    """Create a graph to calculate the restored image

    :param model_imagelist: Model list
    :param psf_imagelist: PSF list
    :param residual_imagelist: Residual list
    :param kwargs: Parameters for functions in components
    :param restore_facets: Number of facets used per axis (used to distribute)
    :param restore_overlap: Overlap in pixels (0 is best)
    :param restore_taper: Type of taper between facets
    :return: list of restored images
    """
    if residual_imagelist is not None:
        log.error("Model and residual list have different lengths")
        raise ValueError("Model and residual list have different lengths")

    psf_list = sum_invert_results(psf_imagelist)
    psf = normalize_sumwt(psf_list[0], psf_list[1])
    clean_beam = fit_psf(psf)

    if residual_imagelist is not None:
        residual_list = remove_sumwt(residual_imagelist)
        restored_list = [
            restore_cube(
                model_imagelist[i],
                clean_beam=clean_beam,
                residual=residual_list[i],
                **kwargs
            )
            for i, _ in enumerate(model_imagelist)
        ]
    else:
        restored_list = [
            restore_cube(
                model_imagelist[i], clean_beam=clean_beam, residual=None, **kwargs
            )
            for i, _ in enumerate(model_imagelist)
        ]
    return restored_list


def deconvolve_list_serial_workflow(
    dirty_list, psf_list, model_imagelist, prefix="", mask=None, **kwargs
):
    """Create a graph for deconvolution, adding to the model

    :param dirty_list: list of dirty images
    :param psf_list: list of psfs
    :param model_imagelist: list of models
    :param prefix: Informative prefix to log messages
    :param mask: Mask for deconvolution
    :param kwargs: Parameters for functions
    :return: List of deconvolved images

    For example::

        dirty_imagelist = invert_list_serial_workflow(vis_list, model_imagelist, context='2d',
                                                          dopsf=False, normalize=True)
        psf_imagelist = invert_list_serial_workflow(vis_list, model_imagelist, context='2d',
                                                        dopsf=True, normalize=True)
        dec_imagelist = deconvolve_list_serial_workflow(dirty_imagelist, psf_imagelist,
                model_imagelist, niter=1000, fractional_threshold=0.01,
                scales=[0, 3, 10], algorithm='mmclean', nmoment=3, nchan=freqwin,
                threshold=0.1, gain=0.7)

    """
    nchan = len(dirty_list)
    nmoment = get_parameter(kwargs, "nmoment", 0)

    # assert isinstance(dirty_list, list), dirty_list
    # assert isinstance(psf_list, list), psf_list
    # assert isinstance(model_imagelist, list), model_imagelist

    def deconvolve(dirty, psf, model, facet, gthreshold, msk=None):
        if prefix == "":
            lprefix = "subimage %d" % facet
        else:
            lprefix = "%s, subimage %d" % (prefix, facet)

        if nmoment > 0:
            moment0 = calculate_image_frequency_moments(dirty)
            this_peak = (
                numpy.max(numpy.abs(moment0.data[0, ...]))
                / dirty["pixels"].data.shape[0]
            )
        else:
            ref_chan = dirty["pixels"].data.shape[0] // 2
            this_peak = numpy.max(numpy.abs(dirty.data[ref_chan, ...]))

        if this_peak > 1.1 * gthreshold:
            kwargs["threshold"] = gthreshold
            result, _ = deconvolve_cube(dirty, psf, prefix=lprefix, mask=msk, **kwargs)

            if result["pixels"].data.shape[0] == model["pixels"].data.shape[0]:
                result.data += model.data
            return result
        else:

            return model.copy(deep=True)

    deconvolve_facets = get_parameter(kwargs, "deconvolve_facets", 1)
    deconvolve_overlap = get_parameter(kwargs, "deconvolve_overlap", 0)
    deconvolve_taper = get_parameter(kwargs, "deconvolve_taper", None)
    if deconvolve_facets > 1 and deconvolve_overlap > 0:
        deconvolve_number_facets = (deconvolve_facets - 2) ** 2
    else:
        deconvolve_number_facets = deconvolve_facets ** 2

    model_imagelist = image_gather_channels(model_imagelist)

    # Scatter the separate channel images into deconvolve facets and then gather channels for each facet.
    # This avoids constructing the entire spectral cube.
    dirty_list_trimmed = remove_sumwt(dirty_list)
    scattered_channels_facets_dirty_list = [
        image_scatter_facets(
            d,
            facets=deconvolve_facets,
            overlap=deconvolve_overlap,
            taper=deconvolve_taper,
        )
        for d in dirty_list_trimmed
    ]

    # Now we do a transpose and gather
    scattered_facets_list = [
        image_gather_channels(
            [scattered_channels_facets_dirty_list[chan][facet] for chan in range(nchan)]
        )
        for facet in range(deconvolve_number_facets)
    ]

    psf_list_trimmed = remove_sumwt(psf_list)
    psf_list_trimmed = image_gather_channels(psf_list_trimmed)

    scattered_model_imagelist = image_scatter_facets(
        model_imagelist, facets=deconvolve_facets, overlap=deconvolve_overlap
    )

    # Work out the threshold. Need to find global peak over all dirty_list images
    threshold = get_parameter(kwargs, "threshold", 0.0)
    fractional_threshold = get_parameter(kwargs, "fractional_threshold", 0.1)
    nmoment = get_parameter(kwargs, "nmoment", 0)
    use_moment0 = nmoment > 0

    # Find the global threshold. This uses the peak in the average on the frequency axis since we
    # want to use it in a stopping criterion in a moment clean
    global_threshold = threshold_list(
        scattered_facets_list,
        threshold,
        fractional_threshold,
        use_moment0=use_moment0,
        prefix=prefix,
    )

    facet_list = numpy.arange(deconvolve_number_facets).astype("int")
    if mask is None:
        scattered_results_list = [
            deconvolve(d, psf_list_trimmed, m, facet, global_threshold)
            for d, m, facet in zip(
                scattered_facets_list, scattered_model_imagelist, facet_list
            )
        ]
    else:
        mask_list = image_scatter_facets(
            mask, facets=deconvolve_facets, overlap=deconvolve_overlap
        )
        scattered_results_list = [
            deconvolve(d, psf_list_trimmed, m, facet, global_threshold, msk)
            for d, m, facet, msk in zip(
                scattered_facets_list, scattered_model_imagelist, facet_list, mask_list
            )
        ]

    # Gather the results back into one image, correcting for overlaps as necessary. The taper function is is used to
    # feather the facets together
    gathered_results_list = image_gather_facets(
        scattered_results_list,
        model_imagelist,
        facets=deconvolve_facets,
        overlap=deconvolve_overlap,
        taper=deconvolve_taper,
    )

    return image_scatter_channels(gathered_results_list, subimages=nchan)


def deconvolve_channel_list_serial_workflow(
    dirty_list, psf_list, model_imagelist, subimages, **kwargs
):
    """Create a graph for deconvolution by channels, adding to the model

    Does deconvolution channel by channel.
    :param dirty_list: List of dirty images
    :param psf_list: List of PSFs, must be the size of a facet
    :param model_imagelist: list of current model
    :param subimages: Number of channels to split into
    :param kwargs: Parameters for functions in components
    :return: list of updated models
    """

    def deconvolve_subimage(dirty, psf):
        # assert isinstance(dirty, Image)
        # assert isinstance(psf, Image)
        comp, _ = deconvolve_cube(dirty, psf, **kwargs)
        return comp

    def add_model(sum_model, model):
        # assert isinstance(output, Image)
        # assert isinstance(model, Image)
        sum_model.data += model.data
        return sum_model

    output = create_empty_image_like(model_imagelist)
    dirty_lists = image_scatter_channels(dirty_list[0], subimages=subimages)
    results = [
        deconvolve_subimage(dirty_list, psf_list[0]) for dirty_list in dirty_lists
    ]
    result = image_gather_channels(results, output, subimages=subimages)
    return add_model(result, model_imagelist)


def weight_list_serial_workflow(
    vis_list, model_imagelist, gcfcf=None, weighting="uniform", **kwargs
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

    if gcfcf is None:
        gcfcf = [
            create_pswf_convolutionfunction(
                model_imagelist[centre],
                polarisation_frame=vis_list[0].blockvisibility_acc.polarisation_frame,
            )
        ]

    def grid_wt(vis, model, g):
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
        grid_wt(vis_list[i], model_imagelist[i], gcfcf) for i in range(len(vis_list))
    ]

    merged_weight_grid = griddata_merge_weights(weight_list)

    def re_weight(vis, model, gd, g):
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
        re_weight(v, model_imagelist[i], merged_weight_grid, gcfcf)
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
