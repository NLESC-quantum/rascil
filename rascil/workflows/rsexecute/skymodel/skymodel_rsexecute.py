__all__ = [
    "predict_skymodel_list_rsexecute_workflow",
    "restore_skymodel_list_rsexecute_workflow",
    "restore_centre_skymodel_list_rsexecute_workflow",
    "invert_skymodel_list_rsexecute_workflow",
    "deconvolve_skymodel_list_rsexecute_workflow",
]

import logging

from rascil.data_models import get_parameter
from rascil.processing_components import (
    normalise_sumwt,
    image_scatter_facets,
    image_gather_facets,
)
from rascil.processing_components.image import restore_cube, fit_psf
from rascil.processing_components.imaging.imaging_helpers import remove_sumwt
from rascil.processing_components.skycomponent.operations import restore_skycomponent
from rascil.processing_components.skycomponent.taylor_terms import (
    find_skycomponents_frequency_taylor_terms,
)
from rascil.processing_components.skymodel.skymodel_imaging import (
    skymodel_predict_calibrate,
    skymodel_calibrate_invert,
)
from rascil.workflows.rsexecute import (
    invert_list_rsexecute_workflow,
    predict_list_rsexecute_workflow,
    subtract_list_rsexecute_workflow,
    zero_list_rsexecute_workflow,
    sum_invert_results_rsexecute,
    deconvolve_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger("rascil-logger")


def predict_skymodel_list_rsexecute_workflow(obsvis, skymodel_list, **kwargs):
    """Predict from a list of skymodels

    If obsvis is a list then we pair obsvis element and skymodel_list element and predict
    If obvis is BlockVisibility then we calculate BlockVisibility for each skymodel

    :param obsvis: Observed Block Visibility or list or graph
    :param skymodel_list: skymodel list
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
    """

    if isinstance(obsvis, list):
        if len(obsvis) != len(skymodel_list):
            raise ValueError("Obsvis and skymodel lists should have the same length")
        return [
            rsexecute.execute(skymodel_predict_calibrate, nout=1)(
                obsvis[ism], sm, **kwargs
            )
            for ism, sm in enumerate(skymodel_list)
        ]
    else:
        return [
            rsexecute.execute(skymodel_predict_calibrate, nout=1)(obsvis, sm, **kwargs)
            for ism, sm in enumerate(skymodel_list)
        ]


def invert_skymodel_list_rsexecute_workflow(vis_list, skymodel_list, **kwargs):
    """Calibrate and invert from a skymodel, iterating over the skymodel

    The function get_pb should have the signature:

        get_pb(BlockVisibility, Image)

    and should return the primary beam for the blockvisibility.

    The return is a graph for a set of tuples of (dirty, sensitivity image)

    :param vis_list: List of Visibility data models
    :param skymodel_list: skymodel list
    :param kwargs: Parameters for functions in components
    :return: List of (image, weight) tuples)
    """

    return [
        rsexecute.execute(skymodel_calibrate_invert, nout=1)(vis_list[i], sm, **kwargs)
        for i, sm in enumerate(skymodel_list)
    ]


def restore_centre_skymodel_list_rsexecute_workflow(
    skymodel_list, psf_imagelist, residual_imagelist=None, clean_beam=None, **kwargs
):
    """Create a graph to calculate the restored skymodel at the centre channel

    :param skymodel_list: Skymodel list (or graph)
    :param psf_imagelist: PSF list (or graph)
    :param residual_imagelist: Residual list (or graph)
    :param kwargs: Parameters for functions in components
    :param clean_beam: Clean beam e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    :return: list of restored images (or graph)
    """
    _check_imagelist_lengths(psf_imagelist, residual_imagelist, skymodel_list)

    # Find the PSF by summing over all channels, fit to this psf
    if clean_beam is None:
        psf = sum_invert_results_rsexecute(psf_imagelist)[0]
        clean_beam = rsexecute.execute(fit_psf, nout=1)(psf)

    # Add the model over all channels
    centre = len(skymodel_list) // 2

    def skymodel_restore(s, res, cb):
        res_image = restore_cube(s.image, residual=res, clean_beam=cb)
        return restore_skycomponent(res_image, s.components, cb)

    residual = sum_invert_results_rsexecute(residual_imagelist)[0]
    restored = rsexecute.execute(skymodel_restore, nout=1)(
        skymodel_list[centre], residual, clean_beam
    )

    return restored


def _check_imagelist_lengths(psf_imagelist, residual_imagelist, skymodel_list):
    """Check that the various image lists are congruent

    Raise ValueError when in error

    :param psf_imagelist:
    :param residual_imagelist:
    :param skymodel_list:
    """
    if residual_imagelist is not None:
        if len(skymodel_list) != len(residual_imagelist):
            errmsg = "Skymodel and residual list have different lengths"
            log.error(errmsg)
            raise ValueError(errmsg)


def restore_skymodel_single_list_rsexecute_workflow(
    skymodel_list, psf_imagelist, residual_imagelist=None, clean_beam=None, **kwargs
):
    """Create a graph to calculate the restored skymodel

    :param skymodel_list: Skymodel list (or graph)
    :param psf_imagelist: PSF list (or graph)
    :param residual_imagelist: Residual list (or graph)
    :param clean_beam: Clean beam e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    :param kwargs: Parameters for functions in components
    :return: list of restored images (or graph)
    """
    _check_imagelist_lengths(psf_imagelist, residual_imagelist, skymodel_list)

    if clean_beam is None:
        psf_list = sum_invert_results_rsexecute(psf_imagelist)
        psf = rsexecute.execute(normalise_sumwt)(psf_list[0], psf_list[1])
        clean_beam = rsexecute.execute(fit_psf, nout=1)(psf)

    def skymodel_restore(s, res, cb):
        res_image = restore_cube(s.image, residual=res, clean_beam=cb)
        return restore_skycomponent(res_image, s.components, cb)

    restored_list = [
        rsexecute.execute(skymodel_restore, nout=1)(
            sm, residual_imagelist[ism][0], clean_beam
        )
        for ism, sm in enumerate(skymodel_list)
    ]

    return restored_list


def restore_skymodel_list_rsexecute_workflow(
    skymodel_list,
    psf_imagelist,
    residual_imagelist=None,
    restore_facets=1,
    restore_overlap=8,
    restore_taper="tukey",
    clean_beam=None,
    **kwargs
):
    """Create a graph to calculate the restored image

    :param model_imagelist: Model list (or graph)
    :param psf_imagelist: PSF list (or graph)
    :param residual_imagelist: Residual list (or graph)
    :param clean_beam: Clean beam e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    :param kwargs: Parameters for functions in components
    :param restore_facets: Number of facets used per axis (used to distribute)
    :param restore_overlap: Overlap in pixels (0 is best)
    :param restore_taper: Type of taper between facets
    :return: list of restored images (or graph)
    """
    _check_imagelist_lengths(psf_imagelist, residual_imagelist, skymodel_list)

    if clean_beam is None:
        clean_beam_list = sum_invert_results_rsexecute(psf_imagelist)
        psf = rsexecute.execute(normalise_sumwt)(clean_beam_list[0], clean_beam_list[1])
        clean_beam = rsexecute.execute(fit_psf)(psf)

    if restore_overlap < 0:
        raise ValueError("Number of pixels for restore overlap must be >= 0")

    if restore_facets % 2 == 0 or restore_facets == 1:
        actual_number_facets = restore_facets
    else:
        actual_number_facets = max(1, (restore_facets - 1))

    # Scatter each list element into a list. We will then run restore_cube on each
    facet_model_list = [
        rsexecute.execute(
            image_scatter_facets, nout=actual_number_facets * actual_number_facets
        )(sm.image, facets=restore_facets, overlap=restore_overlap, taper=restore_taper)
        for sm in skymodel_list
    ]

    if residual_imagelist is not None:
        residual_list = rsexecute.execute(remove_sumwt, nout=len(residual_imagelist))(
            residual_imagelist
        )
        facet_residual_list = [
            rsexecute.execute(
                image_scatter_facets, nout=actual_number_facets * actual_number_facets
            )(
                residual,
                facets=restore_facets,
                overlap=restore_overlap,
                taper=restore_taper,
            )
            for residual in residual_list
        ]
        facet_restored_list = [
            [
                rsexecute.execute(
                    restore_cube, nout=actual_number_facets * actual_number_facets
                )(
                    model=facet_model_list[i][im],
                    residual=facet_residual_list[i][im],
                    clean_beam=clean_beam,
                )
                for im, _ in enumerate(facet_model_list[i])
            ]
            for i, _ in enumerate(skymodel_list)
        ]
    else:
        facet_restored_list = [
            [
                rsexecute.execute(
                    restore_cube, nout=actual_number_facets * actual_number_facets
                )(model=facet_model_list[i][im], clean_beam=clean_beam)
                for im, _ in enumerate(facet_model_list[i])
            ]
            for i, _ in enumerate(skymodel_list)
        ]

    # Now we gather the results across all facets
    restored_imagelist = [
        rsexecute.execute(image_gather_facets)(
            facet_restored_list[i],
            skymodel_list[i].image,
            facets=restore_facets,
            overlap=restore_overlap,
            taper=restore_taper,
        )
        for i, _ in enumerate(skymodel_list)
    ]

    def skymodel_restore_component(s, restored_image, cb):
        return restore_skycomponent(restored_image, s.components, cb)

    restored_imagelist = [
        rsexecute.execute(skymodel_restore_component, nout=1)(
            sm, restored_imagelist[ism], clean_beam
        )
        for ism, sm in enumerate(skymodel_list)
    ]

    def set_clean_beam(r, cb):
        r.attrs["clean_beam"] = cb
        return r

    restored_imagelist = [
        rsexecute.execute(set_clean_beam, nout=1)(r, clean_beam)
        for r in restored_imagelist
    ]
    return rsexecute.optimize(restored_imagelist)


def residual_skymodel_list_rsexecute_workflow(
    vis, model_imagelist, context="ng", skymodel_list=None, get_pb=None, **kwargs
):
    """Create a graph to calculate residual image for a skymodel_list

    The function get_pb should have the signature:

        get_pb(BlockVisibility, Image)

    and should return the primary beam for the blockvisibility e.g. using average
    parallactic angle

    :param vis: List of vis (or graph)
    :param model_imagelist: Model used to determine image parameters (or graph)
    :param context: Imaging context e.g. '2d', 'wstack'
    :param skymodel_list: List of skymodels (or graph)
    :param kwargs: Parameters for functions in components
    :return: list of (image, sumwt) tuples or graph
    """
    model_vis = zero_list_rsexecute_workflow(vis)

    if skymodel_list is not None:
        model_vis = predict_skymodel_list_rsexecute_workflow(
            model_vis,
            skymodel_list,
            context=context,
            docal=True,
            get_pb=get_pb,
            **kwargs
        )
    else:
        model_vis = predict_list_rsexecute_workflow(
            model_vis, model_imagelist, context=context, get_pb=get_pb, **kwargs
        )
    residual_vis = subtract_list_rsexecute_workflow(vis, model_vis)

    if skymodel_list is not None:
        result = invert_skymodel_list_rsexecute_workflow(
            residual_vis,
            skymodel_list,
            docal=True,
            dopsf=False,
            get_pb=get_pb,
            **kwargs
        )
    else:
        result = invert_list_rsexecute_workflow(
            residual_vis,
            model_imagelist,
            context=context,
            dopsf=False,
            normalise=True,
            get_pb=get_pb,
            **kwargs
        )
    return rsexecute.optimize(result)


def deconvolve_skymodel_list_rsexecute_workflow(
    dirty_image_list, psf_list, skymodel_list, prefix="", fit_skymodel=False, **kwargs
):
    """Deconvolve using a skymodel

    This will either fit for the brightest components and add those to the
    skymodel components or use (optionally faceted) CLEAN based deconvolution

    :param dirty_image_list: List of dirty images (or graphs)
    :param psf_list: List of corresponding psf images (or graphs)
    :param skymodel_list: list of skymodels (or graph)
    :param prefix: Informational prefix for logging messages
    :param fit_skymodel: Fit the skymodel?
    :param kwargs:
    :return: list of skymodels (or graph)
    """
    component_method = get_parameter(kwargs, "component_method", None)
    component_threshold = get_parameter(kwargs, "component_threshold", None)
    if fit_skymodel and component_method == "fit" and component_threshold is not None:
        # Update the skymodel with point sources found in moment 0
        # and fitted by a polynomial in frequency.

        skymodel_list = rsexecute.execute(
            convert_skycomponents_taylor_terms_list, nout=len(skymodel_list)
        )(dirty_image_list, skymodel_list, **kwargs)
        return skymodel_list
    elif (
        fit_skymodel
        and component_method == "extract"
        and component_threshold is not None
    ):
        # Update the skymodel with point sources found in moment 0
        # and extracted from the frequency cube without fitting
        kwargs["nmoment"] = len(skymodel_list)
        skymodel_list = rsexecute.execute(
            convert_skycomponents_taylor_terms_list, nout=len(skymodel_list)
        )(dirty_image_list, skymodel_list, **kwargs)
        return skymodel_list
    else:

        def extract_sm_image(s):
            return s.image

        deconvolve_model_imagelist = [
            rsexecute.execute(extract_sm_image, nout=1)(sm) for sm in skymodel_list
        ]

        deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(
            dirty_image_list,
            psf_list,
            deconvolve_model_imagelist,
            prefix=prefix,
            **kwargs
        )

        def skymodel_update_image(sm, im):
            if not sm.fixed:
                sm.image = im
            return sm

        skymodel_list = [
            rsexecute.execute(skymodel_update_image, nout=1)(skymodel_list[i], m)
            for i, m in enumerate(deconvolve_model_imagelist)
        ]
        # Optimize to reduce the size of graph
        return rsexecute.optimize(skymodel_list)


def convert_skycomponents_taylor_terms_list(dirty_image_list, skymodel_list, **kwargs):
    skycomponent_list = find_skycomponents_frequency_taylor_terms(
        dirty_image_list, **kwargs
    )

    def add_skycomponents(sm, scl):
        if len(scl) > 0:
            for sc in scl:
                sm.components.append(sc)
        return sm

    if len(skycomponent_list) > 0:
        skymodel_list = [
            add_skycomponents(sm, skycomponent_list[ism])
            for ism, sm in enumerate(skymodel_list)
        ]
    return skymodel_list
