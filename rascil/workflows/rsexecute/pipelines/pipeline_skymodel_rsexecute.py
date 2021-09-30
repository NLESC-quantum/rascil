""" Pipeline functions using SkyModel. SDP standard pipelines expressed as functions.
"""

__all__ = [
    "ical_skymodel_list_rsexecute_workflow",
    "continuum_imaging_skymodel_list_rsexecute_workflow",
    "spectral_line_imaging_skymodel_list_rsexecute_workflow",
]

import logging

from rascil.data_models import SkyModel
from rascil.data_models.parameters import get_parameter
from rascil.processing_components import (
    copy_visibility,
)
from rascil.processing_components.skycomponent.taylor_terms import (
    calculate_skycomponent_list_taylor_terms,
    gather_skycomponents_from_channels,
)
from rascil.processing_components.image.taylor_terms import (
    calculate_frequency_taylor_terms_from_image_list,
)
from rascil.workflows.rsexecute import calibrate_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import (
    invert_list_rsexecute_workflow,
    predict_list_rsexecute_workflow,
    subtract_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.skymodel.skymodel_rsexecute import (
    predict_skymodel_list_rsexecute_workflow,
    invert_skymodel_list_rsexecute_workflow,
    residual_skymodel_list_rsexecute_workflow,
    restore_skymodel_list_rsexecute_workflow,
    restore_centre_skymodel_list_rsexecute_workflow,
    deconvolve_skymodel_list_rsexecute_workflow,
)

log = logging.getLogger("rascil-logger")


def ical_skymodel_list_rsexecute_workflow(
    vis_list,
    model_imagelist,
    context,
    skymodel_list=None,
    calibration_context="TG",
    controls=None,
    do_selfcal=True,
    pipeline_name="ical",
    **kwargs,
):
    """Create graph for ICAL pipeline using SkyModel

    :param vis_list: List of vis (or graph)
    :param model_imagelist:  list of models (or graph)
    :param skymodel_list: list of SkyModels
    :param context: imaging context e.g. '2d'
    :param calibration_context: Sequence of calibration steps e.g. TGB
    :param do_selfcal: Do the selfcalibration?
    :param kwargs: Parameters for functions in components
    :return:
    """

    gt_list = list()

    # Create PSFs
    psf_imagelist = invert_list_rsexecute_workflow(
        vis_list, model_imagelist, context=context, dopsf=True, **kwargs
    )

    def trim(x):
        return x[0]

    psf_imagelist_trimmed = [rsexecute.execute(trim)(d) for d in psf_imagelist]

    # Create a list of copied input visibilities
    model_vislist = [
        rsexecute.execute(copy_visibility, nout=1)(v, zero=True) for v in vis_list
    ]

    # Create a list of visibilities for the calibration (?)
    if do_selfcal:
        cal_vis_list = [rsexecute.execute(copy_visibility, nout=1)(v) for v in vis_list]
    else:
        cal_vis_list = vis_list

    # Ensure that we always have a skymodel to work with
    if skymodel_list is None:
        skymodel_list = [
            rsexecute.execute(SkyModel)(image=model) for model in model_imagelist
        ]

    if do_selfcal:
        # Make the predicted visibilities, selfcalibrate against it correcting the gains, then
        # form the residual visibility, then make the residual image

        predicted_model_vislist = predict_skymodel_list_rsexecute_workflow(
            model_vislist,
            skymodel_list,
            context=context,
            docal=True,
            **kwargs,
        )
        # Selfcalibrate against it correcting the gains
        cal_vis_list, gt_list = calibrate_list_rsexecute_workflow(
            cal_vis_list,
            predicted_model_vislist,
            gt_list,
            calibration_context=calibration_context,
            controls=controls,
            **kwargs,
        )

        # Erase data in the input model_imagelist
        def pipeline_zero_skymodel_image(sm):
            log.info(
                "ical_list_rsexecute_workflow: setting initial model to zero after initial selfcal"
            )
            if sm.image is not None:
                sm.image["pixels"].data[...] = 0.0
            return sm

        skymodel_list = [
            rsexecute.execute(pipeline_zero_skymodel_image, nout=1)(sm)
            for sm in skymodel_list
        ]

        # Make the residual images for the skymodels
        residual_imagelist = invert_skymodel_list_rsexecute_workflow(
            cal_vis_list,
            skymodel_list,
            docal=True,
            dopsf=False,
            iteration=0,
            **kwargs,
        )
    else:

        residual_imagelist = residual_skymodel_list_rsexecute_workflow(
            cal_vis_list,
            model_imagelist,
            context=context,
            skymodel_list=skymodel_list,
            **kwargs,
        )

    def trim(x):
        return x[0]

    residual_imagelist_trimmed = [
        rsexecute.execute(trim)(d) for d in residual_imagelist
    ]
    skymodel_list = deconvolve_skymodel_list_rsexecute_workflow(
        residual_imagelist_trimmed,
        psf_imagelist_trimmed,
        skymodel_list,
        prefix=f"{pipeline_name} cycle 0",
        fit_skymodel=True,
        **kwargs,
    )

    # Next major cycles, if nmajor>1
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):

            if do_selfcal:
                # Predict the visibility for the skymodel
                model_vislist = predict_skymodel_list_rsexecute_workflow(
                    model_vislist,
                    skymodel_list,
                    context=context,
                    docal=True,
                    **kwargs,
                )

                cal_vis_list = [rsexecute.execute(copy_visibility)(v) for v in vis_list]
                # Calibrate using the observed visibility and the model visibility
                cal_vis_list, gt_list = calibrate_list_rsexecute_workflow(
                    cal_vis_list,
                    model_vislist,
                    gt_list,
                    calibration_context=calibration_context,
                    controls=controls,
                    iteration=cycle,
                    **kwargs,
                )

                # Calculate the residual visibility
                residual_vislist = subtract_list_rsexecute_workflow(
                    cal_vis_list, model_vislist
                )
                # ... and the residual images
                residual_imagelist = invert_skymodel_list_rsexecute_workflow(
                    residual_vislist,
                    skymodel_list,
                    docal=True,
                    dopsf=False,
                    iteration=0,
                    **kwargs,
                )
            else:
                # Calculate the residual images
                residual_imagelist = residual_skymodel_list_rsexecute_workflow(
                    cal_vis_list,
                    skymodel_list,
                    context=context,
                    skymodel_list=skymodel_list,
                    **kwargs,
                )

            def trim(x):
                return x[0]

            residual_imagelist_trimmed = [
                rsexecute.execute(trim)(d) for d in residual_imagelist
            ]

            # Deconvolve to get an updated skymodel
            skymodel_list = deconvolve_skymodel_list_rsexecute_workflow(
                residual_imagelist_trimmed,
                psf_imagelist_trimmed,
                skymodel_list,
                prefix=f"{pipeline_name} cycle {cycle + 1}",
                **kwargs,
            )

            # Most of the computations are done here. We noticed that the
            # graph size increases non-linearly for each major cycle. We
            # faced issues on computing the entire graph at once and so here
            # we are persisting the graph so that a smaller graph size
            # starts executing as it gets constructed for each major cycle.
            # The idea is to introduce a convergence criteria so that we can
            # break the loop when criteria is met. More details can
            # be found here: https://jira.skatelescope.org/browse/SIM-1015
            skymodel_list = rsexecute.persist(skymodel_list)

    # We've finished so now we update the residual images and calculate the restored image
    residual_imagelist = residual_skymodel_list_rsexecute_workflow(
        cal_vis_list,
        skymodel_list,
        context=context,
        skymodel_list=skymodel_list,
        **kwargs,
    )

    (
        skymodel_list,
        residual_imagelist,
        restored_imagelist,
    ) = restore_skymodel_pipeline_rsexecute_workflow(
        skymodel_list, psf_imagelist, residual_imagelist, **kwargs
    )
    return (
        residual_imagelist,
        restored_imagelist,
        skymodel_list,
        gt_list,
    )


def restore_skymodel_pipeline_rsexecute_workflow(
    skymodel_list, psf_imagelist, residual_imagelist, **kwargs
):
    """Restore images using one of a number of methods selected by restored_output

    list: lists of images
    integrated: restored is the central channel image plus the average residual
    taylor: lists of Taylor terms of all images

    :param skymodel_imagelist: List (or graph) of skymodels
    :param psf_imagelist: List (or graph) of point spread functions
    :param residual_imagelist: List (or graph) of residual images
    :param kwargs:
    :return:
    """
    output = get_parameter(kwargs, "restored_output", "list")

    if output == "integrated":
        restored_imagelist = restore_centre_skymodel_list_rsexecute_workflow(
            skymodel_list, psf_imagelist, residual_imagelist, **kwargs
        )
    elif output == "taylor":
        # In this case, we overwrite the originals with Taylor term versions
        nmoment = get_parameter(kwargs, "nmoment", 1)

        def extract_sm_image(s):
            return s.image

        deconvolve_model_imagelist = [
            rsexecute.execute(extract_sm_image)(sm, nout=1)(sm) for sm in skymodel_list
        ]
        deconvolve_model_imagelist = rsexecute.execute(
            calculate_frequency_taylor_terms_from_image_list, nout=nmoment
        )(deconvolve_model_imagelist, nmoment=nmoment)

        def trim(x):
            return x[0]

        residual_imagelist_trimmed = [
            rsexecute.execute(trim)(d) for d in residual_imagelist
        ]
        residual_imagelist = rsexecute.execute(
            calculate_frequency_taylor_terms_from_image_list, nout=nmoment
        )(residual_imagelist_trimmed, nmoment=nmoment)

        def untrim(x):
            return (x, 0.0)

        residual_imagelist = [rsexecute.execute(untrim)(d) for d in residual_imagelist]
        # Now we need to create a skymodel in Taylor space
        skymodel_list = rsexecute.execute(
            convert_skycomponents_taylor_terms_list, nout=nmoment
        )(deconvolve_model_imagelist, nmoment=nmoment, skymodel_list=skymodel_list)

        # Note that the psf has not been converted to Taylor form
        restored_imagelist = restore_skymodel_list_rsexecute_workflow(
            skymodel_list, psf_imagelist, residual_imagelist, **kwargs
        )

    elif output == "list":
        restored_imagelist = restore_skymodel_list_rsexecute_workflow(
            skymodel_list, psf_imagelist, residual_imagelist, **kwargs
        )
    else:
        raise ValueError(
            f"continuum_imaging_skymodel_list_rsexecute_workflow: Unknown restored_output {output}"
        )

    return skymodel_list, residual_imagelist, restored_imagelist


def convert_skycomponents_taylor_terms_list(
    deconvolve_model_imagelist, nmoment, skymodel_list
):
    """Convert skycomponents into Taylor term form and update skymodel

    :param deconvolve_model_imagelist: Deconvolved model in Taylor term sequence
    :param nmoment: Number of moments/Taylor terms
    :param skymodel_list: Skymodel in Frequency space, to be updated
    :return:
    """
    # Extract the skycomponents in frequency space
    skycomponent_list = [sm.components for sm in skymodel_list]

    # Gather the different frequency components into a set of multifrequency skycomponents
    channel_sky_component_list = gather_skycomponents_from_channels(skycomponent_list)
    if len(channel_sky_component_list) == 0:
        skymodel_list = [
            SkyModel(
                image=deconvolve_model_imagelist[moment],
            )
            for moment in range(nmoment)
        ]
        return skymodel_list

    # Convert to Taylor term components [source][taylor term]
    skycomponent_list = calculate_skycomponent_list_taylor_terms(
        channel_sky_component_list, nmoment=nmoment
    )
    skymodel_list = [
        SkyModel(
            image=deconvolve_model_imagelist[moment],
            components=[scl[moment] for scl in skycomponent_list],
        )
        for moment in range(nmoment)
    ]
    return skymodel_list


def continuum_imaging_skymodel_list_rsexecute_workflow(
    vis_list, model_imagelist, context, skymodel_list=None, **kwargs
):
    """Create graph for the continuum imaging pipeline.

    Same as ICAL but with no selfcal.

    :param vis_list: List of vis (or graph)
    :param model_imagelist: List of models (or graph)
    :param skymodel_list: list of SkyModels
    :param context: Imaging context
    :param skymodel_list: list of SkyModels
    :param kwargs: Parameters for functions in components
    :return:
    """
    (
        residual_imagelist,
        restore_imagelist,
        skymodel_list,
        gt_list,
    ) = ical_skymodel_list_rsexecute_workflow(
        vis_list,
        model_imagelist,
        context=context,
        skymodel_list=skymodel_list,
        calibration_context="",
        do_selfcal=False,
        pipeline_name="cip",
        **kwargs,
    )
    return (
        residual_imagelist,
        restore_imagelist,
        skymodel_list,
    )


def spectral_line_imaging_skymodel_list_rsexecute_workflow(
    vis_list,
    model_imagelist,
    context,
    continuum_model_imagelist=None,
    **kwargs,
):
    """Create graph for spectral line imaging pipeline

    Uses the continuum imaging rsexecute pipeline after subtraction of a continuum model

    :param vis_list: List of vis (or graph)
    :param model_imagelist: List of Spectral line model (or graph)
    :param continuum_model_imagelist: Continuum model list (or graph)
    :param context: Imaging context e.g. ng or 2d
    :param kwargs: Parameters for functions in components
    :return: list of (deconvolved model, residual, restored) or graph
    """
    if continuum_model_imagelist is not None:
        vis_list = predict_list_rsexecute_workflow(
            vis_list,
            continuum_model_imagelist,
            context=context,
            **kwargs,
        )

    return continuum_imaging_skymodel_list_rsexecute_workflow(
        vis_list,
        model_imagelist,
        context=context,
        pipeline_name="slip",
        **kwargs,
    )
