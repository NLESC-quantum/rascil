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
    extract_skycomponents_from_skymodel,
)
from rascil.workflows.rsexecute import calibrate_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import (
    invert_list_rsexecute_workflow,
    predict_list_rsexecute_workflow,
    subtract_list_rsexecute_workflow,
    deconvolve_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.skymodel.skymodel_rsexecute import (
    predict_skymodel_list_rsexecute_workflow,
    invert_skymodel_list_rsexecute_workflow,
    residual_skymodel_list_rsexecute_workflow,
    restore_skymodel_list_rsexecute_workflow,
    restore_centre_skymodel_list_rsexecute_workflow
)

log = logging.getLogger("rascil-logger")


def ical_skymodel_list_rsexecute_workflow(
    vis_list,
    model_imagelist,
    context,
    skymodel_list=None,
    gcfcf=None,
    calibration_context="TG",
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
        vis_list, model_imagelist, context=context, dopsf=True, gcfcf=gcfcf, **kwargs
    )

    # Create a list of copied input visibilities
    model_vislist = [
        rsexecute.execute(copy_visibility, nout=1)(v, zero=True) for v in vis_list
    ]

    # Create a list of visibilities for the calibration (?)
    if do_selfcal:
        cal_vis_list = [rsexecute.execute(copy_visibility, nout=1)(v) for v in vis_list]
    else:
        cal_vis_list = vis_list

    if do_selfcal:
        # Make the predicted visibilities, selfcalibrate against it correcting the gains, then
        # form the residual visibility, then make the residual image
        if skymodel_list is None:
            skymodel_list = [
                rsexecute.execute(SkyModel)(image=model) for model in model_imagelist
            ]

        #  Make the predicted visibilities
        predicted_model_vislist = predict_skymodel_list_rsexecute_workflow(
            model_vislist,
            skymodel_list,
            context=context,
            gcfcf=gcfcf,
            docal=True,
            **kwargs,
        )
        # Selfcalibrate against it correcting the gains
        cal_vis_list, gt_list = calibrate_list_rsexecute_workflow(
            cal_vis_list,
            predicted_model_vislist,
            gt_list,
            calibration_context=calibration_context,
            **kwargs,
        )

        # Erase data in the input model_imagelist
        def zero_model_image(im):
            log.info(
                "ical_list_rsexecute_workflow: setting initial model to zero after initial selfcal"
            )
            im["pixels"].data[...] = 0.0
            return im

        model_imagelist = [
            rsexecute.execute(zero_model_image, nout=1)(model)
            for model in model_imagelist
        ]
        skymodel_list = [
            rsexecute.execute(SkyModel)(image=model) for model in model_imagelist
        ]

        # Make the residual images for the skymodels
        residual_imagelist = invert_skymodel_list_rsexecute_workflow(
            cal_vis_list,
            skymodel_list,
            gcfcf=gcfcf,
            docal=True,
            dopsf=False,
            iteration=0,
            **kwargs,
        )
    else:

        if skymodel_list is None:
            skymodel_list = [
                rsexecute.execute(SkyModel)(image=model) for model in model_imagelist
            ]

        residual_imagelist = residual_skymodel_list_rsexecute_workflow(
            cal_vis_list,
            model_imagelist,
            context=context,
            skymodel_list=skymodel_list,
            gcfcf=gcfcf,
            **kwargs,
        )

    # First major cycle: updte model and components. The component list is across the entire frequency range
    deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(
        residual_imagelist,
        psf_imagelist,
        model_imagelist,
        prefix=f"{pipeline_name} cycle 0",
        **kwargs,
    )
    # Set the skymodel image and then extract skycomponents
    def setmodel(sm, im):
        sm.image = im
        sm = extract_skycomponents_from_skymodel(sm, **kwargs)
        return sm

    # Next major cycles, if nmajor>1
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):

            skymodel_list = [
                rsexecute.execute(setmodel, nout=1)(skymodel_list[i], m)
                for i, m in enumerate(deconvolve_model_imagelist)
            ]

            if do_selfcal:
                model_vislist = predict_skymodel_list_rsexecute_workflow(
                    model_vislist,
                    skymodel_list,
                    context=context,
                    gcfcf=gcfcf,
                    docal=True,
                    **kwargs,
                )

                cal_vis_list = [rsexecute.execute(copy_visibility)(v) for v in vis_list]
                cal_vis_list, gt_list = calibrate_list_rsexecute_workflow(
                    cal_vis_list,
                    model_vislist,
                    gt_list,
                    calibration_context=calibration_context,
                    iteration=cycle,
                    **kwargs,
                )
                residual_vislist = subtract_list_rsexecute_workflow(
                    cal_vis_list, model_vislist
                )

                residual_imagelist = invert_skymodel_list_rsexecute_workflow(
                    residual_vislist,
                    skymodel_list,
                    gcfcf=gcfcf,
                    docal=True,
                    dopsf=False,
                    iteration=0,
                    **kwargs,
                )
            else:
                residual_imagelist = residual_skymodel_list_rsexecute_workflow(
                    cal_vis_list,
                    deconvolve_model_imagelist,
                    context=context,
                    skymodel_list=skymodel_list,
                    gcfcf=gcfcf,
                    **kwargs,
                )

            deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(
                residual_imagelist,
                psf_imagelist,
                deconvolve_model_imagelist,
                prefix=f"{pipeline_name} cycle {cycle+1}",
                **kwargs,
            )
            
    # Now recreate the sky models
    skymodel_list = [
        rsexecute.execute(setmodel, nout=1)(skymodel_list[i], m)
        for i, m in enumerate(deconvolve_model_imagelist)
    ]

    residual_imagelist = residual_skymodel_list_rsexecute_workflow(
        cal_vis_list,
        deconvolve_model_imagelist,
        context=context,
        skymodel_list=skymodel_list,
        gcfcf=gcfcf,
        **kwargs,
    )
    output = get_parameter(kwargs, "restored_output", "list")
    if output == "integrated":
        restored_imagelist = restore_centre_skymodel_list_rsexecute_workflow(
            skymodel_list, psf_imagelist, residual_imagelist, **kwargs
        )
    elif output == "list":
        restored_imagelist = restore_skymodel_list_rsexecute_workflow(
            skymodel_list, psf_imagelist, residual_imagelist, **kwargs
        )
    else:
        raise ValueError(f"continuum_imaging_list_rsexecute_workflow: Unknown restored_output {output}")

    return deconvolve_model_imagelist, residual_imagelist, restored_imagelist, \
        skymodel_list, gt_list


def continuum_imaging_skymodel_list_rsexecute_workflow(
    vis_list, model_imagelist, context, skymodel_list=None, gcfcf=None, **kwargs
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
        deconvolve_model_imagelist,
        residual_imagelist,
        restore_imagelist,
        skymodel_list,
        gt_list,
    ) = ical_skymodel_list_rsexecute_workflow(
        vis_list,
        model_imagelist,
        context=context,
        skymodel_list=skymodel_list,
        gcfcf=gcfcf,
        calibration_context="",
        do_selfcal=False,
        pipeline_name="cip",
        **kwargs,
    )
    return (
        deconvolve_model_imagelist,
        residual_imagelist,
        restore_imagelist,
        skymodel_list,
    )


def spectral_line_imaging_skymodel_list_rsexecute_workflow(
    vis_list,
    model_imagelist,
    context,
    continuum_model_imagelist=None,
    vis_slices=1,
    facets=1,
    gcfcf=None,
    **kwargs,
):
    """Create graph for spectral line imaging pipeline

    Uses the continuum imaging rsexecute pipeline after subtraction of a continuum model

    :param vis_list: List of vis (or graph)
    :param model_imagelist: List of Spectral line model (or graph)
    :param continuum_model_imagelist: Continuum model list (or graph)
    :param context: Imaging context
    :param vis_slices: Number of visibility slices (time or w)
    :param facets: Number of facets on each x,y axis
    :param kwargs: Parameters for functions in components
    :return: list of (deconvolved model, residual, restored) or graph
    """
    if continuum_model_imagelist is not None:
        vis_list = predict_list_rsexecute_workflow(
            vis_list,
            continuum_model_imagelist,
            context=context,
            gcfcf=gcfcf,
            vis_slices=vis_slices,
            **kwargs,
        )

    return continuum_imaging_skymodel_list_rsexecute_workflow(
        vis_list,
        model_imagelist,
        context=context,
        gcfcf=gcfcf,
        pipeline_name="slip",
        **kwargs,
    )
