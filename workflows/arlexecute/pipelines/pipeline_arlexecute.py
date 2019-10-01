""" Pipeline functions. SDP standard pipelines expressed as functions.
"""

from data_models.parameters import get_parameter
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.griddata.kernels import create_pswf_convolutionfunction
from wrappers.arlexecute.visibility.base import copy_visibility
from ..calibration.calibration_arlexecute import calibrate_list_arlexecute_workflow
from ..imaging.imaging_arlexecute import invert_list_arlexecute_workflow, residual_list_arlexecute_workflow, \
    predict_list_arlexecute_workflow, subtract_list_arlexecute_workflow, \
    restore_list_arlexecute_workflow, deconvolve_list_arlexecute_workflow


def ical_list_arlexecute_workflow(vis_list, model_imagelist, context, vis_slices=1, facets=1,
                                  gcfcf=None, calibration_context='TG', do_selfcal=True, **kwargs):
    """Create graph for ICAL pipeline

    :param vis_list:
    :param model_imagelist:
    :param context: imaging context e.g. '2d'
    :param calibration_context: Sequence of calibration steps e.g. TGB
    :param do_selfcal: Do the selfcalibration?
    :param kwargs: Parameters for functions in components
    :return:
    """
    
    gt_list = list()
    
    if gcfcf is None:
        gcfcf = [arlexecute.execute(create_pswf_convolutionfunction)(model_imagelist[0])]
    
    psf_imagelist = invert_list_arlexecute_workflow(vis_list, model_imagelist, dopsf=True, context=context,
                                                    vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    
    model_vislist = [arlexecute.execute(copy_visibility, nout=1)(v, zero=True) for v in vis_list]
    
    if do_selfcal:
        cal_vis_list = [arlexecute.execute(copy_visibility, nout=1)(v) for v in vis_list]
    else:
        cal_vis_list = vis_list
    
    if do_selfcal:
        # Make the predicted visibilities, selfcalibrate against it correcting the gains, then
        # form the residual visibility, then make the residual image
        predicted_model_vislist = predict_list_arlexecute_workflow(model_vislist, model_imagelist,
                                                                   context=context, vis_slices=vis_slices,
                                                                   facets=facets,
                                                                   gcfcf=gcfcf, **kwargs)
        recal_vis_list, gt_list = calibrate_list_arlexecute_workflow(cal_vis_list, predicted_model_vislist,
                                                                   calibration_context=calibration_context, **kwargs)
        residual_vislist = subtract_list_arlexecute_workflow(recal_vis_list, predicted_model_vislist)
        residual_imagelist = invert_list_arlexecute_workflow(residual_vislist, model_imagelist,
                                                             context=context, dopsf=False,
                                                             vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                                             iteration=0, **kwargs)
    else:
        # If we are not selfcalibrating it's much easier and we can avoid an unnecessary round of gather/scatter
        # for visibility partitioning such as timeslices and wstack.
        residual_imagelist = residual_list_arlexecute_workflow(cal_vis_list, model_imagelist, context=context,
                                                               vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                                               **kwargs)
    
    deconvolve_model_imagelist = deconvolve_list_arlexecute_workflow(residual_imagelist, psf_imagelist,
                                                                     model_imagelist,
                                                                     prefix='cycle 0',
                                                                     **kwargs)
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            if do_selfcal:
                predicted_model_vislist = predict_list_arlexecute_workflow(model_vislist, deconvolve_model_imagelist,
                                                                           context=context, vis_slices=vis_slices,
                                                                           facets=facets,
                                                                           gcfcf=gcfcf, **kwargs)
                recal_vis_list, gt_list = calibrate_list_arlexecute_workflow(cal_vis_list,
                                                                           predicted_model_vislist,
                                                                           calibration_context=calibration_context,
                                                                           iteration=cycle, **kwargs)
                residual_vislist = subtract_list_arlexecute_workflow(recal_vis_list, model_vislist)
                residual_imagelist = invert_list_arlexecute_workflow(residual_vislist, model_imagelist,
                                                                     context=context,
                                                                     vis_slices=vis_slices, facets=facets,
                                                                     gcfcf=gcfcf, **kwargs)
            else:
                residual_imagelist = residual_list_arlexecute_workflow(cal_vis_list, deconvolve_model_imagelist,
                                                                       context=context,
                                                                       vis_slices=vis_slices, facets=facets,
                                                                       gcfcf=gcfcf,
                                                                       **kwargs)
            
            prefix = "cycle %d" % (cycle + 1)
            deconvolve_model_imagelist = deconvolve_list_arlexecute_workflow(residual_imagelist, psf_imagelist,
                                                                             deconvolve_model_imagelist,
                                                                             prefix=prefix,
                                                                             **kwargs)
    residual_imagelist = residual_list_arlexecute_workflow(cal_vis_list, deconvolve_model_imagelist, context=context,
                                                           vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    restore_imagelist = restore_list_arlexecute_workflow(deconvolve_model_imagelist, psf_imagelist, residual_imagelist)
    return (deconvolve_model_imagelist, residual_imagelist, restore_imagelist, gt_list)


def continuum_imaging_list_arlexecute_workflow(vis_list, model_imagelist, context, gcfcf=None,
                                               vis_slices=1, facets=1, **kwargs):
    """ Create graph for the continuum imaging pipeline.
    
    Same as ICAL but with no selfcal.
    
    :param vis_list:
    :param model_imagelist:
    :param context: Imaging context
    :param kwargs: Parameters for functions in components
    :return:
    """
    if gcfcf is None:
        gcfcf = [arlexecute.execute(create_pswf_convolutionfunction)(model_imagelist[0])]
    
    psf_imagelist = invert_list_arlexecute_workflow(vis_list, model_imagelist, context=context, dopsf=True,
                                                    vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    
    residual_imagelist = residual_list_arlexecute_workflow(vis_list, model_imagelist, context=context, gcfcf=gcfcf,
                                                           vis_slices=vis_slices, facets=facets, **kwargs)
    
    deconvolve_model_imagelist = deconvolve_list_arlexecute_workflow(residual_imagelist, psf_imagelist,
                                                                     model_imagelist,
                                                                     prefix='cycle 0',
                                                                     **kwargs)
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            prefix = "cycle %d" % (cycle + 1)
            residual_imagelist = residual_list_arlexecute_workflow(vis_list, deconvolve_model_imagelist,
                                                                   context=context, vis_slices=vis_slices,
                                                                   facets=facets,
                                                                   gcfcf=gcfcf, **kwargs)
            deconvolve_model_imagelist = deconvolve_list_arlexecute_workflow(residual_imagelist, psf_imagelist,
                                                                             deconvolve_model_imagelist,
                                                                             prefix=prefix,
                                                                             **kwargs)
    
    residual_imagelist = residual_list_arlexecute_workflow(vis_list, deconvolve_model_imagelist, context=context,
                                                           vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    restore_imagelist = restore_list_arlexecute_workflow(deconvolve_model_imagelist, psf_imagelist,
                                                         residual_imagelist, **kwargs)
    return (deconvolve_model_imagelist, residual_imagelist, restore_imagelist)


def spectral_line_imaging_list_arlexecute_workflow(vis_list, model_imagelist, context, continuum_model_imagelist=None,
                                                   vis_slices=1, facets=1, gcfcf=None, **kwargs):
    """Create graph for spectral line imaging pipeline

    Uses the continuum imaging arlexecute pipeline after subtraction of a continuum model
    
    :param vis_list: List of visibility components
    :param model_imagelist: Spectral line model graph
    :param continuum_model_imagelist: Continuum model list
    :param context: Imaging context
    :param kwargs: Parameters for functions in components
    :return: (deconvolved model, residual, restored)
    """
    if continuum_model_imagelist is not None:
        vis_list = predict_list_arlexecute_workflow(vis_list, continuum_model_imagelist, context=context, gcfcf=gcfcf,
                                                    vis_slices=vis_slices, facets=facets, **kwargs)
    
    return continuum_imaging_list_arlexecute_workflow(vis_list, model_imagelist, context=context, gcfcf=gcfcf,
                                                      vis_slices=vis_slices, facets=facets, **kwargs)
