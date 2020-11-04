""" Pipeline functions using SkyModel. SDP standard pipelines expressed as functions.
"""

__all__ = ['ical_skymodel_list_rsexecute_workflow',
           'continuum_imaging_skymodel_list_rsexecute_workflow',
           'spectral_line_imaging_skymodel_list_rsexecute_workflow']

import logging

from rascil.data_models.parameters import get_parameter
from rascil.processing_components.griddata import create_pswf_convolutionfunction
from rascil.processing_components.visibility import copy_visibility
from rascil.workflows.rsexecute.calibration.calibration_rsexecute import calibrate_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import invert_list_rsexecute_workflow, \
    predict_list_rsexecute_workflow, subtract_list_rsexecute_workflow, \
    deconvolve_list_rsexecute_workflow
from rascil.workflows.rsexecute.skymodel.skymodel_rsexecute import predict_skymodel_list_rsexecute_workflow, \
    invert_skymodel_list_rsexecute_workflow, residual_skymodel_list_rsexecute_workflow, \
    restore_skymodel_list_rsexecute_workflow

log = logging.getLogger('rascil-logger')


def ical_skymodel_list_rsexecute_workflow(vis_list, model_imagelist, context, skymodel_list, vis_slices=1, facets=1,
                                          gcfcf=None, calibration_context='TG', do_selfcal=True, **kwargs):
    """Create graph for ICAL pipeline using SkyModel

    :param vis_list: List of vis (or graph)
    :param model_imagelist:  list of models (or graph)
    :param skymodel_list: list of SkyModels
    :param context: imaging context e.g. '2d'
    :param vis_slices: Number of visibility slices (time or w)
    :param facets: Number of facets on each x,y axis
    :param calibration_context: Sequence of calibration steps e.g. TGB
    :param do_selfcal: Do the selfcalibration?
    :param kwargs: Parameters for functions in components
    :return:
    """
    
    # Check if SkyModel list is given
    assert len(skymodel_list) is not None, skymodel_list
    # Note that the following test will fail if skymodel_list is a Future or Delayed
    #assert isinstance(skymodel_list[0], SkyModel), skymodel_list[0]
    
    gt_list = list()
    
    # Function to copy image into a SkyModel
    def set_image(sm, dcm):
        sm.image = dcm.copy()
        return sm
    
    # Make wkernels
    if gcfcf is None:
        gcfcf = [rsexecute.execute(create_pswf_convolutionfunction)(m) for m in model_imagelist]
    
    # Create PSFs
    psf_imagelist = invert_list_rsexecute_workflow(vis_list, model_imagelist, context=context, dopsf=True, gcfcf=gcfcf,
                                                   **kwargs)
    
    # Create a list of copied input visibilities
    model_vislist = [rsexecute.execute(copy_visibility, nout=1)(v, zero=True) for v in vis_list]
    
    # Create a list of visibilities for the calibration (?)
    if do_selfcal:
        cal_vis_list = [rsexecute.execute(copy_visibility, nout=1)(v) for v in vis_list]
    else:
        cal_vis_list = vis_list
    if do_selfcal:
        # Make the predicted visibilities, selfcalibrate against it correcting the gains, then
        # form the residual visibility, then make the residual image
        
        #  Make the predicted visibilities
        
        predicted_model_vislist = predict_skymodel_list_rsexecute_workflow(model_vislist, skymodel_list,
                                                                           context=context, vis_slices=vis_slices,
                                                                           facets=facets,
                                                                           gcfcf=gcfcf, docal=True, **kwargs)
        # Selfcalibrate against it correcting the gains
        cal_vis_list, gt_list = calibrate_list_rsexecute_workflow(cal_vis_list,
                                                                  predicted_model_vislist,
                                                                  gt_list,
                                                                  calibration_context=calibration_context, **kwargs)
        
        def zero_model_image(im):
            log.info("ical_list_rsexecute_workflow: setting initial model to zero after initial selfcal")
            im["pixels"].data[...] = 0.0
            return im
        
        # Erase data in the input model_imagelist
        model_imagelist = [rsexecute.execute(zero_model_image, nout=1)(model) for model in model_imagelist]
        
        # Make the residual image
        skymodel_list = [rsexecute.execute(set_image)(skymodel_list[i], model_imagelist[i])
                           for i in range(len(skymodel_list))]
        
        residual_imagelist = invert_skymodel_list_rsexecute_workflow(cal_vis_list, skymodel_list,
                                                                     context=context, dopsf=False, docal=True,
                                                                     vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                                                     iteration=0, **kwargs)
    else:
        # If we are not selfcalibrating it's much easier and we can avoid an unnecessary round of gather/scatter
        # for visibility partitioning such as timeslices and wstack.
        
        skymodel_list = [rsexecute.execute(set_image)(skymodel_list[i], model_imagelist[i])
                           for i in range(len(skymodel_list))]
        residual_imagelist = residual_skymodel_list_rsexecute_workflow(cal_vis_list, model_imagelist, context=context,
                                                                       skymodel_list=skymodel_list, gcfcf=gcfcf,
                                                                       **kwargs)
    
    # First major cycle
    deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(residual_imagelist, psf_imagelist,
                                                                    model_imagelist,
                                                                    prefix='ical cycle 0',
                                                                    **kwargs)
    
    # Next major cycles, if nmajor>1
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            if do_selfcal:
                
                skymodel_list = [rsexecute.execute(set_image)(skymodel_list[i], deconvolve_model_imagelist[i])
                                   for i in range(len(skymodel_list))]
                model_vislist = predict_skymodel_list_rsexecute_workflow(model_vislist, skymodel_list,
                                                                         context=context,
                                                                         vis_slices=vis_slices,
                                                                         facets=facets, gcfcf=gcfcf, docal=True,
                                                                         **kwargs)
                
                cal_vis_list = [rsexecute.execute(copy_visibility)(v) for v in vis_list]
                cal_vis_list, gt_list = calibrate_list_rsexecute_workflow(cal_vis_list,
                                                                          model_vislist,
                                                                          gt_list,
                                                                          calibration_context=calibration_context,
                                                                          iteration=cycle, **kwargs)
                residual_vislist = subtract_list_rsexecute_workflow(cal_vis_list, model_vislist)
                
                skymodel_list = [rsexecute.execute(set_image)(skymodel_list[i], model_imagelist[i])
                                   for i in range(len(skymodel_list))]
                residual_imagelist = invert_skymodel_list_rsexecute_workflow(residual_vislist, skymodel_list,
                                                                             context=context, dopsf=False, docal=True,
                                                                             vis_slices=vis_slices, facets=facets,
                                                                             gcfcf=gcfcf,
                                                                             iteration=0, **kwargs)
            else:
                skymodel_list = [rsexecute.execute(set_image)(skymodel_list[i], deconvolve_model_imagelist[i])
                                   for i in range(len(skymodel_list))]
                residual_imagelist = residual_skymodel_list_rsexecute_workflow(cal_vis_list, deconvolve_model_imagelist,
                                                                               context=context,
                                                                               skymodel_list=skymodel_list,
                                                                               vis_slices=vis_slices, facets=facets,
                                                                               gcfcf=gcfcf, **kwargs)
            
            prefix = "ical cycle %d" % (cycle + 1)
            deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(residual_imagelist, psf_imagelist,
                                                                            deconvolve_model_imagelist,
                                                                            prefix=prefix,
                                                                            **kwargs)
    # Create residual images
    skymodel_list = [rsexecute.execute(set_image)(skymodel_list[i], deconvolve_model_imagelist[i])
                       for i in range(len(skymodel_list))]
    residual_imagelist = residual_skymodel_list_rsexecute_workflow(cal_vis_list, deconvolve_model_imagelist,
                                                                   context=context,
                                                                   skymodel_list=skymodel_list,
                                                                   vis_slices=vis_slices, facets=facets,
                                                                   gcfcf=gcfcf, **kwargs)
    # Create restored images
    restore_imagelist = restore_skymodel_list_rsexecute_workflow(skymodel_list=skymodel_list,
                                                                 psf_imagelist=psf_imagelist,
                                                                 residual_imagelist=residual_imagelist)
    # restore_imagelist = restore_list_rsexecute_workflow(deconvolve_model_imagelist, psf_imagelist, residual_imagelist)
    return (deconvolve_model_imagelist, residual_imagelist, restore_imagelist, skymodel_list, gt_list)


def continuum_imaging_skymodel_list_rsexecute_workflow(vis_list, model_imagelist, context, skymodel_list, gcfcf=None,
                                                       vis_slices=1, facets=1, **kwargs):
    """ Create graph for the continuum imaging pipeline.
    
    Same as ICAL but with no selfcal.
    
    :param vis_list: List of vis (or graph)
    :param model_imagelist: List of models (or graph)
    :param skymodel_list: list of SkyModels
    :param context: Imaging context
    :param skymodel_list: list of SkyModels
    :param kwargs: Parameters for functions in components
    :return:
    """
    deconvolve_model_imagelist, residual_imagelist, restore_imagelist, skymodel_list, gt_list = \
        ical_skymodel_list_rsexecute_workflow(vis_list, model_imagelist, context, skymodel_list=skymodel_list,
                                              vis_slices=vis_slices, facets=facets,
                                              gcfcf=gcfcf, calibration_context="", do_selfcal=False, **kwargs)
    return (deconvolve_model_imagelist, residual_imagelist, restore_imagelist, skymodel_list)


def spectral_line_imaging_skymodel_list_rsexecute_workflow(vis_list, model_imagelist, context,
                                                           continuum_model_imagelist=None,
                                                           vis_slices=1, facets=1, gcfcf=None, **kwargs):
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
        vis_list = predict_list_rsexecute_workflow(vis_list, continuum_model_imagelist, context=context, gcfcf=gcfcf,
                                                   vis_slices=vis_slices, **kwargs)
    
    return continuum_imaging_skymodel_list_rsexecute_workflow(vis_list, model_imagelist, context=context, gcfcf=gcfcf,
                                                              vis_slices=vis_slices, facets=facets, **kwargs)
