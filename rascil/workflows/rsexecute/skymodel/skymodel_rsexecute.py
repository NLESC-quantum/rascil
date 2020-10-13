__all__ = ['predict_skymodel_list_rsexecute_workflow',
           'predict_skymodel_list_compsonly_rsexecute_workflow',
           'restore_skymodel_list_rsexecute_workflow',
           'crosssubtract_datamodels_skymodel_list_rsexecute_workflow',
           'convolve_skymodel_list_rsexecute_workflow',
           'invert_skymodel_list_rsexecute_workflow']

import logging

import numpy

from rascil.data_models.memory_data_models import Image, GainTable, Visibility, SkyModel, \
    ConvolutionFunction, BlockVisibility
from rascil.processing_components.calibration import apply_gaintable
from rascil.processing_components.image import copy_image
from rascil.processing_components.image import image_scatter_facets, image_gather_facets
from rascil.processing_components.image import restore_cube
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.skycomponent import copy_skycomponent, apply_beam_to_skycomponent, insert_skycomponent
from rascil.processing_components.visibility import copy_visibility, convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
# ToDo - remove non-SkyModel parts
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import invert_list_rsexecute_workflow, \
    predict_list_rsexecute_workflow, subtract_list_rsexecute_workflow, \
    zero_list_rsexecute_workflow
from rascil.workflows.serial.imaging import predict_list_serial_workflow, invert_list_serial_workflow
from rascil.workflows.shared.imaging import remove_sumwt

log = logging.getLogger('logger')


def predict_skymodel_list_rsexecute_workflow(obsvis, skymodel_list, context, vis_slices=1, facets=1,
                                             gcfcf=None, docal=False, **kwargs):
    """Predict from a list of skymodels, producing one visibility per skymodel

    :param obsvis: "Observed Visibility"
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    def ft_cal_sm(ov, sm, g):
        assert isinstance(ov, BlockVisibility), ov
        assert isinstance(sm, SkyModel), sm
        if g is not None:
            assert len(g) == 2, g
            assert isinstance(g[0], Image), g[0]
            assert isinstance(g[1], ConvolutionFunction), g[1]
        
        v = copy_visibility(ov, zero=True)

        if len(sm.components) > 0:
            dftv = copy_visibility(ov, zero=True)
            if isinstance(sm.mask, Image):
                comps = copy_skycomponent(sm.components)
                comps = apply_beam_to_skycomponent(comps, sm.mask)
                dftv = dft_skycomponent_visibility(dftv, comps)
            else:
                dftv = dft_skycomponent_visibility(dftv, sm.components)
            v.data['vis'] += dftv.vis
        
        if isinstance(sm.image, Image):
            if numpy.max(numpy.abs(sm.image.data)) > 0.0:
                imgv = copy_visibility(ov, zero=True)
                if isinstance(sm.mask, Image):
                    model = copy_image(sm.image)
                    model.data *= sm.mask.data
                    imgv = predict_list_serial_workflow([imgv], [model], context=context,
                                                     vis_slices=vis_slices, facets=facets, gcfcf=[g],
                                                     **kwargs)[0]
                else:
                    imgv = predict_list_serial_workflow([imgv], [sm.image], context=context,
                                                     vis_slices=vis_slices, facets=facets, gcfcf=[g],
                                                     **kwargs)[0]
                v.data['vis'] += imgv.vis
        
        if docal and isinstance(sm.gaintable, GainTable):
            v = apply_gaintable(v, sm.gaintable, inverse=True)
       
        return v
    
    if isinstance(obsvis, list):
        assert len(obsvis) == len(skymodel_list)
        if gcfcf is None:
            return [rsexecute.execute(ft_cal_sm, nout=1)(obsvis[ism], sm, None) for ism, sm in enumerate(skymodel_list)]
        else:
            return [rsexecute.execute(ft_cal_sm, nout=1)(obsvis[ism], sm, gcfcf[ism]) for ism, sm in
                    enumerate(skymodel_list)]
    else:
        if gcfcf is None:
            return [rsexecute.execute(ft_cal_sm, nout=1)(obsvis, sm, None) for ism, sm in enumerate(skymodel_list)]
        else:
            return [rsexecute.execute(ft_cal_sm, nout=1)(obsvis, sm, gcfcf[ism]) for ism, sm in
                    enumerate(skymodel_list)]


def predict_skymodel_list_compsonly_rsexecute_workflow(obsvis, skymodel_list, docal=False, **kwargs):
    """Predict from a list of component-only skymodels, producing one visibility per skymodel
    
    This is an optimised version of predict_skymodel_list_rsexecute_workflow, working on block
    visibilities and ignoring the image in a skymodel

    :param obsvis: "Observed Block Visibility"
    :param skymodel_list: skymodel list
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    def ft_cal_sm(obv, sm):
        assert isinstance(obv, BlockVisibility), obv
        bv = copy_visibility(obv)
        
        bv.data['vis'][...] = 0.0 + 0.0j
        
        assert len(sm.components) > 0
        
        if isinstance(sm.mask, Image):
            comps = copy_skycomponent(sm.components)
            comps = apply_beam_to_skycomponent(comps, sm.mask)
            bv = dft_skycomponent_visibility(bv, comps)
        else:
            bv = dft_skycomponent_visibility(bv, sm.components)
        
        if docal and isinstance(sm.gaintable, GainTable):
            bv = apply_gaintable(bv, sm.gaintable, inverse=True)
        
        return bv
    
    return [rsexecute.execute(ft_cal_sm, nout=1)(obsvis, sm) for sm in skymodel_list]


def invert_skymodel_list_rsexecute_workflow(vis_list, skymodel_list, context, vis_slices=1, facets=1,
                                            gcfcf=None, docal=False, **kwargs):
    """Calibrate and invert from a skymodel, iterating over the skymodel

    The visibility and image are scattered, the visibility is predicted and calibrated on each part, and then the
    parts are assembled. The mask if present, is multiplied in at the end.

    :param vis_list: List of Visibility data models
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of (image, weight) tuples)
   """
    
    def ift_ical_sm(v, sm, g):
        assert isinstance(v, Visibility) or isinstance(v, BlockVisibility), v
        assert isinstance(sm, SkyModel), sm
        if g is not None:
            assert len(g) == 2, g
            assert isinstance(g[0], Image), g[0]
            assert isinstance(g[1], ConvolutionFunction), g[1]
        
        if docal and isinstance(sm.gaintable, GainTable):
            v = apply_gaintable(v, sm.gaintable)
            
        result = invert_list_serial_workflow([v], [sm.image], context=context,
                                             vis_slices=vis_slices, facets=facets, gcfcf=[g],
                                             **kwargs)[0]
        if isinstance(sm.mask, Image):
            result[0].data *= sm.mask.data
        
        return result
    
    if gcfcf is None:
        return [rsexecute.execute(ift_ical_sm, nout=1)(vis_list[i], sm, None)
                for i, sm in enumerate(skymodel_list)]
    else:
        return [rsexecute.execute(ift_ical_sm, nout=1)(vis_list[i], sm, gcfcf[i])
                for i, sm in enumerate(skymodel_list)]


def restore_skymodel_list_rsexecute_workflow(skymodel_list, psf_imagelist, residual_imagelist=None, restore_facets=1,
                                             restore_overlap=0, restore_taper='tukey', **kwargs):
    """ Create a graph to calculate the restored skymodel

    :param skymodel_list: Skymodel list (or graph)
    :param psf_imagelist: PSF list (or graph)
    :param residual_imagelist: Residual list (or graph)
    :param kwargs: Parameters for functions in components
    :param restore_facets: Number of facets used per axis (used to distribute)
    :param restore_overlap: Overlap in pixels (0 is best)
    :param restore_taper: Type of taper between facets
    :return: list of restored images (or graph)
    """
    restore_facets=1

    assert len(skymodel_list) == len(psf_imagelist)
    if residual_imagelist is not None:
        assert len(skymodel_list) == len(residual_imagelist)
    
    if restore_facets % 2 == 0 or restore_facets == 1:
        actual_number_facets = restore_facets
    else:
        actual_number_facets = max(1, (restore_facets - 1))
    
    psf_list = rsexecute.execute(remove_sumwt, nout=len(psf_imagelist))(psf_imagelist)
    
    def skymodel_scatter_facets(sm, facets, overlap, taper):
        im = copy_image(sm.image)
        im = insert_skycomponent(im, sm.components, **kwargs)
        return image_scatter_facets(im, facets, overlap, taper)
    
    # Scatter each list element into a list. We will then run restore_cube on each
    facet_model_list = [rsexecute.execute(skymodel_scatter_facets, nout=actual_number_facets * actual_number_facets)
                        (sm, facets=restore_facets, overlap=restore_overlap, taper=restore_taper)
                        for sm in skymodel_list]
    facet_psf_list = [rsexecute.execute(image_scatter_facets, nout=actual_number_facets * actual_number_facets)
                      (psf, facets=restore_facets, overlap=restore_overlap, taper=restore_taper)
                      for psf in psf_list]
    
    if residual_imagelist is not None:
        residual_list = rsexecute.execute(remove_sumwt, nout=len(residual_imagelist))(residual_imagelist)
        facet_residual_list = [
            rsexecute.execute(image_scatter_facets, nout=actual_number_facets * actual_number_facets)
            (residual, facets=restore_facets, overlap=restore_overlap, taper=restore_taper)
            for residual in residual_list]
        facet_restored_list = [[rsexecute.execute(restore_cube, nout=actual_number_facets * actual_number_facets)
                                (model=facet_model_list[i][im], psf=facet_psf_list[i][im],
                                 residual=facet_residual_list[i][im],
                                 **kwargs)
                                for im, _ in enumerate(facet_model_list[i])] for i, _ in enumerate(skymodel_list)]
    else:
        facet_restored_list = [[rsexecute.execute(restore_cube, nout=actual_number_facets * actual_number_facets)
                                (model=facet_model_list[i][im], psf=facet_psf_list[i][im],
                                 **kwargs)
                                for im, _ in enumerate(facet_model_list[i])] for i, _ in enumerate(skymodel_list)]
    
    def skymodel_gather_facets(restored, sm, facets, overlap, taper):
        return image_gather_facets(restored, sm.image, facets, overlap, taper)
    
    # Now we run restore_cube on each and gather the results across all facets
    restored_imagelist = [rsexecute.execute(skymodel_gather_facets)
                          (facet_restored_list[i], skymodel_list[i], facets=restore_facets,
                           overlap=restore_overlap, taper=restore_taper)
                          for i, _ in enumerate(skymodel_list)]
    
    return rsexecute.optimize(restored_imagelist)


def crosssubtract_datamodels_skymodel_list_rsexecute_workflow(obsvis, modelvis_list):
    """Form data models by subtracting sum from the observed and adding back each model in turn
    
    vmodel[p] = vobs - sum(i!=p) modelvis[i]
    
    This is the E step in the Expectation-Maximisation algorithm.

    :param obsvis: "Observed" visibility
    :param modelvis_list: List of Visibility data model predictions
    :return: List of (image, weight) tuples)
   """
    
    # Now do the meaty part. We probably want to refactor this for performance once it works.
    def vsum(ov, mv):
        # Observed vis minus the sum of all predictions
        verr = copy_visibility(ov)
        for m in mv:
            verr.data['vis'] -= m.data['vis']
        # Now add back each model in turn
        result = list()
        for m in mv:
            vr = copy_visibility(verr)
            vr.data['vis'] += m.data['vis']
            result.append(vr)
        assert len(result) == len(mv)
        return result
    
    return rsexecute.execute(vsum, nout=len(modelvis_list))(obsvis, modelvis_list)


def convolve_skymodel_list_rsexecute_workflow(obsvis, skymodel_list, context, vis_slices=1, facets=1,
                                              gcfcf=None, **kwargs):
    """Form residual image from observed visibility and a set of skymodel without calibration

    This is similar to convolving the skymodel images with the PSF

    :param vis_list: List of Visibility data models
    :param skymodel_list: skymodel list
    :param vis_slices: Number of vis slices (w stack or timeslice)
    :param facets: Number of facets (per axis)
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of (image, weight) tuples)
   """
    
    def ft_ift_sm(ov, sm, g):
        assert isinstance(ov, Visibility) or isinstance(ov, BlockVisibility), ov
        assert isinstance(sm, SkyModel), sm
        if g is not None:
            assert len(g) == 2, g
            assert isinstance(g[0], Image), g[0]
            assert isinstance(g[1], ConvolutionFunction), g[1]
        
        v = copy_visibility(ov)
        
        v.data['vis'][...] = 0.0 + 0.0j
        
        if len(sm.components) > 0:
            
            if isinstance(sm.mask, Image):
                comps = copy_skycomponent(sm.components)
                comps = apply_beam_to_skycomponent(comps, sm.mask)
                v = dft_skycomponent_visibility(v, comps)
            else:
                v = dft_skycomponent_visibility(v, sm.components)
        
        if isinstance(sm.image, Image):
            if numpy.max(numpy.abs(sm.image.data)) > 0.0:
                if isinstance(sm.mask, Image):
                    model = copy_image(sm.image)
                    model.data *= sm.mask.data
                else:
                    model = sm.image
                v = predict_list_serial_workflow([v], [model], context=context,
                                                 vis_slices=vis_slices, facets=facets, gcfcf=[g],
                                                 **kwargs)[0]
        
        assert isinstance(sm.image, Image), sm.image
        
        result = invert_list_serial_workflow([v], [sm.image], context=context,
                                             vis_slices=vis_slices, facets=facets, gcfcf=[g],
                                             **kwargs)[0]
        if isinstance(sm.mask, Image):
            result[0].data *= sm.mask.data
        return result
    
    if gcfcf is None:
        return [rsexecute.execute(ft_ift_sm, nout=len(skymodel_list))(obsvis, sm, None)
                for ism, sm in enumerate(skymodel_list)]
    else:
        return [rsexecute.execute(ft_ift_sm, nout=len(skymodel_list))(obsvis, sm, gcfcf[ism])
                for ism, sm in enumerate(skymodel_list)]


def residual_skymodel_list_rsexecute_workflow(vis, model_imagelist, context='2d', skymodel_list=None, gcfcf=None,
                                              **kwargs):
    """ Create a graph to calculate residual image

    :param vis: List of vis (or graph)
    :param model_imagelist: Model used to determine image parameters
    :param context: Imaging context e.g. '2d', 'wstack'
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: list of (image, sumwt) tuples or graph
    """
    model_vis = zero_list_rsexecute_workflow(vis)
    
    if skymodel_list is not None:
        # skymodel_list = rsexecute.scatter(skymodel_list)
        model_vis = predict_skymodel_list_rsexecute_workflow(model_vis, skymodel_list,
                                                             context=context,
                                                             gcfcf=gcfcf, docal=True, **kwargs)
    else:
        model_vis = predict_list_rsexecute_workflow(model_vis, model_imagelist, context=context,
                                                    gcfcf=gcfcf, **kwargs)
    residual_vis = subtract_list_rsexecute_workflow(vis, model_vis)
    
    if skymodel_list is not None:
        result = invert_skymodel_list_rsexecute_workflow(residual_vis, skymodel_list,
                                                         context=context, dopsf=False, docal=True,
                                                         gcfcf=gcfcf,
                                                         **kwargs)
    else:
        result = invert_list_rsexecute_workflow(residual_vis, model_imagelist, dopsf=False, normalize=True,
                                                context=context,
                                                gcfcf=gcfcf, **kwargs)
    return rsexecute.optimize(result)
