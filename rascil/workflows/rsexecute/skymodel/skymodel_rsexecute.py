__all__ = ['predict_skymodel_list_rsexecute_workflow',
           'restore_skymodel_list_rsexecute_workflow',
           'crosssubtract_datamodels_skymodel_list_rsexecute_workflow',
           'convolve_skymodel_list_rsexecute_workflow',
           'invert_skymodel_list_rsexecute_workflow']

import logging

import numpy

from rascil.processing_components import image_scatter_facets, image_gather_facets
from rascil.processing_components.calibration import apply_gaintable
from rascil.processing_components.image import restore_cube
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.skycomponent import copy_skycomponent, apply_beam_to_skycomponent, insert_skycomponent
from rascil.processing_components.visibility import copy_visibility
# ToDo - remove non-SkyModel parts
from rascil.workflows.rsexecute import invert_list_rsexecute_workflow, \
    predict_list_rsexecute_workflow, subtract_list_rsexecute_workflow, \
    zero_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.serial.imaging.imaging_serial import invert_list_serial_workflow, predict_list_serial_workflow
from rascil.workflows.shared.imaging import remove_sumwt

log = logging.getLogger('rascil-logger')


def predict_skymodel_list_rsexecute_workflow(obsvis, skymodel_list, context='ng', gcfcf=None,
                                             docal=False, inverse=True, **kwargs):
    """Predict from a list of skymodels

    If obsvis is a list then we pair obsvis element and skymodel_list element and predict
    If obvis is BlockVisibility then we calculate BlockVisibility for each skymodel

    :param obsvis: Observed Block Visibility or list or graph
    :param skymodel_list: skymodel list
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
   """
    
    def ft_cal_sm(ov, sm, g):
        """ Predict visibility for a skymodel

        :param sm: Skymodel
        :param g: Convolution function
        :param ov: Input visibility
        :return: Visibility with dft of components, fft of image, gaintable
        """
        if g is not None:
            if len(g) != 2:
                raise ValueError("Convolution function value incorrect")
        
        v = copy_visibility(ov, zero=True)
        
        if len(sm.components) > 0:
            if sm.mask is not None:
                comps = copy_skycomponent(sm.components)
                comps = apply_beam_to_skycomponent(comps, sm.mask)
                v = dft_skycomponent_visibility(v, comps, **kwargs)
            else:
                v = dft_skycomponent_visibility(v, sm.components, **kwargs)
        
        if sm.image is not None:
            if numpy.max(numpy.abs(sm.image["pixels"].data)) > 0.0:
                imgv = copy_visibility(ov, zero=True)
                if sm.mask is not None:
                    model = sm.image.copy(deep=True)
                    model["pixels"].data *= sm.mask["pixels"].data
                    imgv = predict_list_serial_workflow([imgv], [model], context=context, gcfcf=[g], **kwargs)[0]
                else:
                    imgv = predict_list_serial_workflow([imgv], [sm.image], context=context, gcfcf=[g], **kwargs)[0]
                v['vis'].data += imgv['vis'].data
        
        if docal and sm.gaintable is not None:
            v = apply_gaintable(v, sm.gaintable, inverse=inverse)
        
        return v
    
    if isinstance(obsvis, list):
        if len(obsvis) != len(skymodel_list):
            raise ValueError("Obsvis and skymodel lists should have the same length")
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


def invert_skymodel_list_rsexecute_workflow(vis_list, skymodel_list, context='ng',
                                            gcfcf=None, docal=False, **kwargs):
    """Calibrate and invert from a skymodel, iterating over the skymodel

    The visibility and image are scattered, the visibility is predicted and calibrated on each part, and then the
    parts are assembled. The mask if present, is multiplied in at the end.

    :param vis_list: List of Visibility data models
    :param skymodel_list: skymodel list
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of (image, weight) tuples)
   """
    
    def ift_ical_sm(v, sm, g):
        """ Inverse Fourier sum of visibility to image and components

        :param v: Visibility to be transformed
        :param sm: Skymodel
        :param g: Convolution function
        :return: Skymodel containing transforms
        """
        if g is not None:
            if len(g) != 2:
                raise ValueError("Convolution function value incorrect")
        
        if docal and sm.gaintable is not None:
            v = apply_gaintable(v, sm.gaintable)
        
        result = invert_list_serial_workflow([v], [sm.image], context=context, gcfcf=[g],
                                             **kwargs)[0]
        if sm.mask is not None:
            result[0]["pixels"].data *= sm.mask["pixels"].data
        
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
    restore_facets = 1
    
    assert len(skymodel_list) == len(psf_imagelist)
    if residual_imagelist is not None:
        assert len(skymodel_list) == len(residual_imagelist)
    
    if restore_facets % 2 == 0 or restore_facets == 1:
        actual_number_facets = restore_facets
    else:
        actual_number_facets = max(1, (restore_facets - 1))
    
    psf_list = rsexecute.execute(remove_sumwt, nout=len(psf_imagelist))(psf_imagelist)
    
    def skymodel_scatter_facets(sm, facets, overlap, taper):
        im = sm.image.copy(deep=True)
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
    :param modelvis_list: List of BlockVisibility data model predictions
    :return: List of (image, weight) tuples)
   """
    
    # Now do the meaty part. We probably want to refactor this for performance once it works.
    def vsum(ov, mv):
        # Observed vis minus the sum of all predictions
        verr = copy_visibility(ov)
        for m in mv:
            verr['vis'].data -= m['vis'].data
        # Now add back each model in turn
        result = list()
        for m in mv:
            vr = copy_visibility(verr)
            vr['vis'].data += m['vis'].data
            result.append(vr)
        assert len(result) == len(mv)
        return result
    
    return rsexecute.execute(vsum, nout=len(modelvis_list))(obsvis, modelvis_list)


def convolve_skymodel_list_rsexecute_workflow(obsvis, skymodel_list, context='ng', gcfcf=None, **kwargs):
    """Form residual image from observed visibility and a set of skymodel without calibration

    This is similar to convolving the skymodel images with the PSF

    :param vis_list: List of BlockVisibility data models
    :param skymodel_list: skymodel list
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of (image, weight) tuples)
   """
    
    def ft_ift_sm(ov, sm, g):
        # assert isinstance(ov, BlockVisibility), ov
        # assert isinstance(sm, SkyModel), sm
        if g is not None:
            assert len(g) == 2, g
            # assert isinstance(g[0], Image), g[0]
            # assert isinstance(g[1], ConvolutionFunction), g[1]
        
        v = copy_visibility(ov)
        
        v['vis'].data[...] = 0.0 + 0.0j
        
        if len(sm.components) > 0:
            
            if sm.mask is not None:
                comps = copy_skycomponent(sm.components)
                comps = apply_beam_to_skycomponent(comps, sm.mask)
                v = dft_skycomponent_visibility(v, comps)
            else:
                v = dft_skycomponent_visibility(v, sm.components)
        
        if sm.image is not None:
            if numpy.max(numpy.abs(sm.image["pixels"].data)) > 0.0:
                if sm.mask is not None:
                    model = sm.image.copy(deep=True)
                    model["pixels"].data *= sm.mask["pixels"].data
                else:
                    model = sm.image
                v = predict_list_serial_workflow([v], [model], context=context, gcfcf=[g], **kwargs)[0]
        
        result = invert_list_serial_workflow([v], [sm.image], context=context, gcfcf=[g], **kwargs)[0]
        if sm.mask is not None:
            result[0]["pixels"].data *= sm.mask["pixels"].data
        return result
    
    if gcfcf is None:
        return [rsexecute.execute(ft_ift_sm, nout=len(skymodel_list))(obsvis, sm, None)
                for ism, sm in enumerate(skymodel_list)]
    else:
        return [rsexecute.execute(ft_ift_sm, nout=len(skymodel_list))(obsvis, sm, gcfcf[ism])
                for ism, sm in enumerate(skymodel_list)]


def residual_skymodel_list_rsexecute_workflow(vis, model_imagelist, context='ng', skymodel_list=None, gcfcf=None,
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
        model_vis = predict_skymodel_list_rsexecute_workflow(model_vis, skymodel_list, context=context, gcfcf=gcfcf,
                                                             docal=True, **kwargs)
    else:
        model_vis = predict_list_rsexecute_workflow(model_vis, model_imagelist, context=context, gcfcf=gcfcf, **kwargs)
    residual_vis = subtract_list_rsexecute_workflow(vis, model_vis)
    
    if skymodel_list is not None:
        result = invert_skymodel_list_rsexecute_workflow(residual_vis, skymodel_list, gcfcf=gcfcf, docal=True,
                                                         dopsf=False, **kwargs)
    else:
        result = invert_list_rsexecute_workflow(residual_vis, model_imagelist, context=context, dopsf=False,
                                                normalize=True, gcfcf=gcfcf, **kwargs)
    return rsexecute.optimize(result)
