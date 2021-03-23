__all__ = [
    "predict_skymodel_list_rsexecute_workflow",
    "restore_skymodel_list_rsexecute_workflow",
    "restore_centre_skymodel_list_rsexecute_workflow",
    "invert_skymodel_list_rsexecute_workflow",
    "deconvolve_skymodel_list_rsexecute_workflow",
]

import logging

import numpy

from rascil.processing_components import (
    copy_skycomponent,
    apply_beam_to_skycomponent,
    normalize_sumwt,
    extract_skycomponents_from_skymodel,
)
from rascil.processing_components.calibration import apply_gaintable
from rascil.processing_components.image import restore_cube, fit_psf
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.visibility import copy_visibility
# ToDo - remove non-SkyModel parts
from rascil.workflows.rsexecute import (
    invert_list_rsexecute_workflow,
    predict_list_rsexecute_workflow,
    subtract_list_rsexecute_workflow,
    zero_list_rsexecute_workflow,
    sum_invert_results_rsexecute,
    deconvolve_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.serial.imaging.imaging_serial import (
    invert_list_serial_workflow,
    predict_list_serial_workflow,
)
from rascil.workflows.shared.imaging import remove_sumwt

log = logging.getLogger("rascil-logger")


def predict_skymodel_list_rsexecute_workflow(
        obsvis, skymodel_list, context="ng", gcfcf=None, docal=False, inverse=True, **kwargs
):
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
        """Predict visibility for a skymodel

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
                    imgv = predict_list_serial_workflow(
                        [imgv], [model], context=context, gcfcf=[g], **kwargs
                    )[0]
                else:
                    imgv = predict_list_serial_workflow(
                        [imgv], [sm.image], context=context, gcfcf=[g], **kwargs
                    )[0]
                v["vis"].data += imgv["vis"].data
        
        if docal and sm.gaintable is not None:
            v = apply_gaintable(v, sm.gaintable, inverse=inverse)
        
        return v
    
    if isinstance(obsvis, list):
        if len(obsvis) != len(skymodel_list):
            raise ValueError("Obsvis and skymodel lists should have the same length")
        if gcfcf is None:
            return [
                rsexecute.execute(ft_cal_sm, nout=1)(obsvis[ism], sm, None)
                for ism, sm in enumerate(skymodel_list)
            ]
        else:
            return [
                rsexecute.execute(ft_cal_sm, nout=1)(obsvis[ism], sm, gcfcf[ism])
                for ism, sm in enumerate(skymodel_list)
            ]
    else:
        if gcfcf is None:
            return [
                rsexecute.execute(ft_cal_sm, nout=1)(obsvis, sm, None)
                for ism, sm in enumerate(skymodel_list)
            ]
        else:
            return [
                rsexecute.execute(ft_cal_sm, nout=1)(obsvis, sm, gcfcf[ism])
                for ism, sm in enumerate(skymodel_list)
            ]


def invert_skymodel_list_rsexecute_workflow(
        vis_list, skymodel_list, context="ng", gcfcf=None, docal=False, **kwargs
):
    """Calibrate and invert from a skymodel, iterating over the skymodel

    :param vis_list: List of Visibility data models
    :param skymodel_list: skymodel list
    :param context: Imaging context 2d or ng
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of (image, weight) tuples)
    """
    
    def ift_ical_sm(v, sm, g):
        """Inverse Fourier sum of visibility to image and components

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
        
        result = invert_list_serial_workflow(
            [v], [sm.image], context=context, gcfcf=[g], **kwargs
        )[0]
        if sm.mask is not None:
            result[0]["pixels"].data *= sm.mask["pixels"].data
        
        return result
    
    if gcfcf is None:
        return [
            rsexecute.execute(ift_ical_sm, nout=1)(vis_list[i], sm, None)
            for i, sm in enumerate(skymodel_list)
        ]
    else:
        return [
            rsexecute.execute(ift_ical_sm, nout=1)(vis_list[i], sm, gcfcf[i])
            for i, sm in enumerate(skymodel_list)
        ]


def restore_centre_skymodel_list_rsexecute_workflow(
        skymodel_list, psf_imagelist, residual_imagelist=None, **kwargs
):
    """Create a graph to calculate the restored skymodel at the centre channel

    :param skymodel_list: Skymodel list (or graph)
    :param psf_imagelist: PSF list (or graph)
    :param residual_imagelist: Residual list (or graph)
    :param kwargs: Parameters for functions in components
    :return: list of restored images (or graph)
    """
    assert len(skymodel_list) == len(psf_imagelist)
    if residual_imagelist is not None:
        assert len(skymodel_list) == len(residual_imagelist)
    
    # Find the PSF by summing over all channels, fit to this psf
    psf = sum_invert_results_rsexecute(psf_imagelist)[0]
    cleanbeam = rsexecute.execute(fit_psf, nout=1)(psf)
    
    # Add the model over all channels
    centre = len(skymodel_list) // 2
    model = skymodel_list[centre].image
    
    if residual_imagelist is not None:
        # Get residual calculated across the band
        residual = sum_invert_results_rsexecute(residual_imagelist)[0]
        restored = rsexecute.execute(restore_cube, nout=1)(
            model, residual=residual, cleanbeam=cleanbeam, **kwargs
        )
    else:
        restored = rsexecute.execute(restore_cube, nout=1)(
            model, cleanbeam=cleanbeam, **kwargs
        )
    
    return restored


def restore_skymodel_list_rsexecute_workflow(
        skymodel_list, psf_imagelist, residual_imagelist=None, **kwargs
):
    """Create a graph to calculate the restored skymodel

    :param skymodel_list: Skymodel list (or graph)
    :param psf_imagelist: PSF list (or graph)
    :param residual_imagelist: Residual list (or graph)
    :param kwargs: Parameters for functions in components
    :return: list of restored images (or graph)
    """
    assert len(skymodel_list) == len(psf_imagelist)
    if residual_imagelist is not None:
        assert len(skymodel_list) == len(residual_imagelist)
    
    psf_list = sum_invert_results_rsexecute(psf_imagelist)
    psf = rsexecute.execute(normalize_sumwt)(psf_list[0], psf_list[1])
    cleanbeam = rsexecute.execute(fit_psf)(psf)
    
    if residual_imagelist is not None:
        residual_list = rsexecute.execute(remove_sumwt, nout=len(residual_imagelist))(
            residual_imagelist
        )
        restored_list = [
            rsexecute.execute(restore_cube, nout=1)(
                skymodel_list[i].image,
                cleanbeam=cleanbeam,
                residual=residual_list[i],
                **kwargs
            )
            for i, _ in enumerate(skymodel_list)
        ]
    else:
        restored_list = [
            rsexecute.execute(restore_cube, nout=1)(
                skymodel_list[i].image, cleanbeam=cleanbeam, residual=None, **kwargs
            )
            for i, _ in enumerate(skymodel_list)
        ]
    
    return restored_list


def residual_skymodel_list_rsexecute_workflow(
        vis, model_imagelist, context="ng", skymodel_list=None, gcfcf=None, **kwargs
):
    """Create a graph to calculate residual image for a skymodel_list

    :param vis: List of vis (or graph)
    :param model_imagelist: Model used to determine image parameters
    :param context: Imaging context e.g. '2d', 'wstack'
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: list of (image, sumwt) tuples or graph
    """
    model_vis = zero_list_rsexecute_workflow(vis)
    
    if skymodel_list is not None:
        model_vis = predict_skymodel_list_rsexecute_workflow(
            model_vis, skymodel_list, context=context, gcfcf=gcfcf, docal=True, **kwargs
        )
    else:
        model_vis = predict_list_rsexecute_workflow(
            model_vis, model_imagelist, context=context, gcfcf=gcfcf, **kwargs
        )
    residual_vis = subtract_list_rsexecute_workflow(vis, model_vis)
    
    if skymodel_list is not None:
        result = invert_skymodel_list_rsexecute_workflow(
            residual_vis, skymodel_list, gcfcf=gcfcf, docal=True, dopsf=False, **kwargs
        )
    else:
        result = invert_list_rsexecute_workflow(
            residual_vis,
            model_imagelist,
            context=context,
            dopsf=False,
            normalize=True,
            gcfcf=gcfcf,
            **kwargs
        )
    return rsexecute.optimize(result)


def deconvolve_skymodel_list_rsexecute_workflow(
        dirty_image_list, psf_list, skymodel_list, prefix="", **kwargs
):
    """

    :param dirty_image_list:
    :param psf_list:
    :param skymodel_list:
    :param prefix:
    :param kwargs:
    :return:
    """
    
    # Set the skymodel image and then if the model is not fixed extract skycomponents
    def update_skymodel(sm, im):
        sm.image = im
        if not sm.fixed:
            sm = extract_skycomponents_from_skymodel(sm, **kwargs)
        return sm
    
    deconvolve_model_imagelist = [sm.image for sm in skymodel_list]
    
    deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(
        dirty_image_list, psf_list, deconvolve_model_imagelist, prefix=prefix, **kwargs
    )
    
    # Now recreate the sky models
    skymodel_list = [
        rsexecute.execute(update_skymodel, nout=1)(skymodel_list[i], m)
        for i, m in enumerate(deconvolve_model_imagelist)
    ]
    
    return skymodel_list
