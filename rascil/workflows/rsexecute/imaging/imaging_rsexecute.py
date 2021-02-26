"""Workflows for imaging, including predict, invert, residual, restore, deconvolve, weight, taper, zero, subtract and sum results from invert

"""

__all__ = ['predict_list_rsexecute_workflow', 'invert_list_rsexecute_workflow', 'residual_list_rsexecute_workflow',
           'restore_list_rsexecute_workflow', 'deconvolve_list_rsexecute_workflow',
           'deconvolve_list_channel_rsexecute_workflow', 'weight_list_rsexecute_workflow',
           'taper_list_rsexecute_workflow', 'zero_list_rsexecute_workflow', 'subtract_list_rsexecute_workflow',
           'sum_invert_results_rsexecute', 'sum_predict_results_rsexecute',
           'restore_rsexecute_workflow']

import collections
import copy
import logging

import numpy

from rascil.data_models.parameters import get_parameter
from rascil.processing_components import calculate_image_frequency_moments
from rascil.processing_components import create_empty_image_like
from rascil.processing_components import create_griddata_from_image
from rascil.processing_components import deconvolve_cube, restore_cube, create_image_from_array
from rascil.processing_components import \
    grid_blockvisibility_weight_to_griddata, griddata_blockvisibility_reweight, \
    griddata_merge_weights, fit_psf, normalize_sumwt
from rascil.processing_components import image_scatter_facets, image_gather_facets, \
    image_scatter_channels
from rascil.processing_components import taper_visibility_gaussian
from rascil.processing_components.visibility import copy_visibility
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.image import image_gather_channels_rsexecute
from rascil.workflows.shared import imaging_context, remove_sumwt, sum_predict_results, \
    threshold_list, sum_invert_results

log = logging.getLogger('rascil-logger')


def predict_list_rsexecute_workflow(vis_list, model_imagelist, context, gcfcf=None, **kwargs):
    """Predict, iterating over both the scattered vis_list and image
    
    The visibility and image are scattered, the visibility is predicted on each part, and then the
    parts are assembled.

    :param vis_list: list of vis (or graph)
    :param model_imagelist: list of models (or graph)
    :param context: Type of processing e.g. 2d, ng, facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists

    For example::

        dprepb_model = [rsexecute.execute(create_low_test_image_from_gleam)
            (npixel=npixel, frequency=[frequency[f]], channel_bandwidth=[channel_bandwidth[f]],
            cellsize=cellsize, phasecentre=phasecentre, polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=3.0, applybeam=True)
            for f, freq in enumerate(frequency)]

        dprepb_model_list = rsexecute.persist(dprepb_model_list)
        predicted_vis_list = predict_list_rsexecute_workflow(vis_list, model_imagelist=dprepb_model_list,
            context='wstack', vis_slices=51)
        predicted_vis_list = rsexecute.compute(predicted_vis_list , sync=True)

   """
    
    # Predict_2d does not clear the vis so we have to do it here.
    vis_list = zero_list_rsexecute_workflow(vis_list)
    
    c = imaging_context(context)
    predict = c['predict']
    
    # Loop over all windows
    assert len(model_imagelist) == len(vis_list)
    predict_results = [rsexecute.execute(predict, pure=True, nout=1)
                       (vis, model_imagelist[ivis], gcfcf=gcfcf, **kwargs)
                       for ivis, vis in enumerate(vis_list)]
    
    return rsexecute.optimize(predict_results)


def invert_list_rsexecute_workflow(vis_list, template_model_imagelist, context, dopsf=False,
                                   normalize=True, gcfcf=None, **kwargs):
    """ Sum results from invert, iterating over the scattered image and vis_list

    :param gcfcf:
    :param vis_list: list of vis (or graph)
    :param template_model_imagelist: list of template models (or graph)
    :param dopsf: Make the PSF instead of the dirty image
    :param normalize: Normalize by sumwt
    :param context: Imaging context
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: List of (image, sumwt) tuples, one per vis in vis_list

    For example::

        model_list = [rsexecute.execute(create_image_from_visibility)
            (v, npixel=npixel, cellsize=cellsize, polarisation_frame=pol_frame)
            for v in vis_list]

        model_list = rsexecute.persist(model_list)
        dirty_list = invert_list_rsexecute_workflow(vis_list, template_model_imagelist=model_list, context='wstack',
                                                    vis_slices=51)
        dirty_sumwt_list = rsexecute.compute(dirty_list, sync=True)
        dirty, sumwt = dirty_sumwt_list[centre]

   """
    if not isinstance(template_model_imagelist, collections.abc.Iterable):
        template_model_imagelist = [template_model_imagelist]
    
    c = imaging_context(context)
    invert = c['invert']
    
    # Loop over all vis_lists independently
    assert len(template_model_imagelist) == len(vis_list)
    invert_results = [rsexecute.execute(invert, nout=2)(vis, template_model_imagelist[ivis], dopsf=dopsf,
                                                        normalise=normalize, gcfcf=gcfcf, **kwargs)
                      for ivis, vis in enumerate(vis_list)]
    
    return rsexecute.optimize(invert_results)


def residual_list_rsexecute_workflow(vis, model_imagelist, context='2d', gcfcf=None, **kwargs):
    """ Create a graph to calculate residual image

    :param vis: List of vis (or graph)
    :param model_imagelist: Model used to determine image parameters
    :param context: Imaging context e.g. '2d', 'wstack'
    :param gcfcg: tuple containing grid correction and convolution function
    :param kwargs: Parameters for functions in components
    :return: list of (image, sumwt) tuples or graph
    """
    model_vis = zero_list_rsexecute_workflow(vis)
    model_vis = predict_list_rsexecute_workflow(model_vis, model_imagelist, context=context, gcfcf=gcfcf, **kwargs)
    residual_vis = subtract_list_rsexecute_workflow(vis, model_vis)
    result = invert_list_rsexecute_workflow(residual_vis, model_imagelist, context=context, dopsf=False, normalize=True,
                                            gcfcf=gcfcf, **kwargs)
    return rsexecute.optimize(result)


def restore_list_rsexecute_workflow(model_imagelist, psf_imagelist, residual_imagelist=None, restore_facets=1,
                                    restore_overlap=0, restore_taper='tukey', **kwargs):
    """ Create a graph to calculate the restored image

    This restores each frequency plane using an cleanbeam fitted from the frequency-summed PSF
    The output is a cube for each frequency. Note that the noise in the residual is that
    (correctly) that for each frequency.

    This will not give any information on the spectral behaviour

    :param model_imagelist: Model list (or graph)
    :param psf_imagelist: PSF list (or graph)
    :param residual_imagelist: Residual list (or graph)
    :param kwargs: Parameters for functions in components
    :return: list of restored images (or graph)
    """
    assert len(model_imagelist) == len(psf_imagelist)
    if residual_imagelist is not None:
        assert len(model_imagelist) == len(residual_imagelist)
    
    psf_list = sum_invert_results_rsexecute(psf_imagelist)
    psf = rsexecute.execute(normalize_sumwt)(psf_list[0], psf_list[1])
    cleanbeam = rsexecute.execute(fit_psf)(psf)
    
    if residual_imagelist is not None:
        residual_list = rsexecute.execute(remove_sumwt, nout=len(residual_imagelist))(residual_imagelist)
        restored_list = [rsexecute.execute(restore_cube, nout=1)(model_imagelist[i], cleanbeam=cleanbeam,
                                                                 residual=residual_list[i], **kwargs)
                         for i, _ in enumerate(model_imagelist)]
    else:
        restored_list = [rsexecute.execute(restore_cube, nout=1)(model_imagelist[i], cleanbeam=cleanbeam,
                                                                 residual=None, **kwargs)
                         for i, _ in enumerate(model_imagelist)]
    
    return rsexecute.optimize(restored_list)


def restore_rsexecute_workflow(model_imagelist, psf_imagelist, residual_imagelist=None, restore_facets=1,
                               restore_overlap=0, restore_taper='tukey', **kwargs):
    """ Create a graph to calculate the restored image
    
    This does the following:
    - Takes the centre frequency slice of the model
    - Integrates the residual across the band
    - Fits to the band-integrated PSF
    - Restores the model, cleanbeam, and residual
    
    This will not give any information on the spectral behaviour

    :param model_imagelist: Model list (or graph)
    :param psf_imagelist: PSF list (or graph)
    :param residual_imagelist: Residual list (or graph)
    :param kwargs: Parameters for functions in components
    :return: list of restored images (or graph)
    """
    assert len(model_imagelist) == len(psf_imagelist)
    if residual_imagelist is not None:
        assert len(model_imagelist) == len(residual_imagelist)
    
    # Find the PSF by summing over all channels, fit to this psf
    psf = sum_invert_results_rsexecute(psf_imagelist)[0]
    cleanbeam = rsexecute.execute(fit_psf, nout=1)(psf)
    
    # Add the model over all channels
    centre = len(model_imagelist) // 2
    model = model_imagelist[centre]
    
    if residual_imagelist is not None:
        residual = sum_invert_results_rsexecute(residual_imagelist)[0]
        restored = rsexecute.execute(restore_cube, nout=1)(model, residual=residual, cleanbeam=cleanbeam,
                                                           **kwargs)
    else:
        restored = rsexecute.execute(restore_cube, nout=1)(model, cleanbeam=cleanbeam, **kwargs)
    
    return restored


def deconvolve_list_singlefacet_rsexecute_workflow(dirty_list, psf_list, model_imagelist, prefix='', mask=None,
                                                   **kwargs):
    """Create a graph for deconvolution, adding to the model

    :param dirty_list: list of dirty images (or graph)
    :param psf_list: list of psfs (or graph)
    :param model_imagelist: list of models (or graph)
    :param prefix: Informative prefix to log messages
    :param mask: Mask for deconvolution
    :param kwargs: Parameters for functions
    :return: graph for the deconvolution

    For example::

        dirty_imagelist = invert_list_rsexecute_workflow(vis_list, model_imagelist, context='2d',
                                                          dopsf=False, normalize=True)
        psf_imagelist = invert_list_rsexecute_workflow(vis_list, model_imagelist, context='2d',
                                                        dopsf=True, normalize=True)
        dirty_imagelist = rsexecute.persist(dirty_imagelist)
        psf_imagelist = rsexecute.persist(psf_imagelist)
        dec_imagelist = deconvolve_list_rsexecute_workflow(dirty_imagelist, psf_imagelist,
                model_imagelist, niter=1000, fractional_threshold=0.01,
                scales=[0, 3, 10], algorithm='mmclean', nmoment=3, nchan=freqwin,
                threshold=0.1, gain=0.7)
        dec_imagelist = rsexecute.persist(dec_imagelist)

    """
    
    nchan = len(dirty_list)
    # Number of moments. 1 is the sum.
    nmoment = get_parameter(kwargs, "nmoment", 1)
    
    # Now do the deconvolution for a single facet.
    def deconvolve(dirty, psf, model, gthreshold, msk):
        
        log.info("deconvolve_list_rsexecute_workflow: Starting clean")
        
        if nmoment > 0:
            moment0 = calculate_image_frequency_moments(dirty)
            this_peak = numpy.max(numpy.abs(moment0["pixels"].data[0, ...])) / dirty["pixels"].data.shape[0]
        else:
            ref_chan = dirty["pixels"].data.shape[0] // 2
            this_peak = numpy.max(numpy.abs(dirty["pixels"].data[ref_chan, ...]))
        
        if this_peak > 1.1 * gthreshold:
            kwargs['threshold'] = gthreshold
            result = deconvolve_cube(dirty, psf, prefix=prefix, mask=msk, **kwargs)
            
            assert result["pixels"].data.shape == model["pixels"].data.shape
            result["pixels"].data = result["pixels"].data + model["pixels"].data
            return result
        else:
            return model.copy(deep=True)
    
    dirty_cube = image_gather_channels_rsexecute([dirty_list[chan][0] for chan in range(nchan)])
    psf_cube = image_gather_channels_rsexecute([psf_list[chan][0] for chan in range(nchan)])
    model_cube = image_gather_channels_rsexecute([model_imagelist[chan] for chan in range(nchan)])
    
    # Work out the threshold. Need to find global peak over all dirty_list images
    threshold = get_parameter(kwargs, "threshold", 0.0)
    nmoment = get_parameter(kwargs, "nmoment", 1)
    
    clean_cube = \
        rsexecute.execute(deconvolve, nout=nchan)(dirty_cube, psf_cube, model_cube, threshold, msk=mask)
    clean_cube = rsexecute.execute(image_scatter_channels, nout=nchan)(clean_cube)
    
    return clean_cube


def deconvolve_list_rsexecute_workflow(dirty_list, psf_list, model_imagelist, prefix='', mask=None,
                                       **kwargs):
    """Create a graph for deconvolution, adding to the model

    :param sc_list: Single component list for all frequency bands
    :param dirty_list: list of dirty images (or graph)
    :param psf_list: list of psfs (or graph)
    :param model_imagelist: list of models (or graph)
    :param prefix: Informative prefix to log messages
    :param mask: Mask for deconvolution
    :param kwargs: Parameters for functions
    :return: graph for the deconvolution

    For example::

        dirty_imagelist = invert_list_rsexecute_workflow(vis_list, model_imagelist, context='2d',
                                                          dopsf=False, normalize=True)
        psf_imagelist = invert_list_rsexecute_workflow(vis_list, model_imagelist, context='2d',
                                                        dopsf=True, normalize=True)
        dirty_imagelist = rsexecute.persist(dirty_imagelist)
        psf_imagelist = rsexecute.persist(psf_imagelist)
        dec_imagelist = deconvolve_list_rsexecute_workflow(dirty_imagelist, psf_imagelist,
                model_imagelist, niter=1000, fractional_threshold=0.01,
                scales=[0, 3, 10], algorithm='mmclean', nmoment=3, nchan=freqwin,
                threshold=0.1, gain=0.7)
        dec_imagelist = rsexecute.persist(dec_imagelist)

    """
    
    nchan = len(dirty_list)
    
    deconvolve_facets = get_parameter(kwargs, 'deconvolve_facets', 1)
    
    if deconvolve_facets == 1:
        return deconvolve_list_singlefacet_rsexecute_workflow(dirty_list, psf_list, model_imagelist, prefix=prefix,
                                                              mask=None, **kwargs)
    
    deconvolve_overlap = get_parameter(kwargs, 'deconvolve_overlap', 0)
    deconvolve_taper = get_parameter(kwargs, 'deconvolve_taper', None)
    if deconvolve_facets > 1 and deconvolve_overlap > 0:
        deconvolve_number_facets = (deconvolve_facets - 2) ** 2
    else:
        deconvolve_number_facets = deconvolve_facets ** 2
    
    scattered_channels_facets_model_list = \
        [rsexecute.execute(image_scatter_facets, nout=deconvolve_number_facets)
         (m, facets=deconvolve_facets, overlap=deconvolve_overlap, taper=deconvolve_taper)
         for m in model_imagelist]
    scattered_facets_model_list = [
        image_gather_channels_rsexecute
        ([scattered_channels_facets_model_list[chan][facet] for chan in range(nchan)])
        for facet in range(deconvolve_number_facets)]
    
    # Scatter the separate channel images into deconvolve facets and then gather channels for each facet.
    # This avoids constructing the entire spectral cube.
    # i.e. SCATTER BY FACET then SCATTER BY CHANNEL
    scattered_channels_facets_dirty_list = \
        [rsexecute.execute(image_scatter_facets, nout=deconvolve_number_facets)(d[0], facets=deconvolve_facets,
                                                                                overlap=deconvolve_overlap,
                                                                                taper=deconvolve_taper)
         for d in dirty_list]
    scattered_facets_dirty_list = [image_gather_channels_rsexecute([scattered_channels_facets_dirty_list[chan][facet]
                                                                    for chan in range(nchan)])
                                   for facet in range(deconvolve_number_facets)]
    
    def extract_psf(psf, facets):
        assert not numpy.isnan(numpy.sum(psf["pixels"].data)), "NaNs present in PSF"
        cx = psf["pixels"].shape[3] // 2
        cy = psf["pixels"].shape[2] // 2
        wx = psf["pixels"].shape[3] // facets
        wy = psf["pixels"].shape[2] // facets
        xbeg = cx - wx // 2
        xend = cx + wx // 2
        ybeg = cy - wy // 2
        yend = cy + wy // 2
        spsf_data = psf["pixels"].isel({"x": slice(xbeg, xend), "y": slice(ybeg, yend)}).data
        wcs = copy.deepcopy(psf.image_acc.wcs)
        wcs.wcs.crpix[0] -= xbeg
        wcs.wcs.crpix[1] -= ybeg
        spsf = create_image_from_array(spsf_data, wcs=wcs, polarisation_frame=psf.image_acc.polarisation_frame)
        return spsf
    
    psf_list_trimmed = [rsexecute.execute(extract_psf)(p[0], deconvolve_facets) for p in psf_list]
    
    psf_centre = image_gather_channels_rsexecute([psf_list_trimmed[chan] for chan in range(nchan)])
    
    # Work out the threshold. Need to find global peak over all dirty_list images
    threshold = get_parameter(kwargs, "threshold", 0.0)
    fractional_threshold = get_parameter(kwargs, "fractional_threshold", 0.1)
    nmoment = get_parameter(kwargs, "nmoment", 1)
    use_moment0 = nmoment > 0
    
    # Find the global threshold. This uses the peak in the average on the frequency axis since we
    # want to use it in a stopping criterion in a moment clean
    global_threshold = rsexecute.execute(threshold_list, nout=1)(scattered_facets_dirty_list, threshold,
                                                                 fractional_threshold,
                                                                 use_moment0=use_moment0, prefix=prefix)
    kwargs["threshold"] = global_threshold
    
    # Do the deconvolution for each facet in turn. Each item of the scattered_results_list
    # contains the clean image cube and lists of list components (a number for each channel)
    if mask is None:
        scattered_results_list = [deconvolve_list_singlefacet_rsexecute_workflow([(d, 0.0)],
                                                                                 [(psf_centre, 0.0)],
                                                                                 [m],
                                                                                 prefix=prefix, msk=None, **kwargs)
                                  for d, m in zip(scattered_facets_dirty_list, scattered_facets_model_list)]
    else:
        mask_list = \
            rsexecute.execute(image_scatter_facets, nout=deconvolve_number_facets)(mask,
                                                                                   facets=deconvolve_facets,
                                                                                   overlap=deconvolve_overlap)
        scattered_results_list = [deconvolve_list_singlefacet_rsexecute_workflow([(d, 0.0)],
                                                                                 [(psf_centre, 0.0)],
                                                                                 [m],
                                                                                 prefix=prefix, msk=msk, **kwargs)
                                  for d, m, msk in
                                  zip(scattered_facets_dirty_list, scattered_facets_model_list, mask_list)]
    
    # We want to avoid constructing the entire cube so we do the inverse of how we got here:
    # i.e. SCATTER BY CHANNEL then GATHER BY FACET
    # Gather the results back into one image, correcting for overlaps as necessary. The taper function is is used to
    # feather the facets together
    
    return [rsexecute.execute(image_gather_facets, nout=1)([scattered_results_list[facet][chan]
                                                            for facet in range(deconvolve_number_facets)],
                                                           model_imagelist[chan], facets=deconvolve_facets,
                                                           overlap=deconvolve_overlap)
            for chan in range(nchan)]


def deconvolve_list_channel_rsexecute_workflow(dirty_list, psf_list, model_imagelist, subimages, **kwargs):
    """Create a graph for deconvolution by channels, adding to the model

    Does deconvolution channel by channel.

    :param dirty_list: list or graph of dirty images
    :param psf_list: list or graph of psf images. The psfs must be the size of a facet
    :param model_imagelist: list of graph of models
    :param subimages: Number of channels to split into
    :param kwargs: Parameters for functions in components
    :return: list of updated models (or graphs)
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
    
    output = rsexecute.execute(create_empty_image_like, nout=1, pure=True)(model_imagelist)
    dirty_lists = rsexecute.execute(image_scatter_channels, nout=subimages, pure=True)(dirty_list[0],
                                                                                       subimages=subimages)
    results = [rsexecute.execute(deconvolve_subimage)(dirty_list, psf_list[0])
               for dirty_list in dirty_lists]
    result = image_gather_channels_rsexecute(results, output)
    result = rsexecute.execute(add_model, nout=1, pure=True)(result, model_imagelist)
    return rsexecute.optimize(result)


def weight_list_rsexecute_workflow(vis_list, model_imagelist, gcfcf=None, weighting='uniform', robustness=0.0,
                                   **kwargs):
    """ Weight the visibility data
    
    This is done collectively so the weights are summed over all vis_lists and then
    corrected

    :param vis_list:
    :param model_imagelist: Model required to determine weighting parameters
    :param weighting: Type of weighting
    :param kwargs: Parameters for functions in graphs
    :return: List of vis_graphs

    For example::

         vis_list = weight_list_rsexecute_workflow(vis_list, model_list, weighting='uniform')

   """
    
    def grid_wt(vis, model):
        if vis is not None:
            if model is not None:
                griddata = create_griddata_from_image(model,
                                                      polarisation_frame=vis.blockvisibility_acc.polarisation_frame)
                griddata = grid_blockvisibility_weight_to_griddata(vis, griddata)
                
                return griddata
            else:
                return None
        else:
            return None
    
    weight_list = [rsexecute.execute(grid_wt, pure=True, nout=1)(vis_list[i], model_imagelist[i])
                   for i in range(len(vis_list))]
    
    merged_weight_grid = rsexecute.execute(griddata_merge_weights, nout=1)(weight_list)
    merged_weight_grid = rsexecute.persist(merged_weight_grid, broadcast=True)
    
    def re_weight(vis, model, gd, g):
        if gd is not None:
            if vis is not None:
                # Ensure that the griddata has the right axes so that the convolution
                # function mapping works
                agd = create_griddata_from_image(model, polarisation_frame=vis.blockvisibility_acc.polarisation_frame)
                agd["pixels"].data = gd[0]["pixels"].data
                vis = griddata_blockvisibility_reweight(vis, agd, weighting=weighting, robustness=robustness)
                return vis
            else:
                return None
        else:
            return vis
    
    result = [rsexecute.execute(re_weight, nout=1)
              (v, model_imagelist[i], merged_weight_grid, gcfcf)
              for i, v in enumerate(vis_list)]
    
    return rsexecute.optimize(result)


def taper_list_rsexecute_workflow(vis_list, size_required):
    """Taper to desired size
    
    :param vis_list: List of vis (or graph)
    :param size_required: Size in radians
    :return: List of vis (or graph)
    """
    result = [rsexecute.execute(taper_visibility_gaussian, nout=1)(v, beam=size_required) for v in vis_list]
    return rsexecute.optimize(result)


def zero_list_rsexecute_workflow(vis_list):
    """ Creates a new vis_list and initialises all to zero

    :param vis_list: List of vis (or graph)
    :return: List of vis (or graph)
   """
    
    def zero(vis):
        if vis is not None:
            zerovis = copy_visibility(vis, zero=True)
            return zerovis
        else:
            return None
    
    result = [rsexecute.execute(zero, pure=True, nout=1)(v) for v in vis_list]
    return rsexecute.optimize(result)


def subtract_list_rsexecute_workflow(vis_list, model_vislist):
    """ Initialise vis to zero

    :param vis_list: List of vis (or graph)
    :param model_vislist: Model to be subtracted (or graph)
    :return: List of vis or graph
   """
    
    def subtract_vis(vis, model_vis):
        if vis is not None and model_vis is not None:
            assert vis.vis.shape == model_vis.vis.shape
            subvis = copy_visibility(vis)
            subvis['vis'].data[...] -= model_vis['vis'].data[...]
            return subvis
        else:
            return None
    
    result = [rsexecute.execute(subtract_vis, pure=True, nout=1)(vis=vis_list[i],
                                                                 model_vis=model_vislist[i])
              for i in range(len(vis_list))]
    return rsexecute.optimize(result)


def sum_predict_results_rsexecute(bvis_list, split=2):
    """ Sum a set of predict results

    :param bvis_list: List of (image, sum weights) tuples
    :param split: Split into
    :return: BlockVis
    """
    if len(bvis_list) > split:
        centre = len(bvis_list) // split
        result = [sum_predict_results_rsexecute(bvis_list[:centre]),
                  sum_predict_results_rsexecute(bvis_list[centre:])]
        return rsexecute.execute(sum_predict_results, nout=2)(result)
    else:
        return rsexecute.execute(sum_predict_results, nout=2)(bvis_list)


def sum_invert_results_rsexecute(image_list, split=2):
    """ Sum a set of invert results with appropriate weighting

    :param image_list: List of (image, sum weights) tuples
    :param split: Split into
    :return: image, sum of weights
    """
    if len(image_list) > split:
        centre = len(image_list) // split
        result = [sum_invert_results_rsexecute(image_list[:centre]), sum_invert_results_rsexecute(image_list[centre:])]
        return rsexecute.execute(sum_invert_results, nout=2)(result)
    else:
        return rsexecute.execute(sum_invert_results, nout=2)(image_list)
