__all__ = [
    "predict_skymodel_list_rsexecute_workflow",
    "restore_skymodel_list_rsexecute_workflow",
    "restore_centre_skymodel_list_rsexecute_workflow",
    "invert_skymodel_list_rsexecute_workflow",
    "deconvolve_skymodel_list_rsexecute_workflow",
]

import logging

import numpy

from rascil.data_models import get_parameter
from rascil.processing_components import (
    copy_skycomponent,
    apply_beam_to_skycomponent,
    normalise_sumwt,
    extract_skycomponents_from_skymodel,
    image_scatter_facets,
    image_gather_facets,
)
from rascil.processing_components.calibration import apply_gaintable
from rascil.processing_components.image import restore_cube, fit_psf
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.visibility import (
    copy_visibility,
    concatenate_visibility,
)
from rascil.processing_components.skycomponent.operations import restore_skycomponent

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
    remove_sumwt,
    invert_list_serial_workflow,
    predict_list_serial_workflow,
)

log = logging.getLogger("rascil-logger")


def predict_skymodel_list_rsexecute_workflow(
    obsvis,
    skymodel_list,
    context="ng",
    docal=False,
    inverse=True,
    get_pb=None,
    **kwargs
):
    """Predict from a list of skymodels

    If obsvis is a list then we pair obsvis element and skymodel_list element and predict
    If obvis is BlockVisibility then we calculate BlockVisibility for each skymodel

    The function get_pb should have the signature:

        get_pb(BlockVisibility, Image)

    and should return the primary beam for the blockvisibility e.g. using average
    parallactic angle

    :param obsvis: Observed Block Visibility or list or graph
    :param skymodel_list: skymodel list
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param docal: Apply calibration table in skymodel
    :param get_pb: Function to get the primary beam for a given image and vis
    :param kwargs: Parameters for functions in components
    :return: List of vis_lists
    """

    def skymodel_predict_calibrate(ov, sm):
        """Predict visibility for a skymodel

        :param sm: Skymodel
        :param ov: Input visibility
        :return: Visibility with dft of components, fft of image, gaintable
        """
        v = copy_visibility(ov, zero=True)

        vis_slices = []
        if get_pb is not None:
            for time, vis_slice in v.groupby("time", squeeze=False):

                pb = get_pb(vis_slice, sm.image)

                # First do the DFT or the components
                if len(sm.components) > 0:
                    if sm.mask is not None or pb is not None:
                        comps = copy_skycomponent(sm.components)
                        if sm.mask is not None:
                            comps = apply_beam_to_skycomponent(comps, sm.mask)
                        if pb is not None:
                            comps = apply_beam_to_skycomponent(comps, pb)
                        vis_slice = dft_skycomponent_visibility(
                            vis_slice, comps, **kwargs
                        )
                    else:
                        vis_slice = dft_skycomponent_visibility(
                            vis_slice, sm.components, **kwargs
                        )

                # Now do the FFT of the image, after multiplying by the mask and primary
                # beam
                if sm.image is not None:
                    if numpy.max(numpy.abs(sm.image["pixels"].data)) > 0.0:
                        imgv = copy_visibility(vis_slice, zero=True)
                        if sm.mask is not None or pb is not None:
                            model = sm.image.copy(deep=True)
                            if sm.mask is not None:
                                model["pixels"].data *= sm.mask["pixels"].data
                            if pb is not None:
                                model["pixels"].data *= pb["pixels"].data
                            imgv = predict_list_serial_workflow(
                                [imgv], [model], context=context, **kwargs
                            )[0]
                        else:
                            imgv = predict_list_serial_workflow(
                                [imgv], [sm.image], context=context, **kwargs
                            )[0]
                        vis_slice["vis"].data += imgv["vis"].data

                vis_slices.append(vis_slice)

            v = concatenate_visibility(vis_slices, "time")

            if docal and sm.gaintable is not None:
                v = apply_gaintable(v, sm.gaintable, inverse=inverse)

            return v

        else:

            # First do the DFT or the components
            if len(sm.components) > 0:
                if sm.mask is not None:
                    comps = copy_skycomponent(sm.components)
                    comps = apply_beam_to_skycomponent(comps, sm.mask)
                    v = dft_skycomponent_visibility(v, comps, **kwargs)
                else:
                    v = dft_skycomponent_visibility(v, sm.components, **kwargs)

            # Now do the FFT of the image, after multiplying by the mask and primary
            # beam
            if sm.image is not None:
                if numpy.max(numpy.abs(sm.image["pixels"].data)) > 0.0:
                    imgv = copy_visibility(v, zero=True)
                    if sm.mask is not None:
                        model = sm.image.copy(deep=True)
                        model["pixels"].data *= sm.mask["pixels"].data
                        imgv = predict_list_serial_workflow(
                            [imgv], [model], context=context, **kwargs
                        )[0]
                    else:
                        imgv = predict_list_serial_workflow(
                            [imgv], [sm.image], context=context, **kwargs
                        )[0]
                    v["vis"].data += imgv["vis"].data

            if docal and sm.gaintable is not None:
                v = apply_gaintable(v, sm.gaintable, inverse=inverse)

            return v

    if isinstance(obsvis, list):
        if len(obsvis) != len(skymodel_list):
            raise ValueError("Obsvis and skymodel lists should have the same length")
        return [
            rsexecute.execute(skymodel_predict_calibrate, nout=1)(obsvis[ism], sm)
            for ism, sm in enumerate(skymodel_list)
        ]
    else:
        return [
            rsexecute.execute(skymodel_predict_calibrate, nout=1)(obsvis, sm)
            for ism, sm in enumerate(skymodel_list)
        ]


def invert_skymodel_list_rsexecute_workflow(
    vis_list,
    skymodel_list,
    context="ng",
    docal=False,
    get_pb=None,
    normalise=True,
    flat_sky=False,
    **kwargs
):
    """Calibrate and invert from a skymodel, iterating over the skymodel

    The function get_pb should have the signature:

        get_pb(BlockVisibility, Image)

    and should return the primary beam for the blockvisibility.

    The return is a graph for a set of tuples of (dirty, sensitivity image)

    :param vis_list: List of Visibility data models
    :param skymodel_list: skymodel list
    :param context: Imaging context 2d or ng
    :param get_pb: Function to get a primary beam
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of (image, weight) tuples)
    """

    def skymodel_calibrate_invert(v, sm):
        """Inverse Fourier sum of visibility to image and components

        :param v: Visibility to be transformed
        :param sm: Skymodel
        :return: Skymodel containing transforms
        """

        if docal and sm.gaintable is not None:
            v = apply_gaintable(v, sm.gaintable)

        if sm.image is None:
            raise ValueError("skymodel image is None")

        sum_flats = sm.image.copy(deep=True)
        sum_flats["pixels"][...] = 0.0
        sum_dirtys = sm.image.copy(deep=True)
        sum_dirtys["pixels"][...] = 0.0

        if get_pb is not None:
            for time, vis_slice in v.groupby("time", squeeze=False):

                pb = get_pb(vis_slice, sm.image)

                # Just do a straightforward invert for just this blockvis
                # and then apply the mask and primary beam if present
                # The return value result contains the weighted image and
                # the weights as an image (including mask and primary beam)
                result = invert_list_serial_workflow(
                    [vis_slice], [sm.image], context=context, normalise=False, **kwargs
                )[0]
                flat = numpy.ones_like(result[0]["pixels"].data)
                if sm.mask is not None:
                    flat *= sm.mask["pixels"].data
                if pb is not None:
                    flat *= pb["pixels"].data

                # We need to apply the flat to the dirty image
                sum_dirtys["pixels"].data += flat * result[0]["pixels"].data
                # The sum_flats should contain the weights and the square of the PB
                sum_flats["pixels"].data += (
                    flat * flat * result[1][:, :, numpy.newaxis, numpy.newaxis]
                )
            if normalise:
                sum_dirtys = normalise_sumwt(sum_dirtys, sum_flats, flat_sky=flat_sky)
                sum_flats["pixels"].data = numpy.sqrt(sum_flats["pixels"].data)

            return (sum_dirtys, sum_flats)

        else:
            result = invert_list_serial_workflow(
                [v], [sm.image], context=context, **kwargs
            )[0]
            if sm.mask is not None:
                result[0]["pixels"].data *= sm.mask["pixels"].data

            return result

    return [
        rsexecute.execute(skymodel_calibrate_invert, nout=1)(vis_list[i], sm)
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
        skymodel_list[centre], residual, clean_beam=clean_beam
    )

    return restored


def _check_imagelist_lengths(psf_imagelist, residual_imagelist, skymodel_list):
    """Check that the various image lists are congruent

    Raise ValueError when in error

    :param psf_imagelist:
    :param residual_imagelist:
    :param skymodel_list:
    """
    if len(skymodel_list) != len(psf_imagelist):
        errmsg = "Skymodel and psf list have different lengths"
        log.error(errmsg)
        raise ValueError(errmsg)
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
    :param kwargs: Parameters for functions in components
    :param restore_facets: Number of facets used per axis (used to distribute)
    :param restore_overlap: Overlap in pixels (0 is best)
    :param restore_taper: Type of taper between facets
    :return: list of restored images (or graph)
    """
    _check_imagelist_lengths(psf_imagelist, residual_imagelist, skymodel_list)

    if restore_overlap < 0:
        raise ValueError("Number of pixels for restore overlap must be >= 0")

    if restore_facets % 2 == 0 or restore_facets == 1:
        actual_number_facets = restore_facets
    else:
        actual_number_facets = max(1, (restore_facets - 1))

    if clean_beam is None:
        clean_beam_list = sum_invert_results_rsexecute(psf_imagelist)
        psf = rsexecute.execute(normalise_sumwt)(clean_beam_list[0], clean_beam_list[1])
        clean_beam = rsexecute.execute(fit_psf)(psf)
        kwargs["clean_beam"] = clean_beam

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
    return rsexecute.optimize((restored_imagelist, clean_beam))


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
    :param kwargs:
    :return: list of skymodels (or graph)
    """

    if fit_skymodel:
        # Now recreate the sky models
        # Set the skymodel image and then if the model is not fixed extract skycomponents
        def skymodel_update_components(sm, d):
            if not sm.fixed:
                sm = extract_skycomponents_from_skymodel(sm, d, **kwargs)
            return sm

        skymodel_list = [
            rsexecute.execute(skymodel_update_components, nout=1)(
                skymodel_list[i], dirty[0]
            )
            for i, dirty in enumerate(dirty_image_list)
        ]
        return skymodel_list

    else:
        deconvolve_model_imagelist = [sm.image for sm in skymodel_list]

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
        return skymodel_list
